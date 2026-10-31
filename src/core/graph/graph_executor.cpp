#include "../executor/ambient_turn.hpp"
#include "../providers/provider_ops.hpp"
#include "../utils/utils.hpp"
#include "graph_internal.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace agent {

namespace {

using ScopedIndex = std::set<std::string, std::less<>>;

constexpr size_t kPayloadMax = 200;

std::string clip(std::string_view text) {
    if (text.size() <= kPayloadMax) return std::string(text);
    std::string out(text.substr(0, kPayloadMax));
    out += "...";
    return out;
}

void log_event(std::vector<GraphEvent> &events, GraphEventType type, std::string_view node, std::string_view key,
               std::string_view payload) {
    events.push_back(GraphEvent{
        .type = type, .node = std::string(node), .key = std::string(key), .timestamp_ms = graph::now_ms(), .payload = clip(payload)});
}

std::span<const uint8_t> bytes_of(std::string_view s) { return {reinterpret_cast<const uint8_t *>(s.data()), s.size()}; }

const GraphNode *find_node(const Graph &graph, std::string_view name) {
    for (const auto &node : graph.nodes)
        if (node.name == name) return &node;
    return nullptr;
}

const GraphEdge *find_edge(const Graph &graph, std::string_view from) {
    for (const auto &edge : graph.edges)
        if (edge.from == from) return &edge;
    return nullptr;
}

Result<void> invalid(const Graph &graph, std::string message) {
    return fail(ErrorCode::InvalidConfig, "graph '" + graph.name + "': " + std::move(message));
}

Result<void> validate(const Graph &graph) {
    if (graph.entry.empty()) return invalid(graph, "no entry node");
    if (!find_node(graph, graph.entry)) return invalid(graph, "entry node '" + graph.entry + "' does not exist");
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const GraphNode &node = graph.nodes[i];
        if (node.name.empty() || node.name == kGraphEnd) return invalid(graph, "invalid node name '" + node.name + "'");
        for (size_t j = 0; j < i; ++j)
            if (graph.nodes[j].name == node.name) return invalid(graph, "duplicate node name '" + node.name + "'");
        if (!node.exit_to.empty() && !find_node(graph, node.exit_to))
            return invalid(graph, "node '" + node.name + "' exits to unknown node '" + node.exit_to + "'");
        switch (node.kind) {
        case GraphNodeKind::Agent: break;
        case GraphNodeKind::Function:
            if (!node.function) return invalid(graph, "function node '" + node.name + "' has no function");
            break;
        case GraphNodeKind::Tool:
            if (node.tool_name.empty()) return invalid(graph, "tool node '" + node.name + "' has no tool_name");
            break;
        case GraphNodeKind::Parallel:
            if (node.branches.empty()) return invalid(graph, "parallel node '" + node.name + "' has no branches");
            for (const auto &b : node.branches)
                if (!find_node(graph, b)) return invalid(graph, "parallel node '" + node.name + "' branch '" + b + "' does not exist");
            break;
        case GraphNodeKind::Loop:
            if (!find_node(graph, node.loop_body)) return invalid(graph, "loop node '" + node.name + "' has no valid loop_body");
            if (node.max_iterations < 1) return invalid(graph, "loop node '" + node.name + "' needs max_iterations >= 1");
            break;
        }
    }
    for (size_t i = 0; i < graph.edges.size(); ++i) {
        const GraphEdge &edge = graph.edges[i];
        if (!find_node(graph, edge.from)) return invalid(graph, "edge from unknown node '" + edge.from + "'");
        if (!edge.router && edge.to != kGraphEnd && !find_node(graph, edge.to))
            return invalid(graph, "edge '" + edge.from + "' -> unknown node '" + edge.to + "'");
        for (size_t j = 0; j < i; ++j)
            if (graph.edges[j].from == edge.from)
                return invalid(graph, "node '" + edge.from + "' has more than one outgoing edge (use a Parallel node to fan out)");
    }
    return ok();
}

// Execution context of one chain of nodes: the top-level run or one parallel branch.
struct Exec {
    Session &session;
    const Graph &graph;
    std::string_view thread_id;
    GraphState &state;
    std::vector<GraphEvent> &events;
    const Exec *parent = nullptr;                      // enclosing chain (branches only)
    std::vector<std::string> *branch_writes = nullptr; // every key this branch wrote (branches only)
    ScopedIndex *scoped_index = nullptr;               // top level only: persists user:/app: writes
    ModelRegistry *models = nullptr;                   // GraphRun::models (GraphNode::model)
};

// Approval for the tool calls an interrupt_before_tools Agent node paused on.
struct PendingResume {
    std::vector<ToolCall> calls;
    bool approved = false;
    std::string human_input;
};

struct NodeResult {
    bool interrupted = false;
    std::string reason;
    std::vector<ToolCall> pending;
    std::string forced_next; // Loop: jump into the body, bypassing edges
};

// The Agent node's conversation in GraphState::histories (history_key, else the node name).
const std::string &history_name(const GraphNode &node) { return node.history_key.empty() ? node.name : node.history_key; }

FlowMemory &history_for(Exec &x, const std::string &node) {
    auto it = x.state.histories.find(node);
    if (it != x.state.histories.end()) return it->second;
    FlowMemory memory;
    // Branches start from the enclosing chain's history (read-only there; maps are not mutated
    // while branches run).
    for (const Exec *p = x.parent; p; p = p->parent) {
        auto found = p->state.histories.find(node);
        if (found != p->state.histories.end()) {
            memory = found->second;
            break;
        }
    }
    if (memory.id.empty()) memory.id = std::string(x.thread_id) + ":" + node;
    return x.state.histories.emplace(node, std::move(memory)).first->second;
}

// Processes the keys written since the last drain: log, record for branch conflict checks, persist
// user:/app: keys at the top level.
Result<void> drain_writes(Exec &x, std::string_view node) {
    std::vector<std::string> written = std::move(x.state.written);
    x.state.written.clear();
    for (const auto &key : written) {
        std::string_view value = get_state(x.state, key);
        log_event(x.events, GraphEventType::StateWrite, node, key, value);
        if (x.branch_writes) x.branch_writes->push_back(key);
        else if (x.scoped_index && graph::is_scoped_key(key)) {
            auto put = graph::persist_scoped_value(x.session, key, value, *x.scoped_index);
            if (!put.ok) return put;
        }
    }
    return ok();
}

// ---- Agent nodes ----

struct ToolLogCtx {
    std::vector<GraphEvent> *events;
    std::string_view node;
};

void log_tool_pre(Session &, FlowMemory &, std::string_view tool, std::string_view args, std::string_view, void *ud) {
    auto *ctx = static_cast<ToolLogCtx *>(ud);
    log_event(*ctx->events, GraphEventType::ToolCall, ctx->node, tool, args);
}

void log_tool_post(Session &, FlowMemory &, std::string_view tool, std::string_view, std::string_view result, void *ud) {
    auto *ctx = static_cast<ToolLogCtx *>(ud);
    log_event(*ctx->events, GraphEventType::ToolResult, ctx->node, tool, result);
}

bool pause_before_tools(Session &session, std::span<const ToolCallView> calls) {
    if (!session.config.permission_check) return true;
    for (const auto &c : calls)
        if (session.config.permission_check(c.name, c.arguments, session.config.permission_user_data) == PermissionDecision::Ask)
            return true;
    return false;
}

// Installs a node's system prompt for the duration of its turn(s); the history keeps whatever
// system message it had before.
struct SystemPromptScope {
    FlowMemory &memory;
    bool injected = false;
    std::optional<Message> saved;

    SystemPromptScope(FlowMemory &m, std::string_view prompt) : memory(m) {
        if (prompt.empty()) return;
        if (!memory.history.empty() && memory.history[0].role == "system") {
            saved = memory.history[0];
            memory.history[0].content = std::string(prompt);
            memory.history[0].token_count = -1;
        } else {
            Message sys;
            sys.role = "system";
            sys.content = std::string(prompt);
            memory.history.insert(memory.history.begin(), std::move(sys));
            injected = true;
        }
    }
    ~SystemPromptScope() {
        if (injected) {
            if (!memory.history.empty() && memory.history[0].role == "system") memory.history.erase(memory.history.begin());
        } else if (saved) {
            memory.history[0] = std::move(*saved);
            memory.history[0].token_count = -1;
        }
    }
    SystemPromptScope(const SystemPromptScope &) = delete;
    SystemPromptScope &operator=(const SystemPromptScope &) = delete;
};

struct AmbientScope {
    std::span<const std::string_view> allowed = ambient::allowed_tools;
    std::span<const ToolHook> hooks = ambient::active_hooks;
    ambient::ToolPauseFn pause = ambient::tool_pause_fn;
    bool paused = ambient::tools_paused;

    ~AmbientScope() {
        ambient::allowed_tools = allowed;
        ambient::active_hooks = hooks;
        ambient::tool_pause_fn = pause;
        ambient::tools_paused = paused;
    }
};

void fire_node_hooks(Session &session, FlowMemory &memory, const GraphNode &node, ToolHook::When when, const ToolCall &tc,
                     std::string_view result) {
    for (const auto &hook : node.hooks) {
        if (hook.when != when || !hook.callback) continue;
        if (!hook.tool_name.empty() && hook.tool_name != tc.name) continue;
        hook.callback(session, memory, tc.name, tc.arguments, result, hook.user_data);
    }
}

std::string run_approved_call(Exec &x, Session &session, FlowMemory &memory, const GraphNode &node, const ToolCall &tc,
                              const PendingResume &resume) {
    log_event(x.events, GraphEventType::ToolCall, node.name, tc.name, tc.arguments);
    trigger_event(session, EventType::OnToolCall, bytes_of(tc.name));

    PermissionDecision policy = PermissionDecision::Ask;
    if (session.config.permission_check)
        policy = session.config.permission_check(tc.name, tc.arguments, session.config.permission_user_data);

    std::string result;
    if (policy == PermissionDecision::Ask && !resume.approved) {
        result = "Error: tool call denied by human reviewer";
        if (!resume.human_input.empty()) result += ": " + resume.human_input;
    } else {
        fire_node_hooks(session, memory, node, ToolHook::When::Pre, tc, {});
        ambient::tool_call_preapproved = (policy == PermissionDecision::Ask);
        auto exec = execute_tool(session, tc.name, tc.arguments);
        ambient::tool_call_preapproved = false;
        result = exec.ok ? std::move(exec.value) : "Error: " + exec.error.message;
        fire_node_hooks(session, memory, node, ToolHook::When::Post, tc, result);
    }

    log_event(x.events, GraphEventType::ToolResult, node.name, tc.name, result);
    trigger_event(session, EventType::OnToolResult, bytes_of(result));
    memory.stats.tool_calls++;
    {
        std::lock_guard<std::mutex> lock(*session.stats_mutex);
        session.stats.total_tool_calls++;
    }
    return result;
}

Result<NodeResult> run_agent_node(Exec &x, const GraphNode &node, const PendingResume *resume) {
    if (node.interrupt_before_tools && x.parent)
        return fail<NodeResult>(ErrorCode::InvalidConfig,
                                "agent node '" + node.name + "': interrupt_before_tools is not supported inside parallel branches");

    std::string system_prompt = node.system_prompt;
    if (node.context_provider) {
        auto extra = node.context_provider(x.session, x.state, node.user_data);
        if (!extra.ok) return fail<NodeResult>(extra.error.code, extra.error.message);
        if (!extra.value.empty()) system_prompt = system_prompt.empty() ? std::move(extra.value) : extra.value + "\n" + system_prompt;
    }

    // GraphNode::model: the turn runs on that model's Session, leased for the whole node. Every lease
    // shares one Session, so turns on it are serialized.
    Session *session = &x.session;
    ModelLease lease;
    std::unique_lock<std::mutex> turn_lock;
    if (!node.model.empty()) {
        if (!x.models)
            return fail<NodeResult>(ErrorCode::InvalidConfig,
                                    "agent node '" + node.name + "' names model '" + node.model + "' but GraphRun::models is not set");
        auto acquired = acquire_model(*x.models, node.model);
        if (!acquired.ok) return fail<NodeResult>(acquired.error.code, "agent node '" + node.name + "': " + acquired.error.message);
        lease = std::move(acquired.value);
        session = lease.session;
        turn_lock = std::unique_lock<std::mutex>(*lease.turn_mutex);
    }

    FlowMemory &memory = history_for(x, history_name(node));
    std::string prompt = node.prompt;
    if (!node.input_key.empty() && has_state(x.state, node.input_key)) prompt = std::string(get_state(x.state, node.input_key));

    std::vector<std::string_view> allowed(node.allowed_tools.begin(), node.allowed_tools.end());
    ToolLogCtx log_ctx{&x.events, node.name};
    // The event-log hooks wrap the node's own hooks.
    std::vector<ToolHook> hooks;
    hooks.reserve(node.hooks.size() + 2);
    hooks.push_back(ToolHook{.when = ToolHook::When::Pre, .tool_name = {}, .callback = log_tool_pre, .user_data = &log_ctx});
    hooks.insert(hooks.end(), node.hooks.begin(), node.hooks.end());
    hooks.push_back(ToolHook{.when = ToolHook::When::Post, .tool_name = {}, .callback = log_tool_post, .user_data = &log_ctx});

    SystemPromptScope system_scope(memory, system_prompt);
    AmbientScope ambient_scope;
    ambient::allowed_tools = allowed;
    ambient::active_hooks = hooks;
    ambient::tool_pause_fn = node.interrupt_before_tools ? pause_before_tools : nullptr;
    ambient::tools_paused = false;

    Result<std::string> reply;
    if (resume) {
        // The paused assistant message is the last one in the history; answer each of its calls.
        for (const auto &tc : resume->calls) {
            Message tool_msg;
            tool_msg.role = "tool";
            tool_msg.content = run_approved_call(x, *session, memory, node, tc, *resume);
            tool_msg.tool_call_id = tc.id;
            memory.history.push_back(std::move(tool_msg));
        }
        reply = run_turn(*session, memory, "");
    } else {
        reply = run_turn(*session, memory, prompt);
    }
    if (!reply.ok) return fail<NodeResult>(reply.error.code, reply.error.message);

    NodeResult result;
    if (ambient::tools_paused) {
        result.interrupted = true;
        result.pending = memory.history.back().tool_calls;
        result.reason = "approval required for tool call:";
        for (const auto &tc : result.pending) result.reason += " " + tc.name;
        return ok(std::move(result));
    }
    if (!node.output_key.empty()) set_state(x.state, node.output_key, std::move(reply.value));
    return ok(std::move(result));
}

// ---- Function / Tool / Loop nodes ----

Result<NodeResult> run_function_node(Exec &x, const GraphNode &node) {
    x.state.interrupt_requested = false;
    auto res = node.function(x.session, x.state, node.user_data);
    const bool interrupt = x.state.interrupt_requested;
    std::string reason = std::move(x.state.interrupt_reason);
    x.state.interrupt_requested = false;
    x.state.interrupt_reason.clear();
    if (!res.ok) return fail<NodeResult>(res.error.code, res.error.message);

    NodeResult result;
    if (interrupt) {
        if (x.parent)
            return fail<NodeResult>(ErrorCode::InvalidConfig,
                                    "function node '" + node.name + "': interrupts are not supported inside parallel branches");
        result.interrupted = true;
        result.reason = std::move(reason);
    }
    return ok(std::move(result));
}

Result<NodeResult> run_tool_node(Exec &x, const GraphNode &node) {
    std::string args = "{}";
    if (node.tool_args) {
        auto built = node.tool_args(x.session, x.state, node.user_data);
        if (!built.ok) return fail<NodeResult>(built.error.code, built.error.message);
        args = std::move(built.value);
    } else if (!node.input_key.empty() && has_state(x.state, node.input_key)) {
        args = std::string(get_state(x.state, node.input_key));
    }

    log_event(x.events, GraphEventType::ToolCall, node.name, node.tool_name, args);
    trigger_event(x.session, EventType::OnToolCall, bytes_of(node.tool_name));
    auto exec = execute_tool(x.session, node.tool_name, args);
    if (!exec.ok) {
        log_event(x.events, GraphEventType::ToolResult, node.name, node.tool_name, "Error: " + exec.error.message);
        return fail<NodeResult>(exec.error.code, exec.error.message);
    }
    log_event(x.events, GraphEventType::ToolResult, node.name, node.tool_name, exec.value);
    trigger_event(x.session, EventType::OnToolResult, bytes_of(exec.value));
    {
        std::lock_guard<std::mutex> lock(*x.session.stats_mutex);
        x.session.stats.total_tool_calls++;
    }
    if (!node.output_key.empty()) set_state(x.state, node.output_key, std::move(exec.value));
    return ok(NodeResult{});
}

Result<NodeResult> run_loop_node(Exec &x, const GraphNode &node) {
    const std::string key = std::string(graph::kInternalPrefix) + "loop:" + node.name;
    int done = 0;
    std::string_view count = get_state(x.state, key, "0");
    std::from_chars(count.data(), count.data() + count.size(), done);

    bool exit = done >= node.max_iterations;
    if (!exit && done > 0 && node.until) {
        auto stop = node.until(x.session, x.state, node.user_data);
        if (!stop.ok) return fail<NodeResult>(stop.error.code, stop.error.message);
        exit = stop.value;
    }

    NodeResult result;
    if (exit) {
        x.state.values.erase(key); // the next visit starts a fresh loop
    } else {
        x.state.values.insert_or_assign(key, std::to_string(done + 1)); // internal: not logged
        result.forced_next = node.loop_body;
    }
    return ok(std::move(result));
}

Result<NodeResult> run_node(Exec &x, const GraphNode &node, const PendingResume *resume);

// ---- Parallel nodes ----

Result<std::string> resolve_next(Exec &x, const GraphNode &node, const NodeResult &result) {
    if (!result.forced_next.empty()) return ok(result.forced_next);
    std::string next(kGraphEnd);
    if (const GraphEdge *edge = find_edge(x.graph, node.name)) {
        if (edge->router) {
            auto routed = edge->router(x.session, x.state, edge->user_data);
            if (!routed.ok) return fail<std::string>(routed.error.code, routed.error.message);
            if (routed.value != kGraphEnd && !find_node(x.graph, routed.value))
                return fail<std::string>(ErrorCode::InvalidConfig,
                                         "router of node '" + node.name + "' returned unknown node '" + std::string(routed.value) + "'");
            next = std::string(routed.value);
        } else {
            next = edge->to;
        }
    }
    if (next == kGraphEnd && !node.exit_to.empty()) next = node.exit_to;
    return ok(std::move(next));
}

// Runs one branch from entry until it reaches the Parallel node (join) or kGraphEnd.
Result<void> run_branch(Exec &x, std::string_view entry, std::string_view join) {
    std::string current(entry);
    int steps = 0;
    while (current != kGraphEnd && current != join) {
        if (++steps > x.graph.max_steps)
            return fail(ErrorCode::InvalidConfig,
                        "branch '" + std::string(entry) + "' exceeded max_steps (" + std::to_string(x.graph.max_steps) + ")");
        const GraphNode *node = find_node(x.graph, current);
        if (!node) return fail(ErrorCode::InvalidConfig, "unknown node '" + current + "'");
        auto res = run_node(x, *node, nullptr);
        if (!res.ok) return fail(res.error.code, res.error.message);
        auto drained = drain_writes(x, node->name);
        if (!drained.ok) return drained;
        auto next = resolve_next(x, *node, res.value);
        if (!next.ok) return fail(next.error.code, next.error.message);
        current = std::move(next.value);
    }
    return ok();
}

struct Branch {
    GraphState state;
    std::vector<GraphEvent> events;
    std::vector<std::string> writes;
    Result<void> result;
};

Result<NodeResult> run_parallel_node(Exec &x, const GraphNode &node) {
    const size_t count = node.branches.size();
    std::vector<Branch> branches(count);
    for (auto &b : branches) b.state.values = x.state.values;

    auto run_one = [&x, &node, &branches](size_t i) {
        Branch &b = branches[i];
        Exec bx{.session = x.session,
                .graph = x.graph,
                .thread_id = x.thread_id,
                .state = b.state,
                .events = b.events,
                .parent = &x,
                .branch_writes = &b.writes,
                .scoped_index = nullptr,
                .models = x.models};
        b.result = run_branch(bx, node.branches[i], node.name);
    };

    // Concurrent only at the top level: worker arenas are a flat pool, one slot per thread.
    const size_t slots = x.session.worker_arenas.size();
    if (!x.parent && count > 1 && slots > 0 && providers::supports_parallel(x.session)) {
        for (size_t base = 0; base < count; base += slots) {
            const size_t end = std::min(count, base + slots);
            std::vector<std::thread> workers;
            workers.reserve(end - base);
            for (size_t i = base; i < end; ++i) {
                const size_t slot = i - base; // indexes Session::worker_arenas
                workers.emplace_back([&run_one, i, slot]() {
                    ambient::worker_arena_index = slot;
                    run_one(i);
                });
            }
            for (auto &w : workers) w.join();
        }
    } else {
        for (size_t i = 0; i < count; ++i) run_one(i);
    }

    for (auto &b : branches)
        for (auto &e : b.events) x.events.push_back(std::move(e));
    for (size_t i = 0; i < count; ++i)
        if (branches[i].result.ok) trigger_event(x.session, EventType::OnSubAgentComplete, bytes_of(node.branches[i]));

    for (size_t i = 0; i < count; ++i) {
        if (!branches[i].result.ok)
            return fail<NodeResult>(branches[i].result.error.code,
                                    "parallel '" + node.name + "' branch '" + node.branches[i] + "': " + branches[i].result.error.message);
    }

    // Join: every key and every Agent history (history_key) must come from a single branch.
    std::map<std::string_view, size_t, std::less<>> key_owner;
    for (size_t i = 0; i < count; ++i) {
        for (const auto &key : branches[i].writes) {
            auto [it, inserted] = key_owner.emplace(key, i);
            if (!inserted && it->second != i)
                return fail<NodeResult>(ErrorCode::InvalidConfig, "parallel '" + node.name + "': state key '" + key +
                                                                      "' written by branches '" + node.branches[it->second] + "' and '" +
                                                                      node.branches[i] + "'; parallel branches must write unique keys");
        }
    }
    std::map<std::string_view, size_t, std::less<>> history_owner;
    for (size_t i = 0; i < count; ++i) {
        for (const auto &[name, memory] : branches[i].state.histories) {
            auto [it, inserted] = history_owner.emplace(name, i);
            if (!inserted)
                return fail<NodeResult>(ErrorCode::InvalidConfig, "parallel '" + node.name + "': agent history '" + name +
                                                                      "' used in branches '" + node.branches[it->second] + "' and '" +
                                                                      node.branches[i] + "'");
        }
    }

    for (const auto &[key, owner] : key_owner) {
        auto &values = branches[owner].state.values;
        auto found = values.find(key);
        if (found == values.end()) continue;
        x.state.values.insert_or_assign(std::string(key), found->second);
        if (x.branch_writes) x.branch_writes->emplace_back(key);
        else if (x.scoped_index && graph::is_scoped_key(key)) {
            auto put = graph::persist_scoped_value(x.session, key, found->second, *x.scoped_index);
            if (!put.ok) return fail<NodeResult>(put.error.code, put.error.message);
        }
    }
    for (auto &b : branches)
        for (auto &[name, memory] : b.state.histories) x.state.histories.insert_or_assign(name, std::move(memory));

    return ok(NodeResult{});
}

Result<NodeResult> run_node(Exec &x, const GraphNode &node, const PendingResume *resume) {
    log_event(x.events, GraphEventType::NodeStart, node.name, {}, {});
    trigger_event(x.session, EventType::OnStateTransition, bytes_of(node.name));

    Result<NodeResult> res;
    switch (node.kind) {
    case GraphNodeKind::Agent: res = run_agent_node(x, node, resume); break;
    case GraphNodeKind::Function: res = run_function_node(x, node); break;
    case GraphNodeKind::Tool: res = run_tool_node(x, node); break;
    case GraphNodeKind::Parallel: res = run_parallel_node(x, node); break;
    case GraphNodeKind::Loop: res = run_loop_node(x, node); break;
    }

    if (!res.ok) log_event(x.events, GraphEventType::NodeFailed, node.name, {}, res.error.message);
    else if (res.value.interrupted) log_event(x.events, GraphEventType::Interrupt, node.name, {}, res.value.reason);
    else log_event(x.events, GraphEventType::NodeEnd, node.name, {}, {});
    return res;
}

// ---- top level ----

void clear_temp(GraphState &state) {
    for (auto it = state.values.begin(); it != state.values.end();) {
        if (graph::is_temp_key(it->first)) it = state.values.erase(it);
        else ++it;
    }
    state.written.clear();
    state.interrupt_requested = false;
    state.interrupt_reason.clear();
}

// One run_graph / resume_graph call: where its checkpoints go.
struct RunCtx {
    Session &session;
    const Graph &graph;
    GraphRun &run;
    graph::Sink sink;
};

std::string checkpoint_metadata(const RunCtx &c) {
    std::string meta = "{\"graph\":\"";
    utils::escape_json_string(c.graph.name, meta);
    meta += "\",\"step\":" + std::to_string(c.run.step) + ",\"next\":\"";
    utils::escape_json_string(c.run.next_node, meta);
    meta += "\",\"status\":\"";
    meta += graph::status_name(c.run.status);
    meta += "\"}";
    return meta;
}

// Writes the run's current checkpoint. through_writer: queue it on the Async writer (when there is
// one); otherwise it is written before this returns.
Result<void> write_checkpoint(RunCtx &c, bool through_writer) {
    GraphRun &run = c.run;
    log_event(run.events, GraphEventType::Checkpoint, run.next_node, graph::checkpoint_key(run.thread_id, std::to_string(run.step)), {});
    std::string json = graph::encode_checkpoint(run, c.graph.name);
    std::string meta = checkpoint_metadata(c);
    if (through_writer && c.sink.writer) {
        c.sink.writer->put(run.thread_id, run.step, std::move(json), std::move(meta));
        return ok();
    }
    return c.sink.ops->put(c.sink.state, run.thread_id, run.step, json, meta);
}

// Ends the call: queued Async checkpoints flushed, final checkpoint, temp: keys dropped.
Result<void> finish(RunCtx &c, std::optional<Error> error) {
    GraphRun &run = c.run;
    if (error) {
        run.status = GraphRunStatus::Failed;
        run.error = *error;
    }
    Result<void> cp = ok();
    if (c.sink.writer) cp = c.sink.writer->flush();
    auto last = write_checkpoint(c, false);
    if (cp.ok) cp = std::move(last);
    clear_temp(run.state);
    if (error) return Result<void>{.ok = false, .error = std::move(*error)};
    if (!cp.ok) {
        run.status = GraphRunStatus::Failed;
        run.error = cp.error;
    }
    return cp;
}

Result<void> drive(RunCtx &c, ScopedIndex &index, const PendingResume *resume) {
    Session &session = c.session;
    const Graph &graph = c.graph;
    GraphRun &run = c.run;
    Exec x{.session = session,
           .graph = graph,
           .thread_id = run.thread_id,
           .state = run.state,
           .events = run.events,
           .parent = nullptr,
           .branch_writes = nullptr,
           .scoped_index = &index,
           .models = run.models};

    auto seeded = drain_writes(x, {}); // caller seeds and human input
    if (!seeded.ok) return finish(c, seeded.error);

    int steps = 0;
    while (run.next_node != kGraphEnd) {
        if (++steps > graph.max_steps)
            return finish(c, make_error(ErrorCode::InvalidConfig,
                                        "graph '" + graph.name + "' exceeded max_steps (" + std::to_string(graph.max_steps) + ")"));
        const GraphNode *node = find_node(graph, run.next_node);
        if (!node) return finish(c, make_error(ErrorCode::InvalidConfig, "unknown node '" + run.next_node + "'"));

        auto res = run_node(x, *node, node->kind == GraphNodeKind::Agent ? resume : nullptr);
        resume = nullptr;
        if (!res.ok) {
            x.state.written.clear(); // a failed node's writes are not logged or persisted
            return finish(c, res.error);
        }
        auto drained = drain_writes(x, node->name);
        if (!drained.ok) return finish(c, drained.error);

        if (res.value.interrupted) {
            run.status = GraphRunStatus::Interrupted;
            run.interrupt_node = node->name;
            run.interrupt_reason = std::move(res.value.reason);
            run.pending_tool_calls = std::move(res.value.pending);
            return finish(c, std::nullopt); // next_node stays on the interrupted node
        }

        auto next = resolve_next(x, *node, res.value);
        if (!next.ok) return finish(c, next.error);
        run.next_node = std::move(next.value);
        run.step++;
        if (run.next_node == kGraphEnd) break; // the final checkpoint is written by finish()
        if (run.durability != GraphDurability::Exit) {
            auto cp = write_checkpoint(c, run.durability == GraphDurability::Async);
            if (!cp.ok) return finish(c, cp.error);
        }
    }

    run.status = GraphRunStatus::Completed;
    run.next_node.clear();
    return finish(c, std::nullopt);
}

void reset_run_status(GraphRun &run) {
    run.status = GraphRunStatus::Running;
    run.error = {};
    run.interrupt_node.clear();
    run.interrupt_reason.clear();
    run.pending_tool_calls.clear();
}

constexpr std::string_view kAbandonedCall =
    "Error: tool call not executed: the run was restarted with run_graph before the call was approved (use resume_graph to answer "
    "pending tool calls)";

// A fresh run_graph on a thread whose last run paused before tool calls (interrupt_before_tools),
// or died mid-round, would otherwise continue a history whose last assistant message has calls
// without results. Each such call gets an error tool message so the conversation stays valid.
void answer_abandoned_calls(GraphState &state, std::vector<GraphEvent> &events) {
    for (auto &[name, memory] : state.histories) {
        auto &history = memory.history;
        size_t i = history.size();
        while (i > 0 && history[i - 1].role == "tool") --i;
        if (i == 0 || history[i - 1].role != "assistant" || history[i - 1].tool_calls.empty()) continue;
        const std::vector<ToolCall> calls = history[i - 1].tool_calls;
        for (const auto &tc : calls) {
            bool answered = false;
            for (size_t j = i; j < history.size() && !answered; ++j) answered = history[j].tool_call_id == tc.id;
            if (answered) continue;
            Message msg;
            msg.role = "tool";
            msg.content = std::string(kAbandonedCall);
            msg.tool_call_id = tc.id;
            history.push_back(std::move(msg));
            log_event(events, GraphEventType::ToolResult, name, tc.name, kAbandonedCall);
        }
    }
}

bool is_approval(std::string_view text) {
    std::string lower;
    for (char c : text)
        if (!std::isspace(static_cast<unsigned char>(c))) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower == "approve" || lower == "approved" || lower == "yes" || lower == "y" || lower == "allow" || lower == "ok";
}

Result<void> resume_impl(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run,
                         std::optional<PermissionDecision> decision, std::string_view human_input_view) {
    auto valid = validate(graph);
    if (!valid.ok) return valid;

    // Copies first: both views may point into run.
    const std::string human_input(human_input_view);
    run.thread_id = thread_id.empty() ? std::string("default") : std::string(thread_id);
    auto sink = graph::resolve_sink(session, run.checkpointer, run.durability == GraphDurability::Async);
    if (!sink.ok) return fail(sink.error.code, sink.error.message);
    RunCtx c{.session = session, .graph = graph, .run = run, .sink = std::move(sink.value)};
    auto stored = c.sink.ops->get_latest(c.sink.state, run.thread_id);
    if (!stored.ok) {
        if (stored.error.code == ErrorCode::KeyNotFound)
            return fail(ErrorCode::KeyNotFound, "resume_graph: no checkpoint for thread '" + run.thread_id + "'");
        return fail(stored.error.code, stored.error.message);
    }
    run.state = GraphState{};
    run.pending_tool_calls.clear();
    auto decoded = graph::decode_checkpoint(stored.value, run);
    if (!decoded.ok) return decoded;
    if (run.status == GraphRunStatus::Completed) return ok();

    ScopedIndex index;
    auto scoped = graph::load_scoped_state(session, run.state, index);
    if (!scoped.ok) return scoped;

    PendingResume pending;
    pending.calls = std::move(run.pending_tool_calls);
    pending.approved = decision ? *decision == PermissionDecision::Allow : is_approval(human_input);
    pending.human_input = human_input;
    reset_run_status(run);

    log_event(run.events, GraphEventType::Resume, run.next_node, {}, human_input);
    if (!human_input.empty()) set_state(run.state, "temp:human_input", human_input);
    return drive(c, index, pending.calls.empty() ? nullptr : &pending);
}

} // namespace

Result<void> run_graph(Session &session, const Graph &graph, GraphRun &run) {
    auto valid = validate(graph);
    if (!valid.ok) return valid;
    if (run.thread_id.empty()) run.thread_id = "default";

    GraphState seed = std::move(run.state);
    run.state = GraphState{};
    run.step = 0;

    auto sink = graph::resolve_sink(session, run.checkpointer, run.durability == GraphDurability::Async);
    if (!sink.ok) return fail(sink.error.code, sink.error.message);
    RunCtx c{.session = session, .graph = graph, .run = run, .sink = std::move(sink.value)};

    // Continue the thread: per-thread values and Agent histories from its latest checkpoint.
    auto stored = c.sink.ops->get_latest(c.sink.state, run.thread_id);
    if (stored.ok) {
        GraphRun previous;
        previous.thread_id = run.thread_id;
        auto decoded = graph::decode_checkpoint(stored.value, previous);
        if (!decoded.ok) return decoded;
        for (auto &[key, value] : previous.state.values)
            if (!graph::is_internal_key(key)) run.state.values.insert_or_assign(key, std::move(value));
        run.state.histories = std::move(previous.state.histories);
        run.step = previous.step;
        answer_abandoned_calls(run.state, run.events);
    } else if (stored.error.code != ErrorCode::KeyNotFound) {
        return fail(stored.error.code, stored.error.message);
    }

    ScopedIndex index;
    auto scoped = graph::load_scoped_state(session, run.state, index);
    if (!scoped.ok) return scoped;

    for (auto &[key, value] : seed.values) run.state.values.insert_or_assign(key, std::move(value));
    for (auto &[name, memory] : seed.histories) run.state.histories.insert_or_assign(name, std::move(memory));
    run.state.written = std::move(seed.written);

    reset_run_status(run);
    run.next_node = graph.entry;
    return drive(c, index, nullptr);
}

Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run, std::string_view human_input) {
    return resume_impl(session, graph, thread_id, run, std::nullopt, human_input);
}

Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run, PermissionDecision decision,
                          std::string_view human_input) {
    return resume_impl(session, graph, thread_id, run, decision, human_input);
}

} // namespace agent
