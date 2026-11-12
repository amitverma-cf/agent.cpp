#include "../executor/ambient_turn.hpp"
#include "../providers/provider_ops.hpp"
#include "../tools/tools_ops.hpp"
#include "../utils/utils.hpp"
#include "graph_internal.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <map>
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

// A tool result keeps its trailing "[attached: <path> (<mime>)]" lines when the text before them is
// clipped, so the event log names the files a tool produced.
std::string clip_tool_result(std::string_view text) {
    if (text.size() <= kPayloadMax) return std::string(text);
    const size_t at = text.find("[attached: ");
    if (at == std::string_view::npos) return clip(text);
    std::string out = clip(text.substr(0, at));
    if (!out.empty() && out.back() != '\n') out += '\n';
    out += text.substr(at, 4 * kPayloadMax);
    return out;
}

void log_event(std::vector<GraphEvent> &events, GraphEventType type, std::string_view node, std::string_view key,
               std::string_view payload) {
    events.push_back(GraphEvent{.type = type,
                                .node = std::string(node),
                                .key = std::string(key),
                                .timestamp_ms = graph::now_ms(),
                                .payload = type == GraphEventType::ToolResult ? clip_tool_result(payload) : clip(payload)});
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

// Approval for the tool calls an interrupt_before_tools Agent node paused on.
struct PendingResume {
    std::vector<ToolCall> calls;
    bool approved = false;
    std::string human_input;
};

// One run_graph / resume_graph call: where its checkpoints go.
struct RunCtx {
    Session &session;
    const Graph &graph;
    GraphRun &run;
    graph::Sink sink;
    size_t persisted_events = run.events.size(); // run.events before this index were stored by earlier calls
    std::string forked_from;                     // fork_graph_run: "<thread>:<step>", recorded in metadata

    // Branch progress of the unfinished top-level fan-out (run.next_node), carried in checkpoints.
    // Guarded by progress_mutex while branches run (they also write checkpoints through it).
    std::optional<graph::ParallelProgress> progress;
    std::mutex progress_mutex;
    // resume_graph: answers for paused branches, by branch entry node.
    std::map<std::string, PendingResume, std::less<>> branch_resume;
    // Session::worker_arenas slots in use by branch threads.
    std::mutex slot_mutex;
    std::vector<bool> slot_used = std::vector<bool>(session.worker_arenas.size(), false);
};

std::string checkpoint_metadata(const RunCtx &c) {
    std::string meta = "{\"graph\":\"";
    utils::escape_json_string(c.graph.name, meta);
    meta += "\",\"step\":" + std::to_string(c.run.step) + ",\"next\":\"";
    utils::escape_json_string(c.run.next_node, meta);
    meta += "\",\"status\":\"";
    meta += graph::status_name(c.run.status);
    if (!c.forked_from.empty()) {
        meta += "\",\"forked_from\":\"";
        utils::escape_json_string(c.forked_from, meta);
    }
    meta += "\"}";
    return meta;
}

// Writes the run's current checkpoint (with the branch progress of an unfinished fan-out), then appends the events logged since the last write to the
// thread's event log. through_writer: queue both on the Async writer (when there is one); otherwise
// they are written before this returns.
Result<void> write_checkpoint(RunCtx &c, bool through_writer) {
    GraphRun &run = c.run;
    log_event(run.events, GraphEventType::Checkpoint, run.next_node, graph::checkpoint_key(run.thread_id, std::to_string(run.step)), {});
    std::string json = graph::encode_checkpoint(run, c.graph.name, c.progress ? &*c.progress : nullptr);
    std::string meta = checkpoint_metadata(c);
    std::vector<GraphEvent> fresh;
    if (c.sink.ops->append_events && c.persisted_events < run.events.size())
        fresh.assign(run.events.begin() + static_cast<std::ptrdiff_t>(c.persisted_events), run.events.end());
    c.persisted_events = run.events.size();
    if (through_writer && c.sink.writer) {
        c.sink.writer->put(run.thread_id, run.step, std::move(json), std::move(meta));
        c.sink.writer->append_events(run.thread_id, std::move(fresh));
        return ok();
    }
    auto put = c.sink.ops->put(c.sink.state, run.thread_id, run.step, json, meta);
    if (!put.ok || fresh.empty()) return put;
    return c.sink.ops->append_events(c.sink.state, run.thread_id, fresh);
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
    RunCtx *ctx = nullptr;
    int depth = 0; // 0 = the top-level chain, 1 = a branch of a top-level fan-out, 2+ = nested branches
};

// log_event, then the caller's GraphRun::on_event (if any). Branch chains call it from worker threads.
void emit(Exec &x, GraphEventType type, std::string_view node, std::string_view key, std::string_view payload) {
    log_event(x.events, type, node, key, payload);
    const GraphRun &run = x.ctx->run;
    if (run.on_event) run.on_event(x.events.back(), run.on_event_user_data);
}

struct NodeResult {
    bool interrupted = false;
    std::string reason;
    std::vector<ToolCall> pending;
    std::string forced_next; // Loop: jump into the body, bypassing edges
    std::vector<GraphInterrupt> interrupts; // Parallel: one per paused branch
};

// The Agent node's conversation in GraphState::histories (history_key, else the node name).
const std::string &history_name(const GraphNode &node) { return node.history_key.empty() ? node.name : node.history_key; }

Conversation &history_for(Exec &x, const std::string &node) {
    auto it = x.state.histories.find(node);
    if (it != x.state.histories.end()) return it->second;
    Conversation memory;
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
    Exec *exec;
    std::string_view node;
};

void log_tool_pre(Session &, Conversation &, std::string_view tool, std::string_view args, std::string_view, void *ud) {
    auto *ctx = static_cast<ToolLogCtx *>(ud);
    emit(*ctx->exec, GraphEventType::ToolCall, ctx->node, tool, args);
}

void log_tool_post(Session &, Conversation &, std::string_view tool, std::string_view, std::string_view result, void *ud) {
    auto *ctx = static_cast<ToolLogCtx *>(ud);
    emit(*ctx->exec, GraphEventType::ToolResult, ctx->node, tool, result);
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
    Conversation &memory;
    bool injected = false;
    std::optional<Message> saved;

    SystemPromptScope(Conversation &m, std::string_view prompt) : memory(m) {
        if (prompt.empty()) return;
        // A compressed-history summary at the front is conversation, not a system prompt: keep it (run_turn
        // merges the leading system messages) and put the node's prompt before it.
        if (!memory.history.empty() && memory.history[0].role == "system" &&
            !std::string_view(memory.history[0].content).starts_with(tools::kCompressedHistoryPrefix)) {
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

void fire_node_hooks(Session &session, Conversation &memory, const GraphNode &node, ToolHook::When when, const ToolCall &tc,
                     std::string_view result) {
    for (const auto &hook : node.hooks) {
        if (hook.when != when || !hook.callback) continue;
        if (!hook.tool_name.empty() && hook.tool_name != tc.name) continue;
        hook.callback(session, memory, tc.name, tc.arguments, result, hook.user_data);
    }
}

ToolResult run_approved_call(Exec &x, Session &session, Conversation &memory, const GraphNode &node, const ToolCall &tc,
                             const PendingResume &resume) {
    emit(x, GraphEventType::ToolCall, node.name, tc.name, tc.arguments);
    trigger_event(session, EventType::OnToolCall, bytes_of(tc.name));

    PermissionDecision policy = PermissionDecision::Ask;
    if (session.config.permission_check)
        policy = session.config.permission_check(tc.name, tc.arguments, session.config.permission_user_data);

    ToolResult result;
    if (policy == PermissionDecision::Ask && !resume.approved) {
        result.text = "Error: tool call denied by human reviewer";
        if (!resume.human_input.empty()) result.text += ": " + resume.human_input;
        result.is_error = true;
    } else {
        fire_node_hooks(session, memory, node, ToolHook::When::Pre, tc, {});
        // Tools that act on the conversation (compress_context) see this node's history, as in run_turn.
        Conversation *previous_memory = ambient::active_memory;
        ambient::active_memory = &memory;
        ambient::tool_call_preapproved = (policy == PermissionDecision::Ask);
        result = tools::execute_for_message(session, tc.name, tc.arguments);
        ambient::tool_call_preapproved = false;
        ambient::active_memory = previous_memory;
        fire_node_hooks(session, memory, node, ToolHook::When::Post, tc, result.text);
    }

    emit(x, GraphEventType::ToolResult, node.name, tc.name, result.text);
    trigger_event(session, EventType::OnToolResult, bytes_of(result.text));
    memory.stats.tool_calls++;
    {
        std::lock_guard<std::mutex> lock(*session.stats_mutex);
        session.stats.total_tool_calls++;
    }
    return result;
}

Result<NodeResult> run_agent_node(Exec &x, const GraphNode &node, const PendingResume *resume) {
    if (node.interrupt_before_tools && x.depth > 1)
        return fail<NodeResult>(ErrorCode::InvalidConfig,
                                "agent node '" + node.name + "': interrupt_before_tools is not supported inside nested parallel branches");

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

    Conversation &memory = history_for(x, history_name(node));
    std::string prompt = node.prompt;
    if (!node.input_key.empty() && has_state(x.state, node.input_key)) prompt = std::string(get_state(x.state, node.input_key));

    std::vector<std::string_view> allowed(node.allowed_tools.begin(), node.allowed_tools.end());
    ToolLogCtx log_ctx{&x, node.name};
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
            ToolResult tr = run_approved_call(x, *session, memory, node, tc, *resume);
            Message tool_msg;
            tool_msg.role = "tool";
            tool_msg.content = std::move(tr.text);
            tool_msg.media = std::move(tr.media);
            tool_msg.tensors = std::move(tr.tensors);
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
        if (x.depth > 1)
            return fail<NodeResult>(ErrorCode::InvalidConfig,
                                    "function node '" + node.name + "': interrupts are not supported inside nested parallel branches");
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

    emit(x, GraphEventType::ToolCall, node.name, node.tool_name, args);
    trigger_event(x.session, EventType::OnToolCall, bytes_of(node.tool_name));
    // The string form: attachment notes are part of the text, an is_error result fails the node.
    auto exec = execute_tool(x.session, node.tool_name, args);
    if (!exec.ok) {
        emit(x, GraphEventType::ToolResult, node.name, node.tool_name, "Error: " + exec.error.message);
        return fail<NodeResult>(exec.error.code, exec.error.message);
    }
    emit(x, GraphEventType::ToolResult, node.name, node.tool_name, exec.value);
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

// One branch's working copy while a fan-out runs.
struct Branch {
    GraphState state;
    std::vector<GraphEvent> events;
    std::vector<std::string> writes;
    Result<void> result;
    std::string next;  // where the branch continues (paused / failed node)
    bool run = false;  // runs in this call
    bool done = false; // reached the join (in this call or an earlier one)
    bool interrupted = false;
    std::string interrupt_node, interrupt_reason;
    std::vector<ToolCall> pending;
    std::optional<PendingResume> resume; // pending tool calls answered on this resume
};

// Records a top-level branch's progress in the run's ParallelProgress and checkpoints it (unless
// durability is Exit). Called from the branch's own thread.
Result<void> save_branch(Exec &x, size_t index, const Branch &b, graph::BranchStatus status, std::string_view next) {
    RunCtx &c = *x.ctx;
    std::lock_guard<std::mutex> lock(c.progress_mutex);
    graph::BranchProgress &rec = c.progress->branches[index];
    rec.status = status;
    rec.next = std::string(next);
    if (status == graph::BranchStatus::Failed) {
        rec.error = b.result.error.message; // values / histories stay as of the last completed node
    } else {
        const auto &parent = x.parent->state.values;
        rec.values.clear();
        rec.erased.clear();
        for (const auto &[key, value] : b.state.values) {
            if (graph::is_temp_key(key)) continue;
            auto it = parent.find(key);
            if (it == parent.end() || it->second != value) rec.values.insert_or_assign(key, value);
        }
        for (const auto &[key, value] : parent)
            if (!graph::is_temp_key(key) && !b.state.values.count(key)) rec.erased.push_back(key);
        rec.writes = b.writes;
        rec.histories = b.state.histories;
        rec.error.clear();
        rec.interrupt_node = b.interrupt_node;
        rec.interrupt_reason = b.interrupt_reason;
        rec.pending_tool_calls = b.pending;
    }
    if (c.run.durability == GraphDurability::Exit) return ok();
    return write_checkpoint(c, c.run.durability == GraphDurability::Async);
}

// Runs one branch from start until it reaches the Parallel node (join) or kGraphEnd, pauses, or
// fails. index: the branch's slot in the run's ParallelProgress (top-level fan-outs), or npos.
void run_branch(Exec &x, Branch &b, std::string current, std::string_view join, size_t index) {
    const bool tracked = index != std::string::npos;
    const PendingResume *resume = b.resume ? &*b.resume : nullptr;
    int steps = 0;
    auto failed = [&](Error error) {
        b.result = Result<void>{.ok = false, .error = std::move(error)};
        b.next = current;
        x.state.written.clear();
        if (tracked) save_branch(x, index, b, graph::BranchStatus::Failed, current);
    };
    while (current != kGraphEnd && current != join) {
        if (++steps > x.graph.max_steps)
            return failed(make_error(ErrorCode::InvalidConfig, "branch exceeded max_steps (" + std::to_string(x.graph.max_steps) + ")"));
        const GraphNode *node = find_node(x.graph, current);
        if (!node) return failed(make_error(ErrorCode::InvalidConfig, "unknown node '" + current + "'"));
        auto res = run_node(x, *node, node->kind == GraphNodeKind::Agent ? resume : nullptr);
        resume = nullptr;
        if (!res.ok) return failed(res.error);
        auto drained = drain_writes(x, node->name);
        if (!drained.ok) return failed(drained.error);
        if (res.value.interrupted) {
            b.interrupted = true;
            b.interrupt_node = node->name;
            b.interrupt_reason = std::move(res.value.reason);
            b.pending = std::move(res.value.pending);
            b.next = current;
            if (tracked) {
                auto saved = save_branch(x, index, b, graph::BranchStatus::Interrupted, current);
                if (!saved.ok) b.result = saved;
            }
            return;
        }
        auto next = resolve_next(x, *node, res.value);
        if (!next.ok) return failed(next.error);
        current = std::move(next.value);
        if (tracked) {
            const bool end = current == kGraphEnd || current == join;
            auto saved = save_branch(x, index, b, end ? graph::BranchStatus::Done : graph::BranchStatus::Running, current);
            if (!saved.ok) return failed(saved.error);
        }
    }
    b.done = true;
}

// Worker slots (Session::worker_arenas) shared by every fan-out of a run, nested ones included.
thread_local bool t_holds_slot = false;

std::vector<size_t> acquire_slots(RunCtx &c, size_t want) {
    std::vector<size_t> got;
    std::lock_guard<std::mutex> lock(c.slot_mutex);
    for (size_t i = 0; i < c.slot_used.size() && got.size() < want; ++i)
        if (!c.slot_used[i]) {
            c.slot_used[i] = true;
            got.push_back(i);
        }
    return got;
}

void release_slots(RunCtx &c, const std::vector<size_t> &slots) {
    std::lock_guard<std::mutex> lock(c.slot_mutex);
    for (size_t s : slots) c.slot_used[s] = false;
}

// Calls fn(i) for every i in which: on worker threads when the engine is concurrent and worker
// slots are free (a thread that holds a slot itself also takes part), otherwise one after another.
template <typename Fn> void run_branches(Exec &x, const std::vector<size_t> &which, Fn &&fn) {
    const size_t own = t_holds_slot ? 1 : 0;
    std::vector<size_t> slots;
    if (which.size() > 1 && x.ctx && !x.ctx->slot_used.empty() && providers::supports_parallel(x.session))
        slots = acquire_slots(*x.ctx, which.size() - own);
    if (slots.empty()) {
        for (size_t i : which) fn(i);
        return;
    }
    std::atomic<size_t> cursor{0};
    auto work = [&which, &cursor, &fn]() {
        for (size_t k = cursor++; k < which.size(); k = cursor++) fn(which[k]);
    };
    std::vector<std::thread> workers;
    workers.reserve(slots.size());
    for (size_t slot : slots)
        workers.emplace_back([&work, slot]() {
            ambient::worker_arena_index = slot;
            t_holds_slot = true;
            work();
        });
    if (own) work();
    for (auto &w : workers) w.join();
    release_slots(*x.ctx, slots);
}

std::string branch_error(const GraphNode &node, size_t i, const Error &e) {
    return "parallel '" + node.name + "' branch '" + node.branches[i] + "': " + e.message;
}

Result<NodeResult> run_parallel_node(Exec &x, const GraphNode &node) {
    const size_t count = node.branches.size();
    // Only fan-outs of the top-level chain keep per-branch progress (and may pause).
    const bool tracked = x.depth == 0 && x.ctx;
    std::vector<Branch> branches(count);
    std::vector<size_t> to_run;

    if (tracked) {
        RunCtx &c = *x.ctx;
        std::lock_guard<std::mutex> lock(c.progress_mutex);
        bool fresh = !c.progress || c.progress->node != node.name || c.progress->branches.size() != count;
        for (size_t i = 0; !fresh && i < count; ++i) fresh = c.progress->branches[i].entry != node.branches[i];
        if (fresh) {
            c.progress.emplace();
            c.progress->node = node.name;
            for (const auto &entry : node.branches) c.progress->branches.push_back(graph::BranchProgress{.entry = entry, .next = entry});
        }
        for (size_t i = 0; i < count; ++i) {
            const graph::BranchProgress &rec = c.progress->branches[i];
            Branch &b = branches[i];
            b.state.values = x.state.values;
            for (const auto &[key, value] : rec.values) b.state.values.insert_or_assign(key, value);
            for (const auto &key : rec.erased) b.state.values.erase(key);
            b.state.histories = rec.histories;
            b.writes = rec.writes;
            b.next = rec.next.empty() ? rec.entry : rec.next;
            switch (rec.status) {
            case graph::BranchStatus::Done: b.done = true; break;
            case graph::BranchStatus::Interrupted: {
                auto input = c.branch_resume.find(rec.entry);
                if (input == c.branch_resume.end()) {
                    b.interrupted = true; // stays paused
                    b.interrupt_node = rec.interrupt_node;
                    b.interrupt_reason = rec.interrupt_reason;
                    b.pending = rec.pending_tool_calls;
                    break;
                }
                if (!input->second.human_input.empty()) {
                    b.state.values.insert_or_assign("temp:human_input", input->second.human_input);
                    log_event(b.events, GraphEventType::StateWrite, rec.interrupt_node, "temp:human_input", input->second.human_input);
                }
                log_event(b.events, GraphEventType::Resume, rec.interrupt_node, rec.entry, input->second.human_input);
                if (!rec.pending_tool_calls.empty()) b.resume = input->second;
                if (b.resume) b.resume->calls = rec.pending_tool_calls;
                b.run = true;
                to_run.push_back(i);
                break;
            }
            default: b.run = true; to_run.push_back(i);
            }
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            branches[i].state.values = x.state.values;
            branches[i].next = node.branches[i];
            branches[i].run = true;
            to_run.push_back(i);
        }
    }

    run_branches(x, to_run, [&x, &node, &branches, tracked](size_t i) {
        Branch &b = branches[i];
        Exec bx{.session = x.session,
                .graph = x.graph,
                .thread_id = x.thread_id,
                .state = b.state,
                .events = b.events,
                .parent = &x,
                .branch_writes = &b.writes,
                .scoped_index = nullptr,
                .models = x.models,
                .ctx = x.ctx,
                .depth = x.depth + 1};
        run_branch(bx, b, b.next, node.name, tracked ? i : std::string::npos);
    });

    for (auto &b : branches)
        for (auto &e : b.events) x.events.push_back(std::move(e));
    for (size_t i = 0; i < count; ++i)
        if (branches[i].run && branches[i].done && branches[i].result.ok)
            trigger_event(x.session, EventType::OnSubAgentComplete, bytes_of(node.branches[i]));

    for (size_t i = 0; i < count; ++i)
        if (!branches[i].result.ok) return fail<NodeResult>(branches[i].result.error.code, branch_error(node, i, branches[i].result.error));

    NodeResult paused;
    for (size_t i = 0; i < count; ++i) {
        Branch &b = branches[i];
        if (!b.interrupted) continue;
        if (!paused.interrupted) paused.reason.clear();
        else paused.reason += "; ";
        paused.interrupted = true;
        paused.reason += "branch '" + node.branches[i] + "': " + b.interrupt_reason;
        paused.pending.insert(paused.pending.end(), b.pending.begin(), b.pending.end());
        paused.interrupts.push_back(GraphInterrupt{
            .branch = node.branches[i], .node = b.interrupt_node, .reason = b.interrupt_reason, .pending_tool_calls = b.pending});
    }
    if (paused.interrupted) return ok(std::move(paused));

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

    if (tracked) {
        std::lock_guard<std::mutex> lock(x.ctx->progress_mutex);
        x.ctx->progress.reset();
    }
    return ok(NodeResult{});
}

Result<NodeResult> run_node(Exec &x, const GraphNode &node, const PendingResume *resume) {
    emit(x, GraphEventType::NodeStart, node.name, {}, {});
    trigger_event(x.session, EventType::OnStateTransition, bytes_of(node.name));

    Result<NodeResult> res;
    switch (node.kind) {
    case GraphNodeKind::Agent: res = run_agent_node(x, node, resume); break;
    case GraphNodeKind::Function: res = run_function_node(x, node); break;
    case GraphNodeKind::Tool: res = run_tool_node(x, node); break;
    case GraphNodeKind::Parallel: res = run_parallel_node(x, node); break;
    case GraphNodeKind::Loop: res = run_loop_node(x, node); break;
    }

    if (!res.ok) emit(x, GraphEventType::NodeFailed, node.name, {}, res.error.message);
    else if (res.value.interrupted) emit(x, GraphEventType::Interrupt, node.name, {}, res.value.reason);
    else emit(x, GraphEventType::NodeEnd, node.name, {}, {});
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
           .models = run.models,
           .ctx = &c,
           .depth = 0};

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
            if (res.value.interrupts.empty())
                run.interrupts.push_back(GraphInterrupt{
                    .branch = {}, .node = node->name, .reason = run.interrupt_reason, .pending_tool_calls = run.pending_tool_calls});
            else run.interrupts = std::move(res.value.interrupts);
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
    run.interrupts.clear();
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

// The paused nodes recorded in a checkpoint (fork_graph_run of an interrupted step).
std::vector<GraphInterrupt> paused_nodes(const GraphRun &run, const std::optional<graph::ParallelProgress> &progress) {
    std::vector<GraphInterrupt> out;
    if (progress) {
        for (const auto &b : progress->branches)
            if (b.status == graph::BranchStatus::Interrupted)
                out.push_back(GraphInterrupt{
                    .branch = b.entry, .node = b.interrupt_node, .reason = b.interrupt_reason, .pending_tool_calls = b.pending_tool_calls});
    } else {
        out.push_back(GraphInterrupt{
            .branch = {}, .node = run.interrupt_node, .reason = run.interrupt_reason, .pending_tool_calls = run.pending_tool_calls});
    }
    return out;
}

// What resume_graph was given: one answer for every paused node (the string / decision overloads),
// or per-branch answers.
struct ResumeRequest {
    bool per_branch = false;
    std::optional<PermissionDecision> decision;
    std::string human_input;
    std::vector<GraphBranchInput> inputs;
};

bool approves(std::optional<PermissionDecision> decision, std::string_view human_input) {
    if (decision && *decision != PermissionDecision::Ask) return *decision == PermissionDecision::Allow;
    return is_approval(human_input); // no decision, or GraphBranchInput's Ask: decide from the text
}

Result<void> resume_impl(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run, ResumeRequest req) {
    auto valid = validate(graph);
    if (!valid.ok) return valid;

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
    graph::ParallelProgress progress;
    bool has_progress = false;
    auto decoded = graph::decode_checkpoint(stored.value, run, &progress, &has_progress);
    if (!decoded.ok) return decoded;
    if (run.status == GraphRunStatus::Completed) return ok();
    if (has_progress) c.progress = std::move(progress);
    const bool was_interrupted = run.status == GraphRunStatus::Interrupted;

    // Work out which paused nodes get which answer.
    PendingResume top; // an interrupt outside a fan-out
    bool resume_top = false;
    std::string top_input;
    if (c.progress) {
        for (const auto &rec : c.progress->branches) {
            if (rec.status != graph::BranchStatus::Interrupted) continue;
            if (req.per_branch) continue;
            // Uniform answer: every paused branch, unless the checkpoint is a failure (then paused
            // branches stay paused while the failed ones re-run).
            if (!was_interrupted) continue;
            c.branch_resume.insert_or_assign(
                rec.entry, PendingResume{.calls = {}, .approved = approves(req.decision, req.human_input), .human_input = req.human_input});
        }
        for (const auto &in : req.inputs) {
            const graph::BranchProgress *rec = nullptr;
            for (const auto &b : c.progress->branches)
                if (b.entry == in.branch && b.status == graph::BranchStatus::Interrupted) rec = &b;
            if (!rec)
                return fail(ErrorCode::InvalidConfig,
                            "resume_graph: no paused branch '" + in.branch + "' in thread '" + run.thread_id + "'");
            c.branch_resume.insert_or_assign(
                in.branch, PendingResume{.calls = {}, .approved = approves(in.decision, in.human_input), .human_input = in.human_input});
        }
    } else if (req.per_branch) {
        for (const auto &in : req.inputs) {
            if (!in.branch.empty() || !was_interrupted)
                return fail(ErrorCode::InvalidConfig,
                            "resume_graph: no paused branch '" + in.branch + "' in thread '" + run.thread_id + "'");
            resume_top = true;
            top_input = in.human_input;
            top.approved = approves(in.decision, in.human_input);
        }
    } else {
        resume_top = true;
        top_input = req.human_input;
        top.approved = approves(req.decision, req.human_input);
    }
    top.human_input = top_input;
    if (resume_top) top.calls = std::move(run.pending_tool_calls);

    ScopedIndex index;
    auto scoped = graph::load_scoped_state(session, run.state, index);
    if (!scoped.ok) return scoped;
    reset_run_status(run);

    log_event(run.events, GraphEventType::Resume, run.next_node, {}, top_input);
    if (!top_input.empty()) set_state(run.state, "temp:human_input", top_input);
    return drive(c, index, top.calls.empty() ? nullptr : &top);
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
    return resume_impl(session, graph, thread_id, run, ResumeRequest{.human_input = std::string(human_input)});
}

Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run, PermissionDecision decision,
                          std::string_view human_input) {
    const PermissionDecision d = decision == PermissionDecision::Allow ? PermissionDecision::Allow : PermissionDecision::Deny;
    return resume_impl(session, graph, thread_id, run, ResumeRequest{.decision = d, .human_input = std::string(human_input)});
}

Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run,
                          std::span<const GraphBranchInput> inputs) {
    return resume_impl(session, graph, thread_id, run,
                       ResumeRequest{.per_branch = true, .inputs = std::vector<GraphBranchInput>(inputs.begin(), inputs.end())});
}

Result<void> fork_graph_run(Session &session, const Graph &graph, std::string_view thread_id_view, uint64_t step,
                            std::string_view new_thread_id_view, GraphRun &run) {
    auto valid = validate(graph);
    if (!valid.ok) return valid;
    // Copies first: both views may point into run.
    const std::string source = thread_id_view.empty() ? std::string("default") : std::string(thread_id_view);
    const std::string target(new_thread_id_view);
    if (target.empty() || target == source)
        return fail(ErrorCode::InvalidConfig, "fork_graph_run: the new thread id must be non-empty and differ from '" + source + "'");

    auto sink = graph::resolve_sink(session, run.checkpointer, run.durability == GraphDurability::Async);
    if (!sink.ok) return fail(sink.error.code, sink.error.message);
    auto existing = sink.value.ops->get_latest(sink.value.state, target);
    if (existing.ok)
        return fail(ErrorCode::InvalidConfig,
                    "fork_graph_run: thread '" + target + "' already has checkpoints (delete_graph_thread it first)");
    if (existing.error.code != ErrorCode::KeyNotFound) return fail(existing.error.code, existing.error.message);
    auto stored = sink.value.ops->get(sink.value.state, source, step);
    if (!stored.ok) return fail(stored.error.code, "fork_graph_run: " + stored.error.message);

    GraphState seed = std::move(run.state);
    run.state = GraphState{};
    run.pending_tool_calls.clear();
    run.thread_id = target;
    RunCtx c{.session = session, .graph = graph, .run = run, .sink = std::move(sink.value)};
    graph::ParallelProgress progress;
    bool has_progress = false;
    auto decoded = graph::decode_checkpoint(stored.value, run, &progress, &has_progress);
    if (!decoded.ok) return decoded;
    run.step = step;
    if (has_progress) c.progress = std::move(progress);

    ScopedIndex index;
    auto scoped = graph::load_scoped_state(session, run.state, index);
    if (!scoped.ok) return scoped;
    for (auto &[key, value] : seed.values) run.state.values.insert_or_assign(key, std::move(value));
    for (auto &[name, memory] : seed.histories) run.state.histories.insert_or_assign(name, std::move(memory));
    run.state.written = std::move(seed.written);

    // The new thread's first checkpoint is the copy, at the same step.
    c.forked_from = source + ":" + std::to_string(step);
    log_event(run.events, GraphEventType::Fork, run.next_node, c.forked_from, {});
    auto copied = write_checkpoint(c, false);
    c.forked_from.clear();
    if (!copied.ok) return copied;
    if (run.status == GraphRunStatus::Completed || run.status == GraphRunStatus::Interrupted) {
        if (run.status == GraphRunStatus::Interrupted) run.interrupts = paused_nodes(run, c.progress);
        clear_temp(run.state);
        return ok();
    }

    reset_run_status(run);
    return drive(c, index, nullptr);
}

} // namespace agent
