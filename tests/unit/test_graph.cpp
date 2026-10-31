#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <atomic>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <string>
#include <thread>

namespace {

const char *kWorkspace = "/tmp/agent_test_graph_ws";

int g_ping_calls = 0;

agent::Tool make_ping_tool() {
    agent::Tool t;
    t.name = "ping";
    t.description = "test tool";
    t.parameter_schema = "{}";
    t.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        ++g_ping_calls;
        return agent::ok(std::string("pong"));
    };
    return t;
}

agent::Session make_session(agent::Config cfg = {}) {
    cfg.provider = agent::AiProvider::Engine;
    if (cfg.engine.empty()) cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.model = "mock.bin";
    cfg.workspace_dir = kWorkspace;
    cfg.tools.push_back(make_ping_tool());
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    return std::move(res.value);
}

agent::GraphNode function_node(std::string name, agent::GraphFunctionFn fn, void *user_data = nullptr) {
    agent::GraphNode n;
    n.name = std::move(name);
    n.kind = agent::GraphNodeKind::Function;
    n.function = fn;
    n.user_data = user_data;
    return n;
}

agent::GraphNode agent_node(std::string name, std::string input_key, std::string output_key) {
    agent::GraphNode n;
    n.name = std::move(name);
    n.kind = agent::GraphNodeKind::Agent;
    n.input_key = std::move(input_key);
    n.output_key = std::move(output_key);
    return n;
}

// Function node that writes user_data (a const char* "key=value") into state.
agent::Result<void> write_kv(agent::Session &, agent::GraphState &state, void *ud) {
    std::string_view kv = static_cast<const char *>(ud);
    auto eq = kv.find('=');
    agent::set_state(state, kv.substr(0, eq), std::string(kv.substr(eq + 1)));
    return agent::ok();
}

agent::Result<void> increment_count(agent::Session &, agent::GraphState &state, void *) {
    int n = std::stoi(std::string(agent::get_state(state, "count", "0")));
    agent::set_state(state, "count", std::to_string(n + 1));
    return agent::ok();
}

size_t count_events(const agent::GraphRun &run, agent::GraphEventType type, std::string_view node = {}, std::string_view key = {}) {
    size_t n = 0;
    for (const auto &e : run.events)
        if (e.type == type && (node.empty() || e.node == node) && (key.empty() || e.key == key)) ++n;
    return n;
}

} // namespace

TEST_CASE("graph: sequential pipeline runs function and agent nodes in order", "[graph]") {
    agent::Session session = make_session();
    int transitions = 0;
    agent::register_event_hook(
        session, agent::EventType::OnStateTransition, [](const agent::Event &, void *ud) { ++*static_cast<int *>(ud); }, &transitions);

    auto writer = agent_node("writer", "topic", "draft");
    writer.system_prompt = "You are a writer.";
    auto post = function_node("post", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        agent::set_state(state, "final", std::string(agent::get_state(state, "draft")) + "!");
        return agent::ok();
    });
    static const char *kTopic = "topic=hello";
    agent::Graph g = agent::make_sequential("pipeline", {agent::make_graph(function_node("prep", write_kv, const_cast<char *>(kTopic))),
                                                         agent::make_graph(writer), agent::make_graph(post)});

    agent::GraphRun run;
    run.thread_id = "seq";
    auto res = agent::run_graph(session, g, run);
    REQUIRE(res.ok);
    REQUIRE(run.status == agent::GraphRunStatus::Completed);
    REQUIRE(agent::get_state(run.state, "draft") == "echo: hello");
    REQUIRE(agent::get_state(run.state, "final") == "echo: hello!");
    REQUIRE(run.step == 3);
    REQUIRE(transitions == 3);

    // The node's system prompt is only installed for the turn.
    const auto &history = run.state.histories.at("writer").history;
    REQUIRE(history.size() == 2);
    REQUIRE(history[0].role == "user");

    // Event log: start/end per node in order, the writes, checkpoints.
    std::vector<std::string> order;
    for (const auto &e : run.events)
        if (e.type == agent::GraphEventType::NodeStart) order.push_back(e.node);
    REQUIRE(order == std::vector<std::string>{"prep", "writer", "post"});
    REQUIRE(count_events(run, agent::GraphEventType::NodeEnd) == 3);
    REQUIRE(count_events(run, agent::GraphEventType::StateWrite, "writer", "draft") == 1);
    REQUIRE(count_events(run, agent::GraphEventType::StateWrite, "prep", "topic") == 1);
    REQUIRE(count_events(run, agent::GraphEventType::Checkpoint) == 3);
    for (const auto &e : run.events) REQUIRE(e.timestamp_ms > 0);
    for (const auto &e : run.events)
        if (e.type == agent::GraphEventType::StateWrite && e.key == "draft") REQUIRE(e.payload == "echo: hello");
}

TEST_CASE("graph: conditional edges route on state", "[graph]") {
    agent::Session session = make_session();

    static const char *kA = "path=a";
    static const char *kB = "path=b";
    agent::Graph g;
    g.name = "router";
    g.entry = "start";
    g.nodes.push_back(function_node("start", [](agent::Session &, agent::GraphState &, void *) { return agent::ok(); }));
    g.nodes.push_back(function_node("a", write_kv, const_cast<char *>(kA)));
    g.nodes.push_back(function_node("b", write_kv, const_cast<char *>(kB)));
    g.edges.push_back(
        agent::GraphEdge{.from = "start",
                         .to = {},
                         .router = [](agent::Session &, const agent::GraphState &state, void *) -> agent::Result<std::string_view> {
                             if (agent::get_state(state, "input") == "go-b") return agent::ok(std::string_view("b"));
                             return agent::ok(std::string_view("a"));
                         },
                         .user_data = nullptr});
    g.edges.push_back(agent::GraphEdge{.from = "a", .to = std::string(agent::kGraphEnd), .router = nullptr, .user_data = nullptr});

    agent::GraphRun run_a;
    run_a.thread_id = "route-a";
    agent::set_state(run_a.state, "input", "go-a");
    REQUIRE(agent::run_graph(session, g, run_a).ok);
    REQUIRE(agent::get_state(run_a.state, "path") == "a");
    REQUIRE(count_events(run_a, agent::GraphEventType::NodeStart, "b") == 0);

    agent::GraphRun run_b;
    run_b.thread_id = "route-b";
    agent::set_state(run_b.state, "input", "go-b");
    REQUIRE(agent::run_graph(session, g, run_b).ok);
    REQUIRE(agent::get_state(run_b.state, "path") == "b");

    // A router naming an unknown node fails the run.
    g.edges[0].router = [](agent::Session &, const agent::GraphState &, void *) -> agent::Result<std::string_view> {
        return agent::ok(std::string_view("nowhere"));
    };
    agent::GraphRun bad;
    bad.thread_id = "route-bad";
    auto res = agent::run_graph(session, g, bad);
    REQUIRE_FALSE(res.ok);
    REQUIRE(bad.status == agent::GraphRunStatus::Failed);
    REQUIRE(res.error.message.find("nowhere") != std::string::npos);
}

TEST_CASE("graph: invalid graphs are rejected", "[graph]") {
    agent::Session session = make_session();
    agent::Graph g;
    g.name = "bad";
    g.entry = "x";
    g.nodes.push_back(function_node("x", write_kv));
    g.nodes.push_back(function_node("x", write_kv));
    agent::GraphRun run;
    auto dup = agent::run_graph(session, g, run);
    REQUIRE_FALSE(dup.ok);
    REQUIRE(dup.error.code == agent::ErrorCode::InvalidConfig);

    g.nodes.pop_back();
    g.edges.push_back(agent::GraphEdge{.from = "x", .to = "missing", .router = nullptr, .user_data = nullptr});
    auto edge = agent::run_graph(session, g, run);
    REQUIRE_FALSE(edge.ok);
    REQUIRE(edge.error.message.find("missing") != std::string::npos);
}

TEST_CASE("graph: loop honours until_fn and max_iterations", "[graph]") {
    agent::Session session = make_session();

    auto until_three = [](agent::Session &, const agent::GraphState &state, void *) -> agent::Result<bool> {
        return agent::ok(agent::get_state(state, "count") == "3");
    };
    static const char *kAfter = "after=yes";
    agent::Graph g = agent::make_sequential(
        "with_loop", {agent::make_loop("loop", agent::make_graph(function_node("inc", increment_count)), 10, until_three),
                      agent::make_graph(function_node("after", write_kv, const_cast<char *>(kAfter)))});
    agent::GraphRun run;
    run.thread_id = "loop-until";
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE(agent::get_state(run.state, "count") == "3");
    REQUIRE(agent::get_state(run.state, "after") == "yes");
    REQUIRE(count_events(run, agent::GraphEventType::NodeStart, "inc") == 3);

    agent::Graph capped = agent::make_loop("capped", agent::make_graph(function_node("inc", increment_count)), 5);
    agent::GraphRun run2;
    run2.thread_id = "loop-capped";
    REQUIRE(agent::run_graph(session, capped, run2).ok);
    REQUIRE(agent::get_state(run2.state, "count") == "5");
    // The loop counter is reset on exit, so the same thread loops again from zero next time.
    REQUIRE_FALSE(agent::has_state(run2.state, "__loop:capped"));
    agent::GraphRun again;
    again.thread_id = "loop-capped";
    REQUIRE(agent::run_graph(session, capped, again).ok);
    REQUIRE(agent::get_state(again.state, "count") == "10");
}

TEST_CASE("graph: parallel fan-out joins branch writes", "[graph]") {
    agent::Session session = make_session();

    static const char *kR1 = "r1=one";
    auto join = function_node("join", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        agent::set_state(state, "joined", std::string(agent::get_state(state, "r1")) + "+" + std::string(agent::get_state(state, "ra")));
        return agent::ok();
    });
    agent::Graph g = agent::make_sequential(
        "fan", {agent::make_parallel("fan_out", {agent::make_graph(function_node("b1", write_kv, const_cast<char *>(kR1))),
                                                 agent::make_graph(agent_node("ba", "q", "ra"))}),
                agent::make_graph(join)});

    agent::GraphRun run;
    run.thread_id = "fan";
    agent::set_state(run.state, "q", "question");
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE(run.status == agent::GraphRunStatus::Completed);
    REQUIRE(agent::get_state(run.state, "joined") == "one+echo: question");
    REQUIRE(run.state.histories.count("ba") == 1);
    REQUIRE(count_events(run, agent::GraphEventType::StateWrite, "b1", "r1") == 1);
    REQUIRE(count_events(run, agent::GraphEventType::NodeEnd, "fan_out") == 1);
}

TEST_CASE("graph: parallel branches writing the same key fail", "[graph]") {
    agent::Session session = make_session();
    static const char *kX = "same=x";
    static const char *kY = "same=y";
    agent::Graph g = agent::make_parallel("conflict", {agent::make_graph(function_node("left", write_kv, const_cast<char *>(kX))),
                                                       agent::make_graph(function_node("right", write_kv, const_cast<char *>(kY)))});
    agent::GraphRun run;
    run.thread_id = "conflict";
    auto res = agent::run_graph(session, g, run);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::InvalidConfig);
    REQUIRE(res.error.message.find("'same'") != std::string::npos);
    REQUIRE(res.error.message.find("left") != std::string::npos);
    REQUIRE(res.error.message.find("right") != std::string::npos);
    REQUIRE(run.status == agent::GraphRunStatus::Failed);
    REQUIRE_FALSE(agent::has_state(run.state, "same"));
}

namespace {

// A concurrent test engine: echoes the last user message and records how many run() calls overlap.
std::atomic<int> g_inflight{0};
std::atomic<int> g_max_inflight{0};

struct ConcurrentReply {
    std::string text;
    agent_part part{};
};

int32_t conc_create(agent_str, void **out, char *, size_t) {
    static int instance = 0;
    *out = &instance;
    return AGENT_OK;
}
void conc_destroy(void *) {}
int32_t conc_run(void *, const agent_request *req, agent_response *resp) {
    int now = ++g_inflight;
    int seen = g_max_inflight.load();
    while (now > seen && !g_max_inflight.compare_exchange_weak(seen, now)) {}
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    thread_local ConcurrentReply reply;
    std::string_view last;
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        if (std::string_view(m.role.data, m.role.size) != "user") continue;
        for (size_t p = 0; p < m.n_parts; ++p)
            if (m.parts[p].kind == AGENT_PART_TEXT) last = std::string_view(m.parts[p].text.data, m.parts[p].text.size);
    }
    reply.text = "conc: " + std::string(last);
    reply.part = agent_part{};
    reply.part.kind = AGENT_PART_TEXT;
    reply.part.text = agent_str{reply.text.data(), reply.text.size()};
    *resp = agent_response{};
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = &reply.part;
    resp->message.n_parts = 1;
    --g_inflight;
    return AGENT_OK;
}
void conc_free(void *, agent_response *resp) { *resp = agent_response{}; }
const char *conc_error(void *) { return ""; }

const agent_engine_v1 kConcurrentEngine = {
    AGENT_ENGINE_ABI_VERSION,
    "graph_concurrent_test",
    "1.0",
    AGENT_CAP_TEXT_GEN,
    AGENT_ENGINE_FLAG_CONCURRENT,
    conc_create,
    conc_destroy,
    conc_run,
    conc_free,
    conc_error,
    nullptr,
    nullptr,
};

} // namespace

TEST_CASE("graph: parallel branches run concurrently on a concurrent engine", "[graph]") {
    REQUIRE(agent::register_engine(&kConcurrentEngine).ok);
    agent::Config cfg;
    cfg.engine = "graph_concurrent_test";
    cfg.max_parallel_subagents = 2; // three branches -> two batches
    agent::Session session = make_session(cfg);
    REQUIRE(session.worker_arenas.size() == 2);

    g_max_inflight = 0;
    agent::Graph g =
        agent::make_parallel("trio", {agent::make_graph(agent_node("c1", "q1", "a1")), agent::make_graph(agent_node("c2", "q2", "a2")),
                                      agent::make_graph(agent_node("c3", "q3", "a3"))});
    agent::GraphRun run;
    run.thread_id = "concurrent";
    agent::set_state(run.state, "q1", "one");
    agent::set_state(run.state, "q2", "two");
    agent::set_state(run.state, "q3", "three");
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE(agent::get_state(run.state, "a1") == "conc: one");
    REQUIRE(agent::get_state(run.state, "a2") == "conc: two");
    REQUIRE(agent::get_state(run.state, "a3") == "conc: three");
    REQUIRE(g_max_inflight.load() >= 2);
}

TEST_CASE("graph: scoped state - temp discarded, user/app persisted, thread state per thread", "[graph]") {
    agent::Session session = make_session();

    auto writer = function_node("write", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        agent::set_state(state, "temp:scratch", "gone");
        agent::set_state(state, "user:name", "Ava");
        agent::set_state(state, "app:version", "1");
        agent::set_state(state, "note", "thread one");
        return agent::ok();
    });
    agent::Graph g1 = agent::make_graph(writer);
    agent::GraphRun first;
    first.thread_id = "t1";
    REQUIRE(agent::run_graph(session, g1, first).ok);
    REQUIRE_FALSE(agent::has_state(first.state, "temp:scratch"));
    REQUIRE(agent::get_state(first.state, "user:name") == "Ava");

    auto reader = function_node("read", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        std::string seen =
            std::string(agent::get_state(state, "user:name", "-")) + "," + std::string(agent::get_state(state, "app:version", "-")) + "," +
            std::string(agent::get_state(state, "note", "-")) + "," + std::string(agent::get_state(state, "temp:scratch", "-"));
        agent::set_state(state, "seen", seen);
        return agent::ok();
    });
    agent::Graph g2 = agent::make_graph(reader);

    agent::GraphRun other;
    other.thread_id = "t2";
    REQUIRE(agent::run_graph(session, g2, other).ok);
    REQUIRE(agent::get_state(other.state, "seen") == "Ava,1,-,-");

    agent::GraphRun same;
    same.thread_id = "t1";
    REQUIRE(agent::run_graph(session, g2, same).ok);
    REQUIRE(agent::get_state(same.state, "seen") == "Ava,1,thread one,-");

    auto stored = agent::retrieve_memory(session, "graph_state:user:name");
    REQUIRE(stored.ok);
    REQUIRE(stored.value == "Ava");
}

namespace {
int g_a_calls = 0;
int g_flaky_calls = 0;
} // namespace

TEST_CASE("graph: checkpoint and resume after a failing node", "[graph]") {
    agent::Session session = make_session();
    g_a_calls = 0;
    g_flaky_calls = 0;

    auto a = function_node("a", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        ++g_a_calls;
        agent::set_state(state, "a", "done");
        return agent::ok();
    });
    auto flaky = function_node("flaky", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        if (++g_flaky_calls == 1) return agent::fail(agent::ErrorCode::NetworkError, "simulated outage");
        agent::set_state(state, "flaky", "recovered");
        return agent::ok();
    });
    static const char *kC = "c=done";
    auto chat = agent_node("chat", "", "reply");
    chat.prompt = "remember me";
    agent::Graph g = agent::make_sequential("durable", {agent::make_graph(a), agent::make_graph(chat), agent::make_graph(flaky),
                                                        agent::make_graph(function_node("c", write_kv, const_cast<char *>(kC)))});

    agent::GraphRun run;
    run.thread_id = "durable";
    run.durability = agent::GraphDurability::Sync;
    auto res = agent::run_graph(session, g, run);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::NetworkError);
    REQUIRE(run.status == agent::GraphRunStatus::Failed);
    REQUIRE(run.next_node == "flaky");
    REQUIRE(count_events(run, agent::GraphEventType::NodeFailed, "flaky") == 1);
    REQUIRE(agent::retrieve_memory(session, "checkpoint:durable:1").ok);
    REQUIRE(agent::retrieve_memory(session, "checkpoint:durable:2").ok);
    REQUIRE(agent::retrieve_memory(session, "checkpoint:durable:latest").ok);

    agent::GraphRun resumed;
    auto again = agent::resume_graph(session, g, "durable", resumed);
    REQUIRE(again.ok);
    REQUIRE(resumed.status == agent::GraphRunStatus::Completed);
    REQUIRE(g_a_calls == 1); // completed nodes are not re-run
    REQUIRE(g_flaky_calls == 2);
    REQUIRE(agent::get_state(resumed.state, "a") == "done");
    REQUIRE(agent::get_state(resumed.state, "reply") == "echo: remember me");
    REQUIRE(agent::get_state(resumed.state, "c") == "done");
    REQUIRE(resumed.state.histories.at("chat").history.size() == 2);
    REQUIRE(resumed.step == 4);
    REQUIRE(count_events(resumed, agent::GraphEventType::Resume) == 1);

    // Resuming a completed thread is a no-op.
    agent::GraphRun done;
    REQUIRE(agent::resume_graph(session, g, "durable", done).ok);
    REQUIRE(done.status == agent::GraphRunStatus::Completed);
    REQUIRE(g_flaky_calls == 2);

    agent::GraphRun none;
    auto missing = agent::resume_graph(session, g, "no-such-thread", none);
    REQUIRE_FALSE(missing.ok);
    REQUIRE(missing.error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("graph: durability Exit checkpoints only when the call returns", "[graph]") {
    agent::Session session = make_session();
    static const char *k1 = "x=1";
    static const char *k2 = "y=2";
    agent::Graph g = agent::make_sequential("exit", {agent::make_graph(function_node("n1", write_kv, const_cast<char *>(k1))),
                                                     agent::make_graph(function_node("n2", write_kv, const_cast<char *>(k2)))});
    agent::GraphRun run;
    run.thread_id = "exit-mode";
    run.durability = agent::GraphDurability::Exit;
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE_FALSE(agent::retrieve_memory(session, "checkpoint:exit-mode:1").ok);
    REQUIRE(agent::retrieve_memory(session, "checkpoint:exit-mode:2").ok);
    REQUIRE(count_events(run, agent::GraphEventType::Checkpoint) == 1);
}

TEST_CASE("graph: function interrupt resumes with human input", "[graph]") {
    agent::Session session = make_session();

    auto ask = function_node("ask", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        if (!agent::has_state(state, "temp:human_input")) {
            agent::request_interrupt(state, "need a name");
            return agent::ok();
        }
        agent::set_state(state, "name", std::string(agent::get_state(state, "temp:human_input")));
        return agent::ok();
    });
    auto greet = function_node("greet", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        agent::set_state(state, "greeting", "hi " + std::string(agent::get_state(state, "name")));
        return agent::ok();
    });
    agent::Graph g = agent::make_sequential("hitl", {agent::make_graph(ask), agent::make_graph(greet)});

    agent::GraphRun run;
    run.thread_id = "hitl";
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE(run.status == agent::GraphRunStatus::Interrupted);
    REQUIRE(run.interrupt_node == "ask");
    REQUIRE(run.interrupt_reason == "need a name");
    REQUIRE(run.next_node == "ask");
    REQUIRE(count_events(run, agent::GraphEventType::Interrupt, "ask") == 1);
    REQUIRE_FALSE(agent::has_state(run.state, "greeting"));

    REQUIRE(agent::resume_graph(session, g, "hitl", run, "Ava").ok);
    REQUIRE(run.status == agent::GraphRunStatus::Completed);
    REQUIRE(agent::get_state(run.state, "greeting") == "hi Ava");
    REQUIRE_FALSE(agent::has_state(run.state, "temp:human_input"));
    REQUIRE(count_events(run, agent::GraphEventType::StateWrite, {}, "temp:human_input") == 1);
}

TEST_CASE("graph: interrupt_before_tools pauses for approval", "[graph]") {
    agent::Session session = make_session();
    g_ping_calls = 0;

    auto worker = agent_node("worker", "", "out");
    worker.prompt = "call:ping";
    worker.allowed_tools = {"ping"};
    worker.interrupt_before_tools = true;
    agent::Graph g = agent::make_graph(worker);

    SECTION("approved") {
        agent::GraphRun run;
        run.thread_id = "approve";
        REQUIRE(agent::run_graph(session, g, run).ok);
        REQUIRE(run.status == agent::GraphRunStatus::Interrupted);
        REQUIRE(run.pending_tool_calls.size() == 1);
        REQUIRE(run.pending_tool_calls[0].name == "ping");
        REQUIRE(run.interrupt_reason.find("ping") != std::string::npos);
        REQUIRE(g_ping_calls == 0);

        agent::GraphRun resumed;
        REQUIRE(agent::resume_graph(session, g, "approve", resumed, "approve").ok);
        REQUIRE(resumed.status == agent::GraphRunStatus::Completed);
        REQUIRE(g_ping_calls == 1);
        REQUIRE(agent::get_state(resumed.state, "out") == "tool done");
        REQUIRE(count_events(resumed, agent::GraphEventType::ToolCall, "worker", "ping") == 1);
        REQUIRE(count_events(resumed, agent::GraphEventType::ToolResult, "worker", "ping") == 1);
        const auto &history = resumed.state.histories.at("worker").history;
        REQUIRE(history.size() == 4); // user, assistant(tool call), tool, assistant
        REQUIRE(history[2].content == "pong");
    }

    SECTION("denied") {
        agent::GraphRun run;
        run.thread_id = "deny";
        REQUIRE(agent::run_graph(session, g, run).ok);
        REQUIRE(run.status == agent::GraphRunStatus::Interrupted);

        REQUIRE(agent::resume_graph(session, g, "deny", run, agent::PermissionDecision::Deny, "not now").ok);
        REQUIRE(run.status == agent::GraphRunStatus::Completed);
        REQUIRE(g_ping_calls == 0);
        const auto &history = run.state.histories.at("worker").history;
        REQUIRE(history[2].role == "tool");
        REQUIRE(history[2].content.find("denied") != std::string::npos);
        REQUIRE(history[2].content.find("not now") != std::string::npos);
    }
}

TEST_CASE("graph: interrupt_before_tools only pauses for Ask under a permission policy", "[graph]") {
    agent::Config cfg;
    cfg.permission_check = [](std::string_view tool, std::string_view, void *) {
        return tool == "ping" ? agent::PermissionDecision::Allow : agent::PermissionDecision::Ask;
    };
    agent::Session session = make_session(cfg);
    g_ping_calls = 0;

    auto worker = agent_node("worker", "", "out");
    worker.prompt = "call:ping";
    worker.interrupt_before_tools = true;
    agent::GraphRun run;
    run.thread_id = "policy";
    REQUIRE(agent::run_graph(session, agent::make_graph(worker), run).ok);
    REQUIRE(run.status == agent::GraphRunStatus::Completed);
    REQUIRE(g_ping_calls == 1);
    // Tool calls made inside run_turn are logged too.
    REQUIRE(count_events(run, agent::GraphEventType::ToolCall, "worker", "ping") == 1);
    REQUIRE(count_events(run, agent::GraphEventType::ToolResult, "worker", "ping") == 1);
}

TEST_CASE("graph: tool node runs a tool with arguments from state", "[graph]") {
    agent::Session session = make_session();
    g_ping_calls = 0;
    int tool_hooks = 0;
    agent::register_event_hook(
        session, agent::EventType::OnToolCall, [](const agent::Event &, void *ud) { ++*static_cast<int *>(ud); }, &tool_hooks);

    agent::GraphNode tool;
    tool.name = "pinger";
    tool.kind = agent::GraphNodeKind::Tool;
    tool.tool_name = "ping";
    tool.input_key = "args";
    tool.output_key = "pong";
    agent::GraphRun run;
    run.thread_id = "tool";
    agent::set_state(run.state, "args", "{\"x\":1}");
    REQUIRE(agent::run_graph(session, agent::make_graph(tool), run).ok);
    REQUIRE(g_ping_calls == 1);
    REQUIRE(tool_hooks == 1);
    REQUIRE(agent::get_state(run.state, "pong") == "pong");
    REQUIRE(count_events(run, agent::GraphEventType::ToolCall, "pinger", "ping") == 1);
    for (const auto &e : run.events)
        if (e.type == agent::GraphEventType::ToolCall) REQUIRE(e.payload == "{\"x\":1}");

    tool.tool_name = "no_such_tool";
    agent::GraphRun bad;
    bad.thread_id = "tool-bad";
    REQUIRE_FALSE(agent::run_graph(session, agent::make_graph(tool), bad).ok);
    REQUIRE(bad.status == agent::GraphRunStatus::Failed);
}
