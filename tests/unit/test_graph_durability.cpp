// Graph durability: CheckpointerOps (in-memory, SQLite, StoreOps adapter), Async writer, resume in a
// new Session.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

const char *kWorkspace = "/tmp/agent_test_graph_durability_ws";

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

agent::GraphNode kv_node(std::string name, const char *kv) { return function_node(std::move(name), write_kv, const_cast<char *>(kv)); }

size_t count_events(const std::vector<agent::GraphEvent> &events, agent::GraphEventType type, std::string_view node = {}) {
    size_t n = 0;
    for (const auto &e : events)
        if (e.type == type && (node.empty() || e.node == node)) ++n;
    return n;
}

std::string temp_db(std::string_view name) {
    std::filesystem::create_directories(kWorkspace);
    std::string path = std::string(kWorkspace) + "/" + std::string(name);
    for (const char *suffix : {"", "-wal", "-shm"}) std::filesystem::remove(path + suffix);
    return path;
}

void check_round_trip(const agent::Checkpointer &cp) {
    const agent::CheckpointerOps *ops = cp.ops;
    void *s = cp.state.get();
    auto missing = ops->get_latest(s, "t");
    REQUIRE_FALSE(missing.ok);
    REQUIRE(missing.error.code == agent::ErrorCode::KeyNotFound);
    REQUIRE(ops->list(s, "t").value.empty());

    REQUIRE(ops->put(s, "t", 1, "one", "{\"n\":1}").ok);
    REQUIRE(ops->put(s, "t", 2, "two", "{\"n\":2}").ok);
    REQUIRE(ops->put(s, "other", 7, "seven", "{}").ok);
    REQUIRE(ops->get_latest(s, "t").value == "two");
    REQUIRE(ops->get(s, "t", 1).value == "one");
    REQUIRE(ops->get(s, "t", 3).error.code == agent::ErrorCode::KeyNotFound);
    REQUIRE(ops->put(s, "t", 2, "two again", "{\"n\":22}").ok); // replace
    REQUIRE(ops->get_latest(s, "t").value == "two again");

    auto listed = ops->list(s, "t");
    REQUIRE(listed.ok);
    REQUIRE(listed.value.size() == 2);
    REQUIRE(listed.value[0].step == 1);
    REQUIRE(listed.value[1].step == 2);
    REQUIRE(listed.value[1].metadata == "{\"n\":22}");
    REQUIRE(listed.value[0].timestamp_ms > 0);

    REQUIRE(ops->delete_thread(s, "t").ok);
    REQUIRE(ops->get_latest(s, "t").error.code == agent::ErrorCode::KeyNotFound);
    REQUIRE(ops->list(s, "t").value.empty());
    REQUIRE(ops->get_latest(s, "other").value == "seven");
}

} // namespace

TEST_CASE("graph durability: checkpointer round trips", "[graph][durability]") {
    SECTION("in memory") {
        auto cp = agent::open_checkpointer(agent::in_memory_checkpointer());
        REQUIRE(cp.ok);
        check_round_trip(cp.value);
    }
    SECTION("sqlite :memory:") {
        auto cp = agent::open_checkpointer(agent::store::sqlite_checkpointer());
        REQUIRE(cp.ok);
        check_round_trip(cp.value);
    }
    SECTION("sqlite file survives reopening") {
        const std::string path = temp_db("roundtrip.sqlite3");
        {
            auto cp = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = path});
            REQUIRE(cp.ok);
            check_round_trip(cp.value);
            REQUIRE(cp.value.ops->put(cp.value.state.get(), "kept", 3, "three", "{}").ok);
        }
        auto again = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = path});
        REQUIRE(again.ok);
        REQUIRE(again.value.ops->get_latest(again.value.state.get(), "kept").value == "three");
        REQUIRE(again.value.ops->get_latest(again.value.state.get(), "other").value == "seven");
    }
    SECTION("incomplete ops are rejected") {
        agent::CheckpointerOps partial = *agent::in_memory_checkpointer();
        partial.list = nullptr;
        auto cp = agent::open_checkpointer(&partial);
        REQUIRE_FALSE(cp.ok);
        REQUIRE(cp.error.code == agent::ErrorCode::InvalidConfig);
    }
}

TEST_CASE("graph durability: runs checkpoint through the selected checkpointer", "[graph][durability]") {
    agent::Session session = make_session();
    auto cp = agent::open_checkpointer(agent::in_memory_checkpointer());
    REQUIRE(cp.ok);

    agent::Graph g = agent::make_sequential("cp", {agent::make_graph(kv_node("n1", "x=1")), agent::make_graph(kv_node("n2", "y=2"))});
    agent::GraphRun run;
    run.thread_id = "selected";
    run.durability = agent::GraphDurability::Sync;
    run.checkpointer = cp.value;
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE(run.status == agent::GraphRunStatus::Completed);

    // Nothing went to the Session's StoreOps.
    REQUIRE_FALSE(agent::retrieve_memory(session, "checkpoint:selected:latest").ok);
    auto listed = agent::list_graph_checkpoints(session, "selected", cp.value);
    REQUIRE(listed.ok);
    REQUIRE(listed.value.size() == 2);
    REQUIRE(listed.value[0].step == 1);
    REQUIRE(listed.value[1].step == 2);
    REQUIRE(listed.value[1].metadata.find("\"status\":\"completed\"") != std::string::npos);
    REQUIRE(listed.value[0].metadata.find("\"next\":\"n2\"") != std::string::npos);

    // A later run of the thread continues from the checkpointer's state.
    agent::GraphRun more;
    more.thread_id = "selected";
    more.checkpointer = cp.value;
    REQUIRE(agent::run_graph(session, agent::make_graph(kv_node("n3", "z=3")), more).ok);
    REQUIRE(agent::get_state(more.state, "x") == "1");
    REQUIRE(more.step == 3);

    REQUIRE(agent::delete_graph_thread(session, "selected", cp.value).ok);
    REQUIRE(agent::list_graph_checkpoints(session, "selected", cp.value).value.empty());

    // An unopened checkpointer is a configuration error.
    agent::GraphRun bad;
    bad.checkpointer.ops = agent::in_memory_checkpointer();
    auto res = agent::run_graph(session, g, bad);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("graph durability: the default StoreOps adapter lists and deletes checkpoints", "[graph][durability]") {
    agent::Session session = make_session();
    agent::Graph g = agent::make_sequential("store", {agent::make_graph(kv_node("s1", "a=1")), agent::make_graph(kv_node("s2", "b=2"))});
    agent::GraphRun run;
    run.thread_id = "store-adapter";
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE(agent::retrieve_memory(session, "checkpoint:store-adapter:1").ok); // v1 key layout kept
    auto listed = agent::list_graph_checkpoints(session, "store-adapter");
    REQUIRE(listed.ok);
    REQUIRE(listed.value.size() == 2);
    REQUIRE(listed.value[1].step == 2);
    REQUIRE(listed.value[1].timestamp_ms > 0);

    REQUIRE(agent::delete_graph_thread(session, "store-adapter").ok);
    REQUIRE(agent::list_graph_checkpoints(session, "store-adapter").value.empty());
    agent::GraphRun resumed;
    REQUIRE(agent::resume_graph(session, g, "store-adapter", resumed).error.code == agent::ErrorCode::KeyNotFound);
    // A fresh run starts the thread over.
    agent::GraphRun fresh;
    fresh.thread_id = "store-adapter";
    REQUIRE(agent::run_graph(session, g, fresh).ok);
    REQUIRE(fresh.step == 2);
}

namespace {

// A slow checkpointer (wraps the in-memory one) that records which threads called put.
std::mutex g_slow_mutex;
std::vector<std::thread::id> g_put_threads;
std::vector<uint64_t> g_put_steps;

agent::Result<void> slow_put(void *state, std::string_view thread_id, uint64_t step, std::string_view blob, std::string_view metadata) {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    {
        std::lock_guard<std::mutex> lock(g_slow_mutex);
        g_put_threads.push_back(std::this_thread::get_id());
        g_put_steps.push_back(step);
    }
    return agent::in_memory_checkpointer()->put(state, thread_id, step, blob, metadata);
}

const agent::CheckpointerOps kSlowCheckpointer = [] {
    agent::CheckpointerOps ops = *agent::in_memory_checkpointer();
    ops.put = slow_put;
    return ops;
}();

} // namespace

TEST_CASE("graph durability: Async writes checkpoints on a background thread and flushes on return", "[graph][durability]") {
    agent::Session session = make_session();
    auto cp = agent::open_checkpointer(&kSlowCheckpointer, {}, 2);
    REQUIRE(cp.ok);
    {
        std::lock_guard<std::mutex> lock(g_slow_mutex);
        g_put_threads.clear();
        g_put_steps.clear();
    }

    std::vector<agent::Graph> steps;
    static const char *kv[] = {"a=1", "b=2", "c=3", "d=4", "e=5"};
    for (int i = 0; i < 5; ++i) steps.push_back(agent::make_graph(kv_node("n" + std::to_string(i), kv[i])));
    agent::Graph g = agent::make_sequential("async", std::move(steps));

    agent::GraphRun run;
    run.thread_id = "async";
    run.durability = agent::GraphDurability::Async;
    run.checkpointer = cp.value;
    REQUIRE(agent::run_graph(session, g, run).ok);

    // Flushed: every checkpoint is stored by the time run_graph returns, in step order.
    auto listed = agent::list_graph_checkpoints(session, "async", cp.value);
    REQUIRE(listed.ok);
    REQUIRE(listed.value.size() == 5);
    {
        std::lock_guard<std::mutex> lock(g_slow_mutex);
        REQUIRE(g_put_steps == std::vector<uint64_t>{1, 2, 3, 4, 5});
        // Steps 1-4 went through the writer thread; the final checkpoint is written by the caller.
        for (size_t i = 0; i + 1 < g_put_threads.size(); ++i) REQUIRE(g_put_threads[i] != std::this_thread::get_id());
        REQUIRE(g_put_threads.back() == std::this_thread::get_id());
        g_put_threads.clear();
        g_put_steps.clear();
    }

    // Sync writes on the calling thread.
    agent::GraphRun sync;
    sync.thread_id = "sync";
    sync.durability = agent::GraphDurability::Sync;
    sync.checkpointer = cp.value;
    REQUIRE(agent::run_graph(session, g, sync).ok);
    std::lock_guard<std::mutex> lock(g_slow_mutex);
    REQUIRE(g_put_threads.size() == 5);
    for (auto id : g_put_threads) REQUIRE(id == std::this_thread::get_id());
}

namespace {
int g_before_calls = 0;
int g_crash_calls = 0;
} // namespace

TEST_CASE("graph durability: resume from the sqlite checkpointer in a new Session after a failure", "[graph][durability]") {
    const std::string path = temp_db("crash.sqlite3");
    g_before_calls = 0;
    g_crash_calls = 0;

    auto before = function_node("before", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        ++g_before_calls;
        agent::set_state(state, "before", "done");
        return agent::ok();
    });
    auto crash = function_node("crash", [](agent::Session &, agent::GraphState &state, void *) -> agent::Result<void> {
        if (++g_crash_calls == 1) return agent::fail(agent::ErrorCode::EngineCrashed, "simulated crash");
        agent::set_state(state, "crash", "recovered");
        return agent::ok();
    });
    auto chat = agent_node("chat", "", "reply");
    chat.prompt = "hello";
    agent::Graph g = agent::make_sequential("crashy", {agent::make_graph(before), agent::make_graph(chat), agent::make_graph(crash),
                                                       agent::make_graph(kv_node("after", "after=yes"))});

    {
        agent::Session first = make_session();
        auto cp = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = path});
        REQUIRE(cp.ok);
        agent::GraphRun run;
        run.thread_id = "crashy";
        run.checkpointer = cp.value; // Async (default)
        auto res = agent::run_graph(first, g, run);
        REQUIRE_FALSE(res.ok);
        REQUIRE(res.error.code == agent::ErrorCode::EngineCrashed);
        REQUIRE(run.next_node == "crash");
    } // "process exit": Session and checkpointer gone

    agent::Session second = make_session();
    auto cp = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = path});
    REQUIRE(cp.ok);
    agent::GraphRun resumed;
    resumed.checkpointer = cp.value;
    auto res = agent::resume_graph(second, g, "crashy", resumed);
    REQUIRE(res.ok);
    REQUIRE(resumed.status == agent::GraphRunStatus::Completed);
    REQUIRE(g_before_calls == 1);
    REQUIRE(g_crash_calls == 2);
    REQUIRE(agent::get_state(resumed.state, "before") == "done");
    REQUIRE(agent::get_state(resumed.state, "reply") == "echo: hello");
    REQUIRE(agent::get_state(resumed.state, "after") == "yes");
    REQUIRE(resumed.state.histories.at("chat").history.size() == 2);
    REQUIRE(count_events(resumed.events, agent::GraphEventType::NodeStart, "before") == 0);
}

TEST_CASE("graph durability: a fresh run after an interrupt answers the abandoned tool calls", "[graph][durability]") {
    agent::Session session = make_session();
    g_ping_calls = 0;

    auto worker = agent_node("worker", "msg", "out");
    worker.allowed_tools = {"ping"};
    worker.interrupt_before_tools = true;
    agent::Graph g = agent::make_graph(worker);

    agent::GraphRun run;
    run.thread_id = "abandoned";
    agent::set_state(run.state, "msg", "call:ping");
    REQUIRE(agent::run_graph(session, g, run).ok);
    REQUIRE(run.status == agent::GraphRunStatus::Interrupted);
    REQUIRE(run.pending_tool_calls.size() == 1);

    // Instead of resume_graph, start a new run on the same thread.
    agent::GraphRun fresh;
    fresh.thread_id = "abandoned";
    agent::set_state(fresh.state, "msg", "hello again");
    REQUIRE(agent::run_graph(session, g, fresh).ok);
    REQUIRE(fresh.status == agent::GraphRunStatus::Completed);
    REQUIRE(fresh.pending_tool_calls.empty());
    REQUIRE(g_ping_calls == 0);
    REQUIRE(agent::get_state(fresh.state, "out") == "echo: hello again");

    // user, assistant(tool call), tool(error), user, assistant
    const auto &history = fresh.state.histories.at("worker").history;
    REQUIRE(history.size() == 5);
    REQUIRE(history[1].tool_calls.size() == 1);
    REQUIRE(history[2].role == "tool");
    REQUIRE(history[2].tool_call_id == history[1].tool_calls[0].id);
    REQUIRE(history[2].content.find("not executed") != std::string::npos);
    REQUIRE(history[3].content == "hello again");
    REQUIRE(count_events(fresh.events, agent::GraphEventType::ToolResult, "worker") == 1);

    // The answered history is what got checkpointed: resuming the completed thread changes nothing.
    agent::GraphRun again;
    REQUIRE(agent::resume_graph(session, g, "abandoned", again).ok);
    REQUIRE(again.state.histories.at("worker").history.size() == 5);
}
