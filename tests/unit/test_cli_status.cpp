// agent run status view (apps/cli/src/status_view.{hpp,cpp}): the pure render / format functions and
// the table fed by GraphRun::on_event.

#include "status_view.hpp"

#include <algorithm>
#include <catch_amalgamated.hpp>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace agent::cli;

namespace {

agent::GraphEvent ev(agent::GraphEventType type, std::string node, int64_t ts, std::string key = {}, std::string payload = {}) {
    agent::GraphEvent e;
    e.type = type;
    e.node = std::move(node);
    e.timestamp_ms = ts;
    e.key = std::move(key);
    e.payload = std::move(payload);
    return e;
}

std::string slurp(std::FILE *f) {
    std::fflush(f);
    std::rewind(f);
    std::string out;
    char buf[512];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    return out;
}

agent::Result<void> noop_node(agent::Session &, agent::GraphState &, void *) { return agent::ok(); }

agent::Result<void> failing_node(agent::Session &, agent::GraphState &, void *) {
    return agent::fail(agent::ErrorCode::InvalidConfig, "boom");
}

agent::GraphNode function_node(std::string name, agent::GraphFunctionFn fn) {
    agent::GraphNode n;
    n.name = std::move(name);
    n.kind = agent::GraphNodeKind::Function;
    n.function = fn;
    return n;
}

const NodeStatus &row_named(const std::vector<NodeStatus> &rows, std::string_view name) {
    const auto it = std::find_if(rows.begin(), rows.end(), [&](const NodeStatus &r) { return r.name == name; });
    REQUIRE(it != rows.end());
    return *it;
}

using T = agent::GraphEventType;

} // namespace

TEST_CASE("cli: status view picks live output only on an interactive terminal", "[cli]") {
    CHECK(choose_status_mode(false, true, true, "xterm-256color", nullptr) == StatusMode::Live);
    CHECK(choose_status_mode(false, true, true, nullptr, nullptr) == StatusMode::Live); // TERM is often unset on Windows
    CHECK(choose_status_mode(false, true, true, "xterm", "") == StatusMode::Live);      // an empty NO_COLOR does not count
    CHECK(choose_status_mode(false, true, true, "dumb", nullptr) == StatusMode::Plain);
    CHECK(choose_status_mode(false, true, true, "xterm", "1") == StatusMode::Plain);
    CHECK(choose_status_mode(false, false, true, "xterm", nullptr) == StatusMode::Plain); // stdout piped
    CHECK(choose_status_mode(false, true, false, "xterm", nullptr) == StatusMode::Plain); // stderr redirected
    CHECK(choose_status_mode(true, true, true, "xterm", nullptr) == StatusMode::Json);
}

TEST_CASE("cli: format_elapsed", "[cli]") {
    CHECK(format_elapsed(-5) == "0.0s");
    CHECK(format_elapsed(400) == "0.4s");
    CHECK(format_elapsed(12345) == "12.3s");
    CHECK(format_elapsed(125000) == "2m05s");
    CHECK(format_elapsed(3720000) == "1h02m");
}

TEST_CASE("cli: render_status turns a node table into text", "[cli]") {
    std::vector<NodeStatus> rows;
    apply_event(rows, ev(T::NodeStart, "plan", 1000), "agent");
    apply_event(rows, ev(T::NodeEnd, "plan", 2500), "agent");
    apply_event(rows, ev(T::NodeStart, "left", 2500), "agent");
    apply_event(rows, ev(T::NodeStart, "right", 2500), "function");
    apply_event(rows, ev(T::ToolCall, "left", 3000, "read_file", "{}"), "agent");
    apply_event(rows, ev(T::NodeFailed, "right", 4000, {}, "boom\nsecond line"), "function");
    apply_event(rows, ev(T::NodeStart, "review", 4000), "agent");
    apply_event(rows, ev(T::Interrupt, "review", 4100, {}, "needs approval"), "agent");

    CHECK(render_status(rows, 6500) == "plan    agent     done         1.5s\n"
                                       "left    agent     running      4.0s  tool: read_file\n"
                                       "right   function  error        1.5s  - boom\n"
                                       "review  agent     interrupted  0.1s  - needs approval\n");

    // Long lines are cut, ending in "...".
    const std::string narrow = render_status(rows, 6500, 30);
    CHECK(narrow.find("tool: read_file") == std::string::npos);
    CHECK(narrow.find("left    agent     running  ...\n") != std::string::npos);

    // Too many rows: the oldest finished ones are summarised.
    const std::string capped = render_status(rows, 6500, 0, 3);
    CHECK(capped.find("plan") == std::string::npos);
    CHECK(capped.find("(1 finished nodes hidden)") != std::string::npos);
    CHECK(render_status({}, 0).empty());
}

TEST_CASE("cli: a node that runs again starts over, tool errors are marked", "[cli]") {
    std::vector<NodeStatus> rows;
    apply_event(rows, ev(T::NodeStart, "step", 100), "loop");
    apply_event(rows, ev(T::ToolResult, "step", 150, "grep", "Error: no such file"), "loop");
    CHECK(rows[0].last_tool == "grep (error)");
    apply_event(rows, ev(T::NodeEnd, "step", 200), "loop");
    apply_event(rows, ev(T::NodeStart, "step", 300), "loop");
    REQUIRE(rows.size() == 1);
    CHECK(rows[0].state == NodeState::Running);
    CHECK(rows[0].last_tool.empty());
    CHECK(apply_event(rows, ev(T::StateWrite, "step", 310, "k", "v"), "loop") == nullptr);
    CHECK(apply_event(rows, ev(T::ToolCall, "unknown", 310, "x"), "loop") == nullptr);
}

TEST_CASE("cli: plain and json event lines", "[cli]") {
    std::vector<NodeStatus> rows;
    auto line = [&](const agent::GraphEvent &e, std::string_view kind) {
        const NodeStatus *r = apply_event(rows, e, kind);
        REQUIRE(r != nullptr);
        return std::pair{format_event_line(e, *r), format_event_json(e, *r)};
    };
    auto start = line(ev(T::NodeStart, "a", 1000), "agent");
    CHECK(start.first == "[node] a (agent) running");
    CHECK(start.second == R"({"event":"node_start","node":"a","kind":"agent","ts":1000})");
    auto tool = line(ev(T::ToolCall, "a", 1200, "write_file"), "agent");
    CHECK(tool.first == "[node] a (agent) tool: write_file");
    CHECK(tool.second == R"({"event":"tool_call","node":"a","kind":"agent","ts":1200,"tool":"write_file"})");
    CHECK(line(ev(T::NodeEnd, "a", 3400), "agent").first == "[node] a (agent) done in 2.4s");

    line(ev(T::NodeStart, "b", 5000), "tool");
    CHECK(line(ev(T::NodeFailed, "b", 5100, {}, "bad \"input\""), "tool").first == "[node] b (tool) error after 0.1s: bad \"input\"");
    line(ev(T::NodeStart, "b", 5200), "tool");
    CHECK(line(ev(T::Interrupt, "b", 5300, {}, "ok?"), "tool").second ==
          R"({"event":"interrupt","node":"b","kind":"tool","ts":5300,"elapsed_ms":100,"reason":"ok?"})");

    const auto write = ev(T::StateWrite, "b", 5400, "k", "v");
    CHECK(format_event_line(write, rows[1]).empty());
    CHECK(format_event_json(write, rows[1]).empty());
}

TEST_CASE("cli: StatusView live mode redraws in place, plain mode prints lines", "[cli]") {
    agent::Graph g = agent::make_graph(function_node("only", noop_node));
    {
        std::FILE *f = std::tmpfile();
        REQUIRE(f != nullptr);
        {
            StatusView view(f, StatusMode::Live, g, 80);
            view.on_event(ev(T::NodeStart, "only", 1000));
            view.on_event(ev(T::NodeEnd, "only", 1500));
            view.finish();
            view.finish(); // safe to repeat
        }
        const std::string out = slurp(f);
        CHECK(out.find("only  function  running") != std::string::npos);
        CHECK(out.find("\x1b[1A") != std::string::npos); // moved back up over the previous frame
        CHECK(out.find("\x1b[2K") != std::string::npos);
        CHECK(out.find("done") != std::string::npos);
        std::fclose(f);
    }
    {
        std::FILE *f = std::tmpfile();
        REQUIRE(f != nullptr);
        StatusView view(f, StatusMode::Plain, g);
        view.on_event(ev(T::NodeStart, "only", 1000));
        view.on_event(ev(T::StateWrite, "only", 1100, "k", "v"));
        view.on_event(ev(T::NodeEnd, "only", 1500));
        view.finish();
        CHECK(slurp(f) == "[node] only (function) running\n[node] only (function) done in 0.5s\n");
        std::fclose(f);
    }
}

TEST_CASE("cli: StatusView tracks events from many threads at once", "[cli]") {
    constexpr int kThreads = 8;
    constexpr int kTools = 50;
    std::FILE *f = std::tmpfile();
    REQUIRE(f != nullptr);
    agent::Graph g;
    for (int i = 0; i < kThreads; ++i) g.nodes.push_back(function_node("n" + std::to_string(i), noop_node));
    StatusView view(f, StatusMode::Plain, g);

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i)
        threads.emplace_back([&view, i] {
            const std::string node = "n" + std::to_string(i);
            view.on_event(ev(T::NodeStart, node, 1000));
            for (int k = 0; k < kTools; ++k) view.on_event(ev(T::ToolCall, node, 1000 + k, "tool" + std::to_string(k)));
            view.on_event(ev(T::NodeEnd, node, 2000));
        });
    for (auto &t : threads) t.join();
    view.finish();

    const auto rows = view.snapshot();
    REQUIRE(rows.size() == kThreads);
    for (const auto &r : rows) {
        CHECK(r.state == NodeState::Done);
        CHECK(r.last_tool == "tool" + std::to_string(kTools - 1));
        CHECK(r.kind == "function");
    }
    // Every output line is whole: the mutex keeps writers from interleaving.
    const std::string out = slurp(f);
    size_t lines = 0;
    for (size_t pos = 0; pos < out.size();) {
        const size_t nl = out.find('\n', pos);
        REQUIRE(nl != std::string::npos);
        CHECK(out.compare(pos, 7, "[node] ") == 0);
        ++lines;
        pos = nl + 1;
    }
    CHECK(lines == static_cast<size_t>(kThreads) * (kTools + 2));
    std::fclose(f);
}

TEST_CASE("cli: StatusView follows a parallel graph through GraphRun::on_event", "[cli]") {
    const auto ws = std::filesystem::temp_directory_path() / "agent_test_cli_status_ws";
    std::filesystem::create_directories(ws);
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.model = "mock.bin";
    cfg.workspace_dir = ws.string();
    auto init = agent::init(cfg);
    REQUIRE(init.ok);
    agent::Session session = std::move(init.value);

    agent::Graph g = agent::make_sequential("status", {agent::make_parallel("fan", {agent::make_graph(function_node("left", noop_node)),
                                                                                    agent::make_graph(function_node("right", noop_node))}),
                                                       agent::make_graph(function_node("join", noop_node))});

    std::FILE *f = std::tmpfile();
    REQUIRE(f != nullptr);
    StatusView view(f, StatusMode::Plain, g);
    agent::GraphRun run;
    run.thread_id = "status";
    run.on_event = StatusView::on_graph_event;
    run.on_event_user_data = &view;
    REQUIRE(agent::run_graph(session, g, run).ok);
    view.finish();

    CHECK(run.status == agent::GraphRunStatus::Completed);
    const auto rows = view.snapshot();
    REQUIRE(rows.size() == 4);
    for (const char *name : {"fan", "left", "right", "join"}) CHECK(row_named(rows, name).state == NodeState::Done);
    CHECK(row_named(rows, "fan").kind == "parallel");
    CHECK(row_named(rows, "left").kind == "function");

    // The live feed carried the node start and end events that ended up in run.events.
    const std::string out = slurp(f);
    size_t expected = 0;
    for (const auto &e : run.events)
        if (e.type == T::NodeStart || e.type == T::NodeEnd) ++expected;
    CHECK(static_cast<size_t>(std::count(out.begin(), out.end(), '\n')) == expected);
    CHECK(out.find("[node] left (function) running") != std::string::npos);
    CHECK(out.find("[node] right (function) done in") != std::string::npos);
    std::fclose(f);

    // A failing node ends as an error row.
    agent::Graph bad = agent::make_graph(function_node("bad", failing_node));
    std::FILE *f2 = std::tmpfile();
    REQUIRE(f2 != nullptr);
    StatusView view2(f2, StatusMode::Plain, bad);
    agent::GraphRun run2;
    run2.thread_id = "status-bad";
    run2.on_event = StatusView::on_graph_event;
    run2.on_event_user_data = &view2;
    CHECK_FALSE(agent::run_graph(session, bad, run2).ok);
    view2.finish();
    const auto bad_rows = view2.snapshot();
    REQUIRE(bad_rows.size() == 1);
    CHECK(bad_rows[0].state == NodeState::Error);
    CHECK(bad_rows[0].detail.find("boom") != std::string::npos);
    std::fclose(f2);
}
