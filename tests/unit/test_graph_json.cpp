#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

const char *kWorkspace = "/tmp/agent_test_graph_json_ws";

agent::Tool make_echo_args_tool() {
    agent::Tool t;
    t.name = "echo_args";
    t.description = "returns its arguments";
    t.parameter_schema = "{}";
    t.callback = [](std::string_view args, void *) -> agent::Result<std::string> { return agent::ok(std::string(args)); };
    return t;
}

agent::Session make_session() {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.model = "mock.bin";
    cfg.workspace_dir = kWorkspace;
    cfg.auto_default_tools = false;
    cfg.tools.push_back(make_echo_args_tool());
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    return std::move(res.value);
}

agent::Result<void> count_up(agent::Session &, agent::GraphState &state, void *) {
    int n = std::stoi(std::string(agent::get_state(state, "count", "0")));
    agent::set_state(state, "count", std::to_string(n + 1));
    if (n + 1 == 2) agent::set_state(state, "status", "done");
    return agent::ok();
}

agent::Result<void> append_trace(agent::Session &, agent::GraphState &state, void *ud) {
    std::string trace(agent::get_state(state, "trace"));
    trace += static_cast<const char *>(ud);
    agent::set_state(state, "trace", std::move(trace));
    return agent::ok();
}

const char kA[] = "a";
const char kB[] = "b";
const char kC[] = "c";
const char kD[] = "d";

struct Registrations {
    Registrations() {
        agent::register_graph_function("count_up", count_up);
        agent::register_graph_function("trace_a", append_trace, const_cast<char *>(kA));
        agent::register_graph_function("trace_b", append_trace, const_cast<char *>(kB));
        agent::register_graph_function("trace_c", append_trace, const_cast<char *>(kC));
        agent::register_graph_function("trace_d", append_trace, const_cast<char *>(kD));
    }
};

void ensure_registered() { static Registrations r; }

std::string error_of(std::string_view json) {
    ensure_registered();
    auto r = agent::parse_graph_json(json);
    if (r.ok) return "<ok>";
    return r.error.message;
}

bool has_warning(const agent::LoadedGraph &g, std::string_view needle) {
    for (const auto &w : g.warnings)
        if (w.find(needle) != std::string::npos) return true;
    return false;
}

} // namespace

TEST_CASE("agent.json: agent nodes, static edges and hints", "[graph_json]") {
    auto session = make_session();
    auto loaded = agent::load_graph_json(session, R"({
        "name": "two_step",
        "engine": "llama",
        "model": "qwen3-4b",
        "checkpointer": "sqlite",
        "description": "test graph",
        "nodes": [
            {"name": "first", "kind": "agent", "system_prompt": "be brief", "input_key": "topic", "output_key": "draft"},
            {"name": "second", "kind": "agent", "input_key": "draft", "output_key": "final",
             "allowed_tools": ["echo_args"], "interrupt_before_tools": false, "history_key": "shared"}
        ],
        "edges": [{"from": "first", "to": "second"}, {"from": "second", "to": "__end__"}]
    })");
    REQUIRE(loaded.ok);
    const agent::LoadedGraph &lg = loaded.value;
    CHECK(lg.warnings.empty());
    CHECK(lg.engine == "llama");
    CHECK(lg.model == "qwen3-4b");
    CHECK(lg.checkpointer == "sqlite");
    CHECK(lg.description == "test graph");
    CHECK(lg.graph.name == "two_step");
    CHECK(lg.graph.entry == "first");
    REQUIRE(lg.graph.nodes.size() == 2);
    CHECK(lg.graph.nodes[0].system_prompt == "be brief");
    CHECK(lg.graph.nodes[1].allowed_tools == std::vector<std::string>{"echo_args"});
    CHECK(lg.graph.nodes[0].history_key.empty());
    CHECK(lg.graph.nodes[1].history_key == "shared");

    agent::GraphRun run;
    run.thread_id = "static";
    agent::set_state(run.state, "topic", "cats");
    auto res = agent::run_graph(session, lg.graph, run);
    REQUIRE(res.ok);
    CHECK(run.status == agent::GraphRunStatus::Completed);
    CHECK(agent::get_state(run.state, "draft") == "echo: cats");
    CHECK(agent::get_state(run.state, "final") == "echo: echo: cats");
    CHECK(run.state.histories.count("first") == 1);
    CHECK(run.state.histories.count("shared") == 1);
    CHECK(run.state.histories.count("second") == 0);
}

TEST_CASE("agent.json: checkpointer hint", "[graph_json]") {
    auto memory = agent::parse_graph_json(R"({"checkpointer": "memory", "nodes": [{"name": "a", "kind": "agent"}]})");
    REQUIRE(memory.ok);
    CHECK(memory.value.checkpointer == "memory");
    auto unset = agent::parse_graph_json(R"({"nodes": [{"name": "a", "kind": "agent"}]})");
    REQUIRE(unset.ok);
    CHECK(unset.value.checkpointer.empty());
    auto bad = agent::parse_graph_json(R"({"checkpointer": "postgres", "nodes": [{"name": "a", "kind": "agent"}]})");
    REQUIRE_FALSE(bad.ok);
    CHECK(bad.error.code == agent::ErrorCode::InvalidConfig);
    CHECK(bad.error.message.find("checkpointer") != std::string::npos);
}

TEST_CASE("agent.json: routes on state equality with a default", "[graph_json]") {
    ensure_registered();
    auto session = make_session();
    auto loaded = agent::load_graph_json(session, R"({
        "nodes": [
            {"name": "classify", "kind": "agent", "input_key": "q", "output_key": "temp:mode"},
            {"name": "a", "kind": "function", "function": "trace_a"},
            {"name": "b", "kind": "function", "function": "trace_b"},
            {"name": "fallback", "kind": "function", "function": "trace_c"}
        ],
        "edges": [
            {"from": "classify", "route": {"key": "temp:mode", "cases": {"echo: go-a": "a", "echo: go-b": "b", "echo: stop": "__end__"},
                                           "default": "fallback"}}
        ]
    })");
    REQUIRE(loaded.ok);
    CHECK(loaded.value.warnings.empty());

    auto run_with = [&](std::string q, std::string thread) {
        agent::GraphRun run;
        run.thread_id = std::move(thread);
        agent::set_state(run.state, "q", std::move(q));
        auto res = agent::run_graph(session, loaded.value.graph, run);
        REQUIRE(res.ok);
        REQUIRE(run.status == agent::GraphRunStatus::Completed);
        return std::string(agent::get_state(run.state, "trace"));
    };
    CHECK(run_with("go-a", "r1") == "a");
    CHECK(run_with("go-b", "r2") == "b");
    CHECK(run_with("something else", "r3") == "c");
    CHECK(run_with("stop", "r4").empty());
}

TEST_CASE("agent.json: tool nodes with static args or input_key", "[graph_json]") {
    auto session = make_session();
    auto loaded = agent::load_graph_json(session, R"({
        "nodes": [
            {"name": "fixed", "kind": "tool", "tool": "echo_args", "args": {"path": "a.txt", "n": [1, 2]}, "output_key": "fixed_out"},
            {"name": "dynamic", "kind": "tool", "tool": "echo_args", "input_key": "dyn_args", "output_key": "dyn_out"}
        ],
        "edges": [{"from": "fixed", "to": "dynamic"}]
    })");
    REQUIRE(loaded.ok);
    agent::GraphRun run;
    run.thread_id = "tools";
    agent::set_state(run.state, "dyn_args", R"({"x":1})");
    REQUIRE(agent::run_graph(session, loaded.value.graph, run).ok);
    CHECK(agent::get_state(run.state, "fixed_out") == R"({"path":"a.txt","n":[1,2]})");
    CHECK(agent::get_state(run.state, "dyn_out") == R"({"x":1})");

    auto unknown = agent::load_graph_json(session, R"({"nodes": [{"name": "t", "kind": "tool", "tool": "nope"}]})");
    REQUIRE_FALSE(unknown.ok);
    CHECK(unknown.error.message.find("tool node 't' uses unknown tool 'nope'") != std::string::npos);

    // parse_graph_json does not need a Session and does not check tool names.
    CHECK(agent::parse_graph_json(R"({"nodes": [{"name": "t", "kind": "tool", "tool": "nope"}]})").ok);
}

TEST_CASE("agent.json: sequential, loop and parallel workflow nodes", "[graph_json]") {
    ensure_registered();
    auto session = make_session();
    auto loaded = agent::load_graph_json(session, R"({
        "entry": "main",
        "nodes": [
            {"name": "main", "kind": "sequential", "steps": ["start", "repeat", "fan"]},
            {"name": "start", "kind": "function", "function": "trace_a"},
            {"name": "repeat", "kind": "loop", "body": "count", "max_iterations": 5, "until": {"key": "status", "equals": "done"}},
            {"name": "count", "kind": "function", "function": "count_up"},
            {"name": "fan", "kind": "parallel", "branches": ["left", "right"]},
            {"name": "left", "kind": "agent", "prompt": "L", "output_key": "left_out"},
            {"name": "right", "kind": "agent", "prompt": "R", "output_key": "right_out"},
            {"name": "after", "kind": "function", "function": "trace_d"}
        ],
        "edges": [{"from": "main", "to": "after"}]
    })");
    REQUIRE(loaded.ok);
    CHECK(loaded.value.warnings.empty());

    agent::GraphRun run;
    run.thread_id = "workflow";
    auto res = agent::run_graph(session, loaded.value.graph, run);
    INFO(res.error.message);
    REQUIRE(res.ok);
    CHECK(run.status == agent::GraphRunStatus::Completed);
    CHECK(agent::get_state(run.state, "count") == "2"); // until stopped the loop before max_iterations
    CHECK(agent::get_state(run.state, "left_out") == "echo: L");
    CHECK(agent::get_state(run.state, "right_out") == "echo: R");
    CHECK(agent::get_state(run.state, "trace") == "ad");

    std::vector<std::string> order;
    for (const auto &e : run.events)
        if (e.type == agent::GraphEventType::NodeStart && e.node != "left" && e.node != "right") order.push_back(e.node);
    CHECK(order == std::vector<std::string>{"main", "start", "repeat", "count", "repeat", "count", "repeat", "fan", "main#done", "after"});
}

TEST_CASE("agent.json: loop without until runs max_iterations", "[graph_json]") {
    ensure_registered();
    auto session = make_session();
    auto loaded = agent::load_graph_json(session, R"({
        "nodes": [
            {"name": "loop", "kind": "loop", "body": "body", "max_iterations": 3},
            {"name": "body", "kind": "sequential", "steps": ["x", "y"]},
            {"name": "x", "kind": "function", "function": "trace_a"},
            {"name": "y", "kind": "function", "function": "trace_b"}
        ]
    })");
    REQUIRE(loaded.ok);
    agent::GraphRun run;
    run.thread_id = "loop3";
    REQUIRE(agent::run_graph(session, loaded.value.graph, run).ok);
    CHECK(agent::get_state(run.state, "trace") == "ababab");
}

TEST_CASE("agent.json: validation errors are specific", "[graph_json]") {
    CHECK(error_of("{not json").find("invalid JSON") != std::string::npos);
    CHECK(error_of("[]").find("top level must be a JSON object") != std::string::npos);
    CHECK(error_of("{}").find("missing required field \"nodes\"") != std::string::npos);
    CHECK(error_of(R"({"nodes": []})").find("\"nodes\" is empty") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"kind": "agent"}]})").find("nodes[0]: missing required field \"name\"") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a"}]})").find("nodes[0] (\"a\"): missing required field \"kind\"") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "robot"}]})").find("unknown kind 'robot'") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent"}, {"name": "a", "kind": "agent"}]})").find("duplicate node name 'a'") !=
          std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a#1", "kind": "agent"}]})").find("invalid node name") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent", "prompt": 5}]})").find("\"prompt\" must be a string") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent", "allowed_tools": "x"}]})").find("must be an array of strings") !=
          std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "t", "kind": "tool"}]})").find("missing required field \"tool\"") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "t", "kind": "tool", "tool": "x", "args": [1]}]})").find("\"args\" must be a JSON object") !=
          std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "f", "kind": "function", "function": "nobody"}]})").find("function 'nobody' is not registered") !=
          std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "p", "kind": "parallel", "branches": []}]})").find("non-empty \"branches\"") !=
          std::string::npos);
    CHECK(
        error_of(R"({"nodes": [{"name": "p", "kind": "parallel", "branches": ["ghost"]}]})").find("unknown node 'ghost' in \"branches\"") !=
        std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "l", "kind": "loop", "body": "l", "max_iterations": 2}]})").find("cannot be its own body") !=
          std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "l", "kind": "loop", "body": "x"}, {"name": "x", "kind": "agent"}]})")
              .find("\"max_iterations\" >= 1") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "s", "kind": "sequential", "steps": ["x", "x"]}, {"name": "x", "kind": "agent"}]})")
              .find("listed twice") != std::string::npos);
    CHECK(error_of(R"({"entry": "zz", "nodes": [{"name": "a", "kind": "agent"}]})").find("\"entry\" names unknown node 'zz'") !=
          std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent"}], "edges": [{"from": "a", "to": "b"}]})")
              .find("edges[0] (from \"a\"): unknown node 'b' in \"to\"") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent"}], "edges": [{"from": "q", "to": "a"}]})")
              .find("unknown node 'q' in \"from\"") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent"}], "edges": [{"from": "a"}]})").find("exactly one of \"to\" or \"route\"") !=
          std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent"}, {"name": "b", "kind": "agent"}],
                       "edges": [{"from": "a", "to": "b"}, {"from": "a", "to": "__end__"}]})")
              .find("already has an outgoing edge (edges[0])") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent"}], "edges": [{"from": "a", "route": {"cases": {}}}]})")
              .find("missing required field \"key\"") != std::string::npos);
    CHECK(
        error_of(R"({"nodes": [{"name": "a", "kind": "agent"}], "edges": [{"from": "a", "route": {"key": "k", "cases": {"x": "nope"}}}]})")
            .find("case \"x\": unknown node 'nope'") != std::string::npos);
    CHECK(error_of(R"({"nodes": [{"name": "a", "kind": "agent"}], "edges": [{"from": "a", "route": {"key": "k", "cases": {"x": 1}}}]})")
              .find("must map to a node name") != std::string::npos);
    CHECK(error_of(R"({"max_steps": 0, "nodes": [{"name": "a", "kind": "agent"}]})").find("\"max_steps\" must be >= 1") !=
          std::string::npos);
    CHECK(error_of(R"({"mcp_servers": [{"name": "x"}], "nodes": [{"name": "a", "kind": "agent"}]})")
              .find("mcp_servers[0]: missing required field \"command\"") != std::string::npos);
    CHECK(error_of(R"({"mcp_servers": [{"name": "x", "command": "y", "env": {"A": 1}}], "nodes": [{"name": "a", "kind": "agent"}]})")
              .find("\"env\" values must be strings") != std::string::npos);

    // One node claimed by two loop bodies.
    CHECK(error_of(R"({"nodes": [
            {"name": "s", "kind": "sequential", "steps": ["l1", "l2"]},
            {"name": "l1", "kind": "loop", "body": "x", "max_iterations": 2},
            {"name": "l2", "kind": "loop", "body": "x", "max_iterations": 2},
            {"name": "x", "kind": "agent"}]})")
              .find("node 'x' is part of both loop 'l1' body and loop 'l2' body") != std::string::npos);
    // A loop body that jumps back into the top-level flow.
    CHECK(error_of(R"({"nodes": [
            {"name": "start", "kind": "agent"},
            {"name": "l", "kind": "loop", "body": "x", "max_iterations": 2},
            {"name": "x", "kind": "agent"}],
            "edges": [{"from": "start", "to": "l"}, {"from": "x", "to": "start"}]})")
              .find("is part of both the top level") != std::string::npos);

    auto missing = agent::parse_graph_json("/nonexistent/agent.json");
    REQUIRE_FALSE(missing.ok);
    CHECK(missing.error.code == agent::ErrorCode::FilesystemError);
    CHECK(agent::parse_graph_json("{oops").error.code == agent::ErrorCode::ParseError);
    CHECK(agent::parse_graph_json(R"({"nodes": 1})").error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("agent.json: warnings for unknown fields, unreachable nodes and static cycles", "[graph_json]") {
    auto parsed = agent::parse_graph_json(R"({
        "//": "comments are fine",
        "colour": "blue",
        "nodes": [
            {"name": "a", "kind": "agent", "prompt": "x", "sytem_prompt": "typo"},
            {"name": "b", "kind": "agent", "prompt": "x"},
            {"name": "c", "kind": "agent", "prompt": "x"},
            {"name": "island", "kind": "agent", "prompt": "x"}
        ],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}, {"from": "c", "to": "b"}]
    })");
    REQUIRE(parsed.ok);
    const auto &lg = parsed.value;
    CHECK(has_warning(lg, "unknown field \"colour\""));
    CHECK(has_warning(lg, "nodes[0] (\"a\"): unknown field \"sytem_prompt\""));
    CHECK(has_warning(lg, "node 'island' is not reachable from entry 'a'"));
    CHECK(has_warning(lg, "cycle b -> c -> b has no conditional edge or loop node"));
    CHECK_FALSE(has_warning(lg, "\"//\""));

    // A cycle through a route is the normal way to loop on a condition: no warning.
    auto routed = agent::parse_graph_json(R"({
        "nodes": [{"name": "a", "kind": "agent", "prompt": "x"}, {"name": "b", "kind": "agent", "prompt": "x"}],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "route": {"key": "done", "cases": {"yes": "__end__"}, "default": "a"}}]
    })");
    REQUIRE(routed.ok);
    CHECK(routed.value.warnings.empty());

    auto no_input = agent::parse_graph_json(R"({"nodes": [{"name": "a", "kind": "agent"}]})");
    REQUIRE(no_input.ok);
    CHECK(has_warning(no_input.value, "neither \"prompt\" nor \"input_key\""));
}

TEST_CASE("agent.json: graph files and the bundled examples", "[graph_json]") {
    for (const char *name : {"research_pipeline.json", "router.json", "mcp_filesystem.json"}) {
        std::string path = std::string(AGENT_TEST_EXAMPLES_DIR) + "/" + name;
        auto parsed = agent::parse_graph_json(path);
        INFO(path << ": " << parsed.error.message);
        REQUIRE(parsed.ok);
        CHECK(parsed.value.warnings.empty());
        CHECK_FALSE(parsed.value.engine.empty());
    }
    auto mcp = agent::parse_graph_json(std::string(AGENT_TEST_EXAMPLES_DIR) + "/mcp_filesystem.json");
    REQUIRE(mcp.value.mcp_servers.size() == 1);
    CHECK(mcp.value.mcp_servers[0].command == "npx");
    CHECK(mcp.value.mcp_servers[0].env.at("NODE_NO_WARNINGS") == "1");
    CHECK(mcp.value.mcp_servers[0].timeout_ms == 60000);

    // The graph name defaults to the file stem.
    std::filesystem::create_directories(kWorkspace);
    std::string path = std::string(kWorkspace) + "/my_flow.json";
    {
        std::ofstream f(path);
        f << R"({"nodes": [{"name": "a", "kind": "agent", "prompt": "hi"}]})";
    }
    auto parsed = agent::parse_graph_json(path);
    REQUIRE(parsed.ok);
    CHECK(parsed.value.graph.name == "my_flow");
}

TEST_CASE("agent.json: mcp_servers are connected at load and their tools used", "[graph_json][mcp]") {
    auto session = make_session();
    const std::string json = std::string(R"({
        "mcp_servers": [{"name": "fake", "command": ")") +
                             AGENT_TEST_FAKE_MCP_SERVER_PATH + R"(", "env": {"FAKE_MCP_VAR": "7"}}],
        "nodes": [
            {"name": "direct", "kind": "tool", "tool": "fake.echo", "args": {"text": "hi"}, "output_key": "direct_out"},
            {"name": "assistant", "kind": "agent", "prompt": "call:fake.echo", "output_key": "reply", "allowed_tools": ["fake.echo"]}
        ],
        "edges": [{"from": "direct", "to": "assistant"}]
    })";
    auto loaded = agent::load_graph_json(session, json);
    INFO(loaded.error.message);
    REQUIRE(loaded.ok);
    CHECK(loaded.value.warnings.empty());
    CHECK(agent::list_mcp_servers(session) == std::vector<std::string>{"fake"});

    agent::GraphRun run;
    run.thread_id = "mcp";
    auto res = agent::run_graph(session, loaded.value.graph, run);
    INFO(res.error.message);
    REQUIRE(res.ok);
    CHECK(agent::get_state(run.state, "direct_out") == "echo: hi env=7");
    CHECK(agent::get_state(run.state, "reply") == "tool done");
    bool saw_tool_result = false;
    for (const auto &e : run.events)
        if (e.type == agent::GraphEventType::ToolResult && e.node == "assistant" && e.key == "fake.echo" && e.payload == "echo:  env=7")
            saw_tool_result = true;
    CHECK(saw_tool_result);

    // Loading again reuses the connected server (with a warning).
    auto again = agent::load_graph_json(session, json);
    REQUIRE(again.ok);
    CHECK(has_warning(again.value, "already connected; reusing it"));
    agent::disconnect_mcp_servers(session);

    // A failing graph disconnects the servers it connected.
    const std::string bad = std::string(R"({
        "mcp_servers": [{"name": "fake2", "command": ")") +
                            AGENT_TEST_FAKE_MCP_SERVER_PATH + R"("}],
        "nodes": [{"name": "t", "kind": "tool", "tool": "fake2.missing"}]
    })";
    auto failed = agent::load_graph_json(session, bad);
    REQUIRE_FALSE(failed.ok);
    CHECK(failed.error.message.find("unknown tool 'fake2.missing'") != std::string::npos);
    CHECK(agent::list_mcp_servers(session).empty());
    CHECK(session.tool_by_name.count("fake2.echo") == 0);

    const std::string broken = R"({"mcp_servers": [{"name": "gone", "command": "/nonexistent/server"}],
                                   "nodes": [{"name": "a", "kind": "agent", "prompt": "x"}]})";
    auto broken_res = agent::load_graph_json(session, broken);
    REQUIRE_FALSE(broken_res.ok);
    CHECK(broken_res.error.message.find("mcp_servers[0] (\"gone\")") != std::string::npos);
}

TEST_CASE("agent.json: mcp_servers sandbox objects are parsed and validated", "[graph_json][mcp][sandbox]") {
    auto parse = [](std::string_view server) {
        return agent::parse_graph_json(std::string(R"({"mcp_servers": [)") + std::string(server) +
                                       R"(], "nodes": [{"name": "a", "kind": "agent", "prompt": "x"}]})");
    };

    auto none = parse(R"({"name": "s", "command": "srv"})");
    REQUIRE(none.ok);
    CHECK_FALSE(none.value.mcp_servers[0].sandbox.enabled); // opt-in

    auto full = parse(R"({"name": "s", "command": "srv", "sandbox": {
        "read_only_paths": ["/opt/node", "data"], "read_write_paths": ["/srv/out"], "network": "deny", "memory_limit_mb": 512}})");
    INFO(full.error.message);
    REQUIRE(full.ok);
    CHECK(full.value.warnings.empty());
    const agent::SandboxPolicy &sb = full.value.mcp_servers[0].sandbox;
    CHECK(sb.enabled); // an object without "enabled" opts in
    CHECK(sb.read_only_paths == std::vector<std::string>{"/opt/node", "data"});
    CHECK(sb.read_write_paths == std::vector<std::string>{"/srv/out"});
    CHECK(sb.network == agent::SandboxNetwork::Deny);
    CHECK(sb.memory_limit_bytes == 512ull << 20);
    CHECK(sb.filesystem);
    CHECK(sb.system_paths);

    auto off = parse(R"({"name": "s", "command": "srv", "sandbox": {"enabled": false, "network": "allow"}})");
    REQUIRE(off.ok);
    CHECK_FALSE(off.value.mcp_servers[0].sandbox.enabled);
    CHECK(off.value.mcp_servers[0].sandbox.network == agent::SandboxNetwork::Allow);

    auto shorthand = parse(R"({"name": "s", "command": "srv", "sandbox": true})");
    REQUIRE(shorthand.ok);
    CHECK(shorthand.value.mcp_servers[0].sandbox.enabled);
    CHECK(shorthand.value.mcp_servers[0].sandbox.network == agent::SandboxNetwork::Allow);

    auto typo = parse(R"({"name": "s", "command": "srv", "sandbox": {"netwrok": "deny"}})");
    REQUIRE(typo.ok);
    CHECK(has_warning(typo.value, "mcp_servers[0]: sandbox: unknown field \"netwrok\""));

    auto err = [&](std::string_view server) {
        auto r = parse(server);
        return r.ok ? std::string("<ok>") : r.error.message;
    };
    CHECK(err(R"({"name": "s", "command": "srv", "sandbox": "yes"})").find("\"sandbox\" must be true, false or an object") !=
          std::string::npos);
    CHECK(err(R"({"name": "s", "command": "srv", "sandbox": {"network": "none"}})").find("\"network\" must be \"allow\" or \"deny\"") !=
          std::string::npos);
    CHECK(err(R"({"name": "s", "command": "srv", "sandbox": {"read_only_paths": "/x"}})")
              .find("\"read_only_paths\" must be an array of strings") != std::string::npos);
    CHECK(err(R"({"name": "s", "command": "srv", "sandbox": {"read_write_paths": [""]}})").find("paths must not be empty") !=
          std::string::npos);
    CHECK(err(R"({"name": "s", "command": "srv", "sandbox": {"memory_limit_mb": -1}})").find("\"memory_limit_mb\" must be an integer") !=
          std::string::npos);
    CHECK(err(R"({"name": "s", "command": "srv", "sandbox": {"memory_limit_mb": 1.5}})").find("\"memory_limit_mb\" must be an integer") !=
          std::string::npos);
    CHECK(err(R"({"name": "s", "command": "srv", "sandbox": {"enabled": 1}})").find("\"enabled\" must be true or false") !=
          std::string::npos);
}

#if defined(__linux__)
TEST_CASE("agent.json: a sandboxed MCP server is started under its policy", "[graph_json][mcp][sandbox]") {
    auto session = make_session();
    const std::string server = AGENT_TEST_FAKE_MCP_SERVER_PATH;
    auto json = [&](std::string_view name, std::string_view sandbox) {
        return std::string(R"({"mcp_servers": [{"name": ")") + std::string(name) + R"(", "command": ")" + server +
               R"(", "sandbox": )" + std::string(sandbox) + R"(}], "nodes": [{"name": "t", "kind": "tool", "tool": ")" +
               std::string(name) + R"(.echo", "args": {"text": "boxed"}, "output_key": "out"}]})";
    };

    auto boxed = agent::load_graph_json(session, json("boxed", R"({"network": "deny"})"));
    INFO(boxed.error.message);
    REQUIRE(boxed.ok);
    agent::GraphRun run;
    run.thread_id = "mcp_sandbox";
    auto res = agent::run_graph(session, boxed.value.graph, run);
    INFO(res.error.message);
    REQUIRE(res.ok);
    CHECK(agent::get_state(run.state, "out") == "echo: boxed");
    agent::disconnect_mcp_servers(session);

    // The policy reaches the spawn: 1 MiB of address space cannot even map the server's libraries.
    auto starved = agent::load_graph_json(session, json("starved", R"({"memory_limit_mb": 1})"));
    CHECK_FALSE(starved.ok);
    CHECK(agent::list_mcp_servers(session).empty());
}
#endif
