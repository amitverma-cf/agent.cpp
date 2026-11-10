// Lazy tool loading (Config::lazy_tools) with the mock engine: "show:tools" echoes the tool list the host sent, and
// "call:<tool> <json>" makes the mock call a tool.

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

#include <filesystem>
#include <string>

namespace {

agent::Config lazy_config() {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.workspace_dir = (std::filesystem::temp_directory_path() / "agent_test_lazy_tools_ws").string();
    cfg.lazy_tools = true;
    return cfg;
}

bool has(const std::string &text, std::string_view needle) { return text.find(needle) != std::string::npos; }

} // namespace

TEST_CASE("lazy tools: the model starts with only the meta tools", "[lazy]") {
    auto s = agent::init(lazy_config());
    REQUIRE(s.ok);
    agent::FlowMemory memory;
    memory.id = "lazy1";
    auto shown = agent::run_turn(s.value, memory, "show:tools");
    REQUIRE(shown.ok);
    CHECK(has(shown.value, "tool_search"));
    CHECK(has(shown.value, "load_tool"));
    CHECK_FALSE(has(shown.value, "read_file"));
    CHECK_FALSE(has(shown.value, "run_command"));
    CHECK(memory.active_tools.empty());
}

TEST_CASE("lazy tools: eager mode is unchanged", "[lazy]") {
    auto cfg = lazy_config();
    cfg.lazy_tools = false;
    auto s = agent::init(cfg);
    REQUIRE(s.ok);
    agent::FlowMemory memory;
    memory.id = "eager";
    auto shown = agent::run_turn(s.value, memory, "show:tools");
    REQUIRE(shown.ok);
    CHECK(has(shown.value, "read_file"));
    CHECK(has(shown.value, "run_command"));
    CHECK_FALSE(has(shown.value, "tool_search"));
}

TEST_CASE("lazy tools: tool_search lists, load_tool loads, and the tool is then advertised and callable", "[lazy]") {
    auto s = agent::init(lazy_config());
    REQUIRE(s.ok);
    agent::FlowMemory memory;
    memory.id = "lazy2";

    // A tool that is not loaded cannot be called, and the error says what to do.
    auto early = agent::execute_tool(s.value, "read_file", R"({"path":"x"})");
    CHECK_FALSE(early.ok);
    CHECK(has(early.error.message, "not loaded"));

    // tool_search lists everything but the meta tools; a query narrows it.
    auto all = agent::run_turn(s.value, memory, "call:tool_search {}");
    REQUIRE(all.ok);
    auto listed = agent::execute_tool(s.value, "tool_search", "{}");
    REQUIRE(listed.ok);
    CHECK(has(listed.value, "read_file"));
    CHECK(has(listed.value, "run_command"));
    CHECK_FALSE(has(listed.value, "- tool_search"));
    auto narrowed = agent::execute_tool(s.value, "tool_search", R"({"query":"shell"})");
    REQUIRE(narrowed.ok);
    CHECK(narrowed.value.find("read_file") == std::string::npos);

    // Loading goes through a turn (the loaded set belongs to the conversation).
    auto loaded = agent::run_turn(s.value, memory, R"(call:load_tool {"name":"read_file"})");
    REQUIRE(loaded.ok);
    REQUIRE(memory.active_tools.size() == 1);
    CHECK(memory.active_tools[0] == "read_file");

    auto shown = agent::run_turn(s.value, memory, "show:tools");
    REQUIRE(shown.ok);
    CHECK(has(shown.value, "read_file"));
    CHECK_FALSE(has(shown.value, "run_command")); // only what was loaded

    // Loading the same tool again changes nothing; an unknown name is an error.
    REQUIRE(agent::run_turn(s.value, memory, R"(call:load_tool {"names":["read_file","list_dir"]})").ok);
    CHECK(memory.active_tools.size() == 2);
    auto bad = agent::execute_tool(s.value, "load_tool", R"({"name":"no_such_tool"})");
    CHECK_FALSE(bad.ok);
}

TEST_CASE("lazy tools: loaded tools survive pruning of the history", "[lazy]") {
    auto s = agent::init(lazy_config());
    REQUIRE(s.ok);
    agent::FlowMemory memory;
    memory.id = "lazy3";
    REQUIRE(agent::run_turn(s.value, memory, R"(call:load_tool {"name":"read_file"})").ok);
    memory.history.clear(); // what compression or pruning leaves behind is a different history, not a different tool set
    auto shown = agent::run_turn(s.value, memory, "show:tools");
    REQUIRE(shown.ok);
    CHECK(has(shown.value, "read_file"));
}

TEST_CASE("lazy tools: a graph node whose allowlist omits tool_search keeps its fixed tools", "[lazy]") {
    auto s = agent::init(lazy_config());
    REQUIRE(s.ok);
    agent::FlowMemory memory;
    memory.id = "lazy4";
    const std::string_view allowed[] = {"read_file", "list_dir"};
    // The way a graph Agent node restricts tools: the ambient allowlist. Here through the public node API.
    agent::GraphNode node;
    node.name = "worker";
    node.kind = agent::GraphNodeKind::Agent;
    node.prompt = "show:tools";
    node.output_key = "out";
    node.allowed_tools = {"read_file", "list_dir"};
    agent::GraphRun run;
    run.thread_id = "lazy_graph";
    auto ran = agent::run_graph(s.value, agent::make_graph(std::move(node)), run);
    REQUIRE(ran.ok);
    const std::string out(agent::get_state(run.state, "out"));
    CHECK(has(out, "read_file"));
    CHECK(has(out, "list_dir"));
    CHECK_FALSE(has(out, "tool_search"));
    (void)allowed;
}
