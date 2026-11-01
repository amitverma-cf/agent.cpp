// Tools that return parts (ToolResult): text + media + tensors, through execute_tool_parts, the
// string execute_tool, run_turn (tool media reaches the engine as FILE parts and persists), graph
// Agent nodes, process-isolated engines and MCP image content.

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

const char *kWorkspace = "/tmp/agent_test_tool_parts_ws";

std::string snap_path() { return std::string(kWorkspace) + "/snap.png"; }

// A tool that "takes a picture": returns text + an image file in the workspace.
agent::Result<agent::ToolResult> snap_tool(std::string_view, void *) {
    std::filesystem::create_directories(kWorkspace);
    std::ofstream(snap_path(), std::ios::binary) << "png";
    agent::ToolResult r;
    r.text = "took a picture";
    r.media.push_back(agent::Media{.path = snap_path(), .mime = "image/png"});
    return agent::ok(std::move(r));
}

agent::Result<agent::ToolResult> pdf_tool(std::string_view, void *) {
    agent::ToolResult r;
    r.text = "made a pdf";
    r.media.push_back(agent::Media{.path = std::string(kWorkspace) + "/doc.pdf", .mime = "application/pdf"});
    return agent::ok(std::move(r));
}

agent::Result<agent::ToolResult> broken_tool(std::string_view, void *) {
    agent::ToolResult r;
    r.text = "lens cap on";
    r.is_error = true;
    return agent::ok(std::move(r));
}

agent::Result<agent::ToolResult> tensor_tool(std::string_view, void *) {
    agent::ToolResult r;
    r.text = "measured";
    r.tensors.push_back(
        agent::TensorData{.name = "depth", .dtype = agent::DType::Float32, .shape = {2, 2}, .data = std::vector<uint8_t>(16)});
    return agent::ok(std::move(r));
}

agent::Tool parts_tool(std::string_view name, agent::Tool::PartsCallbackFn fn) {
    agent::Tool t;
    t.name = name;
    t.description = "test tool returning parts";
    t.parameter_schema = "{}";
    t.parts_callback = fn;
    return t;
}

agent::Config engine_config() {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.model = "mock.bin";
    cfg.workspace_dir = kWorkspace;
    cfg.auto_default_tools = false;
    cfg.tools = {parts_tool("snap", snap_tool), parts_tool("pdf", pdf_tool), parts_tool("broken", broken_tool),
                 parts_tool("measure", tensor_tool)};
    return cfg;
}

agent::Session open(const agent::Config &cfg) {
    auto res = agent::init(cfg);
    INFO(res.error.message);
    REQUIRE(res.ok);
    return std::move(res.value);
}

agent::PermissionDecision deny_snap(std::string_view tool, std::string_view, void *) {
    return tool == "snap" ? agent::PermissionDecision::Deny : agent::PermissionDecision::Allow;
}

struct HookLog {
    std::string pre;
    std::string post;
};

void record_hook(agent::Session &, agent::FlowMemory &, std::string_view tool, std::string_view, std::string_view result, void *ud) {
    auto *log = static_cast<HookLog *>(ud);
    if (result.empty()) log->pre = std::string(tool);
    else log->post = std::string(result);
}

} // namespace

TEST_CASE("execute_tool_parts and execute_tool handle parts tools", "[tools][tool_parts]") {
    auto session = open(engine_config());

    auto parts = agent::execute_tool_parts(session, "snap", "{}");
    REQUIRE(parts.ok);
    CHECK(parts.value.text == "took a picture");
    REQUIRE(parts.value.media.size() == 1);
    CHECK(parts.value.media[0].path == snap_path());
    CHECK(parts.value.media[0].mime == "image/png");
    CHECK_FALSE(parts.value.is_error);

    // The string form lists the attachment after the text.
    auto text = agent::execute_tool(session, "snap", "{}");
    REQUIRE(text.ok);
    CHECK(text.value == "took a picture\n[attached: " + snap_path() + " (image/png)]");

    auto tensors = agent::execute_tool(session, "measure", "{}");
    REQUIRE(tensors.ok);
    CHECK(tensors.value == "measured\n[tensor: depth [2,2]]");

    // A tool-reported failure: ok() with is_error from the parts API, ToolError from the string API.
    auto broken = agent::execute_tool_parts(session, "broken", "{}");
    REQUIRE(broken.ok);
    CHECK(broken.value.is_error);
    auto broken_text = agent::execute_tool(session, "broken", "{}");
    REQUIRE_FALSE(broken_text.ok);
    CHECK(broken_text.error.code == agent::ErrorCode::ToolError);
    CHECK(broken_text.error.message == "lens cap on");

    // A string tool through the parts API.
    agent::Tool plain;
    plain.name = "plain";
    plain.callback = [](std::string_view, void *) -> agent::Result<std::string> { return agent::ok(std::string("hi")); };
    session.config.tools.push_back(plain);
    // Exactly one callback must be set.
    agent::Tool both = plain;
    both.name = "both";
    both.parts_callback = snap_tool;
    session.config.tools.push_back(both);
    agent::Tool none;
    none.name = "none";
    session.config.tools.push_back(none);
    agent::rebuild_tool_index(session);

    auto plain_parts = agent::execute_tool_parts(session, "plain", "{}");
    REQUIRE(plain_parts.ok);
    CHECK(plain_parts.value.text == "hi");
    CHECK(plain_parts.value.media.empty());
    CHECK(agent::execute_tool(session, "both", "{}").error.code == agent::ErrorCode::InvalidConfig);
    CHECK(agent::execute_tool_parts(session, "none", "{}").error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("parts tools go through the permission gate and relative media paths become absolute", "[tools][tool_parts]") {
    auto cfg = engine_config();
    cfg.permission_check = deny_snap;
    auto session = open(cfg);
    auto denied = agent::execute_tool_parts(session, "snap", "{}");
    REQUIRE_FALSE(denied.ok);
    CHECK(denied.error.code == agent::ErrorCode::PermissionDenied);

    agent::Tool rel = parts_tool("rel", [](std::string_view, void *) -> agent::Result<agent::ToolResult> {
        agent::ToolResult r;
        r.media.push_back(agent::Media{.path = "some/dir/../pic.jpg", .mime = "image/jpeg"});
        return agent::ok(std::move(r));
    });
    session.config.tools.push_back(rel);
    agent::rebuild_tool_index(session);
    auto res = agent::execute_tool_parts(session, "rel", "{}");
    REQUIRE(res.ok);
    REQUIRE(res.value.media.size() == 1);
    CHECK(std::filesystem::path(res.value.media[0].path).is_absolute());
    CHECK(res.value.media[0].path == (std::filesystem::current_path() / "some/pic.jpg").string());
}

TEST_CASE("run_turn attaches tool media to the tool message and sends it to the engine", "[tools][tool_parts]") {
    auto session = open(engine_config());
    agent::FlowMemory memory;
    memory.id = "tool_parts_turn";

    auto reply = agent::run_turn(session, memory, "call:snap");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    // The mock lists the FILE parts it received on the tool message.
    CHECK(reply.value == "tool done\ntool media: image/png " + snap_path());

    const agent::Message *tool_msg = nullptr;
    for (const auto &m : memory.history)
        if (m.role == "tool") tool_msg = &m;
    REQUIRE(tool_msg != nullptr);
    CHECK(tool_msg->content == "took a picture\n[attached: " + snap_path() + " (image/png)]");
    REQUIRE(tool_msg->media.size() == 1);
    CHECK(tool_msg->media[0].path == snap_path());

    // Persisted with the history.
    agent::FlowMemory loaded;
    loaded.id = memory.id;
    REQUIRE(agent::load_flow_memory(session, loaded).ok);
    bool found = false;
    for (const auto &m : loaded.history)
        if (m.role == "tool" && m.media.size() == 1 && m.media[0].mime == "image/png") found = true;
    CHECK(found);

    // Later turns still carry the image (the engine sees the whole history).
    auto again = agent::run_turn(session, memory, "hello");
    REQUIRE(again.ok);
    CHECK(again.value == "echo: hello");
}

TEST_CASE("run_turn reports tool-side failures and tensors", "[tools][tool_parts]") {
    auto session = open(engine_config());
    agent::FlowMemory memory;
    REQUIRE(agent::run_turn(session, memory, "call:broken").ok);
    CHECK(memory.history[memory.history.size() - 2].content == "Error: lens cap on");

    agent::FlowMemory m2;
    REQUIRE(agent::run_turn(session, m2, "call:measure").ok);
    const auto &tool_msg = m2.history[m2.history.size() - 2];
    CHECK(tool_msg.role == "tool");
    REQUIRE(tool_msg.tensors.size() == 1);
    CHECK(tool_msg.tensors[0].name == "depth");
}

TEST_CASE("tool media the engine cannot take stays out of the request", "[tools][tool_parts]") {
    SECTION("no matching input capability") {
        auto session = open(engine_config());
        agent::FlowMemory memory;
        auto reply = agent::run_turn(session, memory, "call:pdf");
        REQUIRE(reply.ok);
        CHECK(reply.value == "tool done"); // the mock does not declare application/pdf input
        CHECK(memory.history[memory.history.size() - 2].media.size() == 1);
    }
    SECTION("an engine that fails on media: retried without it, then left out") {
        auto cfg = engine_config();
        cfg.engine_options = R"({"reject-media": true})";
        auto session = open(cfg);
        agent::FlowMemory memory;
        auto reply = agent::run_turn(session, memory, "call:snap");
        INFO(reply.error.message);
        REQUIRE(reply.ok);
        CHECK(reply.value == "tool done");
        auto again = agent::run_turn(session, memory, "still there?");
        REQUIRE(again.ok);
        CHECK(again.value == "echo: still there?");
    }
}

TEST_CASE("tool media reaches a process-isolated engine", "[tools][tool_parts][engine_host]") {
    auto cfg = engine_config();
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_host_path = AGENT_TEST_ENGINE_HOST_PATH;
    auto session = open(cfg);
    agent::FlowMemory memory;
    auto reply = agent::run_turn(session, memory, "call:snap");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    CHECK(reply.value == "tool done\ntool media: image/png " + snap_path());
}

TEST_CASE("graph agent nodes: tool media, hooks, events and allowed_tools", "[tools][tool_parts][graph]") {
    auto session = open(engine_config());
    HookLog hooks;

    agent::GraphNode node;
    node.name = "photographer";
    node.kind = agent::GraphNodeKind::Agent;
    node.prompt = "call:snap";
    node.output_key = "reply";
    node.allowed_tools = {"snap"};
    node.hooks = {agent::ToolHook{.when = agent::ToolHook::When::Pre, .tool_name = {}, .callback = record_hook, .user_data = &hooks},
                  agent::ToolHook{.when = agent::ToolHook::When::Post, .tool_name = {}, .callback = record_hook, .user_data = &hooks}};

    agent::GraphRun run;
    auto res = agent::run_graph(session, agent::make_graph(node), run);
    INFO(res.error.message);
    REQUIRE(res.ok);
    CHECK(agent::get_state(run.state, "reply") == "tool done\ntool media: image/png " + snap_path());
    CHECK(hooks.pre == "snap");
    CHECK(hooks.post.find("[attached: " + snap_path() + " (image/png)]") != std::string::npos);

    bool event = false;
    for (const auto &e : run.events)
        if (e.type == agent::GraphEventType::ToolResult && e.key == "snap" && e.payload.find("[attached: ") != std::string::npos)
            event = true;
    CHECK(event);

    const auto &history = run.state.histories.at("photographer").history;
    bool media = false;
    for (const auto &m : history)
        if (m.role == "tool" && !m.media.empty()) media = true;
    CHECK(media);

    // A parts tool outside the node's allowlist is refused like any other tool.
    agent::GraphNode other = node;
    other.name = "other";
    other.prompt = "call:pdf";
    agent::GraphRun run2;
    REQUIRE(agent::run_graph(session, agent::make_graph(other), run2).ok);
    const auto &h2 = run2.state.histories.at("other").history;
    CHECK(h2[h2.size() - 2].content.find("Tool not allowed") != std::string::npos);
}

TEST_CASE("MCP image content becomes tool media", "[tools][tool_parts][mcp]") {
    auto cfg = engine_config();
    auto session = open(cfg);
    agent::McpServerConfig mcp;
    mcp.name = "fake";
    mcp.command = AGENT_TEST_FAKE_MCP_SERVER_PATH;
    mcp.timeout_ms = 5000;
    REQUIRE(agent::connect_mcp_server(session, mcp).ok);

    auto parts = agent::execute_tool_parts(session, "fake.echo", R"({"text":"pic","image":true})");
    REQUIRE(parts.ok);
    CHECK(parts.value.text == "echo: pic");
    REQUIRE(parts.value.media.size() == 1);
    CHECK(parts.value.media[0].mime == "image/png");
    CHECK(parts.value.media[0].path.find("/artifacts/mcp-fake-echo-") != std::string::npos);
    std::ifstream f(parts.value.media[0].path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(bytes == "hello");

    auto failed = agent::execute_tool_parts(session, "fake.fail", "{}");
    REQUIRE(failed.ok);
    CHECK(failed.value.is_error);
    CHECK(failed.value.text == "boom");

    // Through a turn: the model asks for the image, the engine receives it as a FILE part.
    agent::FlowMemory memory;
    auto reply = agent::run_turn(session, memory, R"(call:fake.echo {"text":"pic","image":true})");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    CHECK(reply.value.starts_with("tool done\ntool media: image/png "));
    CHECK(reply.value.find("/artifacts/mcp-fake-echo-") != std::string::npos);
    agent::disconnect_mcp_servers(session);
}
