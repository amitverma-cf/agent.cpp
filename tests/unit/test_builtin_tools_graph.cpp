// The built-in tools (filesystem, run_command, compress_context, use_skill) driven by a model inside
// graph Agent nodes, with the mock engine plugin: "call:<tool> <json>" makes it call a tool, it
// lists the FILE parts it receives after a tool call, and "show:system" echoes its system messages.

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

const char *kWorkspace = "/tmp/agent_test_builtin_graph_ws";

std::string ws_path(std::string_view rel) { return (std::filesystem::path(kWorkspace) / rel).string(); }

void write(const std::string &path, std::string_view content) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream(path, std::ios::binary) << content;
}

std::string slurp(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

agent::Config base_config() {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.model = "mock.bin";
    cfg.workspace_dir = kWorkspace;
    return cfg;
}

agent::Session open(const agent::Config &cfg) {
    auto res = agent::init(cfg);
    INFO(res.error.message);
    REQUIRE(res.ok);
    return std::move(res.value);
}

agent::GraphNode agent_node(std::string name, std::string prompt, std::vector<std::string> allowed = {}) {
    agent::GraphNode n;
    n.name = std::move(name);
    n.kind = agent::GraphNodeKind::Agent;
    n.prompt = std::move(prompt);
    n.output_key = n.name + "_reply";
    n.allowed_tools = std::move(allowed);
    return n;
}

struct NodeOutcome {
    std::string reply;
    agent::Message tool; // the last tool message of the node's conversation
    agent::GraphRun run;
};

int g_thread = 0;

// Runs one Agent node with `prompt` on a fresh thread and returns its reply and tool message.
NodeOutcome run_node(agent::Session &session, std::string prompt, std::vector<std::string> allowed = {}) {
    NodeOutcome out;
    out.run.thread_id = "builtin-" + std::to_string(++g_thread);
    auto node = agent_node("worker", std::move(prompt), std::move(allowed));
    auto res = agent::run_graph(session, agent::make_graph(node), out.run);
    INFO(res.error.message);
    REQUIRE(res.ok);
    out.reply = std::string(agent::get_state(out.run.state, "worker_reply"));
    for (const auto &m : out.run.state.histories.at("worker").history)
        if (m.role == "tool") out.tool = m;
    return out;
}

const std::vector<std::string> kFsTools = {"read_file",  "write_file",  "append_file", "list_dir",
                                           "create_dir", "delete_path", "move_path",   "file_info"};

} // namespace

TEST_CASE("built-in filesystem tools under graph agent nodes", "[builtin_tools][graph][filesystem_tools]") {
    std::filesystem::remove_all(kWorkspace);
    auto session = open(base_config());

    auto w = run_node(session, R"(call:write_file {"path":"notes/a.txt","content":"alpha"})", kFsTools);
    CHECK(w.reply == "tool done");
    CHECK(w.tool.content.find("wrote 5 bytes") != std::string::npos);
    CHECK(slurp(ws_path("notes/a.txt")) == "alpha");

    auto missing = run_node(session, R"(call:write_file {"path":"notes/a.txt"})", kFsTools);
    CHECK(missing.tool.content == "Error: Missing 'content' (a string)");
    CHECK(slurp(ws_path("notes/a.txt")) == "alpha"); // not truncated

    run_node(session, R"(call:append_file {"path":"notes/a.txt","content":" beta"})", kFsTools);
    auto r = run_node(session, R"(call:read_file {"path":"notes/a.txt"})", kFsTools);
    CHECK(r.tool.content == "alpha beta");
    CHECK(r.tool.media.empty());

    run_node(session, R"(call:create_dir {"path":"empty"})", kFsTools);
    write(ws_path("notes/c.txt"), "c");
    write(ws_path("notes/b.txt"), "bb");
    auto ls = run_node(session, R"(call:list_dir {"path":"notes"})", kFsTools);
    // Sorted by name.
    CHECK(ls.tool.content ==
          R"([{"name":"a.txt","type":"file","size":10},{"name":"b.txt","type":"file","size":2},{"name":"c.txt","type":"file","size":1}])");

    auto info = run_node(session, R"(call:file_info {"path":"notes/b.txt"})", kFsTools);
    CHECK(info.tool.content.find("\"size\":2") != std::string::npos);

    run_node(session, R"(call:move_path {"from":"notes/b.txt","to":"moved/b.txt"})", kFsTools);
    CHECK(std::filesystem::exists(ws_path("moved/b.txt")));
    auto root_move = run_node(session, R"(call:move_path {"from":".","to":"elsewhere"})", kFsTools);
    CHECK(root_move.tool.content == "Error: Cannot move the workspace root directory.");
    CHECK(std::filesystem::exists(ws_path("notes/a.txt")));

    run_node(session, R"(call:delete_path {"path":"moved"})", kFsTools);
    CHECK_FALSE(std::filesystem::exists(ws_path("moved")));

    // allowed_tools: a node limited to read_file cannot write.
    auto refused = run_node(session, R"(call:write_file {"path":"x.txt","content":"x"})", {"read_file"});
    CHECK(refused.tool.content == "Error: Tool not allowed for the current agent node: write_file");
    CHECK_FALSE(std::filesystem::exists(ws_path("x.txt")));

    // Escapes are refused under the path sandbox.
    auto escape = run_node(session, R"(call:read_file {"path":"../../etc/passwd"})", kFsTools);
    CHECK(escape.tool.content.starts_with("Error: Path is outside workspace_dir"));
}

TEST_CASE("read_file: text paging, UTF-8 boundaries and binary files", "[builtin_tools][filesystem_tools]") {
    std::filesystem::remove_all(kWorkspace);
    auto session = open(base_config());

    write(ws_path("utf8.txt"), "h\xC3\xA9llo"); // "héllo": the é is 2 bytes
    auto cut = agent::execute_tool(session, "read_file", R"({"path":"utf8.txt","max_bytes":2})");
    REQUIRE(cut.ok);
    // The cut falls inside the é: only "h" is returned, and the next offset is 1.
    CHECK(cut.value == "h\n[truncated at 2 bytes]\n[6 bytes in total; read more with \"offset\": 1]");
    auto rest = agent::execute_tool(session, "read_file", R"({"path":"utf8.txt","offset":1})");
    REQUIRE(rest.ok);
    CHECK(rest.value == "\xC3\xA9llo");

    write(ws_path("latin1.txt"), "caf\xE9");
    auto latin = agent::execute_tool(session, "read_file", R"({"path":"latin1.txt"})");
    REQUIRE(latin.ok);
    CHECK(latin.value == "caf\xEF\xBF\xBD"); // invalid byte -> U+FFFD

    write(ws_path("blob.bin"), std::string("\x7F"
                                           "ELF\0\0\0",
                                           7));
    auto bin = agent::execute_tool(session, "read_file", R"({"path":"blob.bin"})");
    REQUIRE(bin.ok);
    CHECK(bin.value == "[binary file blob.bin, 7 bytes; not shown as text]");

    auto dir = agent::execute_tool(session, "read_file", R"({"path":"."})");
    CHECK_FALSE(dir.ok);

    // A huge max_bytes is clamped instead of allocating it.
    auto big = agent::execute_tool(session, "read_file", R"({"path":"utf8.txt","max_bytes":9223372036854775807})");
    REQUIRE(big.ok);
    CHECK(big.value == "h\xC3\xA9llo");
}

TEST_CASE("read_file of an image attaches it for the engine", "[builtin_tools][graph][filesystem_tools]") {
    std::filesystem::remove_all(kWorkspace);
    write(ws_path("pics/cat.png"), "\x89PNG fake");

    SECTION("in process") {
        auto session = open(base_config());
        auto out = run_node(session, R"(call:read_file {"path":"pics/cat.png"})", {"read_file"});
        const std::string abs = std::filesystem::weakly_canonical(ws_path("pics/cat.png")).string(); // macOS: /tmp resolves to /private/tmp
        CHECK(out.reply == "tool done\ntool media: image/png " + abs);
        REQUIRE(out.tool.media.size() == 1);
        CHECK(out.tool.media[0].mime == "image/png");
        CHECK(out.tool.content == "image file cat.png (9 bytes), attached\n[attached: " + abs + " (image/png)]");
    }
    SECTION("process-isolated, sandboxed engine") {
        auto cfg = base_config();
        cfg.engine_isolation = agent::EngineIsolation::Process;
        cfg.engine_host_path = AGENT_TEST_ENGINE_HOST_PATH;
        auto session = open(cfg);
        auto out = run_node(session, R"(call:read_file {"path":"pics/cat.png"})", {"read_file"});
        CHECK(out.reply.starts_with("tool done\ntool media: image/png "));
    }
    SECTION("outside the workspace: copied into artifacts") {
        const std::string outside = "/tmp/agent_test_builtin_graph_outside/dog.jpg";
        write(outside, "jpeg");
        auto cfg = base_config();
        cfg.sandbox_filesystem = false;
        auto session = open(cfg);
        auto out = run_node(session, "call:read_file {\"path\":\"" + outside + "\"}", {"read_file"});
        REQUIRE(out.tool.media.size() == 1);
        const std::string artifacts = std::filesystem::absolute(ws_path("artifacts")).string();
        CHECK(std::filesystem::path(out.tool.media[0].path).generic_string().starts_with(std::filesystem::path(artifacts).generic_string() + "/read-"));
        CHECK(out.tool.media[0].path.ends_with("dog.jpg"));
        CHECK(slurp(out.tool.media[0].path) == "jpeg");
        std::filesystem::remove_all("/tmp/agent_test_builtin_graph_outside");
    }
}

TEST_CASE("run_command under a graph node: approval interrupt, then sandboxed execution", "[builtin_tools][graph][terminal_tools]") {
    std::filesystem::remove_all(kWorkspace);
    auto cfg = base_config();
    cfg.permission_check = agent::tools::default_permission_policy; // run_command -> Ask
    auto session = open(cfg);

#ifdef _WIN32 // a Windows command is confined to the system folders and the workspace: no cat from a Git install
    const char *shell_call = R"(call:run_command {"command":"echo hi>out.txt && type out.txt"})";
#else
    const char *shell_call = R"(call:run_command {"command":"echo hi>out.txt && cat out.txt"})";
#endif
    auto node = agent_node("shell", shell_call, {"run_command"});
    node.interrupt_before_tools = true;
    auto graph = agent::make_graph(node);

    agent::GraphRun run;
    run.thread_id = "shell-approve";
    REQUIRE(agent::run_graph(session, graph, run).ok);
    REQUIRE(run.status == agent::GraphRunStatus::Interrupted);
    REQUIRE(run.pending_tool_calls.size() == 1);
    CHECK(run.pending_tool_calls[0].name == "run_command");
    CHECK_FALSE(std::filesystem::exists(ws_path("out.txt")));

    agent::GraphRun resumed;
    auto res = agent::resume_graph(session, graph, "shell-approve", resumed, "approve");
    INFO(res.error.message);
    REQUIRE(res.ok);
    CHECK(resumed.status == agent::GraphRunStatus::Completed);
    CHECK(agent::get_state(resumed.state, "shell_reply") == "tool done");
    std::string tool_content;
    for (const auto &m : resumed.state.histories.at("shell").history)
        if (m.role == "tool") tool_content = m.content;
    INFO(tool_content);
    CHECK(tool_content.find("\"exit_code\":0") != std::string::npos);
    // cmd.exe writes "hi" followed by a space and CRLF, so compare without trailing whitespace.
    CHECK(tool_content.find("\"output\":\"hi") != std::string::npos);
    {
        std::string written = slurp(ws_path("out.txt"));
        while (!written.empty() && std::isspace(static_cast<unsigned char>(written.back()))) written.pop_back();
        CHECK(written == "hi");
    }

    // Denied: nothing runs.
    std::filesystem::remove(ws_path("out.txt"));
    agent::GraphRun run2;
    run2.thread_id = "shell-deny";
    REQUIRE(agent::run_graph(session, graph, run2).ok);
    agent::GraphRun denied;
    REQUIRE(agent::resume_graph(session, graph, "shell-deny", denied, "no").ok);
    CHECK_FALSE(std::filesystem::exists(ws_path("out.txt")));
}

TEST_CASE("compress_context tool on a shared graph conversation", "[builtin_tools][graph][context_compressor]") {
    std::filesystem::remove_all(kWorkspace);
    auto session = open(base_config());

    std::vector<agent::Graph> steps;
    for (int i = 0; i < 6; ++i) {
        auto n = agent_node("turn" + std::to_string(i), "message " + std::to_string(i));
        n.history_key = "chat";
        steps.push_back(agent::make_graph(n));
    }
    auto compress = agent_node("compress", R"(call:compress_context {"keep_recent":2})", {"compress_context"});
    compress.history_key = "chat";
    compress.system_prompt = "You are the archivist.";
    steps.push_back(agent::make_graph(compress));

    agent::GraphRun run;
    run.thread_id = "compress-tool";
    auto res = agent::run_graph(session, agent::make_sequential("pipeline", std::move(steps)), run);
    INFO(res.error.message);
    REQUIRE(res.ok);

    const auto &chat = run.state.histories.at("chat");
    std::string dump;
    for (const auto &m : chat.history) dump += m.role + ":" + m.content.substr(0, 60) + "\n";
    INFO(dump);
    CHECK(chat.stats.compressions == 1);
    // The node's system prompt is gone again; the summary leads, followed by the two most recent
    // messages (the user request and the in-flight assistant tool call), its result and the reply.
    REQUIRE(chat.history.size() == 5);
    CHECK(chat.history[0].role == "system");
    CHECK(chat.history[0].content.starts_with("[Compressed history]: echo: Summarize the following conversation"));
    CHECK(chat.history[0].content.find("message 0") != std::string::npos);
    CHECK(chat.history[0].content.find("echo: message 5") != std::string::npos);
    CHECK(chat.history[1].content.starts_with("call:compress_context"));
    REQUIRE(chat.history[2].tool_calls.size() == 1);
    CHECK(chat.history[3].role == "tool");
    CHECK(chat.history[3].tool_call_id == chat.history[2].tool_calls[0].id);
    CHECK(chat.history[3].content == "Context compressed.");
    CHECK(chat.history[4].content == "tool done");

    // keep_recent 0 still keeps the in-flight tool call with its result.
    auto again = agent_node("again", R"(call:compress_context {"keep_recent":0})", {"compress_context"});
    again.history_key = "chat";
    agent::GraphRun run2;
    run2.thread_id = "compress-tool-0";
    run2.state.histories["chat"] = chat;
    REQUIRE(agent::run_graph(session, agent::make_graph(again), run2).ok);
    const auto &h2 = run2.state.histories.at("chat").history;
    for (size_t i = 0; i < h2.size(); ++i)
        if (h2[i].role == "tool") CHECK((i > 0 && !h2[i - 1].tool_calls.empty()));
}

TEST_CASE("automatic compression in run_turn under a graph uses the session's engine", "[builtin_tools][graph][context_compressor]") {
    std::filesystem::remove_all(kWorkspace);
    auto cfg = base_config();
    cfg.auto_default_tools = false;
    cfg.context_window = 1000;
    cfg.max_tokens = 32;
    auto session = open(cfg);

    const std::string filler(400, 'x');
    std::vector<agent::Graph> steps;
    for (int i = 0; i < 6; ++i) {
        auto n = agent_node("turn" + std::to_string(i), "turn " + std::to_string(i) + " " + filler);
        n.history_key = "chat";
        n.system_prompt = "You are X.";
        steps.push_back(agent::make_graph(n));
    }
    auto probe = agent_node("probe", "show:system");
    probe.history_key = "chat";
    probe.system_prompt = "You are X.";
    steps.push_back(agent::make_graph(probe));

    agent::GraphRun run;
    run.thread_id = "auto-compress";
    auto res = agent::run_graph(session, agent::make_sequential("pipeline", std::move(steps)), run);
    INFO(res.error.message);
    REQUIRE(res.ok);

    const auto &chat = run.state.histories.at("chat");
    std::string dump;
    for (const auto &m : chat.history) dump += m.role + ":" + m.content.substr(0, 60) + "\n";
    INFO(dump);
    CHECK(chat.stats.compressions >= 1);
    CHECK(chat.history[0].content.starts_with("[Compressed history]: echo: Summarize"));
    // The engine got one system message: the node's prompt, then the summary.
    const std::string probe_reply(agent::get_state(run.state, "probe_reply"));
    CHECK(probe_reply.starts_with("1 system: You are X.\n\n[Compressed history]: echo: Summarize"));
    CHECK(agent::get_stats(session).total_compressions >= 1);
}

TEST_CASE("history is compressed inside a tool loop, before the request that follows a large tool result", "[builtin_tools][graph][context_compressor]") {
    std::filesystem::remove_all(kWorkspace);
    std::filesystem::create_directories(kWorkspace);
    { std::ofstream(ws_path("big.txt")) << std::string(3000, 'y'); }
    auto cfg = base_config();
    cfg.context_window = 1000;
    cfg.max_tokens = 32;
    auto session = open(cfg);

    // Short turns first: the history stays below 70% of the window, so the start-of-turn check does nothing.
    std::vector<agent::Graph> steps;
    for (int i = 0; i < 3; ++i) {
        auto n = agent_node("t" + std::to_string(i), "hello " + std::to_string(i));
        n.history_key = "chat";
        n.system_prompt = "You are X.";
        steps.push_back(agent::make_graph(n));
    }
    // This turn calls a tool whose result is large: only the check inside the loop can see it.
    auto reader = agent_node("reader", R"(call:read_file {"path":"big.txt"})", {"read_file"});
    reader.history_key = "chat";
    reader.system_prompt = "You are X.";
    steps.push_back(agent::make_graph(reader));

    agent::GraphRun run;
    run.thread_id = "loop-compress";
    auto res = agent::run_graph(session, agent::make_sequential("pipeline", std::move(steps)), run);
    INFO(res.error.message);
    REQUIRE(res.ok);
    CHECK(run.state.histories.at("chat").stats.compressions >= 1);
}
TEST_CASE("compress_context with a process-isolated engine and a non-chat default task",
          "[builtin_tools][context_compressor][engine_host]") {
    auto cfg = base_config();
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_host_path = AGENT_TEST_ENGINE_HOST_PATH;
    auto session = open(cfg);
    agent::Conversation memory;
    for (int i = 0; i < 6; ++i) {
        memory.history.push_back(agent::Message{.role = "user", .content = "q" + std::to_string(i)});
        memory.history.push_back(agent::Message{.role = "assistant", .content = "a" + std::to_string(i)});
    }
    REQUIRE(agent::compress_context(session, memory, 2).ok);
    REQUIRE(memory.history.size() == 3);
    CHECK(memory.history[0].content.starts_with("[Compressed history]: echo: Summarize"));

    // A second compression folds the old summary in instead of keeping it as a system prompt.
    for (int i = 0; i < 4; ++i) memory.history.push_back(agent::Message{.role = "user", .content = "more" + std::to_string(i)});
    REQUIRE(agent::compress_context(session, memory, 1).ok);
    size_t summaries = 0;
    for (const auto &m : memory.history)
        if (m.content.starts_with("[Compressed history]: ")) ++summaries;
    CHECK(summaries == 1);
    CHECK(memory.stats.compressions == 2);
}

TEST_CASE("graph nodes advertise only their allowed tools and skills", "[builtin_tools][graph][skills]") {
    std::filesystem::remove_all(kWorkspace);
    const std::string skills = ws_path("skills");
    write(skills + "/demo/SKILL.md", "---\nname: demo\ndescription: \"Demo skill.\"\n---\nFull demo instructions.\n");
    write(skills + "/demo/references/notes.md", "bundled notes");
    write(skills + "/demo/logo.png", "png");
    auto cfg = base_config();
    cfg.skills_dir = skills;
    auto session = open(cfg);
    REQUIRE(session.skills.count("demo") == 1);
    CHECK(session.skills.at("demo").description == "Demo skill."); // quotes stripped

    auto all = run_node(session, "show:system");
    CHECK(all.reply.find("- run_command:") != std::string::npos);
    CHECK(all.reply.find("- demo: Demo skill.") != std::string::npos);

    auto reader = run_node(session, "show:system", {"read_file"});
    CHECK(reader.reply.find("- read_file:") != std::string::npos);
    CHECK(reader.reply.find("- run_command:") == std::string::npos);
    CHECK(reader.reply.find("demo") == std::string::npos); // no use_skill, no skills list

    auto skilled = run_node(session, "show:system", {"use_skill"});
    CHECK(skilled.reply.find("- use_skill:") != std::string::npos);
    CHECK(skilled.reply.find("- demo: Demo skill.") != std::string::npos);

    // use_skill from the node: body, a bundled text file, a bundled image, an escape attempt.
    auto body = run_node(session, R"(call:use_skill {"name":"demo"})", {"use_skill"});
    CHECK(body.tool.content.find("Full demo instructions.") != std::string::npos);
    CHECK(body.tool.content.find("Skill directory: " + std::filesystem::absolute(skills + "/demo").string()) == 0);
    auto notes = run_node(session, R"(call:use_skill {"name":"demo","file":"references/notes.md"})", {"use_skill"});
    CHECK(notes.tool.content == "bundled notes");
    auto logo = run_node(session, R"(call:use_skill {"name":"demo","file":"logo.png"})", {"use_skill"});
    CHECK(logo.reply.starts_with("tool done\ntool media: image/png "));
    auto escape = run_node(session, R"(call:use_skill {"name":"demo","file":"../../../etc/passwd"})", {"use_skill"});
    CHECK(escape.tool.content == "Error: use_skill: 'file' must be inside the skill directory");
}
