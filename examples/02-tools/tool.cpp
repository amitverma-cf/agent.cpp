// docs/guides/writing-a-tool.md: a custom tool, the permission gate, event hooks, per-node tool hooks
// and MCP tools.
//
//   guide_tool <engine plugin> [mcp server command] [workspace]
//
// With the mock engine plugin (src/engines/mock) a user message "call:<tool>" makes the model request
// <tool> with {} arguments, so the whole tool loop runs without a real model.

#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const char *what, const std::string &detail = {}) {
    std::printf("[%s] %s%s%s\n", ok ? "ok" : "FAIL", what, detail.empty() ? "" : ": ", detail.c_str());
    if (!ok) ++g_failures;
}

// ---- the tool --------------------------------------------------------------------------------------

// Tool state lives outside the Session and is reached through user_data; it must outlive the Session.
struct WordCounter {
    int calls = 0;
};

// The string value of "key" in a flat JSON object (no escapes). agent.hpp has no JSON parser; use your own.
std::string json_string(std::string_view json, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    size_t at = json.find(needle);
    if (at == std::string_view::npos) return {};
    at = json.find('"', json.find(':', at + needle.size()) + 1);
    if (at == std::string_view::npos) return {};
    const size_t end = json.find('"', at + 1);
    return end == std::string_view::npos ? std::string() : std::string(json.substr(at + 1, end - at - 1));
}

agent::Result<std::string> count_words(std::string_view arguments, void *user_data) {
    auto *counter = static_cast<WordCounter *>(user_data);
    ++counter->calls;
    const std::string text = json_string(arguments, "text");
    int words = 0;
    bool in_word = false;
    for (char c : text) {
        const bool space = c == ' ' || c == '\n' || c == '\t';
        if (!space && !in_word) ++words;
        in_word = !space;
    }
    // ok(...) becomes the "tool" message; fail(...) becomes "Error: <message>" and the turn goes on.
    return agent::ok("{\"words\":" + std::to_string(words) + "}");
}

agent::Tool make_word_count_tool(WordCounter &counter) {
    return agent::Tool{
        .name = "word_count",
        .description = "Count the words in a text.",
        .parameter_schema = R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})",
        .callback = count_words,
        .user_data = &counter,
    };
}

// ---- the permission gate ---------------------------------------------------------------------------

struct Approvals {
    int asked = 0;
};

agent::PermissionDecision policy(std::string_view tool, std::string_view arguments, void *) {
    if (tool == "delete_path") return agent::PermissionDecision::Deny;
    if (tool == "run_command" || arguments.find("secret") != std::string_view::npos) return agent::PermissionDecision::Ask;
    return agent::PermissionDecision::Allow;
}

// Resolves Ask (a CLI would prompt the user here). true = approve.
bool prompt(std::string_view tool, std::string_view, void *user_data) {
    ++static_cast<Approvals *>(user_data)->asked;
    return tool != "run_command"; // approve everything but shell commands
}

// ---- hooks -----------------------------------------------------------------------------------------

void on_tool_event(const agent::Event &event, void *user_data) {
    // OnToolCall / OnToolResult payloads are the tool name / the result text.
    const std::string_view payload(reinterpret_cast<const char *>(event.payload.data()), event.payload.size());
    std::printf("   event %s: %.*s\n", event.type == agent::EventType::OnToolCall ? "OnToolCall" : "OnToolResult",
                static_cast<int>(payload.size()), payload.data());
    ++*static_cast<int *>(user_data);
}

void audit(agent::Session &, agent::Conversation &, std::string_view tool, std::string_view args, std::string_view result,
           void *user_data) {
    std::printf("   post-hook %.*s(%.*s) -> %.*s\n", static_cast<int>(tool.size()), tool.data(), static_cast<int>(args.size()), args.data(),
                static_cast<int>(result.size()), result.data());
    ++*static_cast<int *>(user_data);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <engine plugin> [mcp server command] [workspace]\n", argv[0]);
        return 2;
    }
    agent::AgentRuntime runtime;

    WordCounter counter;
    Approvals approvals;
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = argv[1];
    cfg.workspace_dir = argc > 3 ? argv[3] : (std::filesystem::temp_directory_path() / "agent-guide-tool").string();
    cfg.tools = {make_word_count_tool(counter)}; // added next to the default filesystem/terminal tools
    cfg.permission_check = policy;
    cfg.permission_prompt = prompt;
    cfg.permission_user_data = &approvals;

    auto sess = agent::init(cfg);
    check(sess.ok, "init", sess.error.message);
    if (!sess.ok) return 1;
    agent::Session session = std::move(sess.value);

    int events = 0;
    agent::register_event_hook(session, agent::EventType::OnToolCall, on_tool_event, &events);
    agent::register_event_hook(session, agent::EventType::OnToolResult, on_tool_event, &events);

    // 1. Direct dispatch (always through execute_tool, never Tool::callback): the gate applies.
    auto direct = agent::execute_tool(session, "word_count", R"({"text":"the quick brown fox"})");
    check(direct.ok && direct.value == R"({"words":4})", "execute_tool word_count", direct.ok ? direct.value : direct.error.message);

    auto asked = agent::execute_tool(session, "word_count", R"({"text":"a secret"})");
    check(asked.ok && approvals.asked == 1, "Ask -> permission_prompt approved", asked.ok ? asked.value : asked.error.message);

    auto denied = agent::execute_tool(session, "delete_path", R"({"path":"notes"})");
    check(!denied.ok && denied.error.code == agent::ErrorCode::PermissionDenied, "Deny -> PermissionDenied", denied.error.message);

    auto refused = agent::execute_tool(session, "run_command", R"({"command":"ls"})");
    check(!refused.ok && refused.error.code == agent::ErrorCode::PermissionDenied, "Ask -> prompt refused", refused.error.message);

    // 2. Model-requested call inside run_turn: the mock model asks for word_count, the result goes back
    //    to it as a "tool" message, and the turn continues ("tool done").
    agent::Conversation memory;
    auto turn = agent::run_turn(session, memory, "call:word_count");
    check(turn.ok && turn.value == "tool done", "run_turn dispatches the tool call", turn.ok ? turn.value : turn.error.message);
    check(counter.calls == 3 && events >= 2, "tool ran and OnToolCall/OnToolResult fired",
          "calls=" + std::to_string(counter.calls) + " events=" + std::to_string(events));

    // 3. Per-node tool hooks + allowlist in a graph Agent node.
    int hooked = 0;
    agent::GraphNode counter_node{.name = "counter",
                                  .kind = agent::GraphNodeKind::Agent,
                                  .allowed_tools = {"word_count"},
                                  .prompt = "call:word_count",
                                  .output_key = "answer"};
    counter_node.hooks = {{.when = agent::ToolHook::When::Post, .tool_name = "word_count", .callback = audit, .user_data = &hooked}};
    agent::GraphRun run;
    auto graph = agent::run_graph(session, agent::make_graph(counter_node), run);
    check(graph.ok && hooked == 1 && agent::get_state(run.state, "answer") == "tool done", "ToolHook fires in an Agent node",
          graph.ok ? std::string(agent::get_state(run.state, "answer")) : graph.error.message);

    // 4. MCP: every tool of an MCP stdio server becomes "<server>.<tool>" and goes through the same gate.
    if (argc > 2) {
        auto mcp = agent::connect_mcp_server(session, {.name = "demo", .command = argv[2]});
        check(mcp.ok, "connect_mcp_server", mcp.ok ? std::to_string(mcp.value.size()) + " tools" : mcp.error.message);
        if (mcp.ok) {
            for (const agent::Tool &t : mcp.value) std::printf("   mcp tool: %.*s\n", static_cast<int>(t.name.size()), t.name.data());
            auto echo = agent::execute_tool(session, "demo.echo", R"({"text":"hi"})");
            check(echo.ok && echo.value.find("echo: hi") != std::string::npos, "demo.echo", echo.ok ? echo.value : echo.error.message);
        }
        agent::disconnect_mcp_servers(session);
    }

    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
