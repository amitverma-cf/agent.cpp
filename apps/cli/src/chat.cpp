// agent chat: line-based text TUI over run_turn. Streams tokens, shows tool calls and results,
// asks before tools the default permission policy rates Ask, persists the conversation in the
// workspace's SQLite store, and opens generated artifacts with the OS default app.

#include "catalog_browse.hpp"
#include "chat_extras.hpp"
#include "chat_memory.hpp"
#include "chat_models.hpp"
#include "commands.hpp"
#include "edit_journal.hpp"
#include "mcp_config.hpp"
#include "model_detect.hpp"
#include "registries.hpp"
#include "runtimes.hpp"
#include "schedule.hpp"
#include "secrets.hpp"
#include "settings_tool.hpp"
#include "speed.hpp"
#include "tui.hpp"
#include "voice.hpp"
#include "workflows.hpp"

#include <agent-cpp/hub.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <simdjson.h>
#include <thread>

#if !defined(_WIN32)
#include <signal.h>
#endif

namespace agent::cli {

namespace {

namespace fs = std::filesystem;

constexpr std::string_view kIndexKey = "cli:conversations";
constexpr std::string_view kTitlePrefix = "cli:title:"; // + the conversation id: a name given with /rename
constexpr std::string_view kSystemPrompt =
    "You are agent, a helpful assistant running locally on the user's computer. Use the available tools when they help; "
    "file paths are relative to the workspace.";

std::atomic<int32_t *> g_cancel_flag{nullptr};
std::atomic<bool> g_generating{false};
bool g_at_line_start = true;
std::FILE *g_err = stderr; // the terminal's stderr, also while fd 2 goes to the engine log

void on_sigint(int) {
    // During a turn: cancel engine work (the same atomic store request_cancel does). At the prompt the
    // signal just interrupts the blocking read (no SA_RESTART), which ends the chat cleanly.
    if (!g_generating.load()) return;
    if (int32_t *flag = g_cancel_flag.load()) std::atomic_ref<int32_t>(*flag).store(1, std::memory_order_relaxed);
}

void install_sigint() {
#if defined(_WIN32)
    std::signal(SIGINT, on_sigint);
#else
    struct sigaction sa{};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // no SA_RESTART: a blocked fgets returns EINTR
    sigaction(SIGINT, &sa, nullptr);
#endif
}

bool g_tui = false;      // the full-screen interface is running: output goes to it, not to stdout
bool g_headless = false; // -p: the answer alone goes to stdout; tool lines and notes go to stderr
bool g_jsonl = false;    // --jsonl: every piece of output is one JSON line on stdout
bool g_acp = false;      // --acp: Agent Client Protocol on stdio (JSON-RPC, one message per line)
std::string g_acp_session, g_acp_call;
int g_acp_calls = 0;
bool g_last_turn_failed = false;

void acp_send(const std::string &line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void acp_update(const std::string &update) {
    acp_send("{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{\"sessionId\":\"" + json_escape(g_acp_session) +
             "\",\"update\":" + update + "}}");
}

const char *acp_kind(std::string_view tool) {
    if (tool == "read_file" || tool == "list_dir" || tool == "list_files") return "read";
    if (tool == "search_files" || tool == "grep" || tool == "web_search") return "search";
    if (tool == "write_file" || tool == "append_file" || tool == "create_dir") return "edit";
    if (tool == "delete_path") return "delete";
    if (tool == "move_path") return "move";
    if (tool == "run_command") return "execute";
    if (tool == "web_fetch") return "fetch";
    return "other";
}

void acp_tool_call(std::string_view name, std::string_view arguments) {
    g_acp_call = "call_" + std::to_string(++g_acp_calls);
    const bool is_json = !arguments.empty() && arguments.front() == '{';
    acp_update("{\"sessionUpdate\":\"tool_call\",\"toolCallId\":\"" + g_acp_call + "\",\"title\":\"" + json_escape(name) +
               "\",\"kind\":\"" + acp_kind(name) + "\",\"status\":\"in_progress\"" +
               (is_json ? ",\"rawInput\":" + std::string(arguments) : std::string()) + "}");
}

void acp_tool_result(std::string_view result) {
    acp_update("{\"sessionUpdate\":\"tool_call_update\",\"toolCallId\":\"" + g_acp_call +
               "\",\"status\":\"completed\",\"content\":[{\"type\":\"content\",\"content\":{\"type\":\"text\",\"text\":\"" +
               json_escape(result.substr(0, 4000)) + "\"}}]}");
}

void json_event(const char *type, std::string_view text) {
    std::printf("{\"type\":\"%s\",\"text\":\"%s\"}\n", type, json_escape(text).c_str());
    std::fflush(stdout);
}

TuiKind classify(std::string_view s) {
    if (s.starts_with("[tool]") || s.starts_with("[result]")) return TuiKind::Tool;
    if (s.starts_with("[error]") || s.starts_with("error")) return TuiKind::Error;
    return TuiKind::Note;
}

void out(std::string_view s) {
    if (s.empty() || g_acp) return;
    if (g_jsonl) {
        const TuiKind kind = classify(s);
        json_event(kind == TuiKind::Tool ? "tool" : kind == TuiKind::Error ? "error" : "note", s);
        return;
    }
    if (g_headless) {
        std::fwrite(s.data(), 1, s.size(), stderr);
        return;
    }
#if defined(AGENT_CLI_HAS_TUI)
    if (g_tui) {
        tui_emit(classify(s), s);
        return;
    }
#endif
    std::fwrite(s.data(), 1, s.size(), stdout);
    g_at_line_start = s.back() == '\n';
    std::fflush(stdout);
}

void ensure_newline() {
    if (!g_tui && !g_headless && !g_jsonl && !g_acp && !g_at_line_start) out("\n");
}

// Streamed answer text: the transcript's assistant block in the full-screen interface, stdout otherwise.
void emit_assistant(std::string_view s) {
    if (s.empty()) return;
    if (g_acp) {
        acp_update("{\"sessionUpdate\":\"agent_message_chunk\",\"content\":{\"type\":\"text\",\"text\":\"" + json_escape(s) + "\"}}");
        return;
    }
    if (g_jsonl) {
        json_event("token", s);
        return;
    }
    if (g_headless) {
        std::fwrite(s.data(), 1, s.size(), stdout);
        std::fflush(stdout);
        return;
    }
#if defined(AGENT_CLI_HAS_TUI)
    if (g_tui) {
        tui_emit(TuiKind::Assistant, s);
        return;
    }
#endif
    out(s);
}

void outf(const char *format, ...) {
    char buf[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    out(buf);
}

struct ChatState {
    Context *ctx = nullptr;
    // Models live in the registry (memory_budget_mb, LRU unload); lease is the current one. Every model
    // Session shares the SQLite store opened by `store`, so switching never leaves two write-back caches
    // on one database.
    ModelRegistry registry;
    ModelLease lease;
    Session store;
    Conversation memory;

    ModelRef model;
    LoadedEngine engine;
    std::string engine_flag; // --engine as given (empty = by model file type)
    std::string engine_options;
    std::string workspace;
    int context_window = 8192;
    int max_tokens = 2048;
    bool verbose = false;
    bool auto_approve = false;
    bool inprocess = false; // --inprocess: load the engine into the CLI instead of agent-engine-host
    bool interactive = false;
    bool use_tui = false; // the full-screen interface (a terminal, and not --plain)

    std::shared_ptr<SecretContext> secrets;  // names and bindings; values are read only when an MCP server starts
    std::vector<std::string> always_allowed; // tools the user allowed for good in this project (.agents/permissions.json)
    ChatMode mode = ChatMode::Ask;
    ThinkFilter think;
    bool repaint = false; // the open conversation was replaced by a command
    bool no_tools =
        false; // --json / --grammar: the answer is sampled under a grammar, which cannot share the turn with the tool-call grammar
    std::unique_ptr<EditJournal> journal; // what the tools changed in files this session (/diff, /undo)
    std::vector<Media> artifacts;
    size_t streamed_bytes = 0;
    StreamFilter filter;
    bool saved = false;      // the conversation is in the store (resumable)
    SettingsBridge settings; // what the settings tool uses (set up in open_session)
};

// The system prompt of a new conversation: ours, then the rules the user wrote for the project (AGENTS.md and .agents/rules).
std::string system_prompt(const ChatState &st) {
    // The model must know what computer it is on: a small model otherwise tries `ls` and `find` on Windows and loops on the errors.
#if defined(_WIN32)
    const char *platform =
        " The computer runs Windows: run_command uses cmd.exe, which has no ls, find, cat, grep, touch or mkdir -p (use dir, type and "
        "mkdir). "
        "To work with files prefer the file tools (list_dir, read_file, write_file) over shell commands. If a command fails, change "
        "the approach or tell the user what went wrong; never repeat the same failing command.";
#else
    const char *platform =
        " The computer runs Linux: run_command uses sh. To work with files prefer the file tools (list_dir, read_file, write_file) over "
        "shell commands. If a command fails, change the approach or tell the user what went wrong; never repeat the same failing command.";
#endif
    const char *habits =
        " When the user states a lasting preference, a fact about themselves or the project, or corrects you, save it "
        "with remember without "
        "being asked. When a request is ambiguous, ask with ask_user instead of guessing. For a request with several "
        "steps keep a checklist "
        "with todo. When the user mentions their files, a folder or a document without a path, look in the workspace first "
        "(list_dir, then read what matches) before asking; ask only when nothing matches. For questions about earlier "
        "chats, check recall and the saved notes first. When the user asks to make answers faster, slower, more creative, steadier or "
        "longer, or asks about a small helper model or MTP, use the settings tool (action get first, then set); never claim a "
        "speed feature the settings tool does not list.";
    const char *honesty =
        " After remember or forget, answer with one short sentence and do nothing else unless asked. To delete a note use forget, and say "
        "it was removed only when forget says so. For what was said in earlier chats call recall with the topic: it searches the notes "
        "and the saved conversations and says how many chats it searched. Never claim that a file was written, changed or deleted, or "
        "that a tool ran, unless a tool result in this conversation shows it; after write_file check the result and, when it matters, "
        "read the file back. Questions about the user's own files, folders, settings, models or installed tools are answered with "
        "list_dir, read_file or agent_run, never with web_fetch or web_search: the web is only for facts that are not on this computer. "
        "Skills and MCP servers: agent_run [\"skill\",\"browse\"] and [\"mcp\",\"browse\"] list well-known ones; [\"skill\",\"search\","
        "\"WORDS\"] and [\"mcp\",\"search\",\"WORDS\"] search the catalogues; [\"skill\",\"add\",\"owner/repo/skill\"], "
        "[\"mcp\",\"add\",\"NAME\"] (a name from the browse list) and [\"mcp\",\"add\",\"NAME\",\"--run\",\"npx -y PACKAGE\"] (or "
        "\"--url\",\"https://...\") install them; [\"skill\",\"ls\"] and [\"mcp\",\"ls\"] list what is installed; rm, on and off remove, "
        "or switch them. Their tools and instructions are there from the next message. Workflows: to keep a described process, call "
        "workflow_save with a short name and the graph as JSON; run it later, in any chat, with agent_run [\"run\",\"NAME\"]; "
        "agent_run [\"workflow\",\"ls\"] lists them. When a run ends with \"status: interrupted\" it waits for the user: show what is "
        "pending with ask_user, then continue with agent_run [\"run\",\"NAME\",\"--resume\",\"THREAD\",\"--input\",\"ANSWER\"] (THREAD "
        "is in the output; the answer approve lets pending tool calls run).";
    const auto models = installed_models(catalog_path(*st.ctx));
    return std::string(kSystemPrompt) + platform + habits + honesty + installed_models_prompt(models) + workflows_prompt(*st.ctx) +
           load_project_rules(st.workspace, st.ctx->has_home ? st.ctx->home.root : st.ctx->config_dir);
}

void log_fn(LogLevel level, std::string_view message, void *user_data) {
    const auto *st = static_cast<const ChatState *>(user_data);
    if (!st->verbose && (level == LogLevel::Debug || level == LogLevel::Info)) return;
    const char *prefix = level == LogLevel::Error ? "error" : level == LogLevel::Warning ? "warning" : "log";
    ensure_newline();
    std::fprintf(g_err, "[%s] %.*s\n", prefix, static_cast<int>(message.size()), message.data());
}

void on_token(std::string_view token, void *user_data) {
    auto *st = static_cast<ChatState *>(user_data);
    st->streamed_bytes += token.size();
    emit_assistant(filter_stream(st->filter, filter_think(st->think, token)));
}

// "always" in a permission question: kept for the rest of the session and in <workspace>/.agents/permissions.json for the next.
void remember_always(ChatState &st, const std::string &tool) {
    if (std::find(st.always_allowed.begin(), st.always_allowed.end(), tool) == st.always_allowed.end()) st.always_allowed.push_back(tool);
    save_always_allowed(st.workspace, st.always_allowed);
}

PermissionDecision check_tool(std::string_view name, std::string_view arguments, void *user_data) {
    auto *st = static_cast<ChatState *>(user_data);
    ensure_newline();
    if (g_acp) acp_tool_call(name, arguments);
    out("[tool] " + std::string(name) + " " + one_line(arguments, 300) + "\n");
    if (st->journal) st->journal->before_tool(name, arguments);
    const bool changes_settings = name == "settings" && settings_call_changes(arguments);
    if (st->mode == ChatMode::Plan && (tool_changes_things(name) || changes_settings)) {
        out("[plan mode] " + std::string(name) + " is not allowed; /mode ask or /mode auto lets the agent make changes\n");
        return PermissionDecision::Deny;
    }
    if (st->mode == ChatMode::Auto) return PermissionDecision::Allow;
    PermissionDecision d = tools::default_permission_policy(name, arguments, nullptr);
    if ((name == "web_fetch" || name == "web_search" || name == "agent_run") && d == PermissionDecision::Allow)
        d = PermissionDecision::Ask;                                                     // the network, other models
    if (changes_settings && d == PermissionDecision::Allow) d = PermissionDecision::Ask; // changing a setting reloads the model
    if (d == PermissionDecision::Ask && std::find(st->always_allowed.begin(), st->always_allowed.end(), name) != st->always_allowed.end())
        return PermissionDecision::Allow;
    return d;
}

bool prompt_tool(std::string_view name, std::string_view arguments, void *user_data) {
    auto *st = static_cast<ChatState *>(user_data);
    const std::string n(name);
    if (g_acp && !st->auto_approve) {
        // session/request_permission: the editor answers with the option it picked.
        const std::string rid = "perm_" + std::to_string(++g_acp_calls);
        acp_send("{\"jsonrpc\":\"2.0\",\"id\":\"" + rid + "\",\"method\":\"session/request_permission\",\"params\":{\"sessionId\":\"" +
                 json_escape(g_acp_session) + "\",\"toolCall\":{\"toolCallId\":\"" + g_acp_call +
                 "\"},\"options\":[{\"optionId\":\"allow-once\",\"name\":\"Allow once\",\"kind\":\"allow_once\"},"
                 "{\"optionId\":\"allow-always\",\"name\":\"Always allow\",\"kind\":\"allow_always\"},"
                 "{\"optionId\":\"reject-once\",\"name\":\"Reject\",\"kind\":\"reject_once\"}]}}");
        std::string reply;
        while (read_line(stdin, reply) == ReadStatus::Line) {
            if (reply.find(rid) == std::string::npos || reply.find("\"method\"") != std::string::npos) continue; // not our answer
            if (reply.find("allow-always") != std::string::npos) remember_always(*st, n);
            return reply.find("allow-once") != std::string::npos || reply.find("allow-always") != std::string::npos;
        }
        return false;
    }
    if (st->auto_approve) {
        out("[permission] " + n + ": approved (--yes)\n");
        return true;
    }
    if (g_jsonl) {
        // The editor answers on the next line of stdin: {"type":"permission","allow":true,"always":false}.
        std::printf("{\"type\":\"permission_request\",\"tool\":\"%s\",\"arguments\":\"%s\"}\n", json_escape(n).c_str(),
                    json_escape(arguments).c_str());
        std::fflush(stdout);
        std::string reply;
        if (read_line(stdin, reply) != ReadStatus::Line) return false;
        reply.erase(std::remove(reply.begin(), reply.end(), ' '), reply.end());
        if (reply.find("\"always\":true") != std::string::npos) remember_always(*st, n);
        return reply.find("\"allow\":true") != std::string::npos;
    }
#if defined(AGENT_CLI_HAS_TUI)
    if (g_tui) {
        bool always = false;
        const bool allowed = tui_ask(n + "\n" + pretty_arguments(arguments), &always);
        if (always) remember_always(*st, n);
        return allowed;
    }
#endif
    if (!st->interactive) {
        out("[permission] " + n + ": denied (stdin is not a terminal; pass --yes to approve such calls)\n");
        return false;
    }
    out("[permission] allow " + n + "? [y]es / [N]o / [a]lways this session: ");
    std::string line;
    const ReadStatus rs = read_line(stdin, line);
    g_at_line_start = true;
    if (rs != ReadStatus::Line) {
        out("\n[permission] denied\n");
        return false;
    }
    const size_t i = line.find_first_not_of(" \t");
    const char c = i == std::string::npos ? 'n' : line[i];
    if (c == 'a' || c == 'A') {
        remember_always(*st, n);
        return true;
    }
    return c == 'y' || c == 'Y';
}

void on_tool_result(const Event &event, void *) {
    const std::string_view result(reinterpret_cast<const char *>(event.payload.data()), event.payload.size());
    if (g_acp) {
        acp_tool_result(result);
        return;
    }
    ensure_newline();
    out("[result] " + one_line(result, 300) + "\n");
}

Result<void> acquire_current(ChatState &st) {
    auto lease = acquire_model(st.registry, st.model.name);
    if (!lease.ok) return fail(lease.error.code, lease.error.message);
    st.lease = std::move(lease.value);
    g_cancel_flag.store(st.lease->cancel_flag.get());
    return ok();
}

// Makes the session's MCP servers and skills match the files: the servers of the home's mcp/mcp.json and mcp.d and of the project's
// .agents/mcp.json and mcp.d (each runs sandboxed in its own data folder, its keys put in its environment only as it starts), and the
// skills of <home>/skills and <workspace>/.agents/skills. Runs when a model loads and before every message, so a server or a skill
// added in Settings, by a command or by writing the file shows up in the next message. `quiet`: no notes (the load prints to stderr).
std::string sync_extensions(ChatState &st, Session &session, bool at_load) {
    std::string report; // what changed, for the agent_run result: the agent must not claim a server that did not start
    // The chat may work in a folder other than where it was started, and .agents/ may appear while it runs.
    if (st.ctx->project_root.empty()) st.ctx->project_root = find_project_root(st.workspace);
    std::vector<std::string> warnings;
    const auto servers = collect_mcp_servers(*st.ctx, warnings);
    auto say = [&](const std::string &line) {
        report += line + "\n";
        if (at_load) std::fprintf(g_err, "%s\n", line.c_str());
        else out(line + "\n");
    };
    if (at_load)
        for (const std::string &w : warnings) say("[mcp] warning: " + w);
    const auto connected = list_mcp_servers(session);
    for (const std::string &name : connected)
        if (std::none_of(servers.begin(), servers.end(), [&](const McpServerConfig &c) { return c.name == name; })) {
            disconnect_mcp_server(session, name);
            say("[mcp] " + name + " stopped");
        }
    for (const McpServerConfig &server : servers) {
        if (std::find(connected.begin(), connected.end(), server.name) != connected.end()) continue;
        auto result = connect_mcp_server(session, server);
        if (result.ok) say("[mcp] " + server.name + ": " + std::to_string(result.value.size()) + " tool(s)");
        else say("[mcp] " + server.name + " not started: " + result.error.message);
    }
    for (const std::string &dir : skill_dirs(*st.ctx, st.workspace)) {
        std::error_code ec;
        if (fs::is_directory(dir, ec)) load_skills_dir(session, dir);
    }
    for (auto it = session.skills.begin(); it != session.skills.end();) { // a skill switched off or removed
        std::error_code ec;
        it = fs::exists(fs::path(it->second.path) / "SKILL.md", ec) ? std::next(it) : session.skills.erase(it);
    }
    return report;
}

std::string sync_extensions(ChatState &st) { return st.lease ? sync_extensions(st, *st.lease, false) : std::string(); }

// ModelSpec::on_load: share the chat's store and show tool results.
Result<void> on_model_loaded(Session &session, const ModelSpec &, void *user_data) {
    auto *st = static_cast<ChatState *>(user_data);
    session.config.memory = st->store.config.memory;
    session.memory_state = st->store.memory_state;
    register_event_hook(session, EventType::OnToolResult, on_tool_result, st);
    sync_extensions(*st, session, true);
    return ok();
}

// ---- tools of the web: the model asks, the user is asked first (see check_tool)

constexpr std::string_view kWebFetchSchema =
    R"({"type":"object","properties":{"url":{"type":"string","description":"an http or https address"}},"required":["url"]})";
constexpr std::string_view kWebSearchSchema =
    R"({"type":"object","properties":{"query":{"type":"string","description":"what to search for"}},"required":["query"]})";

std::string url_encode(std::string_view s) {
    static const char *hex = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out += static_cast<char>(c);
        else out += {'%', hex[c >> 4], hex[c & 15]};
    }
    return out;
}

std::string cut_text(std::string text, size_t max) {
    if (text.size() > max) {
        text.resize(max);
        text += "\n[cut]";
    }
    return text;
}

Result<std::string> web_fetch_cb(std::string_view args, void *) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    std::string_view url;
    if (parser.parse(padded).get(doc) || doc["url"].get(url)) return fail<std::string>(ErrorCode::InvalidConfig, "web_fetch needs a url");
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0)
        return fail<std::string>(ErrorCode::InvalidConfig, "web_fetch takes http and https addresses only");
    auto page = agent::hub::http_get_text(url, agent::hub::http_options_from_env(), 1u << 20);
    if (!page.ok) return fail<std::string>(page.error.code, "cannot fetch " + std::string(url) + ": " + page.error.message);
    const std::string &body = page.value;
    std::string head = body.substr(0, 2048);
    for (char &c : head) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool html = head.find("<html") != std::string::npos || head.find("<body") != std::string::npos ||
                      head.find("<!doctype html") != std::string::npos;
    return ok(cut_text(html ? html_to_text(body) : body, 20000));
}

Result<std::string> web_search_cb(std::string_view args, void *user) {
    const auto *tmpl = static_cast<const std::string *>(user);
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    std::string_view query;
    if (parser.parse(padded).get(doc) || doc["query"].get(query))
        return fail<std::string>(ErrorCode::InvalidConfig, "web_search needs a query");
    std::string url = *tmpl;
    const size_t at = url.find("{q}");
    if (at == std::string::npos) return fail<std::string>(ErrorCode::InvalidConfig, "search_url has no {q} in config.json");
    url.replace(at, 3, url_encode(query));
    auto page = agent::hub::http_get_text(url, agent::hub::http_options_from_env(), 1u << 20);
    if (!page.ok) return fail<std::string>(page.error.code, "the search failed: " + page.error.message);
    return ok(cut_text(page.value, 8000));
}

// ---- the agent's own commands as a tool: "draw a lighthouse", "transcribe talk.wav", "run flow.json" from a chat message.
// The command runs as `agent <args>` in a child process (its own engine, its own memory), the output comes back as the tool result,
// and the files it names ("media: MIME PATH") are added to the artifacts that /open lists. The user is asked first (see check_tool).

constexpr std::string_view kAgentRunSchema =
    R"({"type":"object","properties":{"args":{"type":"array","items":{"type":"string"},"description":"The arguments of the agent command as separate strings, for example [\"task\",\"image_gen\",\"--model\",\"sdxs-512\",\"--text\",\"a red lighthouse\"], [\"task\",\"asr\",\"--model\",\"parakeet\",\"--file\",\"talk.wav\"], [\"run\",\"flow.json\"], [\"model\",\"ls\"] or [\"engine\",\"ls\"]. The first word is task, run, model, engine, doctor, skill, mcp, workflow, schedule or wake. A saved workflow runs by name: [\"run\",\"daily-report\"]."}},"required":["args"]})";

std::string shell_quote(const std::string &s) {
#if defined(_WIN32)
    std::string r = "\"";
    for (const char c : s) r += c == '"' ? std::string("\\\"") : std::string(1, c);
    return r + "\"";
#else
    std::string r = "'";
    for (const char c : s) r += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return r + "'";
#endif
}

std::string self_executable() {
    std::error_code ec;
#if defined(_WIN32)
    return (fs::path(executable_dir()) / "agent.exe").string();
#else
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string("agent") : p.string();
#endif
}

// Runs `agent ARGS...`; the text is what the command printed. `failed` is set when it did not succeed.
std::string run_agent_command(ChatState &st, std::vector<std::string> args, bool &failed) {
    failed = true;
    if (args.empty()) return "no command given";
    static const char *const allowed[] = {"task", "run",      "model",   "engine",   "doctor", "skill",
                                          "mcp",  "workflow", "runtime", "schedule", "wake"};
    if (std::find_if(std::begin(allowed), std::end(allowed), [&](const char *a) { return args[0] == a; }) == std::end(allowed))
        return "the command '" + args[0] +
               "' is not available here; use task, run, model, engine, doctor, skill, mcp, workflow, runtime, schedule or wake";
    auto has = [&](const std::string &flag) { return std::find(args.begin(), args.end(), flag) != args.end(); };
    if (st.ctx->config.local_only &&
        (args[0] == "mcp" || args[0] == "skill" || (args[0] == "model" && args.size() > 1 && args[1] != "ls" && args[1] != "info") ||
         (args[0] == "engine" && args.size() > 1 && (args[1] == "pull" || args[1] == "add"))))
        return "local only is on (Settings, The web): commands that download are refused";
    if (args[0] == "run" && args.size() > 1) args[1] = resolve_graph_arg(*st.ctx, args[1]); // the name of a saved workflow
    if ((args[0] == "skill" || args[0] == "mcp") && args.size() > 1 && (args[1] == "add" || args[1] == "rm")) {
        // The user was asked about this agent_run call; the command itself must not wait for a yes on a terminal.
        if (args[1] == "add" && !has("--yes")) args.push_back("--yes");
    }
    if ((args[0] == "mcp" || args[0] == "skill") && !has("--workspace")) args.insert(args.end(), {"--workspace", st.workspace});
    std::string model_note;
    if (args[0] == "task" && args.size() > 1) {
        // A model name that is not installed (a guess) becomes the installed model for the task.
        for (size_t i = 2; i + 1 < args.size(); ++i) {
            if (args[i] != "--model" && args[i] != "-m") continue;
            bool found = true;
            const std::string chosen = model_for_task(installed_models(catalog_path(*st.ctx)), args[1], args[i + 1], found);
            if (!found) {
                failed = true;
                return "No model for this is installed on this computer (" + capability_for_task(args[1]) +
                       "). A model can be downloaded first: ask the user which one they want, or look for one on Hugging Face.";
            }
            if (chosen != args[i + 1]) {
                model_note = "(there is no model named '" + args[i + 1] + "'; used the installed model '" + chosen + "')\n";
                args[i + 1] = chosen;
            }
        }
    }
    if (st.ctx->has_home && !has("--home")) args.insert(args.end(), {"--home", st.ctx->home.root});
    if ((args[0] == "task" || args[0] == "run") && !has("--workspace")) args.insert(args.end(), {"--workspace", st.workspace});
    std::string command = shell_quote(self_executable());
    for (const std::string &a : args) command += " " + shell_quote(a);
    // Only what the command prints on stdout is the result. The engine's loader writes tens of kilobytes of log lines to stderr; they go to
    // a file, and the end of it is shown only when the command fails.
    static std::atomic<unsigned> run_counter{0};
    std::error_code tmp_ec;
    const fs::path err_file =
        fs::temp_directory_path(tmp_ec) / ("agent-run-" + std::to_string(++run_counter) + "-" +
                                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".err");
    command += " 2>" + shell_quote(err_file.string());
#if defined(_WIN32)
    command += " <NUL"; // the command never reads the chat's terminal: a question it would ask is answered "no"
#else
    command += " </dev/null";
#endif
#if defined(_WIN32)
    std::FILE *pipe = _popen(command.c_str(), "r");
#else
    std::FILE *pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) return "cannot start the command";
    std::string output;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        output += buf;
        if (output.size() > 24 * 1024) {
            output.resize(24 * 1024);
            output += "\n[cut]\n";
            break;
        }
    }
#if defined(_WIN32)
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    failed = status != 0;
    {
        std::ifstream err_in(err_file, std::ios::binary);
        std::string err_text((std::istreambuf_iterator<char>(err_in)), std::istreambuf_iterator<char>());
        err_in.close();
        fs::remove(err_file, tmp_ec);
        if (failed && !err_text.empty()) {
            if (err_text.size() > 2000) err_text = "..." + err_text.substr(err_text.size() - 2000);
            output += (output.empty() || output.back() == '\n' ? "" : "\n") + err_text;
        }
    }
    // "media: image/png /path/file.png 1234 bytes": the file is an artifact now (/open lists it).
    size_t at = 0;
    while ((at = output.find("media: ", at)) != std::string::npos) {
        const size_t end = output.find('\n', at);
        const std::string line = output.substr(at + 7, end == std::string::npos ? std::string::npos : end - at - 7);
        const size_t sp = line.find(' ');
        const size_t tail = line.rfind(" bytes");
        if (sp != std::string::npos && tail != std::string::npos && tail > sp) {
            const size_t size_at = line.rfind(' ', tail - 1);
            if (size_at != std::string::npos && size_at > sp)
                st.artifacts.push_back(Media{line.substr(0, sp), line.substr(sp + 1, size_at - sp - 1)});
        }
        at = end == std::string::npos ? output.size() : end;
    }
    if (failed) output = beginner_error(output);
    if (args[0] == "skill" || args[0] == "mcp") { // new tools and skills are there for the next message; say if a server did not start
        const std::string report = sync_extensions(st);
        if (report.find("not started") != std::string::npos) failed = true;
        if (!report.empty()) output += (output.empty() || output.back() == '\n' ? "" : "\n") + report;
    }
    return model_note + (output.empty() ? std::string(failed ? "the command failed" : "done") : output);
}

Result<std::string> agent_run_cb(std::string_view args, void *user) {
    auto *st = static_cast<ChatState *>(user);
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    simdjson::dom::array list;
    if (parser.parse(padded).get(doc) || doc["args"].get(list))
        return fail<std::string>(ErrorCode::InvalidConfig, "agent_run needs args, a list of strings");
    std::vector<std::string> words;
    for (simdjson::dom::element e : list) {
        std::string_view w;
        if (e.get(w)) return fail<std::string>(ErrorCode::InvalidConfig, "every item of args must be a string");
        words.emplace_back(w);
    }
    bool failed = false;
    std::string text = run_agent_command(*st, std::move(words), failed);
    if (failed) return fail<std::string>(ErrorCode::InvalidConfig, cut_text(text, 4000));
    return ok(std::move(text));
}

// ---- notes that outlive the conversation: remember / recall over the session's key-value memory

constexpr std::string_view kRememberSchema =
    R"({"type":"object","properties":{"key":{"type":"string","description":"a short name for the note"},"value":{"type":"string","description":"the text to keep"}},"required":["key","value"]})";
constexpr std::string_view kRecallSchema =
    R"({"type":"object","properties":{"key":{"type":"string","description":"the name a note was kept under"},"query":{"type":"string","description":"words to look for in the notes and in the earlier chats of this project"}}})";
constexpr std::string_view kForgetSchema =
    R"({"type":"object","properties":{"key":{"type":"string","description":"the exact name of the note"},"text":{"type":"string","description":"words that describe the note when its name is not known"}}})";

Result<std::string> remember_cb(std::string_view args, void *user) {
    auto *st = static_cast<ChatState *>(user);
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    std::string_view key, value;
    if (parser.parse(padded).get(doc) || doc["key"].get(key) || doc["value"].get(value) || key.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "remember needs a key and a value");
    if (!st->lease) return fail<std::string>(ErrorCode::InvalidConfig, "no model is loaded");
    if (auto r = note_keep(*st->lease, key, value); !r.ok) return fail<std::string>(r.error.code, r.error.message);
    return ok("kept note '" + std::string(key) + "'. Confirm in one short sentence; do nothing else unless asked.");
}

Result<std::string> forget_cb(std::string_view args, void *user) {
    auto *st = static_cast<ChatState *>(user);
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    std::string_view key, text;
    if (parser.parse(padded).get(doc)) return fail<std::string>(ErrorCode::InvalidConfig, "forget needs a key or a text");
    if (doc["key"].get(key)) key = {};
    if (doc["text"].get(text)) text = {};
    if (key.empty() && text.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "forget needs a key or a text");
    if (!st->lease) return fail<std::string>(ErrorCode::InvalidConfig, "no model is loaded");
    bool removed = false;
    const std::string answer = note_forget(*st->lease, key, text, removed);
    if (!removed) return fail<std::string>(ErrorCode::KeyNotFound, answer);
    return ok(answer);
}

Result<std::string> recall_cb(std::string_view args, void *user) {
    auto *st = static_cast<ChatState *>(user);
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    std::string_view key, query;
    if (parser.parse(padded).get(doc)) return fail<std::string>(ErrorCode::InvalidConfig, "recall needs a key or a query");
    if (doc["key"].get(key)) key = {};
    if (doc["query"].get(query)) query = {};
    if (!st->lease) return fail<std::string>(ErrorCode::InvalidConfig, "no model is loaded");
    return ok(recall_text(*st->lease, key, query, st->memory.id));
}

// A reusable workflow: the graph is checked, then kept as <home>/graphs/NAME.json and run later by name.
constexpr std::string_view kWorkflowSaveSchema =
    R"({"type":"object","properties":{"name":{"type":"string","description":"a short name: letters, digits, - and _, for example daily-report"},"graph":{"type":"object","description":"the graph, see the tool description"}},"required":["name","graph"]})";

Result<std::string> workflow_save_cb(std::string_view args, void *user) {
    auto *st = static_cast<ChatState *>(user);
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc, graph;
    std::string_view name;
    if (parser.parse(padded).get(doc) || doc["name"].get(name) || doc["graph"].get(graph))
        return fail<std::string>(ErrorCode::InvalidConfig, "workflow_save needs a name and a graph");
    std::string json;
    std::string_view as_text;
    if (graph.is_string() && !graph.get(as_text)) json = std::string(as_text);
    else json = simdjson::to_string(graph);
    return workflow_save(*st->ctx, name, json);
}

// What the agent knows from earlier chats that fits this message: the notes kept with `remember` whose name or text shares words with
// it, best first, within a small budget. It goes in front of the message, so the agent follows what the user said before without asking.
std::string memory_pack(ChatState &st, const std::string &prompt) {
    if (!st.lease) return {};
    auto index = retrieve_memory(*st.lease, "note_index");
    if (!index.ok || index.value.empty()) return {};
    auto lower = [](std::string s) {
        for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    std::vector<std::string> words;
    {
        std::string w;
        for (const char c : lower(prompt) + " ") {
            if (std::isalnum(static_cast<unsigned char>(c))) w += c;
            else {
                if (w.size() >= 4 && std::find(words.begin(), words.end(), w) == words.end()) words.push_back(w);
                w.clear();
            }
        }
    }
    if (words.empty()) return {};
    std::vector<std::pair<int, std::string>> found;
    size_t from = 0;
    while (from < index.value.size()) {
        size_t end = index.value.find('\n', from);
        if (end == std::string::npos) end = index.value.size();
        const std::string key = index.value.substr(from, end - from);
        from = end + 1;
        if (key.empty()) continue;
        auto value = retrieve_memory(*st.lease, "note:" + key);
        if (!value.ok) continue;
        const std::string haystack = lower(key + " " + value.value);
        int score = 0;
        for (const std::string &w : words)
            if (haystack.find(w) != std::string::npos) ++score;
        if (score > 0) found.emplace_back(score, "- " + key + ": " + one_line(value.value, 300) + "\n");
    }
    std::stable_sort(found.begin(), found.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::string pack;
    for (const auto &[score, line] : found) {
        if (pack.size() + line.size() > 1200) break;
        pack += line;
    }
    if (pack.empty()) return {};
    return "[Notes kept from earlier chats; use them when they matter]\n" + pack + "[End of notes]\n\n";
}

constexpr std::string_view kTodoSchema =
    R"({"type":"object","properties":{"items":{"type":"array","description":"the whole list, every time","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["content","status"]}}},"required":["items"]})";

// The agent's own checklist for a multi-step request: one call writes the whole list, the result shows it.
Result<std::string> todo_cb(std::string_view args, void *) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    simdjson::dom::array list;
    if (parser.parse(padded).get(doc) || doc["items"].get(list)) return fail<std::string>(ErrorCode::InvalidConfig, "todo needs items");
    std::string text;
    size_t done = 0, total = 0;
    for (simdjson::dom::element e : list) {
        std::string_view content, status;
        if (e["content"].get(content) || e["status"].get(status)) continue;
        ++total;
        if (status == "completed") ++done;
        text += (status == "completed" ? "[x] " : status == "in_progress" ? "[~] " : "[ ] ") + std::string(content) + "\n";
    }
    return ok(std::to_string(done) + "/" + std::to_string(total) + " done\n" + text);
}

// A question from the agent to the user, shown in the chat as a line to answer: the agent asks instead of guessing.
constexpr std::string_view kAskUserSchema =
    R"({"type":"object","properties":{"question":{"type":"string","description":"one precise question"}},"required":["question"]})";

Result<std::string> ask_user_cb(std::string_view args, void *) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    std::string_view question;
    if (parser.parse(padded).get(doc) || doc["question"].get(question) || question.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "ask_user needs a question");
#if defined(AGENT_CLI_HAS_TUI)
    if (g_tui) {
        std::string answer;
        if (!tui_prompt(question, "", answer))
            return ok(std::string("(the user did not answer; decide yourself and say what you assumed)"));
        return ok(answer.empty() ? std::string("(no answer)") : answer);
    }
#endif
    return ok(std::string("(no one to ask in this mode; decide yourself and say what you assumed)"));
}

#if defined(AGENT_CLI_HAS_TUI)
std::string tui_set_setting(ChatState &st, const std::string &key, const std::string &value);
bool reload_model(ChatState &st);

// The settings tool reads and changes settings through these, the same code as the Settings screen.
std::string setting_text(ChatState &st, const std::string &key) {
    const CliConfig &c = st.ctx->config;
    if (key == "temperature") {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(c.temperature));
        return buf;
    }
    if (key == "max_tokens") return std::to_string(st.max_tokens);
    if (key == "context_window") return std::to_string(st.context_window);
    if (key == "mode") return chat_mode_name(st.mode);
    if (key == "draft_model_choices") {
        std::vector<std::pair<std::string, uint64_t>> installed;
        if (auto catalog = agent::hub::list_models(catalog_path(*st.ctx)); catalog.ok)
            for (const auto &m : catalog.value) installed.emplace_back(m.name, m.size);
        std::string text;
        for (const std::string &name : draft_model_candidates(installed, st.model.name)) text += (text.empty() ? "" : ", ") + name;
        return text;
    }
    return speed_setting_text(c, key);
}

void init_settings_bridge(ChatState &st) {
    st.settings.user = &st;
    st.settings.get = [](void *u, const std::string &key) { return setting_text(*static_cast<ChatState *>(u), key); };
    st.settings.apply = [](void *u, const std::string &key, const std::string &value) {
        return tui_set_setting(*static_cast<ChatState *>(u), key, value);
    };
    st.settings.reload = [](void *u) { return reload_model(*static_cast<ChatState *>(u)); };
}
#endif

Result<void> open_session(ChatState &st) {
    const std::string previous = st.lease ? st.lease.name : std::string();
    release_model(st.lease);
    g_cancel_flag.store(nullptr);
    // Without a budget only one model stays loaded: free the old one before loading the next.
    if (st.ctx->config.memory_budget_mb <= 0 && !previous.empty() && previous != st.model.name) unload_model(st.registry, previous);
    if (has_model(st.registry, st.model.name)) return acquire_current(st);

    Config cfg;
    cfg.provider = AiProvider::Engine;
    cfg.engine = st.engine.name;
    cfg.engine_isolation = cli_engine_isolation(st.inprocess, st.engine.isolation);
    cfg.engine_sandbox.gpu = cli_engine_gpu(st.engine.backend);
    apply_sandbox_choice(cfg);
    {
        std::string draft_path;
        if (!st.ctx->config.draft_model.empty())
            if (auto catalog = agent::hub::list_models(catalog_path(*st.ctx)); catalog.ok)
                for (const auto &m : catalog.value)
                    if (m.name == st.ctx->config.draft_model && m.path != st.model.path) draft_path = m.path;
        cfg.engine_options = speed_engine_options(st.ctx->config, st.engine.name, st.engine_options, draft_path);
    }
    cfg.model = st.model.path;
    cfg.workspace_dir = st.workspace;
    {
        // With MCP servers configured the tool list gets long, so the model sees tool_search / load_tool and loads tools on demand. Without
        // them the handful of built-in tools stay directly visible: small models do much better when they can just call list_dir.
        std::vector<std::string> mcp_warnings;
        cfg.lazy_tools = !collect_mcp_servers(*st.ctx, mcp_warnings).empty();
    }
    install_cli_hooks(cfg, *st.ctx);
    if (st.no_tools) cfg.auto_default_tools = false;
    if (!st.no_tools && !st.ctx->config.local_only)
        cfg.tools.push_back(Tool{.name = "web_fetch",
                                 .description = "Fetch a web page and return its text (HTML is reduced to readable text, at most 20000 "
                                                "characters). The user is asked first.",
                                 .parameter_schema = kWebFetchSchema,
                                 .callback = web_fetch_cb});
    if (!st.no_tools && !st.ctx->config.local_only && !st.ctx->config.search_url.empty())
        cfg.tools.push_back(
            Tool{.name = "web_search",
                 .description = "Search the web and return the results as the search service gave them. The user is asked first.",
                 .parameter_schema = kWebSearchSchema,
                 .callback = web_search_cb,
                 .user_data = &st.ctx->config.search_url});
    if (!st.no_tools)
        cfg.tools.push_back(Tool{.name = "agent_run",
                                 .description =
                                     "Run an agent command and return what it printed: images, video, 3D meshes, speech, transcripts, "
                                     "embeddings, ONNX models, graphs (agent run FILE.json), model downloads and listings. For "
                                     "example [\"task\",\"image_gen\",\"--model\",\"NAME\",\"--text\",\"prompt\"]. Use "
                                     "[\"model\",\"ls\"] and [\"engine\",\"ls\"] to see what is installed. Capabilities of "
                                     "task: text_gen, ocr, asr, tts, vad, diarize, image_gen, image_edit, video_gen, mesh_gen, "
                                     "embed, rerank, decide, classify, generic. The user is asked first.",
                                 .parameter_schema = kAgentRunSchema,
                                 .callback = agent_run_cb,
                                 .user_data = &st});
#if defined(AGENT_CLI_HAS_TUI)
    if (!st.no_tools) {
        init_settings_bridge(st);
        cfg.tools.push_back(settings_tool(&st.settings));
    }
#endif
    if (!st.no_tools) {
        cfg.tools.push_back(Tool{.name = "remember",
                                 .description =
                                     "Keep a lasting fact about the user or the project (a preference, a name, a correction) under a "
                                     "short name. Notes that match a later message are shown to the agent automatically. Afterwards "
                                     "answer with one short sentence and do nothing else unless asked.",
                                 .parameter_schema = kRememberSchema,
                                 .callback = remember_cb,
                                 .user_data = &st});
        cfg.tools.push_back(Tool{.name = "ask_user",
                                 .description = "Ask the user one precise question and wait for the answer. Use it when the request is "
                                                "ambiguous or a choice is theirs to make, instead of guessing.",
                                 .parameter_schema = kAskUserSchema,
                                 .callback = ask_user_cb});
        cfg.tools.push_back(Tool{.name = "todo",
                                 .description =
                                     "Keep a checklist for a request with several steps. Send the whole list every time with each item "
                                     "pending, in_progress or completed; one item is in progress while working on it.",
                                 .parameter_schema = kTodoSchema,
                                 .callback = todo_cb});
        cfg.tools.push_back(Tool{.name = "recall",
                                 .description = "Read back a note kept with remember, or search the notes and the saved earlier chats of "
                                                "this project for words (what did we talk about before). It reports how many chats it "
                                                "searched.",
                                 .parameter_schema = kRecallSchema,
                                 .callback = recall_cb,
                                 .user_data = &st});
        cfg.tools.push_back(Tool{.name = "forget",
                                 .description = "Delete a note kept with remember, by its exact name or by words from it. It says what was "
                                                "removed; if it removed nothing, say so.",
                                 .parameter_schema = kForgetSchema,
                                 .callback = forget_cb,
                                 .user_data = &st});
        cfg.tools.push_back(
            Tool{.name = "workflow_save",
                 .description =
                     "Keep a described process as a named workflow that can be run later in any chat with agent_run [\"run\",\"NAME\"]. "
                     "graph is a JSON object: {\"description\":\"...\",\"entry\":\"first\",\"nodes\":[...],\"edges\":[...]}. Node kinds: "
                     "{\"name\",\"kind\":\"agent\",\"prompt\",\"input_key\",\"output_key\",\"allowed_tools\":[...],\"interrupt_before_"
                     "tools\":bool}, "
                     "{\"kind\":\"tool\",\"tool\",\"args\"}, {\"kind\":\"parallel\",\"branches\":[entry node names]}, "
                     "{\"kind\":\"loop\",\"body\":entry node name,\"max_iterations\":N,\"until\":{\"key\",\"equals\"}}, "
                     "{\"kind\":\"sequential\",\"steps\":[entry node names]}, {\"kind\":\"wait\",\"seconds\":N}. Edges: "
                     "{\"from\",\"to\"} (to may be \"__end__\") or {\"from\",\"route\":{\"key\",\"cases\":{value:node},\"default\"}}; a "
                     "node has at most one outgoing edge. The graph is checked; the result lists what is wrong.",
                 .parameter_schema = kWorkflowSaveSchema,
                 .callback = workflow_save_cb,
                 .user_data = &st});
    }
    cfg.secret_resolver = cli_resolve_secrets;
    cfg.secret_user_data = st.secrets.get();
    cfg.context_window = st.context_window;
    cfg.max_tokens = st.max_tokens;
    // A chat has one conversation: one sequence slot. The default of four (for parallel sub-agents) made the memory estimate count the
    // cache four times, and a 9B model was refused on a machine that runs it.
    cfg.max_parallel_subagents = 1;
    cfg.tool_result_limit = 6000; // long tool output is shortened for the model; the whole text stays in .agents/results/
    cfg.temperature = st.ctx->config.temperature;
    cfg.enable_file_logging = true;
    cfg.logger = log_fn;
    cfg.logger_user_data = &st;
    cfg.permission_check = check_tool;
    cfg.permission_prompt = prompt_tool;
    cfg.permission_user_data = &st;

    ModelSpec spec{.name = st.model.name, .config = std::move(cfg), .on_load = on_model_loaded, .on_load_user_data = &st};
    if (auto r = register_model(st.registry, std::move(spec)); !r.ok) return r;
    return acquire_current(st);
}

void collect_artifacts(ChatState &st, size_t from) {
    for (size_t i = from; i < st.memory.history.size(); ++i) {
        const Message &m = st.memory.history[i];
        if (m.role == "user") continue; // media the user attached is not an artifact
        for (const auto &media : m.media) {
            st.artifacts.push_back(media);
            ensure_newline();
            out("[artifact " + std::to_string(st.artifacts.size()) + "] " + media.path + " (" + media.mime + ")  /open " +
                std::to_string(st.artifacts.size()) + "\n");
        }
    }
}

void remember_conversation(ChatState &st) {
    auto index = retrieve_memory(*st.lease, kIndexKey);
    const std::string current = index.ok ? index.value : std::string();
    const std::string updated = add_conversation_id(current, st.memory.id);
    if (updated != current) store_memory(*st.lease, kIndexKey, updated);
    st.saved = true;
}

std::string new_conversation_id() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "chat-%Y%m%d-%H%M%S", &tm);
    return buf;
}

void print_stats(const ChatState &st) {
    const SessionStats s = get_stats(*st.lease);
    const TurnStats &c = st.memory.stats;
    outf("conversation %s: %zu messages, %d inference rounds, %d tool calls, %d prompt + %d completion tokens\n", st.memory.id.c_str(),
         st.memory.history.size(), c.turns, c.tool_calls, c.prompt_tokens, c.completion_tokens);
    outf("this session:  %d inference rounds, %d tool calls, %d prompt + %d completion tokens, %d compressions, %d prunes\n", s.total_turns,
         s.total_tool_calls, s.total_prompt_tokens, s.total_completion_tokens, s.total_compressions, s.total_prune_cycles);
    if (c.cached_tokens > 0 && c.prompt_tokens > 0)
        outf("prefix cache:  %d of %d prompt tokens served from the KV cache (%.0f%%)\n", c.cached_tokens, c.prompt_tokens,
             100.0 * c.cached_tokens / c.prompt_tokens);
    if (const std::string footer = format_turn_footer(c.last); !footer.empty()) outf("last call:     %s\n", footer.c_str());
    outf("model %s (%s), engine %s %s\n", st.model.name.c_str(), st.model.path.c_str(), st.engine.name.c_str(), st.engine.version.c_str());
    const ModelBudget budget = get_budget(st.registry);
    const auto mb = [](uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    outf("loaded models (budget %s, CLI process RSS %.0f MiB; out-of-process engines are not included):\n",
         budget.ram_bytes ? (std::to_string(budget.ram_bytes >> 20) + " MiB").c_str() : "unlimited", mb(process_rss_bytes()));
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (const LoadedModelInfo &m : loaded_models(st.registry))
        outf("  %-24s ~%.0f MiB%s  %s, last used %llds ago, loaded in %.1fs\n", m.name.c_str(), mb(m.ram_bytes + m.vram_bytes),
             m.vram_bytes ? " (VRAM)" : "", m.leases ? "in use" : "idle", static_cast<long long>((now - m.last_used_ms) / 1000),
             static_cast<double>(m.load_time_ms) / 1000.0);
}

void print_chat_help() {
    outf("/exit                 quit (also Ctrl-D, or Ctrl-C at the prompt)\n"
         "/clear                forget the conversation so far\n"
         "/model [NAME|PATH]    show or switch the model (the conversation is kept)\n"
         "/image <path> [text]  send an image (or audio) with an optional prompt\n"
         "/open [n]             list generated artifacts, or open the n-th with the default app\n"
         "/stats                token and tool-call counters\n"
         "/diff                 what the tools changed in files this session\n"
         "/undo                 put back the files of the last tool call\n"
         "/compact              summarise the oldest messages to free context\n"
         "/mode [ask|plan|auto] ask before changes (default), only read, or never ask\n"
         "/thinking [on|off]    show or hide the <think> blocks of reasoning models\n"
         "/permissions [clear]  tools allowed for good in this project\n"
         "/rename TITLE         name this conversation in the sidebar\n"
         "/delete               remove this conversation (asks to confirm)\n"
         "@path                 in a message: add the file's content\n"
         "!command              run a shell command in the project folder; the model sees the output\n"
         "/doctor               what is installed and how much disk it uses\n"
         "/setup                ask the setup questions here, then run agent setup with the answers\n"
         "//text                send a message that starts with '/'\n"
         "Ctrl-C while the model is answering cancels the answer.\n");
}

// The line under an answer in the full-screen interface: how fast it wrote and how long the whole answer took (all its rounds, with
// the tools in between). `elapsed` is the wall time of the turn.
std::string format_metrics(const ChatState &st, double elapsed = 0.0) {
    const TokenUsage &u = st.memory.stats.last;
    char buf[96];
    std::string out;
    if (u.decode_ms > 0.0f && u.completion_tokens > 1) {
        std::snprintf(buf, sizeof(buf), "%.1f tok/s",
                      static_cast<double>(u.completion_tokens - 1) / (static_cast<double>(u.decode_ms) / 1000.0));
        out = buf;
    }
    double seconds = elapsed;
    if (seconds <= 0.0) seconds = (static_cast<double>(u.prefill_ms) + static_cast<double>(u.decode_ms)) / 1000.0;
    if (seconds > 0.0) {
        std::snprintf(buf, sizeof(buf), "%.1f s", seconds);
        out += (out.empty() ? "" : "  \xC2\xB7  ") + std::string(buf);
    }
    if (const std::string spec = speculation_note(u); !spec.empty()) out += (out.empty() ? "" : "  ·  ") + spec;
    return out;
}

void run_one_turn(ChatState &st, const std::string &text, std::span<const Media> media) {
    if (!st.lease) {
        out("[error] no model is loaded; switch with /model NAME\n");
        return;
    }
    const size_t before = st.memory.history.size();
    const auto turn_started = std::chrono::steady_clock::now();
    st.streamed_bytes = 0;
    clear_cancel(*st.lease);
    sync_extensions(st); // servers and skills added since the last message
    g_generating.store(true);
    auto r = media.empty() ? run_turn(*st.lease, st.memory, text, true, on_token, &st)
                           : run_turn_with_media(*st.lease, st.memory, text, media, true, on_token, &st);
    g_generating.store(false);
    g_last_turn_failed = !r.ok;
    // A failed turn with attachments leaves its message in the history; every later turn would send the same media and fail again.
    if (!r.ok && !media.empty() && st.memory.history.size() > before)
        st.memory.history.erase(st.memory.history.begin() + static_cast<std::ptrdiff_t>(before), st.memory.history.end());
    if (!r.ok) {
        ensure_newline();
        if (r.error.code == ErrorCode::Cancelled) out("[cancelled]\n");
        else out("[error] " + r.error.message + "\n");
        clear_cancel(*st.lease);
    } else if (st.streamed_bytes == 0) {
        emit_assistant(r.value); // engine did not stream
    }
    emit_assistant(filter_stream(st.filter, flush_think(st.think)));
    emit_assistant(flush_stream(st.filter));
    ensure_newline();
    collect_artifacts(st, before);
    if (r.ok) {
        const std::string footer = format_turn_footer(st.memory.stats.last);
        if (!footer.empty()) std::fprintf(g_err, "[%s]\n", footer.c_str());
#if defined(AGENT_CLI_HAS_TUI)
        if (g_tui) {
            const std::string metrics =
                format_metrics(st, std::chrono::duration<double>(std::chrono::steady_clock::now() - turn_started).count());
            if (!metrics.empty()) tui_emit(TuiKind::Metrics, metrics);
        }
#endif
    }
    remember_conversation(st);
    // Settings the agent was asked to change are applied now, between answers (they reload the model).
    if (const std::string changed = apply_pending_settings(st.settings); !changed.empty()) out(changed);
}

// ---- "!command": the user's own shell command in the project folder; the output goes into the conversation

std::string shell_capture(const std::string &command, const std::string &workspace) {
#if defined(_WIN32)
    const std::string full = "cd /d \"" + workspace + "\" && " + command + " 2>&1";
    std::FILE *pipe = _popen(full.c_str(), "r");
#else
    const std::string full = "cd '" + workspace + "' && " + command + " 2>&1";
    std::FILE *pipe = popen(full.c_str(), "r");
#endif
    if (!pipe) return "cannot start the command\n";
    std::string output;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        output += buf;
        if (output.size() > 16 * 1024) {
            output.resize(16 * 1024);
            output += "\n[cut]\n";
            break;
        }
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return output;
}

void run_shell_line(ChatState &st, std::string command) {
    const size_t a = command.find_first_not_of(" \t");
    command = a == std::string::npos ? std::string() : command.substr(a);
    if (command.empty()) {
        out("usage: !COMMAND  runs it in the project folder; the model sees the output\n");
        return;
    }
    out("$ " + command + "\n");
    const std::string output = shell_capture(command, st.workspace);
    out(output.empty() ? std::string("(no output)\n") : output + (output.back() == '\n' ? "" : "\n"));
    if (!st.lease) return;
    st.memory.history.push_back(Message{
        .role = "user", .content = "I ran this command in the project folder: " + command + "\nIts output:\n```\n" + output + "```"});
    if (auto r = save_conversation(*st.lease, st.memory); !r.ok) out("[error] " + r.error.message + "\n");
    remember_conversation(st);
}

// ---- deleting a saved conversation

bool delete_conversation(ChatState &st, const std::string &id, const std::string &workspace) {
    const auto update = [&](Session &session) {
        auto index = retrieve_memory(session, kIndexKey);
        if (!index.ok) return false;
        if (!store_memory(session, kIndexKey, remove_conversation_id(index.value, id)).ok) return false;
        store_memory(session, std::string(kTitlePrefix) + id, "");
        return true;
    };
    if (workspace == st.workspace && st.lease) return update(*st.lease);
    Config cfg;
    cfg.provider = AiProvider::Mock;
    cfg.workspace_dir = workspace;
    cfg.memory = {.store = store::sqlite_store()};
    cfg.auto_default_tools = false;
    auto other = init(cfg);
    return other.ok && update(other.value);
}

int list_conversations(const std::string &workspace) {
    Config cfg;
    cfg.provider = AiProvider::Mock;
    cfg.workspace_dir = workspace;
    cfg.memory = {.store = store::sqlite_store()};
    auto s = init(cfg);
    if (!s.ok) return report_error(s.error);
    auto index = retrieve_memory(s.value, kIndexKey);
    const auto ids = conversation_ids(index.ok ? index.value : std::string());
    if (ids.empty()) {
        std::printf("no saved conversations in %s\n", workspace.c_str());
        return 0;
    }
    for (const auto &id : ids) std::printf("%s\n", id.c_str());
    std::printf("resume one with: agent chat --resume <id>\n");
    return 0;
}

Result<int> parse_positive(std::string_view flag, std::string_view text) {
    int v = 0;
    for (char c : text) {
        if (c < '0' || c > '9' || v > 10'000'000)
            return fail<int>(ErrorCode::InvalidConfig, "--" + std::string(flag) + " expects a positive number");
        v = v * 10 + (c - '0');
    }
    if (v <= 0) return fail<int>(ErrorCode::InvalidConfig, "--" + std::string(flag) + " expects a positive number");
    return ok(v);
}

// /model NAME: resolve, load the engine for it, re-create the session. On failure the old model stays.
void switch_model(ChatState &st, const std::string &spec) {
    auto m = resolve_model(spec, {}, catalog_refs(*st.ctx), path_exists);
    if (!m.ok) {
        out("[error] " + m.error.message + "\n");
        return;
    }
    auto e = load_engine_for(*st.ctx, st.engine_flag, m.value.path);
    if (!e.ok) {
        out("[error] " + e.error.message + "\n");
        return;
    }
    const ModelRef old_model = st.model;
    const LoadedEngine old_engine = st.engine;
    const std::string old_options = st.engine_options;
    st.model = m.value;
    st.engine = e.value;
    if (!st.engine_options.empty()) {
        out("note: --engine-options belong to the first model and are not used for " + st.model.name + "\n");
        st.engine_options.clear();
    }
    st.engine_options = catalog_engine_options(*st.ctx, st.model.path, st.engine_options); // the new model brings its own projector
    bool cached = false;
    for (const LoadedModelInfo &info : loaded_models(st.registry)) cached = cached || info.name == st.model.name;
    out((cached ? "switching to " : "loading ") + st.model.path + (cached ? " (already loaded)\n" : " ...\n"));
    if (auto r = open_session(st); !r.ok) {
        out("[error] " + r.error.message + "; going back to " + old_model.name + "\n");
        st.model = old_model;
        st.engine = old_engine;
        st.engine_options = old_options;
        if (auto back = open_session(st); !back.ok) out("[error] " + back.error.message + "; no model is loaded, try /model NAME\n");
        return;
    }
    // The new model may not take media (and never gets the old --engine-options such as mmproj), so
    // earlier attachments stay in the conversation as text. /open keeps working from st.artifacts.
    size_t converted = 0;
    for (Message &msg : st.memory.history) {
        for (const Media &media : msg.media) msg.content += "\n[attached " + media.mime + ": " + media.path + "]";
        converted += msg.media.size();
        msg.media.clear();
    }
    if (converted) {
        out("note: " + std::to_string(converted) + " earlier attachment(s) are now referenced by path only\n");
        if (auto r = save_conversation(*st.lease, st.memory); !r.ok) out("[error] " + r.error.message + "\n");
    }
    out("now using " + st.model.name + " on " + st.engine.name + "\n");
}

// ---- sessions: the sidebar lists the saved conversations of every project the CLI has worked in.

fs::path workspaces_file(const Context &ctx) { return fs::path(ctx.has_home ? ctx.home.root : ctx.config_dir) / "workspaces.txt"; }

std::vector<std::string> known_workspaces(const Context &ctx) {
    std::vector<std::string> list;
    std::ifstream in(workspaces_file(ctx));
    for (std::string line; std::getline(in, line);) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) list.push_back(line);
    }
    return list;
}

// Puts the current project first in the list kept in the home folder.
void remember_workspace(const ChatState &st) {
    std::vector<std::string> list = known_workspaces(*st.ctx);
    list.erase(std::remove(list.begin(), list.end(), st.workspace), list.end());
    list.insert(list.begin(), st.workspace);
    if (list.size() > 12) list.resize(12);
    std::error_code ec;
    fs::create_directories(workspaces_file(*st.ctx).parent_path(), ec);
    std::ofstream out_file(workspaces_file(*st.ctx), std::ios::trunc);
    for (const std::string &w : list) out_file << w << '\n';
}

// What a conversation is called in the sidebar: the first thing the user said, on one line.
std::string conversation_title(const Conversation &c) {
    for (const Message &m : c.history) {
        if (m.role != "user" || m.content.empty()) continue;
        std::string title = m.content.substr(0, 120);
        for (char &ch : title)
            if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        const size_t a = title.find_first_not_of(' ');
        return a == std::string::npos ? std::string() : title.substr(a);
    }
    return {};
}

void list_workspace_sessions(Session &session, const std::string &workspace, const std::string &open_id, const std::string &open_workspace,
                             std::vector<TuiSession> &sessions) {
    auto index = retrieve_memory(session, kIndexKey);
    const auto ids = conversation_ids(index.ok ? index.value : std::string());
    int shown = 0;
    for (auto it = ids.rbegin(); it != ids.rend() && shown < 12; ++it, ++shown) {
        Conversation c;
        c.id = *it;
        if (!load_conversation(session, c).ok) continue;
        TuiSession s;
        s.id = *it;
        s.title = conversation_title(c);
        if (auto named = retrieve_memory(session, std::string(kTitlePrefix) + *it); named.ok && !named.value.empty()) s.title = named.value;
        s.workspace = workspace;
        s.messages = static_cast<int>(c.history.size());
        s.active = *it == open_id && workspace == open_workspace;
        sessions.push_back(std::move(s));
    }
}

std::vector<TuiSession> tui_sessions(ChatState &st) {
    std::vector<TuiSession> sessions;
    // The open project first: its conversations are read through the open store.
    if (st.lease) list_workspace_sessions(*st.lease, st.workspace, st.memory.id, st.workspace, sessions);
    const bool saved = std::any_of(sessions.begin(), sessions.end(), [](const TuiSession &s) { return s.active; });
    if (!saved) sessions.insert(sessions.begin(), TuiSession{st.memory.id, conversation_title(st.memory), st.workspace, 0, true});
    for (const std::string &w : known_workspaces(*st.ctx)) {
        if (w == st.workspace) continue;
        std::error_code ec;
        if (!fs::is_directory(w, ec)) continue;
        Config cfg;
        cfg.provider = AiProvider::Mock;
        cfg.workspace_dir = w;
        cfg.memory = {.store = store::sqlite_store()};
        cfg.auto_default_tools = false;
        auto other = init(cfg);
        if (!other.ok) continue;
        list_workspace_sessions(other.value, w, st.memory.id, st.workspace, sessions);
    }
    return sessions;
}

std::vector<TuiMessage> tui_history(const ChatState &st) {
    std::vector<TuiMessage> messages;
    for (const Message &m : st.memory.history) {
        if (m.role == "user") {
            messages.push_back({TuiKind::User, m.content});
        } else if (m.role == "assistant") {
            for (const ToolCall &call : m.tool_calls)
                messages.push_back({TuiKind::Tool, "[tool] " + std::string(call.name) + " " + std::string(call.arguments)});
            if (!m.content.empty()) messages.push_back({TuiKind::Assistant, m.content});
        } else if (m.role == "tool") {
            messages.push_back({TuiKind::Tool, "[result] " + m.content});
        }
    }
    return messages;
}

// Opens another conversation (an empty id: a new one) in `workspace`. A different project means a different folder for the tools, the
// store and the logs, so the model is loaded again for it.
bool switch_session(ChatState &st, const std::string &id, const std::string &workspace) {
    if (!workspace.empty() && workspace != st.workspace) {
        std::error_code ec;
        if (!fs::is_directory(workspace, ec)) {
            out("[error] the project folder " + workspace + " no longer exists\n");
            return false;
        }
        release_model(st.lease);
        unload_model(st.registry, st.model.name);
        st.workspace = workspace;
        st.journal = std::make_unique<EditJournal>(st.workspace);
        st.always_allowed = load_always_allowed(st.workspace);
        Config store_cfg;
        store_cfg.provider = AiProvider::Mock;
        store_cfg.workspace_dir = st.workspace;
        store_cfg.memory = {.store = store::sqlite_store()};
        store_cfg.auto_default_tools = false;
        auto store_session = init(store_cfg);
        if (!store_session.ok) {
            out("[error] " + store_session.error.message + "\n");
            return false;
        }
        st.store = std::move(store_session.value);
        if (auto r = open_session(st); !r.ok) {
            out("[error] " + r.error.message + "\n");
            return false;
        }
        remember_workspace(st);
    }
    st.artifacts.clear();
    if (id.empty()) {
        st.memory = Conversation{};
        st.memory.id = new_conversation_id();
        st.memory.history.push_back(Message{.role = "system", .content = system_prompt(st)});
        st.saved = false;
        return true;
    }
    Conversation c;
    c.id = id;
    if (auto r = load_conversation(*st.lease, c); !r.ok) {
        out("[error] could not open the session: " + r.error.message + "\n");
        return false;
    }
    st.memory = std::move(c);
    st.saved = true;
    return true;
}

int chat_fail(const Error &e) {
    std::fprintf(g_err, "error: %s\n", e.message.c_str());
    return 1;
}

int run_chat(ChatState &st, const CommandLine &cl, const std::string &engine_log);

} // namespace

int cmd_chat(const CommandLine &cl, Context &ctx) {
    ChatState st;
    st.ctx = &ctx;
    st.workspace = workspace_dir(ctx, cl.get("workspace"));
    st.journal = std::make_unique<EditJournal>(st.workspace);
    st.always_allowed = load_always_allowed(st.workspace);
    if (cl.has("list")) return list_conversations(st.workspace);

    st.verbose = cl.has("verbose");
    st.auto_approve = cl.has("yes");
    st.inprocess = cl.has("inprocess");
    st.interactive = stdin_is_tty();
#if defined(AGENT_CLI_HAS_TUI)
    st.use_tui = st.interactive && stream_is_tty(stdout) && !cl.has("plain") && !st.verbose && !cl.has("prompt") && !cl.has("jsonl") &&
                 !cl.has("acp");
#endif
    {
        // Names and bindings of the secrets; a broken index only means that no server gets a key.
        const std::string home_root = ctx.has_home ? ctx.home.root : ctx.config_dir;
        auto secrets = load_secret_context(home_root, (fs::path(home_root) / "secrets.json").string());
        if (secrets.ok) st.secrets = std::move(secrets.value);
        else std::fprintf(g_err, "[secrets] warning: %s\n", secrets.error.message.c_str());
    }
    st.engine_flag = std::string(cl.get("engine"));
    st.engine_options = std::string(cl.get("engine-options"));
    // --json and --grammar FILE become engine options of the llama engine: every answer is sampled under that grammar.
    if (cl.has("json") || cl.has("grammar")) st.no_tools = true;
    if (cl.has("json")) st.engine_options = with_option(st.engine_options, "json_output", "true");
    if (cl.has("grammar")) {
        std::ifstream gin{std::string(cl.get("grammar")), std::ios::binary};
        if (!gin) return chat_fail(make_error(ErrorCode::InvalidConfig, "cannot read the grammar file " + std::string(cl.get("grammar"))));
        const std::string gbnf((std::istreambuf_iterator<char>(gin)), std::istreambuf_iterator<char>());
        st.engine_options = with_option(st.engine_options, "grammar", "\"" + json_escape(gbnf) + "\"");
    }
    st.context_window = ctx.config.context_window;
    st.max_tokens = ctx.config.max_tokens;
    if (cl.has("ctx")) {
        auto v = parse_positive("ctx", cl.get("ctx"));
        if (!v.ok) return report_error(v.error);
        st.context_window = v.value;
    }
    if (cl.has("max-tokens")) {
        auto v = parse_positive("max-tokens", cl.get("max-tokens"));
        if (!v.ok) return report_error(v.error);
        st.max_tokens = v.value;
    }

    auto model = resolve_model(cl.get("model"), ctx.config.default_model, catalog_refs(ctx), path_exists);
    if (!model.ok) return report_error(model.error);
    st.model = model.value;
    st.engine_options = catalog_engine_options(ctx, st.model.path, st.engine_options);
    auto engine = load_engine_for(ctx, st.engine_flag, st.model.path);
    if (!engine.ok) return report_error(engine.error);
    st.engine = engine.value;

    // Engines log straight to stderr (llama.cpp prints hundreds of lines per model load); send that
    // to <workspace>/logs/engine.log unless --verbose.
    std::string engine_log;
    if (!st.verbose) {
        std::error_code ec;
        fs::create_directories(fs::path(st.workspace) / "logs", ec);
        engine_log = (fs::path(st.workspace) / "logs" / "engine.log").string();
        if (std::FILE *original = redirect_stderr(engine_log)) g_err = original;
        else engine_log.clear();
    }
    const int rc = run_chat(st, cl, engine_log);
    // Unload the models while their teardown logs still go to the log file.
    release_model(st.lease);
    st.registry = ModelRegistry();
    st.store = Session{};
    if (g_err != stderr) restore_stderr(g_err);
    g_err = stderr;
    return rc;
}

namespace {

// A subcommand of this binary with its output shown in the chat: it runs as a child process, so what it prints cannot tear the
// full-screen interface, and it never reads the keyboard. Progress that rewrites a line with a carriage return shows only its last state.
void run_cli_command(ChatState &st, const std::vector<std::string> &argv) {
    if (argv.empty()) return;
    std::string exe = executable_dir();
#if defined(_WIN32)
    exe += "\\agent.exe";
    auto quote = [](const std::string &w) { return quote_windows(w); };
#else
    exe += "/agent";
    auto quote = [](const std::string &w) { return quote_posix(w); };
#endif
    std::string command = quote(exe);
    if (st.ctx->has_home) command += " --home " + quote(st.ctx->home.root);
    std::string shown;
    for (const std::string &a : argv) {
        command += " " + quote(a);
        shown += (shown.empty() ? "" : " ") + a;
    }
#if defined(_WIN32)
    command += " < nul 2>&1";
    std::FILE *pipe = _popen(("\"" + command + "\"").c_str(), "r"); // cmd.exe /c wants the whole line in one more pair of quotes
#else
    command += " < /dev/null 2>&1";
    std::FILE *pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) {
        out("[error] could not start: agent " + shown + "\n");
        return;
    }
    out("$ agent " + shown + "\n");
    std::string line;
    bool after_cr = false;
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
        for (size_t i = 0; i < n; ++i) {
            const char c = buf[i];
            if (c == '\n') {
                out(line + "\n");
                line.clear();
                after_cr = false;
            } else if (c == '\r') {
                after_cr = true;
            } else {
                if (after_cr) line.clear(); // a progress line was rewritten
                after_cr = false;
                line += c;
            }
        }
    }
    if (!line.empty()) out(line + "\n");
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
}

// /setup: setup is a script that asks questions and downloads, which does not fit inside the full-screen chat.
void run_setup_wizard(ChatState &) { out("setup runs in a terminal: leave the chat (/exit) and run `agent setup`\n"); }

// One input line: a message or a slash command. Returns false when the user asked to exit.
bool handle_line(ChatState &st, const std::string &line) {
    {
        // "!command" runs a shell command of the user's own.
        const size_t b = line.find_first_not_of(" \t");
        if (b != std::string::npos && line[b] == '!' && line.size() > b + 1) {
            run_shell_line(st, line.substr(b + 1));
            return true;
        }
    }
    {
        // "/task CAPABILITY --model NAME --text ..." and "/run FILE.json" go straight to `agent task` and `agent run`.
        const size_t b = line.find_first_not_of(" \t");
        if (b != std::string::npos && line[b] == '/') {
            std::vector<std::string> words;
            size_t i = b + 1;
            while (i < line.size()) {
                while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
                if (i >= line.size()) break;
                std::string w;
                if (line[i] == '"' || line[i] == '\'') {
                    const char q = line[i++];
                    while (i < line.size() && line[i] != q) w += line[i++];
                    if (i < line.size()) ++i;
                } else {
                    while (i < line.size() && line[i] != ' ' && line[i] != '\t') w += line[i++];
                }
                words.push_back(w);
            }
            if (!words.empty() && (words[0] == "task" || words[0] == "run")) {
                if (words.size() < 2) {
                    out(words[0] == "task"
                            ? "usage: /task CAPABILITY [--model NAME] [--text TEXT] [--file FILE] [--doc TEXT] [--params JSON]\n"
                            : "usage: /run FILE.json\n");
                    return true;
                }
                bool failed = false;
                out("running: agent " + words[0] + " ...\n");
                out(run_agent_command(st, words, failed));
                ensure_newline();
                return true;
            }
        }
    }
    const SlashCommand cmd = parse_slash_command(line);
    if (cmd.kind == SlashKind::None) {
        if (cmd.text.empty()) return true;
        // "@path" in the message adds the file's content.
        std::vector<std::string> attached;
        const std::string text = expand_mentions(cmd.text, st.workspace, &attached);
        if (!attached.empty()) {
            std::string names;
            for (const std::string &n : attached) names += (names.empty() ? "" : ", ") + n;
            out("attached: " + names + "\n");
        }
        run_one_turn(st, memory_pack(st, text) + text, {});
        return true;
    }
    if (cmd.kind == SlashKind::Exit) return false;
    if (!st.lease && (cmd.kind == SlashKind::Clear || cmd.kind == SlashKind::Stats || cmd.kind == SlashKind::Compact ||
                      cmd.kind == SlashKind::Rename || cmd.kind == SlashKind::Delete)) {
        out("[error] no model is loaded; switch with /model NAME\n");
        return true;
    }
    switch (cmd.kind) {
    case SlashKind::Clear: {
        const bool has_system = !st.memory.history.empty() && st.memory.history.front().role == "system";
        Message system = has_system ? st.memory.history.front() : Message{};
        st.memory.history.clear();
        if (has_system) st.memory.history.push_back(std::move(system));
        st.memory.stats = {};
        st.artifacts.clear();
        if (auto r = save_conversation(*st.lease, st.memory); !r.ok) out("[error] " + r.error.message + "\n");
        clear_provider_kv_cache(*st.lease);
        out("conversation cleared\n");
        break;
    }
    case SlashKind::Model:
        if (cmd.arg.empty()) {
            out("current model: " + st.model.name + " (" + st.model.path + ") on " + st.engine.name + "\n");
            for (const auto &m : catalog_refs(*st.ctx)) out("  " + m.name + "  " + m.path + "\n");
        } else {
            switch_model(st, cmd.arg);
        }
        break;
    case SlashKind::Image: {
        if (cmd.arg.empty()) {
            out("usage: /image <path> [prompt]\n");
            break;
        }
        if (!file_exists(cmd.arg)) {
            out("[error] no such file: " + cmd.arg + "\n");
            break;
        }
        std::error_code ec;
        const Media media[] = {{fs::absolute(cmd.arg, ec).string(), mime_for_path(cmd.arg)}};
        const std::string prompt = cmd.rest.empty() ? "Describe this." : cmd.rest;
        run_one_turn(st, prompt, media);
        break;
    }
    case SlashKind::Open: {
        if (cmd.arg.empty()) {
            if (st.artifacts.empty()) out("no artifacts generated yet\n");
            for (size_t i = 0; i < st.artifacts.size(); ++i)
                out("  " + std::to_string(i + 1) + ". " + st.artifacts[i].path + " (" + st.artifacts[i].mime + ")\n");
            break;
        }
        size_t n = 0;
        for (char c : cmd.arg) n = (c >= '0' && c <= '9' && n < 1'000'000) ? n * 10 + static_cast<size_t>(c - '0') : SIZE_MAX;
        if (n == 0 || n == SIZE_MAX || n > st.artifacts.size()) {
            out("[error] no artifact " + cmd.arg + " (" + std::to_string(st.artifacts.size()) + " generated so far)\n");
            break;
        }
        const std::string &path = st.artifacts[n - 1].path;
        // No desktop opener (headless box, container): the path is still useful to copy.
        if (auto r = open_with_os(path); !r.ok) out("[note] " + r.error.message + "; the file is at " + path + "\n");
        else out("opened " + path + "\n");
        break;
    }
    case SlashKind::Stats: print_stats(st); break;
    case SlashKind::Compact: {
        const size_t before = st.memory.history.size();
        if (auto r = compress_context(*st.lease, st.memory, 6); !r.ok) {
            out("[error] " + r.error.message + "\n");
            break;
        }
        if (auto r = save_conversation(*st.lease, st.memory); !r.ok) out("[error] " + r.error.message + "\n");
        out("compacted: " + std::to_string(before) + " -> " + std::to_string(st.memory.history.size()) +
            " messages (the oldest ones are summarised)\n");
        break;
    }
    case SlashKind::Mode: {
        if (cmd.arg.empty()) {
            out(std::string("mode: ") + chat_mode_name(st.mode) +
                "\n  ask   tools that change things ask first (default)\n  plan  only reading: no file changes, no commands\n  auto  "
                "nothing asks\n");
            break;
        }
        ChatMode mode = ChatMode::Ask;
        if (!parse_chat_mode(cmd.arg, mode)) {
            out("[error] /mode ask, plan or auto\n");
            break;
        }
        st.mode = mode;
        out(std::string("mode: ") + chat_mode_name(mode) + "\n");
        break;
    }
    case SlashKind::Thinking: {
        if (cmd.arg == "on") st.think.show = true;
        else if (cmd.arg == "off") st.think.show = false;
        else st.think.show = !st.think.show;
        out(std::string("thinking blocks are ") + (st.think.show ? "shown" : "hidden") + "\n");
        break;
    }
    case SlashKind::Permissions: {
        if (cmd.arg == "clear") {
            st.always_allowed.clear();
            save_always_allowed(st.workspace, st.always_allowed);
            out("no tool is allowed for good in this project any more\n");
            break;
        }
        if (st.always_allowed.empty())
            out("no tool is allowed for good in this project (answer A, the capital letter, to a question to allow one)\n");
        for (const std::string &t : st.always_allowed) out("always allowed: " + t + "\n");
        break;
    }
    case SlashKind::Rename: {
        const std::string title = cmd.rest.empty() ? cmd.arg : cmd.arg + " " + cmd.rest;
        if (title.empty()) {
            out("usage: /rename NEW TITLE\n");
            break;
        }
        if (auto r = store_memory(*st.lease, std::string(kTitlePrefix) + st.memory.id, title); !r.ok)
            out("[error] " + r.error.message + "\n");
        else out("renamed to: " + title + "\n");
        remember_conversation(st);
        break;
    }
    case SlashKind::Delete: {
        if (cmd.arg != "yes") {
            out("this removes the saved conversation " + st.memory.id + " for good; type /delete yes to do it\n");
            break;
        }
        const std::string id = st.memory.id;
        if (!delete_conversation(st, id, st.workspace)) {
            out("[error] could not delete " + id + "\n");
            break;
        }
        switch_session(st, "", st.workspace);
        st.repaint = true;
        out("deleted " + id + "; this is a new conversation\n");
        break;
    }
    case SlashKind::Diff: out((st.journal ? st.journal->diff() : std::string("no changes to files in this session")) + "\n"); break;
    case SlashKind::Undo: {
        const std::string done = st.journal ? st.journal->undo() : std::string();
        out(done.empty() ? "nothing to undo\n" : done);
        break;
    }
    case SlashKind::Help: print_chat_help(); break;
    case SlashKind::Cli:
        if (cmd.arg == "setup" && cmd.rest.empty()) run_setup_wizard(st);
        else run_cli_command(st, slash_cli_argv(cmd));
        break;
    case SlashKind::Unknown: out("unknown command /" + cmd.arg + " (try /help)\n"); break;
    default: break;
    }
    return true;
}

// --prompt TEXT and --jsonl: chat without a terminal. -p answers once (stdout: the answer; stderr: tool lines and notes) and exits with
// 1 when the turn failed. --jsonl reads one JSON object per line from stdin ({"type":"prompt","text":"..."}, {"type":"exit"}) and writes
// one per line to stdout: ready, token, tool, note, error, permission_request, done.
int run_scripted(ChatState &st, const CommandLine &cl) {
    if (cl.has("jsonl")) {
        g_jsonl = true;
        std::printf("{\"type\":\"ready\",\"session\":\"%s\",\"model\":\"%s\",\"engine\":\"%s\",\"workspace\":\"%s\"}\n",
                    json_escape(st.memory.id).c_str(), json_escape(st.model.name).c_str(), json_escape(st.engine.name).c_str(),
                    json_escape(st.workspace).c_str());
        std::fflush(stdout);
        std::string line;
        while (read_line(stdin, line) == ReadStatus::Line) {
            simdjson::dom::parser parser;
            auto doc = parser.parse(line);
            if (doc.error()) {
                json_event("error", "not a JSON line");
                continue;
            }
            std::string_view type, text;
            if (doc["type"].get(type) != simdjson::SUCCESS) continue;
            if (type == "exit") break;
            if (type != "prompt" || doc["text"].get(text) != simdjson::SUCCESS) continue;
            const bool again = handle_line(st, std::string(text));
            std::printf("{\"type\":\"done\",\"session\":\"%s\",\"metrics\":\"%s\"}\n", json_escape(st.memory.id).c_str(),
                        json_escape(format_metrics(st)).c_str());
            std::fflush(stdout);
            if (!again) break;
        }
        return 0;
    }
    std::string prompt(cl.get("prompt"));
    if (prompt == "-") {
        prompt.clear();
        std::string line;
        while (read_line(stdin, line) == ReadStatus::Line) prompt += line + "\n";
    }
    if (prompt.empty()) return chat_fail(make_error(ErrorCode::InvalidConfig, "--prompt needs a text (or - to read it from stdin)"));
    g_headless = true;
    run_one_turn(st, prompt, {});
    std::printf("\n");
    return g_last_turn_failed ? 1 : 0;
}

// ---- the settings screen (bottom-left button of the full-screen chat)

#if defined(AGENT_CLI_HAS_TUI)
// ---- numbers for the bottom bar, kept up to date by a thread of their own so the screen never waits for them

struct SysStats {
    std::atomic<int> cpu{-1};
    std::atomic<double> ram_used{0}, ram_total{0}, files{-1};
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::string folder;
    std::thread thread;
};
SysStats g_stats;

uint64_t folder_bytes_capped(const std::string &dir, size_t max_files) {
    uint64_t total = 0;
    size_t seen = 0;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return 0;
    for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
         it.increment(ec)) {
        std::error_code ec2;
        if (it->is_regular_file(ec2)) total += static_cast<uint64_t>(it->file_size(ec2));
        if (++seen >= max_files) break;
    }
    return total;
}

void start_stats(const std::string &folder) {
    if (g_stats.thread.joinable()) return;
    g_stats.folder = folder;
    g_stats.thread = std::thread([] {
        CpuSample last;
        cpu_sample(last);
        int tick = 0;
        while (!g_stats.stop.load()) {
            for (int i = 0; i < 10 && !g_stats.stop.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            CpuSample now;
            if (cpu_sample(now) && now.total > last.total) {
                g_stats.cpu.store(static_cast<int>(100 * (now.busy - last.busy) / (now.total - last.total)));
                last = now;
            }
            double used = 0, total = 0;
            if (memory_gb(used, total)) {
                g_stats.ram_used.store(used);
                g_stats.ram_total.store(total);
            }
            if (tick++ % 30 == 0)
                g_stats.files.store(static_cast<double>(folder_bytes_capped(g_stats.folder, 200000)) / (1024.0 * 1024.0 * 1024.0));
        }
    });
}

void stop_stats() {
    g_stats.stop.store(true);
    if (g_stats.thread.joinable()) g_stats.thread.join();
}

TuiTaskbar tui_taskbar(ChatState &st) {
    TuiTaskbar tb;
    tb.cpu_percent = g_stats.cpu.load();
    tb.ram_used_gb = g_stats.ram_used.load();
    tb.ram_total_gb = g_stats.ram_total.load();
    tb.files_gb = g_stats.files.load();
    tb.engine = st.engine.name;
    for (const LoadedModelInfo &m : loaded_models(st.registry)) tb.models.push_back(m.name);
    if (tb.models.empty() && !st.model.name.empty()) tb.models.push_back(st.model.name);
    tb.mode = chat_mode_name(st.mode);
    tb.context_total = st.context_window;
    const TokenUsage &u = st.memory.stats.last;
    tb.context_used = u.kv_tokens > 0 ? u.kv_tokens : u.prompt_tokens + u.completion_tokens;
    return tb;
}

// Removes a message of the user and the answers to it from what the model sees and from the saved conversation.
bool tui_delete_turn(ChatState &st, int user_from_end) {
    auto &h = st.memory.history;
    int seen = 0;
    for (size_t i = h.size(); i-- > 0;) {
        if (h[i].role != "user") continue;
        if (seen++ != user_from_end) continue;
        size_t end = i + 1;
        while (end < h.size() && h[end].role != "user") ++end;
        h.erase(h.begin() + static_cast<std::ptrdiff_t>(i), h.begin() + static_cast<std::ptrdiff_t>(end));
        remember_conversation(st);
        return true;
    }
    return false;
}

// ---- the microphone button: records, then turns the speech into text with the first speech recognition model that is installed

struct MicState {
    std::mutex mu;
    std::vector<float> samples;
    std::atomic<bool> stop{false};
    std::thread thread;
    Session asr;
    bool asr_ready = false;
    std::string asr_model;
};
MicState g_mic;

std::string tui_mic(ChatState &st, bool start) {
    if (start) {
        if (!audio::device_backend_available()) return "This build has no microphone support";
        auto in = audio::open_device_input();
        if (!in.ok) return in.error.message;
        g_mic.samples.clear();
        g_mic.stop.store(false);
        g_mic.thread = std::thread([input = std::move(in.value)]() mutable {
            float buf[4096];
            bool eof = false;
            while (!g_mic.stop.load() && !eof) {
                const size_t n = audio::read(input, buf, 4096, 100, eof);
                std::lock_guard<std::mutex> lock(g_mic.mu);
                g_mic.samples.insert(g_mic.samples.end(), buf, buf + n);
            }
        });
        return {};
    }
    g_mic.stop.store(true);
    if (g_mic.thread.joinable()) g_mic.thread.join();
    std::vector<float> samples;
    {
        std::lock_guard<std::mutex> lock(g_mic.mu);
        samples = std::move(g_mic.samples);
    }
    if (samples.size() < audio::kCaptureRate / 2) return "!Nothing was heard";

    // The first installed model that recognises speech.
    std::string model_path, model_name;
    if (auto catalog = agent::hub::list_models(catalog_path(*st.ctx)); catalog.ok)
        for (const auto &m : catalog.value) {
            const ModelKind kind = detect_model(m.path, m.repo, m.file, false);
            if (kind.engine == "audiocpp" && kind.task == "asr") {
                model_path = m.path;
                model_name = m.name;
                break;
            }
        }
    if (model_path.empty()) return "!No speech recognition model is installed: agent model pull <a speech model link>";
    if (!g_mic.asr_ready || g_mic.asr_model != model_name) {
        auto engine = load_engine_for(*st.ctx, "audiocpp", model_path);
        if (!engine.ok) return "!" + engine.error.message;
        auto options = audiocpp_asr_options(model_path);
        if (!options.ok) return "!" + options.error.message;
        Config cfg;
        cfg.provider = AiProvider::Engine;
        cfg.engine = engine.value.name;
        cfg.engine_isolation = cli_engine_isolation(st.inprocess, engine.value.isolation);
        cfg.engine_sandbox.gpu = cli_engine_gpu(engine.value.backend);
        apply_sandbox_choice(cfg);
        cfg.workspace_dir = st.workspace;
        cfg.engine_options = options.value;
        cfg.auto_default_tools = false;
        auto session = init(cfg);
        if (!session.ok) return "!" + session.error.message;
        g_mic.asr = std::move(session.value);
        g_mic.asr_ready = true;
        g_mic.asr_model = model_name;
    }
    g_mic.asr.arena.reset();
    const int64_t shape[] = {static_cast<int64_t>(samples.size())};
    const TensorView tensor{
        .name = "audio",
        .dtype = DType::Float32,
        .shape = shape,
        .data = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(samples.data()), samples.size() * sizeof(float))};
    const MessageView msg{.role = "user", .content = {}, .tensors = std::span<const TensorView>(&tensor, 1)};
    InferRequest req;
    req.messages = std::span<const MessageView>(&msg, 1);
    req.cancel = g_mic.asr.cancel_flag.get();
    auto r = infer(g_mic.asr, req);
    if (!r.ok) return "!" + r.error.message;
    std::string words(r.value.message.content);
    while (!words.empty() && (words.back() == '\n' || words.back() == ' ')) words.pop_back();
    return words.empty() ? "!Nothing was recognised" : words;
}

bool reload_model(ChatState &st) {
    release_model(st.lease);
    unload_model(st.registry, st.model.name);
    return open_session(st).ok;
}

std::string icons_name() {
    const char *v = std::getenv("AGENT_ICONS");
    return v && *v ? v : "unicode";
}

uint64_t folder_bytes(const std::string &dir) {
    uint64_t total = 0;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return 0;
    for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
         it.increment(ec)) {
        std::error_code ec2;
        if (it->is_regular_file(ec2)) total += static_cast<uint64_t>(it->file_size(ec2));
    }
    return total;
}

std::string size_text(uint64_t bytes) {
    char buf[32];
    if (bytes >= (1ull << 30)) std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / static_cast<double>(1ull << 30));
    else std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(bytes) / static_cast<double>(1ull << 20));
    return buf;
}

size_t count_folders(const std::string &dir) {
    size_t n = 0;
    std::error_code ec;
    if (fs::is_directory(dir, ec))
        for (const auto &e : fs::directory_iterator(dir, ec))
            if (e.is_directory(ec)) ++n;
    return n;
}

std::string join_names(const std::vector<std::string> &names, const char *none) {
    std::string out;
    for (const std::string &n : names) out += (out.empty() ? "" : ", ") + n;
    return out.empty() ? std::string(none) : out;
}

// The results of the last search in Settings > Skills / MCP servers; they are rows until the next search.
struct BrowseResults {
    std::vector<SkillHit> skills;
    std::vector<McpHit> mcp;
} g_browse;

// Settings keys of the skills and MCP categories. nullopt: not one of them. The text returned is shown under the list.
std::optional<std::string> browse_setting(ChatState &st, const std::string &key, const std::string &value) {
    auto text = [](const Result<std::string> &r) { return r.ok ? r.value : r.error.message; };
    const bool local = st.ctx->config.local_only;
    const auto http = agent::hub::http_options_from_env();
    auto after = [&](std::string message) {
        sync_extensions(st);
        return message;
    };
    if (key.rfind("skill.on.", 0) == 0) return after(text(skill_set_enabled(*st.ctx, st.workspace, key.substr(9), value == "on")));
    if (key.rfind("mcp.on.", 0) == 0) return after(text(mcp_set_enabled(*st.ctx, st.workspace, key.substr(7), value == "on")));
    if (key == "skill_remove") return value.empty() ? std::string() : after(text(skill_remove_named(*st.ctx, st.workspace, value)));
    if (key == "mcp_remove") return value.empty() ? std::string() : after(text(mcp_remove_named(*st.ctx, st.workspace, value)));
    const bool wants_network = key.rfind("skill.add.", 0) == 0 || key == "skill_id" || key == "skill_search" || key == "mcp_search" ||
                               key.rfind("mcp.reg.", 0) == 0 || key == "mcp_custom" || key.rfind("mcp.add.", 0) == 0;
    if (!wants_network) return std::nullopt;
    if (local) return std::string("Local only is on (Settings, The web): nothing is downloaded or connected");
    if (key.rfind("skill.add.", 0) == 0 || key == "skill_id") {
        const std::string id = key == "skill_id" ? value : key.substr(10);
        if (value == "not installed" || id.empty()) return std::string();
        return after(text(skill_install_id(*st.ctx, st.workspace, id)));
    }
    if (key == "skill_search") {
        if (value.empty()) return std::string();
        auto body = agent::hub::http_get_text(skill_search_url(value), http);
        if (!body.ok) return "The search failed: " + body.error.message;
        auto hits = parse_skill_search(body.value);
        if (!hits.ok) return hits.error.message;
        g_browse.skills = std::move(hits.value);
        if (g_browse.skills.size() > 12) g_browse.skills.resize(12);
        return g_browse.skills.empty() ? "Nothing found for '" + value + "'"
                                       : std::to_string(g_browse.skills.size()) + " results are listed below";
    }
    if (key == "mcp_search") {
        if (value.empty()) return std::string();
        auto body = agent::hub::http_get_text(mcp_search_url(value, 12), http);
        if (!body.ok) return "The search failed: " + body.error.message;
        auto hits = parse_mcp_search(body.value);
        if (!hits.ok) return hits.error.message;
        g_browse.mcp.clear();
        for (McpHit &h : hits.value)
            if (mcp_entry_for(h).ok) g_browse.mcp.push_back(std::move(h));
        return g_browse.mcp.empty() ? "Nothing that can start here was found for '" + value + "'"
                                    : std::to_string(g_browse.mcp.size()) + " results are listed below";
    }
    if (key.rfind("mcp.reg.", 0) == 0) {
        const size_t at = static_cast<size_t>(std::atoi(key.c_str() + 8));
        if (value == "not added" || at >= g_browse.mcp.size()) return std::string();
        auto entry = mcp_entry_for(g_browse.mcp[at]);
        if (!entry.ok) return entry.error.message;
        return after(text(mcp_add_entry_json(*st.ctx, st.workspace, entry.value.server_name, entry.value.json)));
    }
    if (key == "mcp_custom") {
        if (value.empty()) return std::string();
        auto spec = parse_mcp_spec(value);
        if (!spec.ok) return spec.error.message;
        return after(text(mcp_add_spec(*st.ctx, st.workspace, spec.value)));
    }
    if (value == "not added") return std::string();
    return after(text(mcp_add_curated(*st.ctx, st.workspace, key.substr(8))));
}

// Every setting of the chat, by category. The first group is the category in the sidebar, the second a heading inside it.
std::vector<TuiSetting> tui_settings(ChatState &st) {
    std::vector<TuiSetting> s;
    std::string group, section;
    auto cat = [&](const char *g) {
        group = g;
        section.clear();
    };
    auto head = [&](const char *h) { section = h; };
    auto add = [&](std::string key, std::string label, std::string value, std::string hint, std::vector<std::string> options = {},
                   bool editable = true) {
        s.push_back({group, section, std::move(key), std::move(label), std::move(value), std::move(hint), std::move(options), editable});
    };
    auto info = [&](std::string label, std::string value, std::string hint = {}) {
        add("info", std::move(label), std::move(value), std::move(hint), {}, false);
    };

    const CliConfig &c = st.ctx->config;
    std::vector<std::string> models;
    auto catalog = agent::hub::list_models(catalog_path(*st.ctx));
    if (catalog.ok)
        for (const auto &m : catalog.value) models.push_back(m.name);
    // Every model that can hold a conversation (local, vision, remote api); speech, image and embedding models stay out.
    std::vector<std::string> switchable;
    for (const InstalledModel &m : installed_models(catalog_path(*st.ctx)))
        if (can_chat(m)) switchable.push_back(m.name);
    if (std::find(switchable.begin(), switchable.end(), st.model.name) == switchable.end()) switchable.push_back(st.model.name);
    char temperature[16];
    std::snprintf(temperature, sizeof(temperature), "%g", static_cast<double>(c.temperature));
    const std::string home = st.ctx->has_home ? st.ctx->home.root : st.ctx->config_dir;

    cat("General");
    head("Where things are");
    info("Files folder", home, "Models, engines, skills, memory and logs live here. agent home DIR moves it");
    info("Project folder", st.workspace, "The folder the agent works in; its conversations are kept here");
    info("Models folder", models_dir(*st.ctx));
    info("Conversation", st.memory.id, "Resume it later with: agent chat --resume ID");
    head("Startup");
    if (!models.empty())
        add("default_model", "Model to open", c.default_model.empty() ? "(the first one)" : c.default_model,
            "The model a new chat starts with", models);

    cat("Model");
    head("This chat");
    add("model", "Model", st.model.name, "Left and Right switch the model; it loads now", switchable);
    head("How it answers");
    add("temperature", "Temperature", temperature, "0 is exact, 1 is creative, up to 2. Reloads the model");
    add("max_tokens", "Longest answer (tokens)", std::to_string(st.max_tokens), "Where an answer is cut. Reloads the model");
    add("context_window", "Context window (tokens)", std::to_string(st.context_window),
        "How much of the conversation the model sees; more uses more memory. Reloads the model");
    head("Speed");
    add("speculative", "Speculative decoding", speed_setting_text(c, "speculative"),
        "Faster answers when the model can guess ahead. off; prompt lookup reuses text already in the conversation (code, quotes); draft "
        "model lets a small model guess. Reloads the model",
        {"off", "prompt lookup", "draft model"});
    {
        std::vector<std::pair<std::string, uint64_t>> installed;
        if (catalog.ok)
            for (const auto &m : catalog.value) installed.emplace_back(m.name, m.size);
        std::vector<std::string> small = draft_model_candidates(installed, st.model.name);
        if (!small.empty()) {
            small.insert(small.begin(), "(none)");
            add("draft_model", "Draft model", speed_setting_text(c, "draft_model"),
                "A small installed model that guesses ahead; it must share the big model's vocabulary (same family). Reloads the model",
                small);
        } else {
            info("Draft model", "none installed", "Pull a small model of the same family: agent model pull owner/repo");
        }
    }
    add("mtp", "MTP drafting", speed_setting_text(c, "mtp"),
        "For models whose file has a multi-token-prediction head: the model drafts for itself. Reloads the model", {"on", "off"});
    add("kv_cache", "KV cache type", speed_setting_text(c, "kv_cache"),
        "How the conversation cache is stored: q8_0 and q4_0 use less memory (faster on long conversations), f16 is the most exact, auto "
        "chooses. Reloads the model",
        {"auto", "f16", "q8_0", "q4_0"});
    head("Memory");
    add("memory_budget_mb", "Memory budget (MB)", std::to_string(c.memory_budget_mb),
        "0 lets it use 85% of the RAM. Models that do not fit are unloaded in turn");
    add("engine_options", "Engine options (JSON)", st.engine_options,
        "Extra options for this model's engine, for example {\"flash_attn\": true}. Empty for none. Reloads the model");

    cat("Chat");
    head("Tools and permissions");
    add("mode", "Mode", chat_mode_name(st.mode), "ask: asks before commands, deletes and moves   plan: only reads   auto: never asks",
        {"ask", "plan", "auto"});
    add("use_tools", "Use tools", st.no_tools ? "off" : "on",
        "Off: the model only talks (files, commands and the web are not offered). Reloads the model", {"on", "off"});
    add("auto_approve", "Approve without asking", st.auto_approve ? "on" : "off",
        "On answers yes to every permission question, like agent chat -y", {"off", "on"});
    add("always_allowed", "Always-allowed tools", join_names(st.always_allowed, ""),
        "Tools you answered \"a\" for in this project, separated by commas. Empty forgets them");
    head("Reading the answer");
    add("thinking", "Show reasoning", st.think.show ? "on" : "off", "Show the <think> blocks of reasoning models", {"off", "on"});
    head("The web");
    add("local_only", "Local only", c.local_only ? "on" : "off",
        "On refuses web_fetch, web_search and model downloads asked for by the agent: nothing leaves the computer. Reloads the model",
        {"off", "on"});
    add("search_url", "Web search address", c.search_url,
        "A search service with {q} in it, for example http://localhost:8080/search?format=json&q={q}. Empty: no web_search tool. Reloads "
        "the model");

    cat("Tooling");
    head("JavaScript and Python tooling");
    add("js_runtime", "JavaScript tooling", c.js_runtime, "For MCP servers and skills. Install it with: agent runtime use <name>",
        {"bun", "node", "nub"});
    if (st.ctx->has_home)
        for (const char *name : {"bun", "node", "pnpm", "nub", "uv"}) {
            const std::string path = runtime_binary_path(st.ctx->home, name);
            info(name, path.empty() ? "not installed" : "installed", path.empty() ? std::string("agent runtime install ") + name : path);
        }

    cat("Skills");
    {
        const auto installed = list_skills(*st.ctx, st.workspace);
        head("Installed (Left and Right switch one on or off)");
        for (const SkillInstalled &k : installed)
            add("skill.on." + k.name, k.name, k.enabled ? "on" : "off",
                (k.description.empty() ? std::string() : k.description + "  ") + "(" + k.where + ")", {"on", "off"});
        if (installed.empty()) info("None yet", "", "Pick one below, or ask the agent: find a skill for commit messages and add it");
        head("Remove");
        add("skill_remove", "Remove a skill", "", "Press Enter and type the name of an installed skill");
        head("Well-known skills (Right installs)");
        for (const CuratedSkill &c : curated_skills()) {
            const std::string id(c.id), leaf = id.substr(id.rfind('/') + 1);
            const bool have = std::any_of(installed.begin(), installed.end(), [&](const SkillInstalled &k) { return k.name == leaf; });
            if (have) info(leaf, "installed", std::string(c.description));
            else add("skill.add." + id, leaf, "not installed", std::string(c.description) + "  (" + id + ")", {"not installed", "install"});
        }
        head("Find more");
        add("skill_search", "Search skills.sh", "", "Press Enter and type words, for example: commit messages. The results appear below");
        add("skill_id", "Add by name", "", "Press Enter and type owner/repo/skill, for example anthropics/skills/pdf");
        if (!g_browse.skills.empty()) {
            head("Search results (Right installs)");
            for (const SkillHit &h : g_browse.skills)
                add("skill.add." + h.id, h.skill_id, "not installed", h.id + "  " + std::to_string(h.installs) + " installs",
                    {"not installed", "install"});
        }
    }

    cat("MCP servers");
    {
        const auto installed = list_mcp(*st.ctx, st.workspace);
        head("Installed (Left and Right switch one on or off)");
        for (const McpInstalled &m : installed) {
            if (m.removable) add("mcp.on." + m.name, m.name, m.enabled ? "on" : "off", m.detail + "  (" + m.where + ")", {"on", "off"});
            else info(m.name, "on", m.detail + "  (" + m.where + ": edit that file to change it)");
        }
        if (installed.empty()) info("None yet", "", "Pick one below, or ask the agent: connect an MCP server that reads files");
        head("Remove");
        add("mcp_remove", "Remove a server", "", "Press Enter and type the name of a server added here");
        head("Well-known servers (Right adds)");
        for (const CuratedMcp &c : curated_mcp()) {
            const bool have = std::any_of(installed.begin(), installed.end(), [&](const McpInstalled &m) { return m.name == c.id; });
            if (have) info(std::string(c.id), "added", std::string(c.description));
            else add("mcp.add." + std::string(c.id), std::string(c.id), "not added", std::string(c.description), {"not added", "add"});
        }
        head("Add your own");
        add("mcp_custom", "Package, command or address", "",
            "Press Enter and type, for example: @modelcontextprotocol/server-filesystem   or   uvx mcp-server-time   or   "
            "https://example.com/mcp");
        add("mcp_search", "Search the official registry", "", "Press Enter and type words, for example: sqlite. The results appear below");
        if (!g_browse.mcp.empty()) {
            head("Search results (Right adds)");
            for (size_t k = 0; k < g_browse.mcp.size(); ++k) {
                const McpHit &h = g_browse.mcp[k];
                add("mcp.reg." + std::to_string(k), h.name, "not added", h.description.substr(0, 110), {"not added", "add"});
            }
        }
    }

    cat("Engines and GPU");
    head("Backend");
    add("backend", "GPU backend", c.backend, "cpu, vulkan or cuda: what the engines are asked to use. Applies to models loaded from now on",
        {"cpu", "vulkan", "cuda"});
    info("This engine", st.engine.name + " " + st.engine.version, st.engine.path);
    head("Engines installed");
    if (auto engines = load_engines(st.ctx->engines_path); engines.ok)
        for (const EngineEntry &e : engines.value) info(e.name, e.version + (file_exists(e.path) ? "" : "  (file missing)"), e.path);
    head("This computer");
    info("CPU cores", std::to_string(c.hardware.cpu_cores));
    info("RAM", size_text(total_ram_bytes()));

    cat("Models on disk");
    head("Installed models");
    if (catalog.ok)
        for (const auto &m : catalog.value)
            info(m.name, size_text(m.size) + (m.name == c.default_model ? "   (opens by default)" : ""), m.path);
    if (!catalog.ok || catalog.value.empty()) info("None yet", "", "agent model pull <Hugging Face link, owner/repo or a file>");

    cat("Storage");
    head("Disk use");
    info("Models", size_text(folder_bytes(models_dir(*st.ctx))));
    if (st.ctx->has_home) {
        info("Cache", size_text(folder_bytes(st.ctx->home.cache)), "agent clean removes partial downloads and caches");
        info("Logs", size_text(folder_bytes(st.ctx->home.logs)));
        info("Runtimes", size_text(folder_bytes(st.ctx->home.runtimes)));
        info("Log files", st.ctx->home.logs, "Engine and chat logs, for bug reports");
    }
    info("This project", size_text(folder_bytes((fs::path(st.workspace) / "memory").string())), "Conversations and memory of this project");

    cat("Shortcuts");
    head("Press Enter on a shortcut, then the keys you want instead");
    for (const TuiShortcut &a : tui_shortcut_actions()) {
        const auto changed = c.shortcuts.find(a.action);
        const std::string key = changed != c.shortcuts.end() && tui_key_valid(changed->second) ? changed->second : a.default_key;
        s.push_back({group,
                     section,
                     std::string("shortcut.") + a.action,
                     a.label,
                     tui_key_pretty(key),
                     std::string("Default: ") + tui_key_pretty(a.default_key) + ". F1 to F12, Ctrl+letter or Alt+letter",
                     {},
                     true,
                     true});
    }
    add("shortcut_reset", "Put the defaults back", "keep", "Choose reset to forget every shortcut you changed", {"keep", "reset"});

    cat("About");
    head("agent.cpp");
    info("Version", std::string(kCliVersion), "agent update checks for a newer one");
    info("Source", "github.com/amitverma-cf/agent.cpp");
    info("Licence", "MIT");
    info("Set up again", "agent setup", "Asks for the folder, the optional runtimes and a model");
    info("Icons", icons_name(), "AGENT_ICONS=nerd|ascii|unicode before starting agent");
    return s;
}

std::string tui_set_setting(ChatState &st, const std::string &key, const std::string &value) {
    if (auto handled = browse_setting(st, key, value)) return *handled;
    CliConfig &c = st.ctx->config;
    auto persist = [&]() -> std::string {
        if (auto r = save_config(st.ctx->config_path, c); !r.ok) return "could not save config.json: " + r.error.message;
        return {};
    };
    if (auto bad = check_setting_value(key, value); !bad.empty()) return bad; // the checks the settings tool shares
    auto reload = [&]() {
        if (settings_defer_reload(true)) return std::string(); // several changes at once: one reload at the end
        return reload_model(st) ? std::string() : std::string("the model did not reload; try /model NAME");
    };
    auto whole = [&](long lowest, long &out) -> std::string {
        char *end = nullptr;
        const long v = std::strtol(value.c_str(), &end, 10);
        if (value.empty() || *end != 0 || v < lowest || v > 1048576) return "a whole number of at least " + std::to_string(lowest);
        out = v;
        return {};
    };

    if (key == "shortcut_reset") {
        if (value == "reset") {
            c.shortcuts.clear();
            return persist();
        }
        return {};
    }
    if (key.rfind("shortcut.", 0) == 0) {
        const std::string action = key.substr(9);
        if (!tui_key_valid(value)) return "that key cannot be used";
        for (const TuiShortcut &a : tui_shortcut_actions()) {
            if (action == a.action) continue;
            const auto other = c.shortcuts.find(a.action);
            const std::string taken = other != c.shortcuts.end() ? other->second : a.default_key;
            if (taken == value) return tui_key_pretty(value) + " is already used for: " + a.label;
        }
        c.shortcuts[action] = value;
        return persist();
    }
    if (key == "model") {
        if (value == st.model.name) return {};
        switch_model(st, value);
        return st.model.name == value ? std::string() : std::string("could not load " + value + "; see the messages above");
    }
    if (key == "default_model") {
        c.default_model = value;
        return persist();
    }
    if (key == "kv_cache" || key == "speculative" || key == "draft_model" || key == "mtp") {
        set_speed_setting(c, key, value);
        if (auto e = persist(); !e.empty()) return e;
        return reload();
    }
    if (key == "mode") {
        ChatMode m;
        if (!parse_chat_mode(value, m)) return "choose ask, plan or auto";
        st.mode = m;
        return {};
    }
    if (key == "thinking") {
        st.think.show = value == "on";
        return {};
    }
    if (key == "use_tools") {
        st.no_tools = value == "off";
        return reload();
    }
    if (key == "auto_approve") {
        st.auto_approve = value == "on";
        return {};
    }
    if (key == "always_allowed") {
        std::vector<std::string> tools;
        size_t from = 0;
        while (from <= value.size()) {
            const size_t comma = value.find(',', from);
            std::string name = value.substr(from, comma == std::string::npos ? std::string::npos : comma - from);
            while (!name.empty() && name.front() == ' ') name.erase(0, 1);
            while (!name.empty() && name.back() == ' ') name.pop_back();
            if (!name.empty()) tools.push_back(name);
            if (comma == std::string::npos) break;
            from = comma + 1;
        }
        st.always_allowed = tools;
        return save_always_allowed(st.workspace, tools) ? std::string() : std::string("could not save .agents/permissions.json");
    }
    if (key == "temperature") {
        char *end = nullptr;
        const float v = std::strtof(value.c_str(), &end);
        if (value.empty() || end == value.c_str() || *end != 0 || v < 0.0f || v > 2.0f) return "a number from 0 to 2";
        c.temperature = v;
        if (auto e = persist(); !e.empty()) return e;
        return reload();
    }
    if (key == "max_tokens" || key == "context_window") {
        long v = 0;
        if (auto e = whole(key == "max_tokens" ? 16 : 2048, v); !e.empty()) return e;
        if (key == "max_tokens") st.max_tokens = c.max_tokens = static_cast<int>(v);
        else st.context_window = c.context_window = static_cast<int>(v);
        if (auto e = persist(); !e.empty()) return e;
        return reload();
    }
    if (key == "memory_budget_mb") {
        long v = 0;
        if (auto e = whole(0, v); !e.empty()) return e;
        c.memory_budget_mb = v;
        const uint64_t bytes = v > 0 ? static_cast<uint64_t>(v) << 20 : (total_ram_bytes() / 100 * 85);
        set_budget(st.registry, ModelBudget{.ram_bytes = bytes});
        return persist();
    }
    if (key == "engine_options") {
        if (!value.empty()) {
            simdjson::dom::parser parser;
            simdjson::dom::element doc;
            if (parser.parse(value).get(doc) || !doc.is_object()) return "a JSON object such as {\"flash_attn\": true}, or empty";
        }
        st.engine_options = value;
        return reload();
    }
    if (key == "local_only") {
        c.local_only = value == "on";
        if (auto e = persist(); !e.empty()) return e;
        return reload();
    }
    if (key == "search_url") {
        if (!value.empty() && value.find("{q}") == std::string::npos) return "the address needs {q} where the search text goes";
        c.search_url = value;
        if (auto e = persist(); !e.empty()) return e;
        return reload();
    }
    if (key == "js_runtime") {
        if (value != "bun" && value != "node" && value != "nub") return "choose bun, node or nub";
        c.js_runtime = value;
        return persist();
    }
    if (key == "backend") {
        if (value != "cpu" && value != "vulkan" && value != "cuda") return "choose cpu, vulkan or cuda";
        c.backend = value;
        return persist();
    }
    return "this setting cannot be changed here";
}
#endif

// --acp: the Agent Client Protocol, for editors (Zed and others): JSON-RPC 2.0, one message per line on stdio. initialize, session/new,
// session/prompt (streams session/update notifications, asks session/request_permission). session/cancel is read between turns only.
int run_acp(ChatState &st) {
    g_acp = true;
    std::string line;
    while (read_line(stdin, line) == ReadStatus::Line) {
        simdjson::dom::parser parser;
        simdjson::padded_string padded(line);
        simdjson::dom::element msg;
        if (parser.parse(padded).get(msg) || !msg.is_object()) continue;
        std::string_view method;
        if (msg["method"].get(method)) continue;
        simdjson::dom::element id_el;
        const bool has_id = !msg["id"].get(id_el) && !id_el.is_null();
        const std::string id = has_id ? simdjson::to_string(id_el) : std::string("null");
        auto reply = [&](const std::string &result) { acp_send("{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":" + result + "}"); };
        auto error = [&](int code, const std::string &text) {
            acp_send("{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"error\":{\"code\":" + std::to_string(code) + ",\"message\":\"" +
                     json_escape(text) + "\"}}");
        };
        if (method == "initialize") {
            reply("{\"protocolVersion\":1,\"agentCapabilities\":{\"loadSession\":false,\"promptCapabilities\":{\"image\":false,\"audio\":"
                  "false,"
                  "\"embeddedContext\":true}},\"agentInfo\":{\"name\":\"agent.cpp\",\"title\":\"agent.cpp\",\"version\":\"" +
                  std::string(kCliVersion) + "\"},\"authMethods\":[]}");
        } else if (method == "session/new") {
            g_acp_session = st.memory.id;
            reply("{\"sessionId\":\"" + json_escape(g_acp_session) + "\"}");
        } else if (method == "session/prompt") {
            simdjson::dom::array blocks;
            std::string text;
            if (msg["params"]["prompt"].get(blocks)) {
                error(-32602, "session/prompt needs a prompt array");
                continue;
            }
            for (simdjson::dom::element b : blocks) {
                std::string_view type, t, uri;
                if (b["type"].get(type)) continue;
                if (type == "text" && !b["text"].get(t)) {
                    text += std::string(t) + "\n";
                } else if (type == "resource" && !b["resource"]["text"].get(t)) {
                    if (b["resource"]["uri"].get(uri)) uri = "";
                    text += "\nFile " + std::string(uri) + ":\n```\n" + std::string(t) + "\n```\n";
                }
            }
            run_one_turn(st, text, {});
            reply(std::string("{\"stopReason\":\"") + (g_last_turn_failed ? "refusal" : "end_turn") + "\"}");
        } else if (has_id) {
            error(-32601, "method not found: " + std::string(method));
        }
    }
    return 0;
}

int run_chat(ChatState &st, const CommandLine &cl, const std::string &engine_log) {
    Config store_cfg;
    store_cfg.provider = AiProvider::Mock;
    store_cfg.workspace_dir = st.workspace;
    store_cfg.memory = {.store = store::sqlite_store()};
    store_cfg.auto_default_tools = false;
    auto store_session = init(store_cfg);
    if (!store_session.ok) return chat_fail(store_session.error);
    st.store = std::move(store_session.value);
    remember_workspace(st);
    set_budget(st.registry, ModelBudget{.ram_bytes = static_cast<uint64_t>(std::max<int64_t>(0, st.ctx->config.memory_budget_mb)) << 20});

    std::fprintf(g_err, "loading %s ...\n", st.model.path.c_str());
    if (auto r = open_session(st); !r.ok) return chat_fail(r.error);

    if (cl.has("resume")) {
        st.memory.id = std::string(cl.get("resume"));
        st.saved = true;
        if (auto r = load_conversation(*st.lease, st.memory); !r.ok) {
            if (r.error.code == ErrorCode::KeyNotFound)
                return chat_fail(make_error(ErrorCode::KeyNotFound, "no saved conversation '" + st.memory.id + "' in " + st.workspace +
                                                                        " (see `agent chat --list`)"));
            return chat_fail(r.error);
        }
    } else {
        st.memory.id = new_conversation_id();
        st.memory.history.push_back(Message{.role = "system", .content = system_prompt(st)});
    }

#if defined(AGENT_CLI_HAS_TUI)
    if (st.use_tui) {
        if (cl.has("resume")) collect_artifacts(st, 0);
        // Anything written to stderr (engine logs, warnings) would tear the screen: it goes to a file for the time being.
        std::FILE *log = std::fopen((fs::path(st.workspace) / "logs" / "chat.log").string().c_str(), "a");
        std::FILE *saved_err = g_err;
        if (log) g_err = log;
        TuiHost host;
        host.user = &st;
        host.header = [](void *u) {
            auto *c = static_cast<ChatState *>(u);
            return c->model.name + "  \xC2\xB7  " + c->engine.name + "  \xC2\xB7  " + c->memory.id;
        };
        host.subtitle = [](void *u) { return static_cast<ChatState *>(u)->workspace; };
        static constexpr TuiCommand kCommands[] = {
            {"/help", "list the commands"},
            {"/model", "show or switch the model"},
            {"/image", "send an image or audio file"},
            {"/task", "run an engine capability: image, speech, 3D, embed, ONNX"},
            {"/run", "run a graph file"},
            {"/open", "list or open generated files"},
            {"/stats", "token and tool-call counters"},
            {"/diff", "what the tools changed in files"},
            {"/undo", "put back the last file change"},
            {"/compact", "summarise old messages"},
            {"/mode", "ask, plan or auto"},
            {"/thinking", "show or hide reasoning"},
            {"/permissions", "tools allowed for good"},
            {"/rename", "name this conversation"},
            {"/delete", "remove this conversation"},
            {"/doctor", "what is installed and disk use"},
            {"/setup", "first-time setup questions"},
            {"/clear", "forget the conversation"},
            {"/exit", "quit"},
        };
        host.commands = kCommands;
        host.footer = [](void *u) {
            auto *c = static_cast<ChatState *>(u);
            return format_turn_footer(c->memory.stats.last);
        };
        host.submit = [](void *u, const std::string &line) { return handle_line(*static_cast<ChatState *>(u), line); };
        host.sessions = [](void *u) { return tui_sessions(*static_cast<ChatState *>(u)); };
        host.switch_session = [](void *u, const std::string &id, const std::string &workspace) {
            return switch_session(*static_cast<ChatState *>(u), id, workspace);
        };
        host.notice = [](void *) { return sandbox_notice(); };
        host.mode = [](void *u) { return std::string(chat_mode_name(static_cast<ChatState *>(u)->mode)); };
        host.taskbar = [](void *u) { return tui_taskbar(*static_cast<ChatState *>(u)); };
        host.delete_turn = [](void *u, int from_end) { return tui_delete_turn(*static_cast<ChatState *>(u), from_end); };
        host.mic = [](void *u, bool start) { return tui_mic(*static_cast<ChatState *>(u), start); };
        host.shortcuts = [](void *u) {
            std::vector<std::pair<std::string, std::string>> changed;
            for (const auto &[action, key] : static_cast<ChatState *>(u)->ctx->config.shortcuts) changed.emplace_back(action, key);
            return changed;
        };
        host.settings = [](void *u) { return tui_settings(*static_cast<ChatState *>(u)); };
        host.set_setting = [](void *u, const std::string &key, const std::string &value) {
            return tui_set_setting(*static_cast<ChatState *>(u), key, value);
        };
        host.delete_session = [](void *u, const std::string &id, const std::string &workspace) {
            auto &c = *static_cast<ChatState *>(u);
            if (!delete_conversation(c, id, workspace)) return false;
            if (id == c.memory.id && workspace == c.workspace) {
                switch_session(c, "", c.workspace);
                c.repaint = true;
            }
            return true;
        };
        host.take_repaint = [](void *u) {
            auto &c = *static_cast<ChatState *>(u);
            const bool r = c.repaint;
            c.repaint = false;
            return r;
        };
        host.history = [](void *u) { return tui_history(*static_cast<ChatState *>(u)); };
        host.cancel = [](void *) {
            if (!g_generating.load()) return;
            if (int32_t *flag = g_cancel_flag.load()) std::atomic_ref<int32_t>(*flag).store(1, std::memory_order_relaxed);
        };
        g_tui = true;
        start_stats(st.ctx->has_home ? st.ctx->home.root : st.workspace);
        const int rc = run_tui(host);
        stop_stats();
        g_tui = false;
        g_err = saved_err;
        if (log) std::fclose(log);
        if (st.saved || cl.has("resume"))
            std::printf("saved as %s; resume with: agent chat --resume %s\n", st.memory.id.c_str(), st.memory.id.c_str());
        return rc;
    }
#endif
    if (cl.has("acp")) return run_acp(st);
    if (cl.has("prompt") || cl.has("jsonl")) return run_scripted(st, cl);
    std::printf("agent chat  (%s)\n", st.memory.id.c_str());
    std::printf("  model:     %s  %s\n", st.model.name.c_str(), st.model.path.c_str());
    std::printf("  engine:    %s %s  %s  (%s)\n", st.engine.name.c_str(), st.engine.version.c_str(), st.engine.path.c_str(),
                cli_engine_isolation(st.inprocess, st.engine.isolation) == EngineIsolation::Process ? "agent-engine-host" : "in-process");
    std::printf("  workspace: %s\n", st.workspace.c_str());
    std::printf("  sandbox:   %s\n", sandbox_status_line().c_str());
    if (!engine_log.empty()) std::printf("  engine log: %s (--verbose prints it here)\n", engine_log.c_str());
    std::string tool_names;
    for (const auto &t : st.lease->config.tools) tool_names += (tool_names.empty() ? "" : " ") + std::string(t.name);
    std::printf("  tools:     %s\n", tool_names.c_str());
    if (cl.has("resume")) {
        std::printf("  resumed:   %zu messages\n", st.memory.history.size());
        collect_artifacts(st, 0);
    }
    std::printf("Type a message. /help lists commands, /exit quits.\n\n");
    std::fflush(stdout);
    g_at_line_start = true;

    install_sigint();
    std::string line;
    for (;;) {
        if (st.interactive) out("> ");
        const ReadStatus rs = read_line(stdin, line);
        if (rs != ReadStatus::Line) {
            if (st.interactive) out("\n");
            break;
        }
        if (!st.interactive) out("> " + line + "\n"); // echo piped input so transcripts read naturally
        g_at_line_start = true;
        if (!handle_line(st, line)) break;
    }

    std::signal(SIGINT, SIG_DFL);
    if (st.saved || cl.has("resume")) out("saved as " + st.memory.id + "; resume with: agent chat --resume " + st.memory.id + "\n");
    return 0;
}

} // namespace

} // namespace agent::cli
