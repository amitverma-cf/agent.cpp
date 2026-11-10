// agent chat: line-based text TUI over run_turn. Streams tokens, shows tool calls and results,
// asks before tools the default permission policy rates Ask, persists the conversation in the
// workspace's SQLite store, and opens generated artifacts with the OS default app.

#include "commands.hpp"

#include <agent-cpp/store_sqlite.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <cstdio>
#include <filesystem>

#if !defined(_WIN32)
#include <signal.h>
#endif

namespace agent::cli {

namespace {

namespace fs = std::filesystem;

constexpr std::string_view kIndexKey = "cli:conversations";
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
    struct sigaction sa {};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // no SA_RESTART: a blocked fgets returns EINTR
    sigaction(SIGINT, &sa, nullptr);
#endif
}

void out(std::string_view s) {
    if (s.empty()) return;
    std::fwrite(s.data(), 1, s.size(), stdout);
    g_at_line_start = s.back() == '\n';
    std::fflush(stdout);
}

void ensure_newline() {
    if (!g_at_line_start) out("\n");
}

struct ChatState {
    Context *ctx = nullptr;
    // Models live in the registry (memory_budget_mb, LRU unload); lease is the current one. Every model
    // Session shares the SQLite store opened by `store`, so switching never leaves two write-back caches
    // on one database.
    ModelRegistry registry;
    ModelLease lease;
    Session store;
    FlowMemory memory;

    ModelRef model;
    LoadedEngine engine;
    std::string engine_flag; // --engine as given (empty = by model file type)
    std::string engine_options;
    std::string workspace;
    int context_window = 4096;
    int max_tokens = 512;
    bool verbose = false;
    bool auto_approve = false;
    bool inprocess = false; // --inprocess: load the engine into the CLI instead of agent-engine-host
    bool interactive = false;

    std::vector<std::string> always_allowed;
    std::vector<Media> artifacts;
    size_t streamed_bytes = 0;
    StreamFilter filter;
    bool saved = false; // the conversation is in the store (resumable)
};

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
    out(filter_stream(st->filter, token));
}

PermissionDecision check_tool(std::string_view name, std::string_view arguments, void *user_data) {
    auto *st = static_cast<ChatState *>(user_data);
    ensure_newline();
    out("[tool] " + std::string(name) + " " + one_line(arguments, 300) + "\n");
    PermissionDecision d = tools::default_permission_policy(name, arguments, nullptr);
    if (d == PermissionDecision::Ask && std::find(st->always_allowed.begin(), st->always_allowed.end(), name) != st->always_allowed.end())
        return PermissionDecision::Allow;
    return d;
}

bool prompt_tool(std::string_view name, std::string_view, void *user_data) {
    auto *st = static_cast<ChatState *>(user_data);
    const std::string n(name);
    if (st->auto_approve) {
        out("[permission] " + n + ": approved (--yes)\n");
        return true;
    }
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
        st->always_allowed.push_back(n);
        return true;
    }
    return c == 'y' || c == 'Y';
}

void on_tool_result(const Event &event, void *) {
    const std::string_view result(reinterpret_cast<const char *>(event.payload.data()), event.payload.size());
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

// ModelSpec::on_load: share the chat's store and show tool results.
Result<void> on_model_loaded(Session &session, const ModelSpec &, void *user_data) {
    auto *st = static_cast<ChatState *>(user_data);
    session.config.memory = st->store.config.memory;
    session.memory_state = st->store.memory_state;
    register_event_hook(session, EventType::OnToolResult, on_tool_result, st);
    return ok();
}

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
    cfg.engine_options = st.engine_options;
    cfg.model = st.model.path;
    cfg.workspace_dir = st.workspace;
    cfg.lazy_tools = true; // the model sees tool_search / load_tool, not every tool, until it needs one
    cfg.context_window = st.context_window;
    cfg.max_tokens = st.max_tokens;
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
    std::printf("conversation %s: %zu messages, %d inference rounds, %d tool calls, %d prompt + %d completion tokens\n",
                st.memory.id.c_str(), st.memory.history.size(), c.turns, c.tool_calls, c.prompt_tokens, c.completion_tokens);
    std::printf("this session:  %d inference rounds, %d tool calls, %d prompt + %d completion tokens, %d compressions, %d prunes\n",
                s.total_turns, s.total_tool_calls, s.total_prompt_tokens, s.total_completion_tokens, s.total_compressions,
                s.total_prune_cycles);
    if (c.cached_tokens > 0 && c.prompt_tokens > 0)
        std::printf("prefix cache:  %d of %d prompt tokens served from the KV cache (%.0f%%)\n", c.cached_tokens, c.prompt_tokens,
                    100.0 * c.cached_tokens / c.prompt_tokens);
    if (const std::string footer = format_turn_footer(c.last); !footer.empty()) std::printf("last call:     %s\n", footer.c_str());
    std::printf("model %s (%s), engine %s %s\n", st.model.name.c_str(), st.model.path.c_str(), st.engine.name.c_str(),
                st.engine.version.c_str());
    const ModelBudget budget = get_budget(st.registry);
    const auto mb = [](uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    std::printf("loaded models (budget %s, CLI process RSS %.0f MiB; out-of-process engines are not included):\n",
                budget.ram_bytes ? (std::to_string(budget.ram_bytes >> 20) + " MiB").c_str() : "unlimited", mb(process_rss_bytes()));
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    for (const LoadedModelInfo &m : loaded_models(st.registry))
        std::printf("  %-24s ~%.0f MiB%s  %s, last used %llds ago, loaded in %.1fs\n", m.name.c_str(), mb(m.ram_bytes + m.vram_bytes),
                    m.vram_bytes ? " (VRAM)" : "", m.leases ? "in use" : "idle", static_cast<long long>((now - m.last_used_ms) / 1000),
                    static_cast<double>(m.load_time_ms) / 1000.0);
}

void print_chat_help() {
    std::printf("/exit                 quit (also Ctrl-D, or Ctrl-C at the prompt)\n"
                "/clear                forget the conversation so far\n"
                "/model [NAME|PATH]    show or switch the model (the conversation is kept)\n"
                "/image <path> [text]  send an image (or audio) with an optional prompt\n"
                "/open [n]             list generated artifacts, or open the n-th with the default app\n"
                "/stats                token and tool-call counters\n"
                "//text                send a message that starts with '/'\n"
                "Ctrl-C while the model is answering cancels the answer.\n");
}

void run_one_turn(ChatState &st, const std::string &text, std::span<const Media> media) {
    if (!st.lease) {
        out("[error] no model is loaded; switch with /model NAME\n");
        return;
    }
    const size_t before = st.memory.history.size();
    st.streamed_bytes = 0;
    clear_cancel(*st.lease);
    g_generating.store(true);
    auto r = media.empty() ? run_turn(*st.lease, st.memory, text, true, on_token, &st)
                           : run_turn_with_media(*st.lease, st.memory, text, media, true, on_token, &st);
    g_generating.store(false);
    if (!r.ok) {
        ensure_newline();
        if (r.error.code == ErrorCode::Cancelled) out("[cancelled]\n");
        else out("[error] " + r.error.message + "\n");
        clear_cancel(*st.lease);
    } else if (st.streamed_bytes == 0) {
        out(r.value); // engine did not stream
    }
    out(flush_stream(st.filter));
    ensure_newline();
    collect_artifacts(st, before);
    if (r.ok) {
        const std::string footer = format_turn_footer(st.memory.stats.last);
        if (!footer.empty()) std::fprintf(g_err, "[%s]\n", footer.c_str());
    }
    remember_conversation(st);
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
        if (c < '0' || c > '9' || v > 10'000'000) return fail<int>(ErrorCode::InvalidConfig, "--" + std::string(flag) + " expects a positive number");
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
        if (auto r = save_flow_memory(*st.lease, st.memory); !r.ok) out("[error] " + r.error.message + "\n");
    }
    out("now using " + st.model.name + " on " + st.engine.name + "\n");
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
    if (cl.has("list")) return list_conversations(st.workspace);

    st.verbose = cl.has("verbose");
    st.auto_approve = cl.has("yes");
    st.inprocess = cl.has("inprocess");
    st.interactive = stdin_is_tty();
    st.engine_flag = std::string(cl.get("engine"));
    st.engine_options = std::string(cl.get("engine-options"));
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

int run_chat(ChatState &st, const CommandLine &cl, const std::string &engine_log) {
    Config store_cfg;
    store_cfg.provider = AiProvider::Mock;
    store_cfg.workspace_dir = st.workspace;
    store_cfg.memory = {.store = store::sqlite_store()};
    store_cfg.auto_default_tools = false;
    auto store_session = init(store_cfg);
    if (!store_session.ok) return chat_fail(store_session.error);
    st.store = std::move(store_session.value);
    set_budget(st.registry, ModelBudget{.ram_bytes = static_cast<uint64_t>(std::max<int64_t>(0, st.ctx->config.memory_budget_mb)) << 20});

    std::fprintf(g_err, "loading %s ...\n", st.model.path.c_str());
    if (auto r = open_session(st); !r.ok) return chat_fail(r.error);

    if (cl.has("resume")) {
        st.memory.id = std::string(cl.get("resume"));
        st.saved = true;
        if (auto r = load_flow_memory(*st.lease, st.memory); !r.ok) {
            if (r.error.code == ErrorCode::KeyNotFound)
                return chat_fail(make_error(ErrorCode::KeyNotFound, "no saved conversation '" + st.memory.id + "' in " + st.workspace +
                                                                        " (see `agent chat --list`)"));
            return chat_fail(r.error);
        }
    } else {
        st.memory.id = new_conversation_id();
        st.memory.history.push_back(Message{.role = "system", .content = std::string(kSystemPrompt)});
    }

    std::printf("agent chat  (%s)\n", st.memory.id.c_str());
    std::printf("  model:     %s  %s\n", st.model.name.c_str(), st.model.path.c_str());
    std::printf("  engine:    %s %s  %s  (%s)\n", st.engine.name.c_str(), st.engine.version.c_str(), st.engine.path.c_str(),
                cli_engine_isolation(st.inprocess, st.engine.isolation) == EngineIsolation::Process ? "agent-engine-host" : "in-process");
    std::printf("  workspace: %s\n", st.workspace.c_str());
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

        const SlashCommand cmd = parse_slash_command(line);
        if (cmd.kind == SlashKind::None) {
            if (cmd.text.empty()) continue;
            run_one_turn(st, cmd.text, {});
            continue;
        }
        if (cmd.kind == SlashKind::Exit) break;
        if (!st.lease && (cmd.kind == SlashKind::Clear || cmd.kind == SlashKind::Stats)) {
            out("[error] no model is loaded; switch with /model NAME\n");
            continue;
        }
        switch (cmd.kind) {
        case SlashKind::Clear: {
            const bool has_system = !st.memory.history.empty() && st.memory.history.front().role == "system";
            Message system = has_system ? st.memory.history.front() : Message{};
            st.memory.history.clear();
            if (has_system) st.memory.history.push_back(std::move(system));
            st.memory.stats = {};
            st.artifacts.clear();
            if (auto r = save_flow_memory(*st.lease, st.memory); !r.ok) out("[error] " + r.error.message + "\n");
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
        case SlashKind::Help: print_chat_help(); break;
        case SlashKind::Unknown: out("unknown command /" + cmd.arg + " (try /help)\n"); break;
        default: break;
        }
    }

    std::signal(SIGINT, SIG_DFL);
    if (st.saved || cl.has("resume")) out("saved as " + st.memory.id + "; resume with: agent chat --resume " + st.memory.id + "\n");
    return 0;
}

} // namespace

} // namespace agent::cli
