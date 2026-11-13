// agent: the agent.cpp command-line app (setup wizard, engine/model manager, chat TUI).

#include "commands.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
    using namespace agent::cli;

    std::vector<std::string_view> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

    auto parsed = parse_command_line(args);
    if (!parsed.ok) {
        std::fprintf(stderr, "agent: %s\n", parsed.error.message.c_str());
        std::string first;
        for (std::string_view a : args)
            if (!a.empty() && a[0] != '-') {
                first = std::string(a);
                break;
            }
        const std::string detail = first.empty() ? std::string() : usage_text(first);
        if (!detail.empty() && detail != usage_text()) {
            std::fprintf(stderr, "\n%s", detail.c_str());
        } else {
            const std::string guess = closest_command(first);
            if (!guess.empty()) std::fprintf(stderr, "Did you mean '%s'?\n", guess.c_str());
            std::fprintf(stderr, "Run 'agent --help' for the list of commands.\n");
        }
        return 2;
    }
    // `agent` on its own in a terminal opens the chat screen (like other agent CLIs); piped or scripted it still prints the usage.
    std::vector<std::string_view> chat_args;
    if (parsed.value.command.empty() && !parsed.value.help && stdin_is_tty() && stream_is_tty(stdout)) {
        chat_args.push_back("chat");
        chat_args.insert(chat_args.end(), args.begin(), args.end());
        auto as_chat = parse_command_line(chat_args);
        if (as_chat.ok) parsed = std::move(as_chat);
    }
    const CommandLine &cl = parsed.value;
    set_no_sandbox(cl.has("no-sandbox"));
    // Commands that run models, tools or engines say once when nothing is confined (the chat screen shows it on its welcome screen).
    if (cl.command == "task" || cl.command == "run" || cl.command == "voice" || (cl.command == "chat" && !stdin_is_tty())) {
        const std::string notice = sandbox_notice();
        if (!notice.empty()) std::fprintf(stderr, "%s\n", notice.c_str());
    }
    if (cl.command.empty() || cl.command == "help" || cl.help) {
        std::string topic = cl.command == "help" ? (cl.positionals.empty() ? std::string() : cl.positionals[0]) : cl.command;
        std::printf("%s", usage_text(topic).c_str());
        return cl.command.empty() && !cl.help ? 2 : 0;
    }
    if (cl.command == "completion") {
        const std::string script = completion_script(cl.positionals.empty() ? std::string_view() : std::string_view(cl.positionals[0]));
        if (script.empty()) {
            std::fprintf(stderr, "agent: completion SHELL: bash, zsh, fish or powershell\n");
            return 2;
        }
        std::printf("%s", script.c_str());
        return 0;
    }
    if (cl.command == "version") {
        std::printf("agent %.*s\n", static_cast<int>(kCliVersion.size()), kCliVersion.data());
        return 0;
    }

    Context ctx;
    // An explicit config folder (--config-dir or $AGENT_CONFIG_DIR) keeps the old layout; otherwise the global home
    // (--home, $AGENT_HOME, or the one chosen at `agent setup`) holds config.json, engines.json, models, skills, ...
    const char *env_config = std::getenv("AGENT_CONFIG_DIR");
    const char *env_home = std::getenv("AGENT_HOME");
    const bool explicit_config = cl.has("config-dir") || (env_config && *env_config);
    const std::string home_root =
        explicit_config ? std::string() : resolve_home_root(cl.get("home"), env_home ? env_home : "", home_pointer_path());
    if (!home_root.empty()) {
        ctx.home = home_layout(home_root);
        ctx.has_home = true;
        ctx.config_dir = ctx.home.root;
        ctx.config_path = ctx.home.config;
        ctx.engines_path = ctx.home.engines_json;
    } else {
        ctx.config_dir = cl.has("config-dir") ? std::string(cl.get("config-dir")) : default_config_dir();
        ctx.config_path = (std::filesystem::path(ctx.config_dir) / "config.json").string();
        ctx.engines_path = (std::filesystem::path(ctx.config_dir) / "engines.json").string();
    }
    {
        std::error_code cwd_ec;
        ctx.project_root = find_project_root(std::filesystem::current_path(cwd_ec).string());
    }
    auto config = load_config(ctx.config_path);
    if (!config.ok) return report_error(config.error);
    ctx.config = std::move(config.value);

    agent::AgentRuntime runtime;
    if (cl.command == "setup") return cmd_setup(cl, ctx);
    if (cl.command == "engine") return cmd_engine(cl, ctx);
    if (cl.command == "model") return cmd_model(cl, ctx);
    // No memory_budget_mb in config.json: use 85% of the installed RAM, so models that do not fit together are swapped
    // (a large main model next to small sub-agent models) instead of the run failing or the machine swapping to disk.
    if ((cl.command == "chat" || cl.command == "run") && ctx.config.memory_budget_mb == 0) {
        if (const uint64_t ram = total_ram_bytes(); ram) ctx.config.memory_budget_mb = static_cast<int64_t>((ram / 100 * 85) >> 20);
    }
    if (cl.command == "chat") return cmd_chat(cl, ctx);
    if (cl.command == "run") return cmd_run(cl, ctx);
    if (cl.command == "task") return cmd_task(cl, ctx);
    if (cl.command == "init") return cmd_init(cl, ctx);
    if (cl.command == "secret") return cmd_secret(cl, ctx);
    if (cl.command == "doctor") return cmd_doctor(cl, ctx);
    if (cl.command == "update") return cmd_update(cl, ctx);
    if (cl.command == "mcp") return cmd_mcp(cl, ctx);
    if (cl.command == "runtime") return cmd_runtime(cl, ctx);
    if (cl.command == "skill") return cmd_skill(cl, ctx);
    if (cl.command == "sandbox") return cmd_sandbox(cl, ctx);
    if (cl.command == "schedule") return cmd_schedule(cl, ctx);
    if (cl.command == "wake") return cmd_wake(cl, ctx);
    if (cl.command == "clean") return cmd_clean(cl, ctx);
    if (cl.command == "voice") return cmd_voice(cl, ctx);
    std::fprintf(stderr, "agent: unhandled command '%s'\n", cl.command.c_str());
    return 2;
}
