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
        std::fprintf(stderr, "agent: %s\n\n%s", parsed.error.message.c_str(), usage_text().c_str());
        return 2;
    }
    const CommandLine &cl = parsed.value;
    if (cl.command.empty() || cl.command == "help" || cl.help) {
        std::printf("%s", usage_text().c_str());
        return cl.command.empty() && !cl.help ? 2 : 0;
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
    if (cl.command == "chat") return cmd_chat(cl, ctx);
    if (cl.command == "run") return cmd_run(cl, ctx);
    if (cl.command == "task") return cmd_task(cl, ctx);
    if (cl.command == "init") return cmd_init(cl, ctx);
    if (cl.command == "secret") return cmd_secret(cl, ctx);
    if (cl.command == "doctor") return cmd_doctor(cl, ctx);
    if (cl.command == "mcp") return cmd_mcp(cl, ctx);
    if (cl.command == "runtime") return cmd_runtime(cl, ctx);
    if (cl.command == "skill") return cmd_skill(cl, ctx);
    if (cl.command == "clean") return cmd_clean(cl, ctx);
    if (cl.command == "voice") return cmd_voice(cl, ctx);
    std::fprintf(stderr, "agent: unhandled command '%s'\n", cl.command.c_str());
    return 2;
}
