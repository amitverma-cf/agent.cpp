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
    ctx.config_dir = cl.has("config-dir") ? std::string(cl.get("config-dir")) : default_config_dir();
    ctx.config_path = (std::filesystem::path(ctx.config_dir) / "config.json").string();
    ctx.engines_path = (std::filesystem::path(ctx.config_dir) / "engines.json").string();
    auto config = load_config(ctx.config_path);
    if (!config.ok) return report_error(config.error);
    ctx.config = std::move(config.value);

    agent::AgentRuntime runtime;
    if (cl.command == "setup") return cmd_setup(cl, ctx);
    if (cl.command == "engine") return cmd_engine(cl, ctx);
    if (cl.command == "model") return cmd_model(cl, ctx);
    if (cl.command == "chat") return cmd_chat(cl, ctx);
    if (cl.command == "run") return cmd_run(cl, ctx);
    if (cl.command == "voice") return cmd_voice(cl, ctx);
    std::fprintf(stderr, "agent: unhandled command '%s'\n", cl.command.c_str());
    return 2;
}
