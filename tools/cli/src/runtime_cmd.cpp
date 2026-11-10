// agent runtime: the private bun and uv installs that MCP servers and skills use instead of Node / Python.

#include "commands.hpp"
#include "doctor.hpp"
#include "runtimes.hpp"

#include <agent-cpp/hub.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

bool print_progress(uint64_t done, uint64_t total, void *) {
    if (total) std::fprintf(stderr, "\r  %s of %s   ", format_bytes(done).c_str(), format_bytes(total).c_str());
    return true;
}

std::string installed_version(const HomeLayout &home, const std::string &name) {
    std::ifstream in(fs::path(home.runtimes) / name / "runtime.json");
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t at = text.find("\"version\":\"");
    if (at == std::string::npos) return "?";
    const size_t start = at + 11;
    return text.substr(start, text.find('"', start) - start);
}

} // namespace

void install_cli_hooks(Config &cfg, Context &ctx) {
    cfg.mcp_launcher = [](McpServerConfig &server, void *user) {
        auto *c = static_cast<Context *>(user);
        if (c->has_home) rewrite_launcher(server, c->home, c->config.js_runtime);
    };
    cfg.mcp_launcher_user_data = &ctx;
}

int cmd_runtime(const CommandLine &cl, Context &ctx) {
    if (!ctx.has_home) {
        std::fprintf(stderr, "agent runtime: no home is set. Run `agent setup` to choose one.\n");
        return 1;
    }
    const std::string &sub = cl.subcommand;
    if (sub == "ls" || sub.empty()) {
        const auto chosen = runtimes_for_js(ctx.config.js_runtime);
        for (const char *name : {"bun", "node", "pnpm", "nub", "uv"}) {
            const std::string path = runtime_binary_path(ctx.home, name);
            const bool selected = std::find(chosen.begin(), chosen.end(), name) != chosen.end();
            if (path.empty()) std::printf("%-5s not installed   (agent runtime install %s)%s\n", name, name, selected ? "   <- needed for your JavaScript choice" : "");
            else std::printf("%-5s %-10s %s%s\n", name, installed_version(ctx.home, name).c_str(), path.c_str(), selected ? "   <- selected" : "");
        }
        std::printf("\nJavaScript tooling: %s (agent runtime use bun|node|nub). npx / bunx / npm / node run on it, uvx on uv, once installed.\nEvery server keeps its caches, HOME and temp files in its own data folder.\n", ctx.config.js_runtime.c_str());
        return 0;
    }
    if (cl.positionals.empty()) {
        std::fprintf(stderr, "agent runtime %s: name a runtime (bun, node, pnpm, nub, uv or js)\n", sub.c_str());
        return 2;
    }
    const std::string name = cl.positionals[0];
    if (sub == "use") { // choose the JavaScript tooling and install what it needs
        if (name != "bun" && name != "node" && name != "nub") {
            std::fprintf(stderr, "agent runtime use: choose bun, node (with pnpm) or nub\n");
            return 2;
        }
        ctx.config.js_runtime = name;
        if (auto r = save_config(ctx.config_path, ctx.config); !r.ok) return report_error(r.error);
        std::printf("JavaScript tooling is now %s\n", name.c_str());
        std::vector<std::string> need;
        for (const std::string &n : runtimes_for_js(name))
            if (runtime_binary_path(ctx.home, n).empty()) need.push_back(n);
        for (const std::string &n : need) {
            std::fprintf(stderr, "downloading %s ...\n", n.c_str());
            InstallProgress progress;
            progress.on_progress = print_progress;
            auto r = install_runtime(ctx.home, n, "latest", agent::hub::http_options_from_env(), progress);
            std::fprintf(stderr, "\n");
            if (!r.ok) return report_error(r.error);
            std::printf("installed %s at %s\n", n.c_str(), r.value.c_str());
        }
        return 0;
    }
    if (name != "js" && !is_runtime_name(name)) {
        std::fprintf(stderr, "agent runtime: unknown runtime '%s' (bun, node, pnpm, nub, uv or js)\n", name.c_str());
        return 2;
    }
    if (sub == "path") {
        const std::string path = runtime_binary_path(ctx.home, name);
        if (path.empty()) {
            std::fprintf(stderr, "%s is not installed\n", name.c_str());
            return 1;
        }
        std::printf("%s\n", path.c_str());
        return 0;
    }
    if (sub == "rm") {
        auto r = remove_runtime(ctx.home, name);
        if (!r.ok) return report_error(r.error);
        std::printf("removed %s\n", name.c_str());
        return 0;
    }
    if (sub == "install") { // "js" = whatever the chosen JavaScript tooling needs
        const std::vector<std::string> names = name == "js" ? runtimes_for_js(ctx.config.js_runtime) : std::vector<std::string>{name};
        for (const std::string &n : names) {
            std::fprintf(stderr, "downloading %s %s ...\n", n.c_str(), std::string(cl.get("version", "latest")).c_str());
            InstallProgress progress;
            progress.on_progress = print_progress;
            auto r = install_runtime(ctx.home, n, cl.get("version", "latest"), agent::hub::http_options_from_env(), progress);
            std::fprintf(stderr, "\n");
            if (!r.ok) return report_error(r.error);
            std::printf("installed %s at %s\n", n.c_str(), r.value.c_str());
        }
        return 0;
    }
    std::fprintf(stderr, "agent runtime: unknown subcommand '%s'\n", sub.c_str());
    return 2;
}

} // namespace agent::cli
