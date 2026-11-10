// agent runtime: the private bun and uv installs that MCP servers and skills use instead of Node / Python.

#include "commands.hpp"
#include "doctor.hpp"
#include "runtimes.hpp"

#include <agent-cpp/hub.hpp>

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

int cmd_runtime(const CommandLine &cl, Context &ctx) {
    if (!ctx.has_home) {
        std::fprintf(stderr, "agent runtime: no home is set. Run `agent setup` to choose one.\n");
        return 1;
    }
    const std::string &sub = cl.subcommand;
    if (sub == "ls" || sub.empty()) {
        for (const char *name : {"bun", "uv"}) {
            const std::string path = runtime_binary_path(ctx.home, name);
            if (path.empty()) std::printf("%-4s not installed   (agent runtime install %s)\n", name, name);
            else std::printf("%-4s %-10s %s\n", name, installed_version(ctx.home, name).c_str(), path.c_str());
        }
        std::printf("\nnpx / bunx run through bun x, and uvx through uv tool run, once the runtime is installed. Their caches stay in %s\n", ctx.home.cache.c_str());
        return 0;
    }
    if (cl.positionals.empty()) {
        std::fprintf(stderr, "agent runtime %s: name a runtime (bun or uv)\n", sub.c_str());
        return 2;
    }
    const std::string name = cl.positionals[0];
    if (name != "bun" && name != "uv") {
        std::fprintf(stderr, "agent runtime: unknown runtime '%s' (bun or uv)\n", name.c_str());
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
    if (sub == "install") {
        std::error_code ec;
        fs::create_directories(ctx.home.runtimes, ec);
        std::fprintf(stderr, "downloading %s %s ...\n", name.c_str(), cl.get("version", "latest").empty() ? "latest" : std::string(cl.get("version", "latest")).c_str());
        InstallProgress progress;
        progress.on_progress = print_progress;
        auto r = install_runtime(ctx.home, name, cl.get("version", "latest"), agent::hub::http_options_from_env(), progress);
        std::fprintf(stderr, "\n");
        if (!r.ok) return report_error(r.error);
        std::printf("installed %s at %s\n", name.c_str(), r.value.c_str());
        return 0;
    }
    std::fprintf(stderr, "agent runtime: unknown subcommand '%s'\n", sub.c_str());
    return 2;
}

} // namespace agent::cli
