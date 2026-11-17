#include "home.hpp"

#include "cli_core.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {} // namespace

HomeLayout home_layout(std::string_view root) {
    const fs::path r{std::string(root)};
    HomeLayout h;
    h.root = r.string();
    h.config = (r / "config.json").string();
    h.engines = (r / "engines").string();
    h.engines_json = (r / "engines.json").string();
    h.models = (r / "models").string();
    h.skills = (r / "skills").string();
    h.mcp = (r / "mcp").string();
    h.plugins = (r / "plugins").string();
    h.runtimes = (r / "runtimes").string();
    h.cache = (r / "cache").string();
    h.logs = (r / "logs").string();
    h.workspaces = (r / "workspaces").string();
    return h;
}

std::string install_root() {
    const std::string exe_dir = executable_dir();
    if (exe_dir.empty()) return {};
    const fs::path bin(exe_dir);
    if (bin.filename() != "bin") return {};
    std::error_code ec;
    const fs::path root = bin.parent_path();
    return fs::is_regular_file(root / ".agent-root", ec) ? root.string() : std::string();
}

std::string resolve_home_root(std::string_view flag, std::string_view env_home) {
    if (!flag.empty()) return std::string(flag);
    if (!env_home.empty()) return std::string(env_home);
    return install_root(); // started by its full path before AGENT_HOME reached this shell
}

Result<void> create_home_layout(const HomeLayout &home) {
    std::error_code ec;
    for (const std::string *dir : {&home.root, &home.engines, &home.models, &home.skills, &home.mcp, &home.plugins, &home.runtimes,
                                   &home.cache, &home.logs, &home.workspaces}) {
        fs::create_directories(*dir, ec);
        if (ec) return fail(ErrorCode::FilesystemError, "cannot create " + *dir + ": " + ec.message());
    }
    return ok();
}

ProjectLayout project_layout(std::string_view project_root) {
    const fs::path r{std::string(project_root)};
    const fs::path a = r / std::string(kProjectFolderName);
    ProjectLayout p;
    p.root = r.string();
    p.agents = a.string();
    p.skills = (a / "skills").string();
    p.mcp_config = (a / "mcp.json").string();
    p.memory = (a / "memory").string();
    p.cache = (a / "cache").string();
    p.local = (a / "local").string();
    return p;
}

std::string find_project_root(std::string_view start_dir) {
    std::error_code ec;
    fs::path dir = fs::absolute(fs::path(std::string(start_dir)), ec).lexically_normal();
    if (ec) return {};
    for (;;) {
        if (fs::is_directory(dir / std::string(kProjectFolderName), ec)) return dir.string();
        const fs::path parent = dir.parent_path();
        if (parent == dir || parent.empty()) return {};
        dir = parent;
    }
}

Result<ProjectLayout> init_project(std::string_view dir) {
    std::error_code ec;
    const fs::path abs = fs::absolute(fs::path(std::string(dir)), ec).lexically_normal();
    if (ec) return fail<ProjectLayout>(ErrorCode::FilesystemError, "cannot resolve " + std::string(dir) + ": " + ec.message());
    ProjectLayout p = project_layout(abs.string());
    // Only what the owner of the project writes: the agent's own data about the project is in the home (state_dir_for).
    for (const std::string *d : {&p.agents, &p.skills}) {
        fs::create_directories(*d, ec);
        if (ec) return fail<ProjectLayout>(ErrorCode::FilesystemError, "cannot create " + *d + ": " + ec.message());
    }
    const fs::path ignore = fs::path(p.agents) / ".gitignore";
    if (!fs::exists(ignore, ec)) {
        std::ofstream out(ignore, std::ios::binary);
        if (!out) return fail<ProjectLayout>(ErrorCode::FilesystemError, "cannot write " + ignore.string());
        out << "# Private to this machine: conversations, caches, personal notes.\nmemory/\ncache/\nlocal/\n";
    }
    return ok(std::move(p));
}

} // namespace agent::cli
