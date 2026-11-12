#include "home.hpp"

#include "cli_core.hpp"

#include <simdjson.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string env_value(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

} // namespace

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

std::string default_home_suggestion() {
#if defined(_WIN32)
    std::string base = env_value("USERPROFILE");
#else
    std::string base = env_value("HOME");
#endif
    if (base.empty()) base = fs::current_path().string();
    return (fs::path(base) / std::string(kHomeFolderName)).string();
}

std::string home_pointer_path() { return (fs::path(default_config_dir()) / "home.json").string(); }

std::string resolve_home_root(std::string_view flag, std::string_view env_home, const std::string &pointer_path) {
    if (!flag.empty()) return std::string(flag);
    if (!env_home.empty()) return std::string(env_home);
    std::error_code ec;
    if (pointer_path.empty() || !fs::is_regular_file(pointer_path, ec)) return {};
    std::ifstream in(pointer_path, std::ios::binary);
    if (!in) return {};
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(text).get(doc)) return {};
    std::string_view home;
    if (doc["home"].get(home)) return {};
    return std::string(home);
}

Result<void> save_home_pointer(const std::string &pointer_path, std::string_view root) {
    std::error_code ec;
    const fs::path p(pointer_path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path(), ec);
    const std::string tmp = pointer_path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return fail(ErrorCode::FilesystemError, "cannot write " + tmp);
        out << "{\"home\": \"" << json_escape(root) << "\"}\n";
        if (!out.flush()) return fail(ErrorCode::FilesystemError, "failed writing " + tmp);
    }
    fs::rename(tmp, p, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot replace " + pointer_path + ": " + ec.message());
    return ok();
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
    for (const std::string *d : {&p.agents, &p.skills, &p.memory, &p.cache, &p.local}) {
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
