#include "sandbox_policy.hpp"

#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <vector>

namespace agent::sandbox {
namespace {

namespace fs = std::filesystem;

#if defined(_WIN32)
constexpr char kPathListSep = ';';
#else
constexpr char kPathListSep = ':';
#endif

void push_unique(std::vector<std::string> &list, std::string p) {
    if (p.empty()) return;
    for (const auto &e : list)
        if (e == p) return;
    list.push_back(std::move(p));
}

fs::path absolute_or_empty(const fs::path &p) {
    std::error_code ec;
    auto a = fs::absolute(p, ec);
    return ec ? fs::path() : a;
}

// A program name without a directory, looked up on PATH the way spawn_child does.
fs::path resolve_on_path(std::string_view program) {
    fs::path p{std::string(program)};
    if (program.empty() || p.has_parent_path()) return absolute_or_empty(p);
    const char *env = std::getenv("PATH");
    std::string path = env ? env : "";
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find(kPathListSep, start);
        if (end == std::string::npos) end = path.size();
        fs::path candidate = fs::path(path.substr(start, end - start)) / p;
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec)) return absolute_or_empty(candidate);
        start = end + 1;
    }
    return {};
}

// The directory holding file, and the directory of its symlink target when that differs (Landlock
// checks the resolved inode, so a symlinked file needs the target's directory).
void add_file_dirs(std::vector<std::string> &list, const fs::path &file) {
    if (file.empty()) return;
    std::error_code ec;
    fs::path abs = absolute_or_empty(file);
    if (abs.empty() || !fs::exists(abs, ec)) return;
    if (fs::is_directory(abs, ec)) {
        push_unique(list, abs.string());
        return;
    }
    push_unique(list, abs.parent_path().string());
    auto real = fs::canonical(abs, ec);
    if (!ec) push_unique(list, real.parent_path().string());
}

// Every JSON string literal in text (no full parse: the options are engine-defined and only strings
// that name an existing path matter).
std::vector<std::string> json_strings(std::string_view text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '"') continue;
        std::string s;
        size_t j = i + 1;
        for (; j < text.size() && text[j] != '"'; ++j) {
            if (text[j] == '\\' && j + 1 < text.size()) {
                ++j;
                switch (text[j]) {
                case 'n': s.push_back('\n'); break;
                case 't': s.push_back('\t'); break;
                case '/': s.push_back('/'); break;
                case '\\': s.push_back('\\'); break;
                case '"': s.push_back('"'); break;
                default: s.push_back(text[j]); break;
                }
            } else {
                s.push_back(text[j]);
            }
        }
        out.push_back(std::move(s));
        i = j;
    }
    return out;
}

bool looks_like_path(const std::string &s) {
    if (s.empty() || s.size() > 4096) return false;
    if (s.find("://") != std::string::npos) return false;
    return s.find('/') != std::string::npos || s.find('\\') != std::string::npos;
}

} // namespace

SandboxPolicy engine_policy(const SandboxPolicy &base, const EnginePaths &paths) {
    SandboxPolicy p = base;
    if (!p.enabled) return p;

    // The host executable, the libraries next to it (shared libagent builds, installed lib/), the plugin
    // and its bundled runtime libraries ($ORIGIN RPATH: same directory).
    const fs::path host = resolve_on_path(paths.host_path);
    if (!host.empty()) {
        add_file_dirs(p.read_only_paths, host);
        const fs::path prefix = host.parent_path().parent_path();
        // Only directories that exist: a build tree has no ../lib, and warning about it on every
        // host start is noise.
        for (const char *lib : {"lib", "lib64"}) {
            std::error_code ec;
            if (std::filesystem::is_directory(prefix / lib, ec)) push_unique(p.read_only_paths, (prefix / lib).string());
        }
    }
    add_file_dirs(p.read_only_paths, paths.plugin_path);

    // The model: its directory, so multi-file models (projectors, tokenizers, ...) next to it load.
    add_file_dirs(p.read_only_paths, paths.model);
    for (const auto &s : json_strings(paths.engine_options))
        if (looks_like_path(s)) add_file_dirs(p.read_only_paths, s);

    if (!paths.workspace_dir.empty()) push_unique(p.read_only_paths, absolute_or_empty(paths.workspace_dir).string());
    if (!paths.artifacts_dir.empty()) push_unique(p.read_write_paths, absolute_or_empty(paths.artifacts_dir).string());
    // Engine host v2 passes large tensors / media through POSIX shared memory (shm_open lives in
    // /dev/shm). Without access it silently falls back to copying through the pipe.
    {
        std::error_code ec;
        if (fs::is_directory("/dev/shm", ec)) push_unique(p.read_write_paths, "/dev/shm");
    }
    return p;
}

SandboxPolicy mcp_policy(const SandboxPolicy &base, std::string_view program, std::string_view working_directory) {
    SandboxPolicy p = base;
    if (!p.enabled) return p;
    add_file_dirs(p.read_only_paths, resolve_on_path(program));
    if (!working_directory.empty()) push_unique(p.read_write_paths, absolute_or_empty(std::string(working_directory)).string());
    return p;
}

} // namespace agent::sandbox
