#pragma once

// Private bun and uv installs, so MCP servers and skills written for npx / uvx run without Node, npm or Python on the
// machine. They live in <home>/runtimes/<name>/, are downloaded from the projects' GitHub releases (checked against the
// published SHA-256), and keep every cache inside <home>/cache so nothing lands in the user's own profile.

#include "home.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/hub.hpp>

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

struct RuntimeAsset {
    std::string url;          // the archive
    std::string checksum_url; // a text file that holds its SHA-256
    std::string archive_name; // file name of the archive
    std::string binary;       // the executable inside it (bun, uv; with .exe on Windows)
};

// Pure. name: "bun" or "uv"; version: "latest" or a release tag such as "1.2.3" / "0.5.1"; os: "linux", "macos" or
// "windows"; arch: "x86_64" or "aarch64".
Result<RuntimeAsset> runtime_asset(std::string_view name, std::string_view version, std::string_view os, std::string_view arch);

std::string host_os();   // "linux", "macos" or "windows"
std::string host_arch(); // "x86_64" or "aarch64"

// Pure. The SHA-256 of `file_name` in a checksum file ("<hash>  <name>" lines, or just one hash); empty when absent.
std::string find_sha256(std::string_view text, std::string_view file_name);

// Path of the installed executable, or "" when the runtime is not installed.
std::string runtime_binary_path(const HomeLayout &home, std::string_view name);

struct InstallProgress {
    bool (*on_progress)(uint64_t downloaded, uint64_t total, void *user) = nullptr;
    void *user = nullptr;
};

// Downloads, verifies and unpacks the runtime, replacing an older install. Returns the executable's path.
Result<std::string> install_runtime(const HomeLayout &home, std::string_view name, std::string_view version, const agent::hub::HttpOptions &http,
                                    const InstallProgress &progress = {});

Result<void> remove_runtime(const HomeLayout &home, std::string_view name);

// Environment that keeps bun's and uv's caches, tools and Python installs inside <home>/cache.
std::map<std::string, std::string> runtime_env(const HomeLayout &home);
// Same, with every cache under `cache_root` (a server's own data folder: nothing is shared or written elsewhere).
std::map<std::string, std::string> runtime_env_in(const std::string &cache_root);

// npx / bunx -> `bun x`, uvx -> `uv tool run`, uv / bun -> the private executable, with the cache environment added.
// Leaves the command alone (and says why in `note`) when the runtime is not installed.
struct LauncherRewrite {
    bool rewritten = false;
    std::string note;
};
// `cache_root` empty = <home>/cache. npm / node run on bun too (npm install|add|run|exec map to bun's own commands).
LauncherRewrite rewrite_launcher(McpServerConfig &server, const HomeLayout &home, const std::string &cache_root = {});

} // namespace agent::cli
