#pragma once

// Private JavaScript and Python tooling, so MCP servers and skills written for npx / uvx run without anything installed
// on the machine. Installs live in <home>/runtimes/<name>/, are downloaded from the projects' releases and checked against
// the published SHA-256 (SHA-512 integrity for the pnpm package), and every cache they write stays in the folder of the
// server that runs them.
//
// JavaScript: the user picks one in `agent setup` (CliConfig::js_runtime):
//   bun   bun x / bun install / bun run
//   node  a private Node.js with pnpm (pnpm dlx / install / run). npm is never used.
//   nub   the nub toolkit (a pnpm-compatible CLI over Node.js)
// Python: uv (uvx -> uv tool run).

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
    std::string archive_name; // file name of the archive; empty for node (found in the checksum file)
    std::string binary;       // the executable inside it
};

// Pure. name: "bun", "uv", "nub" or "node"; version: "latest" or a release tag such as "1.2.3"; os: "linux", "macos" or
// "windows"; arch: "x86_64" or "aarch64".
Result<RuntimeAsset> runtime_asset(std::string_view name, std::string_view version, std::string_view os, std::string_view arch);

std::string host_os();   // "linux", "macos" or "windows"
std::string host_arch(); // "x86_64" or "aarch64"

// Pure. The SHA-256 of `file_name` in a checksum file ("<hash>  <name>" lines, or just one hash); empty when absent.
std::string find_sha256(std::string_view text, std::string_view file_name);
// Pure. From Node's SHASUMS256.txt: the archive for this platform, e.g. "node-v24.22.2-linux-x64.tar.gz".
std::string find_node_archive(std::string_view shasums, std::string_view os, std::string_view arch);

// Names accepted by install/remove: "bun", "uv", "node", "pnpm", "nub".
bool is_runtime_name(std::string_view name);
// What `agent runtime install js` installs for a js_runtime choice: bun -> {bun}, node -> {node, pnpm}, nub -> {nub}.
std::vector<std::string> runtimes_for_js(std::string_view js_runtime);

// Path of the installed executable (for pnpm: its bin/pnpm.cjs, run with node), or "" when not installed.
std::string runtime_binary_path(const HomeLayout &home, std::string_view name);

struct InstallProgress {
    bool (*on_progress)(uint64_t downloaded, uint64_t total, void *user) = nullptr;
    void *user = nullptr;
};

// Downloads, verifies and unpacks the runtime, replacing an older install. Returns the executable's path.
Result<std::string> install_runtime(const HomeLayout &home, std::string_view name, std::string_view version, const agent::hub::HttpOptions &http,
                                    const InstallProgress &progress = {});

Result<void> remove_runtime(const HomeLayout &home, std::string_view name);

// Environment that keeps every cache, tool, Python install, HOME and temp folder under `cache_root` (a server's own data
// folder: nothing is shared with other servers or written outside it).
std::map<std::string, std::string> runtime_env_in(const std::string &cache_root);
std::map<std::string, std::string> runtime_env(const HomeLayout &home); // cache_root = <home>/cache

struct LauncherRewrite {
    bool rewritten = false;
    std::string note;
};
// Rewrites npx / bunx / npm / node / uvx / uv / bun to the private runtime for `js_runtime` ("bun", "node" or "nub"),
// adding the contained environment. Safe to call twice. Leaves the command alone, with a note, when the runtime is not
// installed. The cache folder is <server.working_directory>/cache, else <home>/cache.
LauncherRewrite rewrite_launcher(McpServerConfig &server, const HomeLayout &home, const std::string &js_runtime = "bun");

} // namespace agent::cli
