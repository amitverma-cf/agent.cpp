#pragma once

// Private JavaScript and Python tooling, so MCP servers and skills written for npx / uvx run without anything installed
// on the machine. Installs live in <home>/runtimes/<name>/ and are put there by the installer script (scripts/install.sh|ps1,
// pinned in scripts/components.tsv); this file only finds and uses them. Every cache they write stays in the folder of the
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

// Names accepted by install/remove: "bun", "uv", "node", "pnpm", "nub".
bool is_runtime_name(std::string_view name);
// What `agent runtime install js` installs for a js_runtime choice: bun -> {bun}, node -> {node, pnpm}, nub -> {nub}.
std::vector<std::string> runtimes_for_js(std::string_view js_runtime);

// Path of the installed executable (for pnpm: its bin/pnpm.cjs, run with node), or "" when not installed.
std::string runtime_binary_path(const HomeLayout &home, std::string_view name);

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
