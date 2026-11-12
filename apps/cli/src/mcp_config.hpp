#pragma once

// MCP servers the user configured, in the common "mcpServers" JSON shape so existing configs can be copied in:
//
//   { "mcpServers": { "github": {
//        "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
//        "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${secret:github}" },
//        "sandbox": { "network": "allow", "read": [], "write": [] } } } }
//
// They live in <home>/mcp/mcp.json (every project) and <project>/.agents/mcp.json (one project; a server with the
// same name replaces the global one). Keys are referenced by ${secret:NAME} and resolved only when the server starts.

#include "commands.hpp"

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

struct McpFile {
    std::vector<McpServerConfig> servers;
    std::vector<std::string> warnings;    // entries that were skipped, and why
    std::vector<std::string> own_sandbox; // servers whose entry has a "sandbox" object (it replaces the defaults)
};

// Pure: parses the JSON text. `source` names the file in warnings.
Result<McpFile> parse_mcp_config(std::string_view json, const std::string &source);

// Gives a parsed server the sandbox every third-party process gets by default: it may write only in its own data folder
// (<home>/plugins/<name>/data, or <project>/.agents/local/mcp/<name>), read the private runtimes, and use <home>/cache for
// package caches; the network stays on because most servers talk to a service. An explicit "sandbox" object in the file
// replaces the defaults (so does "sandbox": {"enabled": false}).
void apply_mcp_defaults(McpServerConfig &server, const std::string &data_dir, const std::string &runtimes_dir, const std::string &cache_dir,
                        bool had_sandbox_object);

// The servers of the global home and the project, project entries replacing global ones of the same name. Reads the
// files and applies the defaults; unreadable or invalid entries become warnings.
std::vector<McpServerConfig> collect_mcp_servers(const Context &ctx, std::vector<std::string> &warnings);

// Where `agent mcp add` writes a server: <home>/mcp/mcp.d/<name>.json, or <project>/.agents/mcp.d/<name>.json.
std::string mcp_dropin_path(const Context &ctx, bool project, std::string_view server_name);

} // namespace agent::cli
