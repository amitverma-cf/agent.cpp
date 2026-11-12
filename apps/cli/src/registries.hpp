#pragma once

// Search and install from the two public catalogues, with our own small clients (no Node, no npx):
//   * skills.sh      https://skills.sh/api/search?q=...        skills live in GitHub repositories
//   * MCP registry   https://registry.modelcontextprotocol.io/v0.1/servers?search=...
//
// The parsing functions are pure so they can be tested without a network. Nothing downloaded here is run: a skill is
// copied to the skills folder as files, and an MCP server becomes a config entry that still goes through the
// secrets and sandbox rules when it starts.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/hub.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// ---------------------------------------------------------------- skills.sh

struct SkillHit {
    std::string id;       // "owner/repo/skill-id"
    std::string source;   // "owner/repo": the GitHub repository
    std::string skill_id; // folder name of the skill
    std::string name;
    uint64_t installs = 0;
};

std::string skill_search_url(std::string_view query);
Result<std::vector<SkillHit>> parse_skill_search(std::string_view json);

// "owner/repo/skill-id" -> its parts; the repository part may not contain extra slashes.
Result<SkillHit> parse_skill_id(std::string_view id);

struct SkillFiles {
    std::string dir;                // folder of the skill inside the repository, "" when it is the repository root
    std::vector<std::string> files; // repository paths of every file in that folder
    uint64_t bytes = 0;             // their total size
};
// `tree_json` is GitHub's git/trees?recursive=1 answer. The skill is the folder with a SKILL.md whose name is
// `skill_id`; when several match, the shortest path wins.
Result<SkillFiles> find_skill_in_tree(std::string_view tree_json, std::string_view skill_id);

struct SkillInstall {
    std::string dir;                  // where the skill was written
    std::vector<std::string> files;   // relative paths
    std::vector<std::string> scripts; // files that are not documentation (they can only run through run_command)
};
// Finds the skill's folder in its GitHub repository (one API call) without downloading it, so the user can be shown
// what would be written. Refuses skills with more than 200 files or 20 MiB.
Result<SkillFiles> plan_skill(const SkillHit &hit, const agent::hub::HttpOptions &http);

// Downloads the planned files into <skills_dir>/<skill>/ and writes .source.json (the repository and a SHA-256 per
// file) next to them. The skill appears only when every file arrived.
Result<SkillInstall> install_skill(const SkillHit &hit, const SkillFiles &files, const std::string &skills_dir,
                                   const agent::hub::HttpOptions &http);

// ---------------------------------------------------------------- MCP registry

struct McpEnvVar {
    std::string name;
    std::string description;
    std::string default_value;
    bool secret = false;
    bool required = false;
};

struct McpPackage {
    std::string type; // "npm", "pypi", "oci", ...
    std::string identifier;
    std::string version;
    std::string transport;              // "stdio", ...
    std::vector<std::string> arguments; // positional package arguments
    std::vector<McpEnvVar> env;
};

struct McpHit {
    std::string name; // "com.example/files"
    std::string description;
    std::string version;
    std::string repository;
    std::vector<McpPackage> packages;
    std::vector<std::string> remotes; // hosted endpoints (not started by us)
};

std::string mcp_search_url(std::string_view query, int limit = 20);
// Keeps only the latest version of each server.
Result<std::vector<McpHit>> parse_mcp_search(std::string_view json);

struct McpEntry {
    std::string server_name;          // key in mcpServers: the last part of the registry name
    std::string json;                 // a complete {"mcpServers": {...}} document
    std::vector<std::string> secrets; // secret names the entry refers to (agent secret set NAME, then bind)
    std::vector<std::string> to_fill; // required settings that have no default (edit the file)
};
// Builds the config for the first package we can start (npm -> npx, pypi -> uvx; `agent runtime` provides both).
Result<McpEntry> mcp_entry_for(const McpHit &hit);

} // namespace agent::cli
