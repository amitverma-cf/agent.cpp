#pragma once

// Skills and MCP servers for someone who never types a command: a built-in list of well-known ones with one line each, what is installed
// (with on / off and remove), and adding by name, package or address. Settings and `agent skill|mcp` share these functions.

#include "commands.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// ---------------------------------------------------------------- MCP servers

struct CuratedMcp {
    std::string_view id;          // the name it gets (letters, digits, - and _)
    std::string_view description; // one line
    std::string_view command;     // npx or uvx
    std::vector<std::string_view> args;
    bool needs_folder = false; // the last argument is a folder the server may use: the project folder unless one is given
    std::string_view secret;   // name of the key the server needs ("" for none)
};
std::span<const CuratedMcp> curated_mcp();

// A server specification typed by a person: "@scope/package" or "package" (npx -y), "npx ARGS...", "uvx ARGS...", any "program ARGS...", or
// an http(s) address. Returns the command and arguments, or the url. A name is derived from the package when `name` is empty.
struct McpSpec {
    std::string name, command, url;
    std::vector<std::string> args;
};
Result<McpSpec> parse_mcp_spec(std::string_view text, std::string_view name = {});

struct McpInstalled {
    std::string name;
    std::string detail; // the command line or address
    std::string where;  // "added here" (mcp.d) or "mcp.json"
    bool enabled = true;
    bool removable = false; // a file of mcp.d: toggled and removed here; servers written into mcp.json by hand are edited there
};
std::vector<McpInstalled> list_mcp(const Context &ctx, const std::string &workspace);

// Each returns a sentence for the person (what happened and what to expect), or an error.
Result<std::string> mcp_add_curated(const Context &ctx, const std::string &workspace, std::string_view id, std::string_view folder = {});
Result<std::string> mcp_add_spec(const Context &ctx, const std::string &workspace, const McpSpec &spec);
// Servers from JSON pasted out of a README: {"mcpServers": {"name": {...}}}, or one entry {"command": "npx", "args": [...], "env": {...}}
// or {"url": "https://...", "headers": {...}} (named by "name", else from its command or address). Keys go in as ${secret:NAME}.
Result<std::string> mcp_add_json_text(const Context &ctx, const std::string &workspace, std::string_view text);
// A server from the official registry: `json` is a complete {"mcpServers": {...}} document (see mcp_entry_for).
Result<std::string> mcp_add_entry_json(const Context &ctx, const std::string &workspace, std::string_view name, const std::string &json);
Result<std::string> mcp_set_enabled(const Context &ctx, const std::string &workspace, std::string_view name, bool enabled);
Result<std::string> mcp_remove_named(const Context &ctx, const std::string &workspace, std::string_view name);

// ---------------------------------------------------------------- skills

struct CuratedSkill {
    std::string_view id; // owner/repo/skill, as `agent skill add` takes it
    std::string_view description;
};
std::span<const CuratedSkill> curated_skills();

struct SkillInstalled {
    std::string name;
    std::string where;       // "every project" or "this project"
    std::string description; // the line after "description:" in SKILL.md
    bool enabled = true;
    bool from_registry = false;
};
std::vector<SkillInstalled> list_skills(const Context &ctx, const std::string &workspace);

// Installs from skills.sh (the repository is read from GitHub). `id` is owner/repo/skill. A skill is text; it is shown nothing here, the
// caller has the user's go-ahead.
// What a person pastes to name a skill, as owner/repo/skill: that, a skills.sh address, or the line `npx skills add owner/repo --skill name`.
std::string skill_id_from_text(std::string_view text);
Result<std::string> skill_install_id(const Context &ctx, const std::string &workspace, std::string_view id, bool project = false);
Result<std::string> skill_set_enabled(const Context &ctx, const std::string &workspace, std::string_view name, bool enabled);
Result<std::string> skill_remove_named(const Context &ctx, const std::string &workspace, std::string_view name);

// The folders skills are read from, global first: <home>/skills and <workspace>/.agents/skills.
std::vector<std::string> skill_dirs(const Context &ctx, const std::string &workspace);

} // namespace agent::cli
