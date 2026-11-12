#pragma once

// Permission gate for paths outside the workspace. A sandboxed tool normally cannot touch them; with a grant, or when the host
// approves a request, it can. Used by the file tools, run_command and (grants only) MCP servers.

#include <agent-cpp/agent.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace agent::tools {

// True when `consumer` may use `path` (write = also modify it): a grant covers it, or the gate allowed it (an AllowAlways
// answer is remembered in the context's grant list for the rest of the session). Never true without a grant or a gate.
bool path_access_allowed(const WorkspaceContext &ctx, std::string_view consumer, const std::filesystem::path &path, bool write);

// Adds the grants that apply to `consumer` to the policy's read-only / read-write lists.
void apply_grants_to_policy(const WorkspaceContext &ctx, std::string_view consumer, SandboxPolicy &policy);

// Absolute paths a shell command line mentions that are neither inside `workspace` nor in the system locations a sandboxed
// program may always read. A heuristic over the text (quotes, redirections and ~ are handled; variables are not).
std::vector<std::filesystem::path> external_paths_in_command(std::string_view command, const std::filesystem::path &workspace);

// True when the command line looks like it changes files (a redirection, cp, mv, rm, mkdir, touch, tee, sed -i, chmod, ...).
bool command_looks_like_write(std::string_view command);

// Grants that apply to a consumer, for code that has no WorkspaceContext (MCP servers).
bool grant_covers(const std::vector<PathGrant> &grants, std::string_view consumer, const std::filesystem::path &path, bool write);

} // namespace agent::tools
