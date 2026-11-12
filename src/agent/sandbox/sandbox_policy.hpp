#pragma once

// Per-use default sandbox policies (docs/security-sandbox.md): what the engine host, run_command and
// an opted-in MCP server need on top of the user's Config / McpServerConfig policy. Enforcement is in
// hal/sandbox.hpp; these only fill in the allow-lists.

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>

namespace agent::sandbox {

struct EnginePaths {
    std::string host_path;      // agent-engine-host (bare name = looked up on PATH)
    std::string plugin_path;    // engine plugin shared library
    std::string model;          // Config::model
    std::string engine_options; // Config::engine_options JSON; existing paths in it become read-only
    std::string workspace_dir;  // read-only (media inputs, models under the workspace)
    std::string artifacts_dir;  // read-write (engine outputs)
    std::string scratch_dir;    // read-write temp space for the engine (given to it as "scratch_dir" in its config; TMPDIR)
};

// base (Config::engine_sandbox) plus: host and plugin dirs (and the lib dirs next to the host), the
// model (its directory, also the resolved one when it is a symlink), paths found in engine_options
// read-only; workspace read-only; artifacts read-write.
SandboxPolicy engine_policy(const SandboxPolicy &base, const EnginePaths &paths);

// base (McpServerConfig::sandbox) plus the program's directory (also the resolved one) read-only and
// working_directory read-write.
SandboxPolicy mcp_policy(const SandboxPolicy &base, std::string_view program, std::string_view working_directory);

} // namespace agent::sandbox
