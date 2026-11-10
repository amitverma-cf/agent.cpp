#pragma once

// Where agent.cpp keeps its files.
//
//   Global home  ".agent.cpp-workspace": everything shared by all projects, in one folder the user chose at
//                setup. Its location is remembered in a small pointer file in the platform config folder
//                (home.json); $AGENT_HOME and --home override it.
//   Project      ".agents/" in a project folder: that project's skills, MCP servers and private data.
//                The nearest ancestor of the working directory that holds a ".agents" folder is the project.
//
// These functions only compute and create paths; nothing here reads the model or engine files.

#include <agent-cpp/agent.hpp>

#include <string>
#include <string_view>

namespace agent::cli {

inline constexpr std::string_view kHomeFolderName = ".agent.cpp-workspace";
inline constexpr std::string_view kProjectFolderName = ".agents";

struct HomeLayout {
    std::string root;       // <home>
    std::string config;     // <home>/config.json
    std::string engines;    // <home>/engines          engine plugins, one folder per engine and version
    std::string engines_json; // <home>/engines.json
    std::string models;     // <home>/models           model files and models.json
    std::string skills;     // <home>/skills           skills available in every project
    std::string mcp;        // <home>/mcp              MCP server definitions (secrets are NOT stored here)
    std::string plugins;    // <home>/plugins          third-party plugins, each with its own data folder
    std::string runtimes;   // <home>/runtimes         private bun / uv installs
    std::string cache;      // <home>/cache            downloads in progress, prefix KV states, GPU kernel caches
    std::string logs;       // <home>/logs
    std::string workspaces; // <home>/workspaces       default workspace and per-project state outside the project
};

HomeLayout home_layout(std::string_view root);

// <user home>/.agent.cpp-workspace
std::string default_home_suggestion();
// <platform config folder>/home.json
std::string home_pointer_path();

// The home to use: flag, then the AGENT_HOME value, then the one recorded in the pointer file. Empty when none.
std::string resolve_home_root(std::string_view flag, std::string_view env_value, const std::string &pointer_path);

Result<void> save_home_pointer(const std::string &pointer_path, std::string_view root);
Result<void> create_home_layout(const HomeLayout &home);

struct ProjectLayout {
    std::string root;       // the project folder
    std::string agents;     // <root>/.agents
    std::string skills;     // <root>/.agents/skills       committed with the project
    std::string mcp_config; // <root>/.agents/mcp.json     committed with the project (secrets are referenced by name)
    std::string memory;     // <root>/.agents/memory       conversations and memory database, not committed
    std::string cache;      // <root>/.agents/cache        not committed
    std::string local;      // <root>/.agents/local        personal notes and overrides, not committed
};

ProjectLayout project_layout(std::string_view project_root);

// The nearest ancestor of start_dir (inclusive) that contains a ".agents" folder; empty when there is none.
std::string find_project_root(std::string_view start_dir);

// Creates <dir>/.agents with its folders and a .gitignore that keeps memory/, cache/ and local/ out of git.
// Does nothing to files that already exist.
Result<ProjectLayout> init_project(std::string_view dir);

} // namespace agent::cli
