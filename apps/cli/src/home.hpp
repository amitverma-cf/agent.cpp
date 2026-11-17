#pragma once

// Where agent.cpp keeps its files.
//
//   Global home  everything agent.cpp keeps, in the one folder chosen when it was installed: the program (bin/), engines/,
//                runtimes/, models/, workspaces/, scripts/, ... Its location is the environment variable AGENT_HOME, which the
//                installer sets; --home overrides it for one run. Nothing else is written outside it.
//   Project      ".agents/" in a project folder: that project's skills, MCP servers and private data.
//                The nearest ancestor of the working directory that holds a ".agents" folder is the project.
//
// These functions only compute and create paths; nothing here reads the model or engine files.

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>

namespace agent::cli {

inline constexpr std::string_view kHomeFolderName = ".agent.cpp";
inline constexpr std::string_view kProjectFolderName = ".agents";

struct HomeLayout {
    std::string root;         // <home>
    std::string config;       // <home>/config.json
    std::string engines;      // <home>/engines          engine plugins, one folder per engine and version
    std::string engines_json; // <home>/engines.json
    std::string models;       // <home>/models           model files and models.json
    std::string skills;       // <home>/skills           skills available in every project
    std::string mcp;          // <home>/mcp              MCP server definitions (secrets are NOT stored here)
    std::string plugins;      // <home>/plugins          third-party plugins, each with its own data folder
    std::string runtimes;     // <home>/runtimes         private bun / uv installs
    std::string cache;        // <home>/cache            downloads in progress, prefix KV states, GPU kernel caches
    std::string logs;         // <home>/logs
    std::string workspaces;   // <home>/workspaces       default workspace and per-project state outside the project
};

HomeLayout home_layout(std::string_view root);

// The folder the running program was installed into: <root>/bin/agent with a ".agent-root" marker in <root>. Empty when the program
// does not run from such an install (a build tree).
std::string install_root();

// The home to use: the flag, then the AGENT_HOME value, then the install folder the program sits in. Empty when none.
std::string resolve_home_root(std::string_view flag, std::string_view env_value);

Result<void> create_home_layout(const HomeLayout &home);

// ---- The state of a workspace
//
// A workspace is the folder the tools work in. When it is a project folder of the user (anywhere outside the home), what the agent
// itself writes about it does not go into that folder but into <home>/workspaces/<folder name>-<hash of its path>/: the conversations
// and notes (memory/), logs/, tmp/, cache/, generated files (artifacts/), saved tool results and the permissions given for good. A
// workspace inside the home (workspaces/default) keeps its state in itself, and so does every workspace when there is no home (the
// layout of --config-dir). What an earlier version left in the project folder is moved over the first time.

// Called once at start: the home and its workspaces folder; empty strings = no home.
void set_state_home(std::string_view home_root, std::string_view workspaces_dir);
// The state folder of a workspace (created). Equal to `workspace` when the state stays in it.
std::string state_dir_for(const std::string &workspace);
// <state>/sessions/<id>/ (created): the folder one conversation works in when it has no project to work on.
std::string session_dir_for(const std::string &workspace, std::string_view session_id);
// Points a Config at a workspace: workspace_dir, state_dir, and the grant that lets the file tools use the state folder.
void use_workspace(Config &cfg, const std::string &workspace);

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
