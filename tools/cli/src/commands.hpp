#pragma once

// Subcommands of the `agent` binary. Each returns the process exit code.

#include "cli_core.hpp"

#include <string>
#include <vector>

namespace agent::cli {

// From project(VERSION) in the top-level CMakeLists.txt.
#ifndef AGENT_CLI_VERSION
#define AGENT_CLI_VERSION "0.0.0-unknown"
#endif
inline constexpr std::string_view kCliVersion = AGENT_CLI_VERSION;

struct Context {
    std::string config_dir;
    std::string config_path;  // <config_dir>/config.json
    std::string engines_path; // <config_dir>/engines.json
    CliConfig config;
};

std::string models_dir(const Context &ctx);
std::string catalog_path(const Context &ctx);
std::string workspace_dir(const Context &ctx, std::string_view override_dir = {});
// models.json entries as ModelRefs (empty when there is no catalogue yet).
std::vector<ModelRef> catalog_refs(const Context &ctx);

struct LoadedEngine {
    std::string name;
    std::string path;
    std::string version;
    uint64_t caps = 0;
    std::string isolation; // engines.json "isolation" of the entry, if any
};
// Resolves the engine for a model (engines.json first, then the plugin search dirs) and loads it.
Result<LoadedEngine> load_engine_for(const Context &ctx, std::string_view engine_flag, std::string_view model_path);

// Prints `question [y/N] ` on stderr and reads the answer from stdin. false on EOF / Ctrl-C.
bool ask_yes_no(std::string_view question);

int report_error(const Error &e);

int cmd_setup(const CommandLine &cl, Context &ctx);
int cmd_engine(const CommandLine &cl, Context &ctx);
int cmd_model(const CommandLine &cl, Context &ctx);
int cmd_chat(const CommandLine &cl, Context &ctx);
int cmd_run(const CommandLine &cl, Context &ctx);
int cmd_voice(const CommandLine &cl, Context &ctx); // voice.cpp

} // namespace agent::cli
