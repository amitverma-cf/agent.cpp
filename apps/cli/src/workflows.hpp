#pragma once

// Workflows by name: a graph a person described in words is saved as <home>/graphs/NAME.json, listed, and run later by its name
// (`agent run NAME`) from any chat or folder.

#include "commands.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

struct WorkflowInfo {
    std::string name;
    std::string description;
    std::string path;
    size_t nodes = 0;
    std::string problem; // set when the file does not load
};

// <home>/graphs (the config folder's graphs when there is no home)
std::string workflows_dir(const Context &ctx);

// A name is 1-64 letters, digits, '-' and '_'.
bool valid_workflow_name(std::string_view name);

// Checks `json` as an agent graph and writes it as NAME.json (replacing a workflow of that name). The answer lists the warnings.
Result<std::string> workflow_save(const Context &ctx, std::string_view name, std::string_view json);

std::vector<WorkflowInfo> workflow_list(const Context &ctx);
Result<std::string> workflow_remove(const Context &ctx, std::string_view name);
Result<std::string> workflow_show(const Context &ctx, std::string_view name);

// What `agent run ARG` opens: ARG itself when it is a file; otherwise, for a bare name, <home>/graphs/NAME.json when that exists.
std::string resolve_graph_arg(const Context &ctx, const std::string &arg);

// One line per saved workflow for the system prompt ("daily-report: summarises ..."); empty when there are none.
std::string workflows_prompt(const Context &ctx);

} // namespace agent::cli
