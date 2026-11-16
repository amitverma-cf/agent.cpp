#pragma once

// Plugins: tools written as a script in a folder, run by a script runtime (bun, node, uv/python) as a child process. The gateway reads
// <home>/plugins/NAME/plugin.json and <workspace>/.agents/plugins/NAME/plugin.json, offers each declared tool to the model, and runs the
// script once per call: the call arrives on stdin as {"tool": "...", "arguments": {...}, "workspace": "..."} and the script prints the
// result (text or JSON) on stdout. A non-zero exit code is an error; what it printed is the message.
//
//   { "name": "weather", "description": "Weather lookups", "runtime": "bun", "entry": "main.js", "timeout_seconds": 60,
//     "tools": [ { "name": "forecast", "description": "...", "parameters": { "type": "object", "properties": { ... } } } ] }

#include "home.hpp"

#include <string>
#include <vector>

namespace agent::cli {

struct PluginTool {
    std::string plugin;      // plugin name
    std::string tool;        // tool name inside the plugin
    std::string full_name;   // "<plugin>_<tool>": what the model calls
    std::string description; // tool description, with the plugin named
    std::string schema;      // JSON schema of the arguments
    std::string dir;         // the plugin folder (the script's working directory)
    std::string runtime;     // bun, node, uv, python
    std::string entry;       // script path inside the folder
    int timeout_seconds = 60;
};

// Every tool of every valid plugin under the given plugin folders. Problems (a bad manifest, an unknown runtime) are added to `problems`
// as one line each; the other plugins still load.
std::vector<PluginTool> load_plugins(const std::vector<std::string> &plugin_dirs, std::vector<std::string> &problems);

// The command line that runs a plugin call: [program, arg, ...]. Empty when the runtime is not installed (`why` says so).
std::vector<std::string> plugin_command(const PluginTool &tool, const HomeLayout &home, std::string &why);

// Runs one call. Returns what the script printed; `failed` is set on a non-zero exit, a timeout or a runtime that is not installed.
std::string run_plugin_tool(const PluginTool &tool, const HomeLayout &home, const std::string &arguments_json, const std::string &workspace,
                            bool &failed);

} // namespace agent::cli
