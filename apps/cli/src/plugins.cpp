#include "plugins.hpp"

#include "cli_core.hpp"
#include "runtimes.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

namespace agent::cli {

namespace {

namespace fs = std::filesystem;

std::string quote(const std::string &s) {
#if defined(_WIN32)
    std::string r = "\"";
    for (const char c : s) r += c == '"' ? std::string("\\\"") : std::string(1, c);
    return r + "\"";
#else
    std::string r = "'";
    for (const char c : s) r += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return r + "'";
#endif
}

bool plain_name(std::string_view s) {
    if (s.empty() || s.size() > 48) return false;
    for (const char c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) return false;
    return true;
}

} // namespace

std::vector<PluginTool> load_plugins(const std::vector<std::string> &plugin_dirs, std::vector<std::string> &problems) {
    std::vector<PluginTool> out;
    for (const std::string &root : plugin_dirs) {
        std::error_code ec;
        if (!fs::is_directory(root, ec)) continue;
        for (const auto &entry : fs::directory_iterator(root, ec)) {
            if (!entry.is_directory(ec)) continue;
            const fs::path manifest = entry.path() / "plugin.json";
            if (!fs::is_regular_file(manifest, ec)) continue;
            std::ifstream in(manifest, std::ios::binary);
            const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            simdjson::dom::parser parser;
            simdjson::dom::element doc;
            const std::string where = manifest.string();
            if (parser.parse(body).get(doc)) {
                problems.push_back(where + ": not valid JSON");
                continue;
            }
            std::string_view name, description, runtime, entry_file;
            if (doc["name"].get(name) || !plain_name(name)) {
                problems.push_back(where + ": \"name\" is missing or has other characters than letters, digits, - and _");
                continue;
            }
            if (doc["description"].get(description)) description = "";
            if (doc["runtime"].get(runtime)) runtime = "bun";
            if (runtime != "bun" && runtime != "node" && runtime != "nub" && runtime != "uv" && runtime != "python") {
                problems.push_back(where + ": \"runtime\" must be bun, node, nub, uv or python");
                continue;
            }
            if (doc["entry"].get(entry_file) || entry_file.empty() || entry_file.find("..") != std::string_view::npos) {
                problems.push_back(where + ": \"entry\" (the script inside the plugin folder) is missing");
                continue;
            }
            if (!fs::is_regular_file(entry.path() / std::string(entry_file), ec)) {
                problems.push_back(where + ": the script " + std::string(entry_file) + " is not in the plugin folder");
                continue;
            }
            int64_t timeout = 60;
            if (doc["timeout_seconds"].get(timeout) || timeout < 1 || timeout > 3600) timeout = 60;
            simdjson::dom::array tools;
            if (doc["tools"].get(tools)) {
                problems.push_back(where + ": \"tools\" (an array) is missing");
                continue;
            }
            for (simdjson::dom::element t : tools) {
                std::string_view tool_name, tool_description;
                if (t["name"].get(tool_name) || !plain_name(tool_name)) {
                    problems.push_back(where + ": a tool has no valid \"name\"");
                    continue;
                }
                if (t["description"].get(tool_description)) tool_description = "";
                PluginTool tool;
                tool.plugin = std::string(name);
                tool.tool = std::string(tool_name);
                tool.full_name = tool.plugin + "_" + tool.tool;
                tool.description = std::string(tool_description) + " (plugin " + tool.plugin +
                                   (description.empty() ? std::string() : ": " + std::string(description)) + ")";
                simdjson::dom::element params;
                tool.schema =
                    t["parameters"].get(params) ? std::string("{\"type\":\"object\",\"properties\":{}}") : simdjson::minify(params);
                tool.dir = entry.path().string();
                tool.runtime = std::string(runtime);
                tool.entry = std::string(entry_file);
                tool.timeout_seconds = static_cast<int>(timeout);
                out.push_back(std::move(tool));
            }
        }
    }
    return out;
}

std::vector<std::string> plugin_command(const PluginTool &tool, const HomeLayout &home, std::string &why) {
    auto installed = [&](std::string_view name) { return runtime_binary_path(home, name); };
    if (tool.runtime == "uv" || tool.runtime == "python") {
        if (const std::string uv = installed("uv"); !uv.empty()) return {uv, "run", tool.entry};
        if (tool.runtime == "python") return {"python3", tool.entry};
        why = "uv is not installed; it can be added in Settings (Tools) or with agent runtime install uv";
        return {};
    }
    if (const std::string exe = installed(tool.runtime); !exe.empty()) return {exe, tool.entry};
    if (tool.runtime != "bun")
        if (const std::string bun = installed("bun"); !bun.empty()) return {bun, tool.entry}; // bun runs node scripts too
    why = tool.runtime + " is not installed; it can be added in Settings (Tools) or with agent runtime install " + tool.runtime;
    return {};
}

std::string run_plugin_tool(const PluginTool &tool, const HomeLayout &home, const std::string &arguments_json, const std::string &workspace,
                            bool &failed) {
    failed = true;
    std::string why;
    const std::vector<std::string> argv = plugin_command(tool, home, why);
    if (argv.empty()) return "the plugin " + tool.plugin + " cannot run: " + why;

    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path(ec);
    const std::string stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(++counter);
    const fs::path call_file = tmp / ("agent-plugin-" + stamp + ".json");
    {
        std::ofstream call(call_file, std::ios::binary);
        call << "{\"tool\":\"" << json_escape(tool.tool) << "\",\"arguments\":" << (arguments_json.empty() ? "{}" : arguments_json)
             << ",\"workspace\":\"" << json_escape(workspace) << "\"}\n";
    }
    std::string command;
#if defined(_WIN32)
    command = "cd /d " + quote(tool.dir) + " && ";
#else
    command = "cd " + quote(tool.dir) + " && ";
    if (fs::exists("/usr/bin/timeout", ec) || fs::exists("/bin/timeout", ec))
        command += "timeout " + std::to_string(tool.timeout_seconds) + " ";
#endif
    for (const std::string &a : argv) command += quote(a) + " ";
    command += "< " + quote(call_file.string()) + " 2>&1";
#if defined(_WIN32)
    std::FILE *pipe = _popen(command.c_str(), "r");
#else
    std::FILE *pipe = popen(command.c_str(), "r");
#endif
    if (!pipe) {
        fs::remove(call_file, ec);
        return "the plugin " + tool.plugin + " could not be started";
    }
    std::string output;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        output += buf;
        if (output.size() > 64 * 1024) {
            output.resize(64 * 1024);
            output += "\n[cut]\n";
            break;
        }
    }
#if defined(_WIN32)
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    fs::remove(call_file, ec);
    failed = status != 0;
    if (failed && output.empty())
        output =
            "the plugin " + tool.plugin + " failed without a message (or took longer than " + std::to_string(tool.timeout_seconds) + " s)";
    return output.empty() ? std::string("done") : output;
}

} // namespace agent::cli
