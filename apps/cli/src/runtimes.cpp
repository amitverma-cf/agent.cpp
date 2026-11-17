#include "runtimes.hpp"

#include "archive.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <openssl/evp.h>
#include <simdjson.h>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string exe_name(std::string_view name) {
#ifdef _WIN32
    return std::string(name) + ".exe";
#else
    return std::string(name);
#endif
}

// File to look for in <runtimes>/<name>/.
std::string entry_name(std::string_view name) { return name == "pnpm" ? std::string("bin/pnpm.cjs") : exe_name(name); }

// "npx", "npx.cmd", "C:\x\npx.exe" -> "npx"
std::string command_stem(const std::string &command) {
    std::string stem = fs::path(command).filename().string();
    for (const char *ext : {".exe", ".cmd", ".bat"})
        if (lower(stem).ends_with(ext)) stem.resize(stem.size() - std::string_view(ext).size());
    return lower(stem);
}

} // namespace

bool is_runtime_name(std::string_view name) { return name == "bun" || name == "uv" || name == "node" || name == "pnpm" || name == "nub"; }

std::vector<std::string> runtimes_for_js(std::string_view js_runtime) {
    if (js_runtime == "node") return {"node", "pnpm"};
    if (js_runtime == "nub") return {"nub"};
    return {"bun"};
}

std::string runtime_binary_path(const HomeLayout &home, std::string_view name) {
    const fs::path p = fs::path(home.runtimes) / std::string(name) / entry_name(name);
    std::error_code ec;
    return fs::is_regular_file(p, ec) ? p.string() : std::string();
}

std::map<std::string, std::string> runtime_env_in(const std::string &cache_root) {
    const fs::path c(cache_root);
    std::map<std::string, std::string> env = {
        // bun
        {"BUN_INSTALL", (c / "bun" / "install").string()},
        {"BUN_INSTALL_CACHE_DIR", (c / "bun" / "packages").string()},
        {"BUN_RUNTIME_TRANSPILER_CACHE_PATH", (c / "bun" / "transpiler").string()},
        // uv
        {"UV_CACHE_DIR", (c / "uv").string()},
        {"UV_TOOL_DIR", (c / "uv-tools").string()},
        {"UV_TOOL_BIN_DIR", (c / "uv-bin").string()},
        {"UV_PYTHON_INSTALL_DIR", (c / "uv-python").string()},
        // pnpm / node / nub (npm_config_* is how pnpm reads its settings; npm itself is never run)
        {"PNPM_HOME", (c / "pnpm-home").string()},
        {"npm_config_store_dir", (c / "pnpm-store").string()},
        {"npm_config_cache", (c / "npm-cache").string()},
        {"npm_config_update_notifier", "false"},
        {"NODE_COMPILE_CACHE", (c / "node-compile").string()},
        // whatever else wants a home or a temp folder
        {"HOME", (c / "home").string()},
        {"XDG_CACHE_HOME", (c / "xdg" / "cache").string()},
        {"XDG_DATA_HOME", (c / "xdg" / "data").string()},
        {"XDG_STATE_HOME", (c / "xdg" / "state").string()},
        {"XDG_CONFIG_HOME", (c / "xdg" / "config").string()},
        {"TMPDIR", (c / "tmp").string()},
#ifdef _WIN32
        {"USERPROFILE", (c / "home").string()},
        {"APPDATA", (c / "home" / "AppData" / "Roaming").string()},
        {"LOCALAPPDATA", (c / "home" / "AppData" / "Local").string()},
        {"TEMP", (c / "tmp").string()},
        {"TMP", (c / "tmp").string()},
#endif
    };
    return env;
}

LauncherRewrite rewrite_launcher(McpServerConfig &server, const HomeLayout &home, const std::string &js_runtime) {
    LauncherRewrite result;
    const std::string stem = command_stem(server.command);
    const bool js_command =
        stem == "npx" || stem == "bunx" || stem == "npm" || stem == "node" || stem == "bun" || stem == "nub" || stem == "pnpm";
    const bool py_command = stem == "uvx" || stem == "uv";
    if (!js_command && !py_command) return result;
    if (fs::path(server.command).is_absolute() && server.command.starts_with(home.runtimes)) return result; // already rewritten

    auto missing = [&](const std::string &what) {
        result.note = "'" + server.name + "' runs " + server.command + ", but the private " + what +
                      " is not installed; using the system command. Run `agent runtime install " + what + "`.";
        return result;
    };
    std::vector<std::string> args;
    std::string command;
    if (py_command) {
        command = runtime_binary_path(home, "uv");
        if (command.empty()) return missing("uv");
        if (stem == "uvx") args = {"tool", "run"};
        args.insert(args.end(), server.args.begin(), server.args.end());
    } else {
        auto dlx_args = [&](std::vector<std::string> prefix) {
            for (const std::string &a : server.args)
                if (a != "-y" && a != "--yes") prefix.push_back(a); // the runner never asks before installing
            return prefix;
        };
        auto pnpm_style = [&](std::vector<std::string> prefix) { // npm's subcommands in pnpm's words
            std::vector<std::string> out = std::move(prefix);
            for (size_t i = 0; i < server.args.size(); ++i) {
                std::string a = server.args[i];
                if (i == 0) {
                    if (a == "i" || a == "install" || a == "ci") a = "install";
                    else if (a == "run-script") a = "run";
                    else if (a == "exec" || a == "x") a = "dlx";
                }
                out.push_back(std::move(a));
            }
            return out;
        };
        if (js_runtime == "node") {
            const std::string node = runtime_binary_path(home, "node"), pnpm = runtime_binary_path(home, "pnpm");
            if (node.empty()) return missing("node");
            command = node;
            if (stem == "node") args = server.args;
            else if (stem == "npx" || stem == "bunx") {
                if (pnpm.empty()) return missing("pnpm");
                args = dlx_args({pnpm, "dlx"});
            } else if (stem == "npm" || stem == "pnpm") {
                if (pnpm.empty()) return missing("pnpm");
                args = pnpm_style({pnpm});
            } else return result; // bun / nub written explicitly: left alone
        } else if (js_runtime == "nub") {
            command = runtime_binary_path(home, "nub");
            if (command.empty()) return missing("nub");
            if (stem == "npx" || stem == "bunx") args = dlx_args({"dlx"});
            else if (stem == "npm" || stem == "pnpm") args = pnpm_style({});
            else if (stem == "nub") args = server.args;
            else return result; // node / bun written explicitly: left alone
        } else {                // bun
            command = runtime_binary_path(home, "bun");
            if (command.empty()) return missing("bun");
            if (stem == "npx" || stem == "bunx") args = dlx_args({"x"});
            else if (stem == "npm" || stem == "pnpm") {
                args = pnpm_style({});
                if (!args.empty() && args[0] == "dlx") args[0] = "x"; // bun's name for dlx
            } else args = server.args;                                // node, bun
        }
    }
    const std::string cache_root = server.working_directory.empty() ? home.cache : (fs::path(server.working_directory) / "cache").string();
    server.command = command;
    server.args = std::move(args);
    for (const auto &[key, value] : runtime_env_in(cache_root)) server.env.emplace(key, value); // an explicit value in the file wins
    std::error_code ec;
    for (const char *dir : {"home", "tmp", "xdg", "bun", "uv", "pnpm-home", "npm-cache"})
        fs::create_directories(fs::path(cache_root) / dir, ec);
    result.rewritten = true;
    return result;
}

} // namespace agent::cli
