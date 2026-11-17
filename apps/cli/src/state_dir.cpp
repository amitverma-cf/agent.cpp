// Where the agent keeps what it writes about a workspace (see home.hpp). A project folder of the user holds only the user's files: the
// conversations, logs, caches and generated files of that project live under <home>/workspaces/<name>-<hash>/.

#include "home.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace agent::cli {
namespace fs = std::filesystem;

namespace {

std::mutex g_mutex;
std::string g_home;                                   // the home folder; empty = no home, the state stays in the workspace
std::string g_workspaces;                             // <home>/workspaces
std::unordered_map<std::string, std::string> g_known; // workspace as given -> its state folder

fs::path canonical(const fs::path &p) {
    std::error_code ec;
    fs::path full = fs::weakly_canonical(fs::absolute(p, ec), ec);
    if (ec || full.empty()) full = fs::absolute(p, ec);
    return full.lexically_normal();
}

bool is_under(const fs::path &path, const fs::path &folder) {
    const fs::path rel = path.lexically_relative(folder);
    return !rel.empty() && rel.native().rfind(fs::path("..").native(), 0) != 0;
}

// The same folder always gets the same name: its own name, then eight hex digits of a hash of its full path.
std::string folder_key(const fs::path &workspace) {
    std::string text = workspace.generic_string();
#if defined(_WIN32)
    for (char &c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); // C:\Work and c:\work are one folder
#endif
    uint64_t hash = 1469598103934665603ULL; // FNV-1a
    for (const unsigned char c : text) hash = (hash ^ c) * 1099511628211ULL;
    std::string name;
    for (const char c : workspace.filename().string()) {
        const bool plain = std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.';
        if (name.size() < 40) name += plain ? c : '-';
    }
    if (name.empty() || name == "." || name == "..") name = "folder";
    char hex[9];
    std::snprintf(hex, sizeof(hex), "%08x", static_cast<unsigned>(hash & 0xffffffffULL));
    return name + "-" + hex;
}

// Moves `from` to `to` unless `to` is already there. False when it could not be moved (the file is open in another program).
bool move_over(const fs::path &from, const fs::path &to) {
    std::error_code ec;
    if (!fs::exists(from, ec)) return true;
    if (fs::exists(to, ec)) return true; // the state folder has its own already: the old one is left where it is
    fs::create_directories(to.parent_path(), ec);
    fs::rename(from, to, ec);
    if (!ec) return true;
    // Another drive: copy, then remove the original.
    std::error_code copy_ec;
    fs::copy(from, to, fs::copy_options::recursive, copy_ec);
    if (copy_ec) {
        std::error_code undo;
        fs::remove_all(to, undo);
        return false;
    }
    fs::remove_all(from, ec);
    return true;
}

// Only files this program writes into a logs folder: a project's own logs folder is left alone.
bool only_agent_logs(const fs::path &dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return false;
    for (const auto &entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name != "engine.log" && name != "chat.log" && name.rfind("agent_", 0) != 0) return false;
    }
    return true;
}

// What earlier versions wrote into the project folder goes to the state folder once. False when the conversations could not be moved:
// the project folder is then used as before, so no conversation is hidden.
bool adopt_old_state(const fs::path &workspace, const fs::path &state) {
    std::error_code ec;
    if (fs::exists(workspace / "memory" / "db.sqlite3", ec) && !fs::exists(state / "memory" / "db.sqlite3", ec)) {
        fs::remove(state / "memory", ec); // an empty folder made by a run that found nothing to move
        if (!move_over(workspace / "memory", state / "memory")) return false;
    }
    move_over(workspace / ".agents" / "permissions.json", state / ".agents" / "permissions.json");
    move_over(workspace / ".agents" / "results", state / ".agents" / "results");
    move_over(workspace / "waiting.json", state / "waiting.json");
    if (only_agent_logs(workspace / "logs")) move_over(workspace / "logs", state / "logs");
    return true;
}

} // namespace

void set_state_home(std::string_view home_root, std::string_view workspaces_dir) {
    const std::lock_guard<std::mutex> lock(g_mutex);
    g_home = std::string(home_root);
    g_workspaces = std::string(workspaces_dir);
    g_known.clear();
}

std::string state_dir_for(const std::string &workspace) {
    const std::lock_guard<std::mutex> lock(g_mutex);
    if (g_workspaces.empty() || workspace.empty()) return workspace;
    if (const auto it = g_known.find(workspace); it != g_known.end()) return it->second;
    const fs::path ws = canonical(workspace);
    std::string result = workspace;
    if (!is_under(ws, canonical(g_home)) && ws != canonical(g_home)) {
        const fs::path state = fs::path(g_workspaces) / folder_key(ws);
        std::error_code ec;
        fs::create_directories(state, ec);
        if (!ec && adopt_old_state(ws, state)) {
            // Which project this folder belongs to, for a person looking into the home.
            if (!fs::exists(state / "project.txt", ec)) std::ofstream(state / "project.txt", std::ios::binary) << ws.string() << "\n";
            result = state.string();
        }
    }
    g_known.emplace(workspace, result);
    return result;
}

std::string session_dir_for(const std::string &workspace, std::string_view session_id) {
    std::string name;
    for (const char c : session_id)
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') name += c;
    if (name.empty()) name = "session";
    const fs::path dir = fs::path(state_dir_for(workspace)) / "sessions" / name;
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir.string();
}

void use_workspace(Config &cfg, const std::string &workspace) {
    cfg.workspace_dir = workspace;
    const std::string state = state_dir_for(workspace);
    if (state == workspace) {
        cfg.state_dir.clear();
        return;
    }
    cfg.state_dir = state;
    // The file tools read what the library saved there (a long tool result, a generated file) as they would in the workspace.
    const bool granted = std::any_of(cfg.path_grants.begin(), cfg.path_grants.end(), [&](const PathGrant &g) { return g.path == state; });
    if (!granted) {
        cfg.path_grants.push_back({.path = state, .write = true, .consumer = "files"});
        // Commands may work in the conversations' own folders (sessions/<id>/), not in the rest of the state.
        cfg.path_grants.push_back({.path = (fs::path(state) / "sessions").string(), .write = true, .consumer = "run_command"});
    }
}

} // namespace agent::cli
