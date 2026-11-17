// Where the CLI keeps the state of a workspace, and how far the changes of a conversation reach.

#include "chat_extras.hpp"
#include "home.hpp"

#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>

using namespace agent;
using namespace agent::cli;
namespace fs = std::filesystem;

namespace {

fs::path fresh_tree(const char *name) {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / "agent_test_workspace" / name;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

void put(const fs::path &p, const std::string &text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}

// Ends the home of a test: later tests (and the rest of the suite) keep their state in the workspace again.
struct HomeForTest {
    explicit HomeForTest(const fs::path &home) { set_state_home(home.string(), (home / "workspaces").string()); }
    ~HomeForTest() { set_state_home({}, {}); }
};

} // namespace

TEST_CASE("workspace: without a home the state stays in the workspace", "[cli][workspace]") {
    const fs::path root = fresh_tree("nohome");
    set_state_home({}, {});
    CHECK(state_dir_for((root / "proj").string()) == (root / "proj").string());
    Config cfg;
    use_workspace(cfg, (root / "proj").string());
    CHECK(cfg.workspace_dir == (root / "proj").string());
    CHECK(cfg.state_dir.empty());
    CHECK(cfg.path_grants.empty());
}

TEST_CASE("workspace: the state of a project folder lives in the home", "[cli][workspace]") {
    const fs::path root = fresh_tree("project");
    const fs::path home = root / "home", proj = root / "code" / "my app";
    fs::create_directories(home / "workspaces" / "default");
    fs::create_directories(proj);
    const HomeForTest scope(home);

    const std::string state = state_dir_for(proj.string());
    CHECK(state != proj.string());
    CHECK(fs::path(state).parent_path() == home / "workspaces");
    CHECK(fs::path(state).filename().string().rfind("my-app-", 0) == 0); // the folder's name, made safe, then the hash
    CHECK(fs::is_directory(state));
    CHECK(fs::exists(fs::path(state) / "project.txt"));
    CHECK(state_dir_for(proj.string()) == state);                      // the same folder every time
    CHECK(state_dir_for((root / "code" / "other").string()) != state); // another project, another folder
    CHECK(state_dir_for((home / "workspaces" / "default").string()) == (home / "workspaces" / "default").string()); // inside the home

    Config cfg;
    use_workspace(cfg, proj.string());
    CHECK(cfg.workspace_dir == proj.string());
    CHECK(cfg.state_dir == state);
    REQUIRE(cfg.path_grants.size() == 2);
    CHECK(cfg.path_grants[0].path == state);
    CHECK(cfg.path_grants[0].consumer == "files");
    CHECK(cfg.path_grants[1].consumer == "run_command");
    CHECK(fs::path(cfg.path_grants[1].path) == fs::path(state) / "sessions");

    const std::string session = session_dir_for(proj.string(), "chat-2026/../x");
    CHECK(fs::path(session).parent_path() == fs::path(state) / "sessions");
    CHECK(fs::path(session).filename() == "chat-2026x"); // nothing of the id leaves the sessions folder
    CHECK(fs::is_directory(session));
    // Nothing was written into the project.
    CHECK(fs::is_empty(proj));
}

TEST_CASE("workspace: what an earlier version left in the project is moved once", "[cli][workspace]") {
    const fs::path root = fresh_tree("adopt");
    const fs::path home = root / "home", proj = root / "proj";
    fs::create_directories(home / "workspaces");
    put(proj / "memory" / "db.sqlite3", "conversations");
    put(proj / "memory" / "index.sqlite3", "index");
    put(proj / ".agents" / "permissions.json", R"({"always":["run_command"]})");
    put(proj / ".agents" / "skills" / "mine" / "SKILL.md", "the user's skill");
    put(proj / "logs" / "engine.log", "log");
    put(proj / "src" / "main.cpp", "int main() {}");
    const HomeForTest scope(home);

    const fs::path state = state_dir_for(proj.string());
    CHECK(fs::exists(state / "memory" / "db.sqlite3"));
    CHECK(fs::exists(state / "memory" / "index.sqlite3"));
    CHECK(fs::exists(state / ".agents" / "permissions.json"));
    CHECK(fs::exists(state / "logs" / "engine.log"));
    CHECK_FALSE(fs::exists(proj / "memory"));
    CHECK_FALSE(fs::exists(proj / "logs"));
    CHECK_FALSE(fs::exists(proj / ".agents" / "permissions.json"));
    // The user's own files stay.
    CHECK(fs::exists(proj / ".agents" / "skills" / "mine" / "SKILL.md"));
    CHECK(fs::exists(proj / "src" / "main.cpp"));
    CHECK(load_always_allowed(proj.string()) == std::vector<std::string>{"run_command"});
}

TEST_CASE("workspace: a logs folder of the project itself is left alone", "[cli][workspace]") {
    const fs::path root = fresh_tree("ownlogs");
    const fs::path home = root / "home", proj = root / "proj";
    fs::create_directories(home / "workspaces");
    put(proj / "logs" / "server-2026.log", "the project's own log");
    put(proj / "logs" / "engine.log", "ours");
    const HomeForTest scope(home);
    const fs::path state = state_dir_for(proj.string());
    CHECK(fs::exists(proj / "logs" / "server-2026.log"));
    CHECK_FALSE(fs::exists(state / "logs" / "server-2026.log"));
}

TEST_CASE("access: the scope a change needs", "[cli][workspace]") {
    const fs::path root = fresh_tree("scope");
    AccessPlaces places;
    places.workspace = (root / "proj").string();
    places.home_workspaces = (root / "home" / "workspaces").string();
    places.session_dir = (root / "home" / "workspaces" / "proj-1234" / "sessions" / "chat-1").string();
    places.other_projects = {(root / "other").string()};
    fs::create_directories(places.workspace);
    fs::create_directories(places.session_dir);
    fs::create_directories(places.other_projects[0]);

    AccessScope needed = AccessScope::Workspace;
    REQUIRE(access_scope_needed(places, (fs::path(places.session_dir) / "draft.md").string(), needed));
    CHECK(needed == AccessScope::Session);
    REQUIRE(access_scope_needed(places, "src/main.cpp", needed)); // relative: in the project
    CHECK(needed == AccessScope::Project);
    REQUIRE(access_scope_needed(places, (root / "proj" / "a" / ".." / "b.txt").string(), needed));
    CHECK(needed == AccessScope::Project);
    REQUIRE(access_scope_needed(places, (root / "other" / "x.txt").string(), needed));
    CHECK(needed == AccessScope::Workspace);
    REQUIRE(access_scope_needed(places, (root / "home" / "workspaces" / "default" / "x.txt").string(), needed));
    CHECK(needed == AccessScope::Workspace);
    // Not one of the three places: the agent never widens to it by itself.
    CHECK_FALSE(access_scope_needed(places, (root / "home" / "engines" / "evil.so").string(), needed));
    CHECK_FALSE(access_scope_needed(places, (root / "elsewhere" / "x").string(), needed));
    CHECK_FALSE(access_scope_needed(places, "../escape.txt", needed));
    CHECK_FALSE(access_scope_needed(places, (root / "project-two" / "x").string(), needed)); // a name that only starts like the project
    // The conversation's folder inside the workspace (a chat in the default workspace) is still the narrowest place.
    AccessPlaces inner;
    inner.workspace = (root / "home" / "workspaces" / "default").string();
    inner.session_dir = (fs::path(inner.workspace) / "sessions" / "chat-9").string();
    REQUIRE(access_scope_needed(inner, "sessions/chat-9/notes.txt", needed));
    CHECK(needed == AccessScope::Session);
    REQUIRE(access_scope_needed(inner, "shared.txt", needed));
    CHECK(needed == AccessScope::Project);
}

TEST_CASE("access: the paths a tool call changes, and the mode names", "[cli][workspace]") {
    CHECK(tool_changed_paths("write_file", R"({"path":"a.txt","content":"x"})") == std::vector<std::string>{"a.txt"});
    CHECK(tool_changed_paths("move_path", R"({"from":"a.txt","to":"b/a.txt"})") == std::vector<std::string>{"a.txt", "b/a.txt"});
    CHECK(tool_changed_paths("read_file", R"({"path":"a.txt"})").empty());         // reading changes nothing
    CHECK(tool_changed_paths("run_command", R"({"command":"rm a.txt"})").empty()); // commands are held by the OS sandbox
    CHECK(tool_changed_paths("write_file", "not json").empty());
    ChatMode mode = ChatMode::Plan;
    CHECK(parse_chat_mode("manual", mode));
    CHECK(mode == ChatMode::Ask);
    CHECK(parse_chat_mode("AUTO", mode));
    CHECK(mode == ChatMode::Auto);
    CHECK(std::string(access_scope_name(AccessScope::Session)) == "session");
    CHECK(std::string(access_scope_name(AccessScope::Workspace)) == "workspace");
}
