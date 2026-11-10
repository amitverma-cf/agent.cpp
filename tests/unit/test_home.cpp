#include "home.hpp"

#include <catch_amalgamated.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace agent::cli;

namespace {

fs::path temp_root(const char *name) {
    const fs::path p = fs::temp_directory_path() / (std::string("agent-home-test-") + name);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p, ec);
    return p;
}

} // namespace

TEST_CASE("home: layout puts every area under one root", "[home]") {
    const HomeLayout h = home_layout((fs::path("base") / ".agent.cpp-workspace").string());
    const fs::path root = fs::path("base") / ".agent.cpp-workspace";
    CHECK(h.root == root.string());
    CHECK(h.config == (root / "config.json").string());
    CHECK(h.engines == (root / "engines").string());
    CHECK(h.models == (root / "models").string());
    CHECK(h.skills == (root / "skills").string());
    CHECK(h.mcp == (root / "mcp").string());
    CHECK(h.plugins == (root / "plugins").string());
    CHECK(h.runtimes == (root / "runtimes").string());
    CHECK(h.cache == (root / "cache").string());
    CHECK(h.workspaces == (root / "workspaces").string());
    CHECK(fs::path(default_home_suggestion()).filename() == ".agent.cpp-workspace");
}

TEST_CASE("home: flag beats environment beats the pointer file", "[home]") {
    const fs::path dir = temp_root("pointer");
    const std::string pointer = (dir / "home.json").string();
    CHECK(resolve_home_root("", "", pointer).empty()); // nothing configured yet

    REQUIRE(agent::cli::save_home_pointer(pointer, "/from/pointer").ok);
    CHECK(resolve_home_root("", "", pointer) == "/from/pointer");
    CHECK(resolve_home_root("", "/from/env", pointer) == "/from/env");
    CHECK(resolve_home_root("/from/flag", "/from/env", pointer) == "/from/flag");
    fs::remove_all(dir);
}

TEST_CASE("home: the pointer file keeps backslashes and quotes in a path", "[home]") {
    const fs::path dir = temp_root("escape");
    const std::string pointer = (dir / "home.json").string();
    const std::string path = R"(C:\Users\a "b"\.agent.cpp-workspace)";
    REQUIRE(save_home_pointer(pointer, path).ok);
    CHECK(resolve_home_root("", "", pointer) == path);
    fs::remove_all(dir);
}

TEST_CASE("home: creating the layout makes every folder and is repeatable", "[home]") {
    const fs::path dir = temp_root("create");
    const HomeLayout h = home_layout((dir / ".agent.cpp-workspace").string());
    REQUIRE(create_home_layout(h).ok);
    for (const std::string *d : {&h.engines, &h.models, &h.skills, &h.mcp, &h.plugins, &h.runtimes, &h.cache, &h.logs, &h.workspaces})
        CHECK(fs::is_directory(*d));
    REQUIRE(create_home_layout(h).ok); // running setup again changes nothing
    fs::remove_all(dir);
}

TEST_CASE("project: the nearest .agents folder up the tree is the project", "[home]") {
    const fs::path dir = temp_root("project");
    const fs::path deep = dir / "a" / "b" / "c";
    fs::create_directories(deep);
    auto made = init_project(dir.string());
    REQUIRE(made.ok);
    CHECK(fs::is_directory(made.value.skills));
    CHECK(fs::is_directory(made.value.memory));
    CHECK(fs::is_directory(made.value.cache));
    CHECK(fs::is_directory(made.value.local));
    CHECK(fs::exists(fs::path(made.value.agents) / ".gitignore"));

    CHECK(fs::path(find_project_root(deep.string())) == fs::absolute(dir).lexically_normal());
    CHECK(fs::path(find_project_root(dir.string())) == fs::absolute(dir).lexically_normal());

    // A nested project wins over the outer one.
    auto inner = init_project((dir / "a").string());
    REQUIRE(inner.ok);
    CHECK(fs::path(find_project_root(deep.string())) == fs::absolute(dir / "a").lexically_normal());
    fs::remove_all(dir);
}

TEST_CASE("project: init keeps what is already there", "[home]") {
    const fs::path dir = temp_root("keep");
    REQUIRE(init_project(dir.string()).ok);
    const fs::path ignore = dir / ".agents" / ".gitignore";
    {
        std::ofstream out(ignore, std::ios::trunc);
        out << "custom\n";
    }
    REQUIRE(init_project(dir.string()).ok);
    std::string line;
    {
        std::ifstream in(ignore);
        std::getline(in, line);
    } // closed before the folder is removed (Windows will not delete an open file)
    CHECK(line == "custom"); // not overwritten
    std::error_code ec;
    fs::remove_all(dir, ec);
}
