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
    const HomeLayout h = home_layout((fs::path("base") / ".agent.cpp").string());
    const fs::path root = fs::path("base") / ".agent.cpp";
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
}

TEST_CASE("home: the flag beats the environment, and a build tree has no home of its own", "[home]") {
    CHECK((resolve_home_root("", "").empty() || !install_root().empty())); // nothing configured, nothing guessed
    CHECK(resolve_home_root("", "/from/env") == "/from/env");
    CHECK(resolve_home_root("/from/flag", "/from/env") == "/from/flag");
}

TEST_CASE("home: creating the layout makes every folder and is repeatable", "[home]") {
    const fs::path dir = temp_root("create");
    const HomeLayout h = home_layout((dir / ".agent.cpp").string());
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
    CHECK_FALSE(fs::exists(made.value.memory)); // the agent's own data about a project is kept in the home
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
