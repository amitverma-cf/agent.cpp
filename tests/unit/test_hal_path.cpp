#include "hal/path.hpp"

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <string>

namespace {

std::string make_workspace(const char *name) {
    std::string dir = "/tmp/agent_test_halpath_" + std::string(name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

} // namespace

TEST_CASE("resolve_under_workspace accepts a relative path inside the workspace", "[hal][path]") {
    std::string ws = make_workspace("valid");
    auto res = agent::hal::resolve_under_workspace(ws, "sub/file.txt", true);
    REQUIRE(res.ok);
    REQUIRE(res.value.string().find(ws) == 0);
}

TEST_CASE("resolve_under_workspace rejects .. traversal that escapes the workspace", "[hal][path][sandbox]") {
    std::string ws = make_workspace("traversal");
    auto res = agent::hal::resolve_under_workspace(ws, "../../etc/passwd", true);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::SandboxViolation);
}

TEST_CASE("resolve_under_workspace rejects an absolute path outside the workspace", "[hal][path][sandbox]") {
    std::string ws = make_workspace("absolute");
    auto res = agent::hal::resolve_under_workspace(ws, "/etc/passwd", true);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::SandboxViolation);
}

TEST_CASE("resolve_under_workspace allows escapes when sandbox is disabled", "[hal][path]") {
    std::string ws = make_workspace("nosandbox");
    auto res = agent::hal::resolve_under_workspace(ws, "../../etc/passwd", false);
    REQUIRE(res.ok);
}

TEST_CASE("resolve_under_workspace rejects an empty workspace_dir", "[hal][path]") {
    auto res = agent::hal::resolve_under_workspace("", "file.txt", true);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("is_workspace_root identifies the workspace root and rejects other paths", "[hal][path]") {
    std::string ws = make_workspace("root");
    REQUIRE(agent::hal::is_workspace_root(ws, ws));
    REQUIRE_FALSE(agent::hal::is_workspace_root(ws, ws + "/subdir"));
}

TEST_CASE("is_safe_identifier validates SQL-identifier-shaped strings", "[hal][path]") {
    REQUIRE(agent::hal::is_safe_identifier("valid_name"));
    REQUIRE(agent::hal::is_safe_identifier("_leading_underscore"));
    REQUIRE(agent::hal::is_safe_identifier("Name123"));

    REQUIRE_FALSE(agent::hal::is_safe_identifier(""));
    REQUIRE_FALSE(agent::hal::is_safe_identifier("123leading_digit"));
    REQUIRE_FALSE(agent::hal::is_safe_identifier("has space"));
    REQUIRE_FALSE(agent::hal::is_safe_identifier("has-dash"));
    REQUIRE_FALSE(agent::hal::is_safe_identifier("has;semicolon"));
    REQUIRE_FALSE(agent::hal::is_safe_identifier(std::string(65, 'a')));
}
