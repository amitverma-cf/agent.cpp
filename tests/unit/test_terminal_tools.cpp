#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <string>

namespace {

std::string make_workspace(const char *name) {
    std::string dir = "/tmp/agent_test_term_" + std::string(name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

const agent::Tool *find_tool(const agent::Session &session, std::string_view name) {
    for (const auto &t : session.config.tools)
        if (t.name == name) return &t;
    return nullptr;
}

} // namespace

TEST_CASE("run_command executes and captures output + exit code", "[terminal_tools]") {
    std::string ws = make_workspace("basic");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
    REQUIRE(run_tool != nullptr);

    auto res = run_tool->callback(R"({"command":"echo hello"})", run_tool->user_data);
    REQUIRE(res.ok);
    REQUIRE(res.value.find("\"exit_code\":0") != std::string::npos);
    REQUIRE(res.value.find("hello") != std::string::npos);
}

TEST_CASE("run_command reports a nonzero exit code", "[terminal_tools]") {
    std::string ws = make_workspace("exitcode");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
    auto res = run_tool->callback(R"({"command":"exit 3"})", run_tool->user_data);
    REQUIRE(res.ok);
    REQUIRE(res.value.find("\"exit_code\":3") != std::string::npos);
}

TEST_CASE("run_command enforces a timeout", "[terminal_tools]") {
    std::string ws = make_workspace("timeout");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
#ifdef _WIN32
    auto res = run_tool->callback(R"({"command":"ping -n 10 127.0.0.1 > NUL","timeout_seconds":1})", run_tool->user_data);
#else
    auto res = run_tool->callback(R"({"command":"sleep 10","timeout_seconds":1})", run_tool->user_data);
#endif
    REQUIRE(res.ok);
    REQUIRE(res.value.find("\"timed_out\":true") != std::string::npos);
}

TEST_CASE("run_command requires a 'command' field", "[terminal_tools]") {
    std::string ws = make_workspace("missing_cmd");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
    auto res = run_tool->callback(R"({})", run_tool->user_data);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::InvalidConfig);
}
