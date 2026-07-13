#include <catch_amalgamated.hpp>

#include <agent-cpp/agent.hpp>

TEST_CASE("execute_tool dispatches a registered tool by name", "[tools_ops]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto res = agent::execute_tool(session, "run_command", R"({"command":"echo hi"})");
    REQUIRE(res.ok);
    REQUIRE(res.value.find("\"exit_code\":0") != std::string::npos);
}

TEST_CASE("execute_tool returns KeyNotFound for an unregistered tool", "[tools_ops]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto res = agent::execute_tool(session, "does_not_exist", "{}");
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("rebuild_tool_index picks up tools added after init", "[tools_ops]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    // Not registered yet.
    auto before = agent::execute_tool(session, "custom_echo", "{}");
    REQUIRE_FALSE(before.ok);
    REQUIRE(before.error.code == agent::ErrorCode::KeyNotFound);

    agent::Tool echo_tool;
    echo_tool.name = "custom_echo";
    echo_tool.description = "echoes back a constant";
    echo_tool.parameter_schema = "{}";
    echo_tool.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        return agent::ok(std::string("echo"));
    };
    session.config.tools.push_back(echo_tool);
    agent::rebuild_tool_index(session);

    auto after = agent::execute_tool(session, "custom_echo", "{}");
    REQUIRE(after.ok);
    REQUIRE(after.value == "echo");
}

TEST_CASE("init_tools registers default tools when auto_default_tools is true", "[tools_ops]") {
    auto session_result = agent::init(
        {.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws", .auto_default_tools = true});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    bool found_run_command = false;
    for (const auto &t : session.config.tools)
        if (t.name == "run_command")
            found_run_command = true;
    REQUIRE(found_run_command);
}

TEST_CASE("Disabling auto_default_tools registers no built-in tools", "[tools_ops]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = "/tmp/agent_test_ws",
                                       .auto_default_tools = false});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    REQUIRE(session.config.tools.empty());
}
