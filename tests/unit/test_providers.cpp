#include <catch_amalgamated.hpp>

#include <agent-cpp/agent.hpp>
#include <string>

namespace {

std::string g_stream_buffer;
void test_token_callback(std::string_view token, void *) { g_stream_buffer += token; }

} // namespace

TEST_CASE("Mock provider generates a response", "[providers][mock]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::MessageView user_msg;
    user_msg.role = "user";
    user_msg.content = "Hello World";
    user_msg.tool_calls = {};
    user_msg.tool_call_id = "";

    agent::InferRequest request{.messages = std::span<const agent::MessageView>(&user_msg, 1), .stream = false};

    auto generate_result = agent::infer(session, request);
    REQUIRE(generate_result.ok);
    REQUIRE(generate_result.value.message.content == "Mock Response");
    REQUIRE(generate_result.value.usage.prompt_tokens == 2);
    REQUIRE(generate_result.value.usage.completion_tokens == 2);
    REQUIRE(generate_result.value.usage.total_tokens == 4);
}

TEST_CASE("Mock provider streams tokens", "[providers][mock]") {
    g_stream_buffer.clear();
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::MessageView user_msg;
    user_msg.role = "user";
    user_msg.content = "Hello";
    user_msg.tool_calls = {};
    user_msg.tool_call_id = "";

    agent::InferRequest request{.messages = std::span<const agent::MessageView>(&user_msg, 1),
                                .stream = true,
                                .on_token = test_token_callback,
                                .token_user_data = nullptr};

    auto stream_result = agent::infer(session, request);
    REQUIRE(stream_result.ok);
    REQUIRE(g_stream_buffer == "Mock Response Streamed");
}

TEST_CASE("Custom tool registers and dispatches", "[providers][tools]") {
    agent::Tool mock_tool;
    mock_tool.name = "get_weather";
    mock_tool.description = "Get the current weather";
    mock_tool.parameter_schema = "{\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}}}";
    mock_tool.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        return agent::ok(std::string("Sunny, 22C"));
    };

    auto session_result = agent::init(
        {.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws", .tools = {mock_tool}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const agent::Tool *weather_tool = nullptr;
    for (const auto &t : session.config.tools)
        if (t.name == "get_weather") {
            weather_tool = &t;
            break;
        }
    REQUIRE(weather_tool != nullptr);
    REQUIRE(weather_tool->callback != nullptr);

    auto exec_res = weather_tool->callback("{}", nullptr);
    REQUIRE(exec_res.ok);
    REQUIRE(exec_res.value == "Sunny, 22C");
}
