#include "../test_assert.hpp"

#include <agent-cpp/agent.hpp>
#include <iostream>
#include <string>

static std::string g_stream_buffer;
static void test_token_callback(std::string_view token, void *) {
    g_stream_buffer += token;
}

void test_provider_mock_generation() {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock});
    TEST_ASSERT(session_result.ok && "Mock initialization failed");
    agent::Session session = std::move(session_result.value);

    agent::MessageView user_msg;
    user_msg.role = "user";
    user_msg.content = "Hello World";
    user_msg.tool_calls = {};
    user_msg.tool_call_id = "";

    agent::ChatRequest request{.messages = std::span<const agent::MessageView>(&user_msg, 1), .stream = false};

    auto generate_result = agent::execute_turn(session, request);
    TEST_ASSERT(generate_result.ok && "Mock generate failed");
    TEST_ASSERT(generate_result.value.message.content == "Mock Response" && "Mock returned wrong string");
    TEST_ASSERT(generate_result.value.usage.prompt_tokens == 2);
    TEST_ASSERT(generate_result.value.usage.completion_tokens == 2);
    TEST_ASSERT(generate_result.value.usage.total_tokens == 4);

    std::cout << "  [PASS] Unit: Providers - Mock Generation\n";
}

void test_provider_mock_streaming() {
    g_stream_buffer.clear();
    auto session_result = agent::init({.provider = agent::AiProvider::Mock});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::MessageView user_msg;
    user_msg.role = "user";
    user_msg.content = "Hello";
    user_msg.tool_calls = {};
    user_msg.tool_call_id = "";

    agent::ChatRequest request{.messages = std::span<const agent::MessageView>(&user_msg, 1),
                               .stream = true,
                               .on_token = test_token_callback,
                               .token_user_data = nullptr};

    auto stream_result = agent::execute_turn(session, request);
    TEST_ASSERT(stream_result.ok && "Mock stream failed");
    TEST_ASSERT(g_stream_buffer == "Mock Response Streamed" && "Stream buffer mismatch");

    std::cout << "  [PASS] Unit: Providers - Mock Streaming\n";
}

void test_tool_configuration() {
    agent::Tool mock_tool;
    mock_tool.name = "get_weather";
    mock_tool.description = "Get the current weather";
    mock_tool.parameter_schema = "{\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}}}";
    mock_tool.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        return agent::ok(std::string("Sunny, 22C"));
    };

    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .tools = {mock_tool}});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    TEST_ASSERT(session.config.tools.size() == 1);
    TEST_ASSERT(session.config.tools[0].name == "get_weather");
    TEST_ASSERT(session.config.tools[0].callback != nullptr);

    auto exec_res = session.config.tools[0].callback("{}", nullptr);
    TEST_ASSERT(exec_res.ok);
    TEST_ASSERT(exec_res.value == "Sunny, 22C");

    std::cout << "  [PASS] Unit: Providers - Tool Configuration\n";
}
