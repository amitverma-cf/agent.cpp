#include "../test_assert.hpp"

#include <agent-cpp/agent.hpp>
#include <iostream>

static int g_log_count = 0;
static void test_logger(agent::LogLevel, std::string_view, void *) {
    g_log_count++;
}

void test_utils_logging() {
    g_log_count = 0;
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .logger = test_logger});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::MessageView user_msg;
    user_msg.role = "user";
    user_msg.content = "test";
    user_msg.tool_calls = {};
    user_msg.tool_call_id = "";

    agent::ChatRequest request{.messages = std::span<const agent::MessageView>(&user_msg, 1),
                               .stream = true,
                               .on_token = [](std::string_view, void *) {},
                               .token_user_data = nullptr};

    auto stream_res = agent::execute_turn(session, request);
    TEST_ASSERT(stream_res.ok);
    TEST_ASSERT(g_log_count > 0 && "Logger callback was not invoked");

    std::cout << "  [PASS] Unit: Logging Utilities\n";
}
