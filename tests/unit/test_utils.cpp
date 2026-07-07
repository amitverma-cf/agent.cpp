#include <agent-cpp/agent.hpp>
#include "../test_assert.hpp"
#include <iostream>

static int g_log_count = 0;
static void test_logger(agent::LogLevel, std::string_view, void*) {
    g_log_count++;
}

void test_utils_logging() {
    g_log_count = 0;
    auto session_result = agent::init({
        .provider = agent::AiProvider::Mock,
        .logger = test_logger
    });
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto stream_res = agent::stream_text(session, "test", [](std::string_view, void*){}, nullptr);
    TEST_ASSERT(stream_res.ok);
    TEST_ASSERT(g_log_count > 0 && "Logger callback was not invoked");

    std::cout << "  [PASS] Unit: Logging Utilities\n";
}
