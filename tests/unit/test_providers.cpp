#include <agent-cpp/agent.hpp>
#include "../test_assert.hpp"
#include <iostream>
#include <string>

static std::string g_stream_buffer;
static void test_token_callback(std::string_view token, void *) {
    g_stream_buffer += token;
}

void test_provider_mock_generation() {
    auto session_result = agent::init({.provider = agent::Provider::Mock});
    TEST_ASSERT(session_result.ok && "Mock initialization failed");
    agent::Session session = std::move(session_result.value);

    auto generate_result = agent::generate_text(session, "Hello World");
    TEST_ASSERT(generate_result.ok && "Mock generate failed");
    TEST_ASSERT(generate_result.value.text == "Mock Response" && "Mock returned wrong string");
    TEST_ASSERT(generate_result.value.usage.prompt_tokens == 2);
    TEST_ASSERT(generate_result.value.usage.completion_tokens == 2);
    TEST_ASSERT(generate_result.value.usage.total_tokens == 4);

    std::cout << "  [PASS] Unit: Providers - Mock Generation\n";
}

void test_provider_mock_streaming() {
    g_stream_buffer.clear();
    auto session_result = agent::init({.provider = agent::Provider::Mock});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto stream_result = agent::stream_text(session, "Hello", test_token_callback, nullptr);
    TEST_ASSERT(stream_result.ok && "Mock stream failed");
    TEST_ASSERT(g_stream_buffer == "Mock Response Streamed" && "Stream buffer mismatch");

    std::cout << "  [PASS] Unit: Providers - Mock Streaming\n";
}
