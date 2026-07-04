#include <agent-cpp/agent.hpp>
#include "../test_assert.hpp"
#include <iostream>

void test_session_lifecycle() {
    // Correct Mock Init
    auto res = agent::init({.provider = agent::Provider::Mock});
    TEST_ASSERT(res.ok);

    // Invalid Config (e.g. OpenAI Compatible without base_url)
    auto res_invalid = agent::init({.provider = agent::Provider::OpenAICompatible});
    TEST_ASSERT(!res_invalid.ok);
    TEST_ASSERT(res_invalid.error.code == agent::ErrorCode::InvalidConfig ||
                res_invalid.error.code == agent::ErrorCode::UnsupportedProvider);
    std::cout << "  [PASS] Integration: Session Lifecycle\n";
}
