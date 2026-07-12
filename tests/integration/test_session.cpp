#include "../test_assert.hpp"

#include <agent-cpp/agent.hpp>
#include <iostream>

void test_session_lifecycle() {
    auto res = agent::init({.provider = agent::AiProvider::Mock});
    TEST_ASSERT(res.ok);

    auto res_invalid = agent::init({.provider = agent::AiProvider::OpenAICompatible});
    TEST_ASSERT(!res_invalid.ok);
    TEST_ASSERT(res_invalid.error.code == agent::ErrorCode::InvalidConfig ||
                res_invalid.error.code == agent::ErrorCode::UnsupportedProvider);
    std::cout << "  [PASS] Integration: Session Lifecycle\n";
}
