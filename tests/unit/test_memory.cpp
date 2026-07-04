#include <agent-cpp/agent.hpp>
#include "../test_assert.hpp"
#include <iostream>

void test_memory_in_memory() {
    auto session_result = agent::init({.provider = agent::Provider::Mock,
                                       .memory = {.provider = agent::MemoryProvider::InMemory}});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto store_res = agent::store_memory(session, "key", "val");
    TEST_ASSERT(store_res.ok);
    auto res = agent::retrieve_memory(session, "key");
    TEST_ASSERT(res.ok && res.value == "val");

    auto clear_res = agent::clear_memory(session);
    TEST_ASSERT(clear_res.ok);
    auto res2 = agent::retrieve_memory(session, "key");
    TEST_ASSERT(!res2.ok);

    std::cout << "  [PASS] Unit: InMemory Provider\n";
}
