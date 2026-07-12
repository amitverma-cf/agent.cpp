#include "../test_assert.hpp"

#include <agent-cpp/agent.hpp>
#include <filesystem>
#include <iostream>

void test_memory_in_memory() {
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock, .memory = {.provider = agent::MemoryProvider::InMemory}});
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

void test_memory_rocksdb() {
    const std::string db_path = "test_rocksdb_db";
    std::error_code ec;
    std::filesystem::remove_all(db_path, ec);

    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .memory = {.provider = agent::MemoryProvider::RocksDB, .db_path = db_path}});

    if (!session_result.ok && session_result.error.code == agent::ErrorCode::UnsupportedProvider) {
        std::cout << "  [SKIP] Unit: RocksDB Provider (not compiled in)\n";
        return;
    }

    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto store_res = agent::store_memory(session, "test_key", "test_val");
    TEST_ASSERT(store_res.ok);
    auto res = agent::retrieve_memory(session, "test_key");
    TEST_ASSERT(res.ok && res.value == "test_val");

    auto clear_res = agent::clear_memory(session);
    TEST_ASSERT(clear_res.ok);
    auto res2 = agent::retrieve_memory(session, "test_key");
    TEST_ASSERT(!res2.ok);

    session = {};
    std::filesystem::remove_all(db_path, ec);

    std::cout << "  [PASS] Unit: RocksDB Provider\n";
}
