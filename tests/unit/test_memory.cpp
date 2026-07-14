#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

TEST_CASE("SQLite memory provider basic store/retrieve/clear", "[memory][sqlite]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto store_res = agent::store_memory(session, "key", "val");
    REQUIRE(store_res.ok);
    auto res = agent::retrieve_memory(session, "key");
    REQUIRE(res.ok);
    REQUIRE(res.value == "val");

    auto clear_res = agent::clear_memory(session);
    REQUIRE(clear_res.ok);
    auto res2 = agent::retrieve_memory(session, "key");
    REQUIRE_FALSE(res2.ok);
}

TEST_CASE("SQLite memory provider persists across reopen", "[memory][sqlite]") {
    const std::string db_path = "/tmp/agent_test_ws/test_sqlite_persist.sqlite3";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path + "-wal", ec);
    std::filesystem::remove(db_path + "-shm", ec);

    {
        auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                           .workspace_dir = "/tmp/agent_test_ws",
                                           .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = db_path}});
        REQUIRE(session_result.ok);
        agent::Session session = std::move(session_result.value);

        auto store_res = agent::store_memory(session, "persist_key", "persist_val");
        REQUIRE(store_res.ok);
    }

    auto session_result2 = agent::init({.provider = agent::AiProvider::Mock,
                                        .workspace_dir = "/tmp/agent_test_ws",
                                        .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = db_path}});
    REQUIRE(session_result2.ok);
    agent::Session session2 = std::move(session_result2.value);

    auto res = agent::retrieve_memory(session2, "persist_key");
    REQUIRE(res.ok);
    REQUIRE(res.value == "persist_val");

    session2 = {};
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path + "-wal", ec);
    std::filesystem::remove(db_path + "-shm", ec);
}

TEST_CASE("SQLite memory provider bounded cache evicts and reads through", "[memory][sqlite]") {
    const std::string db_path = "/tmp/agent_test_ws/test_sqlite_evict.sqlite3";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path + "-wal", ec);
    std::filesystem::remove(db_path + "-shm", ec);

    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = "/tmp/agent_test_ws",
                                       .memory = {.provider = agent::MemoryProvider::Sqlite,
                                                  .db_path = db_path,
                                                  .max_cache_bytes = 256,
                                                  .flush_dirty_threshold_bytes = 64,
                                                  .flush_interval_ms = 20}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    std::vector<std::string> keys;
    for (int i = 0; i < 30; ++i) {
        std::string key = "evict_key_" + std::to_string(i);
        std::string val(64, 'x');
        auto res = agent::store_memory(session, key, val);
        REQUIRE(res.ok);
        keys.push_back(key);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    for (const auto &key : keys) {
        auto res = agent::retrieve_memory(session, key);
        REQUIRE(res.ok);
        REQUIRE(res.value.size() == 64);
    }

    session = {};
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path + "-wal", ec);
    std::filesystem::remove(db_path + "-shm", ec);
}

TEST_CASE("SQLite memory provider handles concurrent writes from multiple threads", "[memory][sqlite]") {
    const std::string db_path = "/tmp/agent_test_ws/test_sqlite_concurrent.sqlite3";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path + "-wal", ec);
    std::filesystem::remove(db_path + "-shm", ec);

    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = "/tmp/agent_test_ws",
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = db_path}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    constexpr int kThreads = 8;
    constexpr int kPerThread = 50;
    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&session, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                std::string key = "t" + std::to_string(t) + "_k" + std::to_string(i);
                std::string val = "v" + std::to_string(t) + "_" + std::to_string(i);
                auto res = agent::store_memory(session, key, val);
                (void)res;
            }
        });
    }
    for (auto &w : workers) w.join();

    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kPerThread; ++i) {
            std::string key = "t" + std::to_string(t) + "_k" + std::to_string(i);
            std::string expected = "v" + std::to_string(t) + "_" + std::to_string(i);
            auto res = agent::retrieve_memory(session, key);
            REQUIRE(res.ok);
            REQUIRE(res.value == expected);
        }
    }

    session = {};
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path + "-wal", ec);
    std::filesystem::remove(db_path + "-shm", ec);
}
