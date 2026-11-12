#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

TEST_CASE("Session lifecycle: valid Mock init", "[session]") {
    auto res = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(res.ok);
}

TEST_CASE("get_stats and reset_stats accumulate across turns", "[session][stats]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto baseline = agent::get_stats(session);
    REQUIRE(baseline.total_turns == 0);

    agent::Conversation conv;
    conv.id = "stats_chat";
    auto r1 = agent::run_turn(session, conv, "hello");
    REQUIRE(r1.ok);
    auto r2 = agent::run_turn(session, conv, "again");
    REQUIRE(r2.ok);

    auto after = agent::get_stats(session);
    REQUIRE(after.total_turns == 2);
    REQUIRE(after.total_prompt_tokens > 0);
    REQUIRE(after.total_completion_tokens > 0);

    agent::reset_stats(session);
    auto reset = agent::get_stats(session);
    REQUIRE(reset.total_turns == 0);
    REQUIRE(reset.total_prompt_tokens == 0);
}

TEST_CASE("clear_provider_kv_cache is a no-op that succeeds for Mock", "[session][kv_cache]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto res = agent::clear_provider_kv_cache(session);
    REQUIRE(res.ok);
}
