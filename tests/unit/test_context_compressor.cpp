#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

namespace {

agent::Message make_msg(std::string_view role, std::string_view content) {
    agent::Message m;
    m.role = std::string(role);
    m.content = std::string(content);
    return m;
}

} // namespace

TEST_CASE("compress_context replaces older history with a summary, keeping recent messages", "[context_compressor]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::FlowMemory conv;
    conv.id = "compress_test";
    conv.history.push_back(make_msg("system", "You are a helpful assistant."));
    for (int i = 0; i < 10; ++i) {
        conv.history.push_back(make_msg("user", "message " + std::to_string(i)));
        conv.history.push_back(make_msg("assistant", "reply " + std::to_string(i)));
    }
    size_t history_before = conv.history.size();

    auto res = agent::compress_context(session, conv, /*keep_recent=*/2);
    REQUIRE(res.ok);

    REQUIRE(conv.history.size() < history_before);
    REQUIRE(conv.history[0].role == "system");
    REQUIRE(conv.history[1].role == "system");
    REQUIRE(conv.history[1].content.find("[Compressed history]:") == 0);

    // Last two messages (most recent) survive uncompressed.
    REQUIRE(conv.history.back().content == "reply 9");

    REQUIRE(conv.stats.compressions == 1);
    auto session_stats = agent::get_stats(session);
    REQUIRE(session_stats.total_compressions == 1);
}

TEST_CASE("compress_context is a no-op when history is too short", "[context_compressor]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::FlowMemory conv;
    conv.id = "compress_short";
    conv.history.push_back(make_msg("system", "sys"));
    conv.history.push_back(make_msg("user", "hi"));

    size_t history_before = conv.history.size();
    auto res = agent::compress_context(session, conv, /*keep_recent=*/6);
    REQUIRE(res.ok);
    REQUIRE(conv.history.size() == history_before);
    REQUIRE(conv.stats.compressions == 0);
}

TEST_CASE("compress_context tool invokes compress_context on the active conversation", "[context_compressor][tools]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const agent::Tool *compress_tool = nullptr;
    for (const auto &t : session.config.tools)
        if (t.name == "compress_context") {
            compress_tool = &t;
            break;
        }
    REQUIRE(compress_tool != nullptr);

    // Outside of an active run_turn, there's no ambient conversation to compress against.
    auto res = compress_tool->callback("{}", compress_tool->user_data);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::InvalidConfig);
}
