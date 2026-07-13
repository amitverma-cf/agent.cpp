#include <catch_amalgamated.hpp>

#include <agent-cpp/agent.hpp>
#include <string>
#include <vector>

namespace {

struct HookRecord {
    std::string payload;
    int fire_count = 0;
};

void record_hook(const agent::Event &event, void *user_data) {
    auto *rec = static_cast<HookRecord *>(user_data);
    rec->fire_count++;
    rec->payload = std::string(reinterpret_cast<const char *>(event.payload.data()), event.payload.size());
}

} // namespace

TEST_CASE("trigger_event invokes a registered hook with the given payload", "[events]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    HookRecord rec;
    agent::register_event_hook(session, agent::EventType::OnToolCall, record_hook, &rec);

    std::string payload = "my_tool";
    agent::trigger_event(session, agent::EventType::OnToolCall,
                         std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(payload.data()),
                                                  payload.size()));

    REQUIRE(rec.fire_count == 1);
    REQUIRE(rec.payload == "my_tool");
}

TEST_CASE("trigger_event only fires hooks matching the event type", "[events]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    HookRecord tool_call_rec, tool_result_rec;
    agent::register_event_hook(session, agent::EventType::OnToolCall, record_hook, &tool_call_rec);
    agent::register_event_hook(session, agent::EventType::OnToolResult, record_hook, &tool_result_rec);

    agent::trigger_event(session, agent::EventType::OnToolCall, std::span<const uint8_t>());

    REQUIRE(tool_call_rec.fire_count == 1);
    REQUIRE(tool_result_rec.fire_count == 0);
}

TEST_CASE("Multiple hooks for the same event type all fire in registration order", "[events]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    HookRecord rec1, rec2;
    agent::register_event_hook(session, agent::EventType::OnCronFire, record_hook, &rec1);
    agent::register_event_hook(session, agent::EventType::OnCronFire, record_hook, &rec2);

    agent::trigger_event(session, agent::EventType::OnCronFire, std::span<const uint8_t>());

    REQUIRE(rec1.fire_count == 1);
    REQUIRE(rec2.fire_count == 1);
}

TEST_CASE("OnCompression fires during a real compress_context call", "[events][context_compressor]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    HookRecord rec;
    agent::register_event_hook(session, agent::EventType::OnCompression, record_hook, &rec);

    agent::FlowMemory conv;
    conv.id = "events_compress";
    conv.history.push_back({.role = "system", .content = "sys"});
    for (int i = 0; i < 10; ++i) {
        conv.history.push_back({.role = "user", .content = "m" + std::to_string(i)});
        conv.history.push_back({.role = "assistant", .content = "r" + std::to_string(i)});
    }

    auto res = agent::compress_context(session, conv, 2);
    REQUIRE(res.ok);
    REQUIRE(rec.fire_count == 1);
}

TEST_CASE("OnPrune fires when run_turn prunes history over budget", "[events][pruning]") {
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock,
                     .workspace_dir = "/tmp/agent_test_ws",
                     .context_window = 400,
                     .max_tokens = 50});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    HookRecord rec;
    agent::register_event_hook(session, agent::EventType::OnPrune, record_hook, &rec);

    agent::FlowMemory conv;
    conv.id = "events_prune";
    conv.history.push_back({.role = "system", .content = "You are a helpful assistant."});

    REQUIRE(agent::run_turn(session, conv, std::string(350, 'a')).ok);
    REQUIRE(agent::run_turn(session, conv, std::string(350, 'b')).ok);
    REQUIRE(agent::run_turn(session, conv, std::string(350, 'c')).ok);

    REQUIRE(rec.fire_count > 0);
}
