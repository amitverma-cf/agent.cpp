#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

namespace {
const char *kWorkspace = "/tmp/agent_test_workspace";
}

TEST_CASE("run_turn updates history and persists to memory", "[turn]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = kWorkspace});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::FlowMemory state;
    state.id = "chat_1";

    auto add_res = agent::run_turn(session, state, "turn 1");
    REQUIRE(add_res.ok);
    REQUIRE(add_res.value == "Mock Response");

    REQUIRE(state.history.size() == 2);
    REQUIRE(state.history[0].role == "user");
    REQUIRE(state.history[0].content == "turn 1");
    REQUIRE(state.history[1].role == "assistant");
    REQUIRE(state.history[1].content == "Mock Response");

    auto mem_res = agent::retrieve_memory(session, "history:" + state.id);
    REQUIRE(mem_res.ok);
    REQUIRE(mem_res.value.find("turn 1") != std::string::npos);
    REQUIRE(mem_res.value.find("Mock Response") != std::string::npos);
}

TEST_CASE("Sliding window pruning drops oldest turns while preserving system + recent", "[turn][pruning]") {
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = kWorkspace, .context_window = 400, .max_tokens = 50});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::FlowMemory state;
    state.id = "chat_2";

    agent::Message sys_msg;
    sys_msg.role = "system";
    sys_msg.content = "You are a helpful assistant.";
    state.history.push_back(std::move(sys_msg));

    auto res1 = agent::run_turn(session, state, std::string(300, 'a'));
    REQUIRE(res1.ok);
    auto res2 = agent::run_turn(session, state, std::string(300, 'b'));
    REQUIRE(res2.ok);
    auto res3 = agent::run_turn(session, state, std::string(300, 'c'));
    REQUIRE(res3.ok);

    REQUIRE(state.history[0].role == "system");

    for (const auto &msg : state.history) {
        if (msg.role == "system") continue;
        INFO("Turn 1 should have been pruned: " << msg.content);
        REQUIRE(msg.content.find('a') == std::string::npos);
    }

    bool found_b = false, found_c = false;
    for (const auto &msg : state.history) {
        if (msg.content.find('b') != std::string::npos) found_b = true;
        if (msg.content.find('c') != std::string::npos) found_c = true;
    }
    REQUIRE(found_b);
    REQUIRE(found_c);
}
