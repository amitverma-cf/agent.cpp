#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

namespace {
const char *kWorkspace = "/tmp/agent_test_workspace";
}

TEST_CASE("run_turn updates history and persists to memory", "[flow]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
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

TEST_CASE("Sliding window pruning drops oldest turns while preserving system + recent", "[flow][pruning]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"},
                                       .context_window = 400,
                                       .max_tokens = 50});
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

namespace {

agent::Result<std::string_view> state_a_transition(agent::Session &session, agent::FlowMemory &conv, void *) {
    if (!conv.history.empty() && conv.history[0].role == "system") {
        conv.history[0].content = "State A prompt";
        conv.history[0].token_count = -1;
    } else {
        agent::Message sys;
        sys.role = "system";
        sys.content = "State A prompt";
        conv.history.insert(conv.history.begin(), std::move(sys));
    }

    auto turn_res = agent::run_turn(session, conv, "");
    if (!turn_res.ok) return agent::fail<std::string_view>(turn_res.error.code, turn_res.error.message);
    return agent::ok(std::string_view("state_B"));
}

agent::Result<std::string_view> state_b_transition(agent::Session &session, agent::FlowMemory &conv, void *) {
    if (!conv.history.empty() && conv.history[0].role == "system") {
        conv.history[0].content = "State B prompt";
        conv.history[0].token_count = -1;
    } else {
        agent::Message sys;
        sys.role = "system";
        sys.content = "State B prompt";
        conv.history.insert(conv.history.begin(), std::move(sys));
    }

    auto turn_res = agent::run_turn(session, conv, "");
    if (!turn_res.ok) return agent::fail<std::string_view>(turn_res.error.code, turn_res.error.message);
    return agent::ok(std::string_view("exit"));
}

} // namespace

TEST_CASE("Flow transitions between states and mutates system prompt per state", "[flow][fsm]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::FlowState state_a{.state_id = "state_A", .on_transition = state_a_transition};
    agent::FlowState state_b{.state_id = "state_B", .on_transition = state_b_transition};

    std::vector<agent::FlowState> states = {state_a, state_b};

    agent::Flow flow;
    flow.states = states;
    flow.memory.id = "chat_fsm";

    auto flow_res = agent::run_flow(session, nullptr, flow);
    REQUIRE(flow_res.ok);

    const auto &history = flow.memory.history;
    REQUIRE(history.size() >= 3);
    REQUIRE(history[0].role == "system");
    REQUIRE(history[0].content.find("State B prompt") != std::string::npos);
}

namespace {

agent::Result<std::string_view> scheduler_single_shot(agent::Session &session, agent::FlowMemory &conv, void *) {
    auto r = agent::run_turn(session, conv, "go");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("exit"));
}

} // namespace

TEST_CASE("AgentScheduler runs multiple Flows in parallel to completion", "[scheduler][flow]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    static const agent::FlowState kStates[] = {
        {.state_id = "go", .on_transition = scheduler_single_shot},
    };

    agent::AgentScheduler scheduler;
    for (int i = 0; i < 3; ++i) {
        agent::Flow flow;
        flow.states = kStates;
        flow.memory.id = "sched_" + std::to_string(i);
        auto spawn_res = scheduler.spawn("worker_" + std::to_string(i), std::move(flow));
        REQUIRE(spawn_res.ok);
    }

    auto run_res = scheduler.run_until_done(session);
    REQUIRE(run_res.ok);
    REQUIRE(scheduler.all_done());

    auto results = scheduler.results();
    REQUIRE(results.size() == 3);
    for (const auto &r : results) {
        REQUIRE(r.ok);
        REQUIRE(r.stats.turns == 1);
    }
}
