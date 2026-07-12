#include "../test_assert.hpp"

#include <agent-cpp/agent.hpp>
#include <iostream>

void test_conversation_flow() {
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock, .memory = {.provider = agent::MemoryProvider::InMemory}});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::ConversationState state;
    state.id = "chat_1";

    auto add_res = agent::run_conversation_turn(session, state, "turn 1");
    TEST_ASSERT(add_res.ok);
    TEST_ASSERT(add_res.value == "Mock Response");

    TEST_ASSERT(state.history.size() == 2);
    TEST_ASSERT(state.history[0].role == "user" && state.history[0].content == "turn 1");
    TEST_ASSERT(state.history[1].role == "assistant" && state.history[1].content == "Mock Response");

    auto mem_res = agent::retrieve_memory(session, "history:" + state.id);
    TEST_ASSERT(mem_res.ok);
    TEST_ASSERT(mem_res.value.find("turn 1") != std::string::npos);
    TEST_ASSERT(mem_res.value.find("Mock Response") != std::string::npos);

    std::cout << "  [PASS] Integration: Conversation Flow (Direct Session Memory)\n";
}

void test_conversation_sliding_window() {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .memory = {.provider = agent::MemoryProvider::InMemory},
                                       .context_window = 400,
                                       .max_tokens = 50});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::ConversationState state;
    state.id = "chat_2";

    agent::Message sys_msg;
    sys_msg.role = "system";
    sys_msg.content = "You are a helpful assistant.";
    state.history.push_back(std::move(sys_msg));

    auto res1 = agent::run_conversation_turn(session, state, std::string(300, 'a'));
    TEST_ASSERT(res1.ok);

    auto res2 = agent::run_conversation_turn(session, state, std::string(300, 'b'));
    TEST_ASSERT(res2.ok);

    auto res3 = agent::run_conversation_turn(session, state, std::string(300, 'c'));
    TEST_ASSERT(res3.ok);

    TEST_ASSERT(state.history[0].role == "system");

    for (const auto &msg : state.history) {
        if (msg.role == "system")
            continue;
        bool pruned = msg.content.find('a') == std::string::npos;
        TEST_ASSERT(pruned && "Turn 1 should have been pruned");
    }

    bool found_b = false;
    bool found_c = false;
    for (const auto &msg : state.history) {
        if (msg.content.find('b') != std::string::npos)
            found_b = true;
        if (msg.content.find('c') != std::string::npos)
            found_c = true;
    }
    TEST_ASSERT(found_b && "Turn 2 should be preserved");
    TEST_ASSERT(found_c && "Turn 3 should be preserved");

    std::cout << "  [PASS] Integration: Conversation Sliding Window Pruning\n";
}

namespace {

// Demonstrates that FSMState/run_fsm is a bare, generic primitive: the executor knows nothing
// about prompts, tools, or ConversationState. All of that is user code living inside
// on_transition, driven through the opaque `void *context`.
struct FsmContext {
    agent::ConversationState *state;
};

void set_system_prompt(agent::ConversationState &state, std::string_view prompt) {
    if (!state.history.empty() && state.history[0].role == "system") {
        state.history[0].content = std::string(prompt);
        state.history[0].token_count = -1;
    } else {
        agent::Message sys_msg;
        sys_msg.role = "system";
        sys_msg.content = std::string(prompt);
        state.history.insert(state.history.begin(), std::move(sys_msg));
    }
}

agent::Result<std::string_view> state_a_transition(agent::Session &session, void *context) {
    auto *ctx = static_cast<FsmContext *>(context);
    set_system_prompt(*ctx->state, "State A prompt");

    auto original_tools = session.config.tools;
    session.config.tools = {agent::tools::get_calculator_tool()};
    auto turn_res = agent::run_conversation_turn(session, *ctx->state, "", false, nullptr, nullptr);
    session.config.tools = original_tools;

    if (!turn_res.ok)
        return agent::fail<std::string_view>(turn_res.error.code, turn_res.error.message);
    return agent::ok(std::string_view("state_B"));
}

agent::Result<std::string_view> state_b_transition(agent::Session &session, void *context) {
    auto *ctx = static_cast<FsmContext *>(context);
    set_system_prompt(*ctx->state, "State B prompt");

    auto turn_res = agent::run_conversation_turn(session, *ctx->state, "", false, nullptr, nullptr);
    if (!turn_res.ok)
        return agent::fail<std::string_view>(turn_res.error.code, turn_res.error.message);
    return agent::ok(std::string_view("exit"));
}

} // namespace

void test_conversation_fsm() {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .memory = {.provider = agent::MemoryProvider::InMemory},
                                       .tools = {agent::tools::get_calculator_tool()}});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::ConversationState state;
    state.id = "chat_fsm";

    agent::FSMState state_a{.state_id = "state_A", .on_transition = state_a_transition};
    agent::FSMState state_b{.state_id = "state_B", .on_transition = state_b_transition};

    std::vector<agent::FSMState> states = {state_a, state_b};
    agent::FSMExecutor executor{.registered_states = states};

    FsmContext ctx{.state = &state};
    auto fsm_res = agent::run_fsm(session, &ctx, executor);
    TEST_ASSERT(fsm_res.ok);

    TEST_ASSERT(state.history.size() >= 3);
    TEST_ASSERT(state.history[0].role == "system");
    TEST_ASSERT(state.history[0].content.find("State B prompt") != std::string::npos);

    std::cout << "  [PASS] Integration: Conversation FSM Executor\n";
}
