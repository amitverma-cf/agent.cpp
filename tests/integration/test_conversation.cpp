#include <agent-cpp/agent.hpp>
#include "../test_assert.hpp"
#include <iostream>

void test_conversation_flow() {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .memory = {.provider = agent::MemoryProvider::InMemory}});
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::Conversation chat{session};
    auto add_res = agent::add_message(chat, "user", "turn 1");
    TEST_ASSERT(add_res.ok);
    auto res1 = agent::complete_conversation(chat);
    TEST_ASSERT(res1.ok);

    // Verify it is saved in memory and matches get_conversation_history
    auto history_res = agent::get_conversation_history(chat);
    TEST_ASSERT(history_res.ok);
    TEST_ASSERT(history_res.value.size() == 2);
    TEST_ASSERT(history_res.value[0].role == "user" && history_res.value[0].content == "turn 1");
    TEST_ASSERT(history_res.value[1].role == "assistant" && history_res.value[1].content == "Mock Response");

    // Verify raw memory storage
    auto mem_res = agent::retrieve_memory(session, "history");
    TEST_ASSERT(mem_res.ok);
    TEST_ASSERT(mem_res.value.find("turn 1") != std::string::npos);
    TEST_ASSERT(mem_res.value.find("Mock Response") != std::string::npos);

    // Create a new stateless Conversation linked to the same session
    agent::Conversation chat2{session};
    auto history_res2 = agent::get_conversation_history(chat2);
    TEST_ASSERT(history_res2.ok);
    TEST_ASSERT(history_res2.value.size() == 2);
    TEST_ASSERT(history_res2.value[0].role == "user" && history_res2.value[0].content == "turn 1");
    TEST_ASSERT(history_res2.value[1].role == "assistant" && history_res2.value[1].content == "Mock Response");

    std::cout << "  [PASS] Integration: Conversation Flow (Direct Session Memory)\n";
}

void test_conversation_sliding_window() {
    auto session_result = agent::init({
        .provider = agent::AiProvider::Mock,
        .memory = {.provider = agent::MemoryProvider::InMemory},
        .context_window = 400,
        .max_tokens = 50
    });
    TEST_ASSERT(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::Conversation chat{session};

    auto add_sys = agent::add_message(chat, "system", "You are a helpful assistant.");
    TEST_ASSERT(add_sys.ok);

    auto add_user1 = agent::add_message(chat, "user", std::string(300, 'a'));
    TEST_ASSERT(add_user1.ok);
    auto res1 = agent::complete_conversation(chat);
    TEST_ASSERT(res1.ok);

    auto add_user2 = agent::add_message(chat, "user", std::string(300, 'b'));
    TEST_ASSERT(add_user2.ok);
    auto res2 = agent::complete_conversation(chat);
    TEST_ASSERT(res2.ok);

    auto add_user3 = agent::add_message(chat, "user", std::string(300, 'c'));
    TEST_ASSERT(add_user3.ok);
    auto res3 = agent::complete_conversation(chat);
    TEST_ASSERT(res3.ok);

    auto history_res = agent::get_conversation_history(chat);
    TEST_ASSERT(history_res.ok);

    TEST_ASSERT(history_res.value[0].role == "system");

    for (const auto &msg : history_res.value) {
        if (msg.role == "system") continue;
        bool pruned = msg.content.find('a') == std::string::npos;
        TEST_ASSERT(pruned && "Turn 1 should have been pruned");
    }

    bool found_b = false;
    bool found_c = false;
    for (const auto &msg : history_res.value) {
        if (msg.content.find('b') != std::string::npos) found_b = true;
        if (msg.content.find('c') != std::string::npos) found_c = true;
    }
    TEST_ASSERT(found_b && "Turn 2 should be preserved");
    TEST_ASSERT(found_c && "Turn 3 should be preserved");

    std::cout << "  [PASS] Integration: Conversation Sliding Window Pruning\n";
}
