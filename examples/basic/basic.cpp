#include <agent-cpp/agent.hpp>
#include <iostream>

namespace {

void log_message(agent::LogLevel level, std::string_view message, void *) {
    const char *prefix = "info";
    switch (level) {
    case agent::LogLevel::Debug:
        prefix = "debug";
        break;
    case agent::LogLevel::Info:
        prefix = "info";
        break;
    case agent::LogLevel::Warning:
        prefix = "warning";
        break;
    case agent::LogLevel::Error:
        prefix = "error";
        break;
    }
    std::cerr << "[" << prefix << "] " << message << "\n";
}

void print_token(std::string_view token, void *) {
    std::cout << token;
}

void print_error(const agent::Error &error) {
    std::cerr << "agent error: " << error.message << "\n";
}

} // namespace

int main() {
    agent::AgentRuntime runtime;
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock,
                     .workspace_dir = ".workspace",
                     .logger = log_message});
    if (!session_result.ok) {
        print_error(session_result.error);
        return 1;
    }

    agent::Session session = std::move(session_result.value);
    auto result = agent::generate_text(session, "hello");
    if (!result.ok) {
        print_error(result.error);
        return 1;
    }
    std::cout << result.value.text << "\n";

    auto stream_result = agent::stream_text(session, "hello", print_token);
    if (!stream_result.ok) {
        print_error(stream_result.error);
        return 1;
    }

    // Conversation
    std::cout << "\nConversations:\n";
    agent::Conversation chat{session};
    auto add_res1 = agent::add_message(chat, "user", "Hello, how are you?");
    if (!add_res1.ok) {
        print_error(add_res1.error);
        return 1;
    }
    auto reply = agent::complete_conversation(chat);
    if (reply.ok) {
        std::cout << "Agent replied: " << reply.value << "\n";
    } else {
        print_error(reply.error);
        return 1;
    }
    auto add_res2 = agent::add_message(chat, "user", "What was my first question?");
    if (!add_res2.ok) {
        print_error(add_res2.error);
        return 1;
    }
    auto reply2 = agent::complete_conversation(chat);
    if (reply2.ok) {
        std::cout << "Agent replied: " << reply2.value << "\n";
    } else {
        print_error(reply2.error);
        return 1;
    }

    return 0;
}
