#include <agent-cpp/agent.hpp>
#include <cstdio>
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
    std::fprintf(stderr, "[%s] %s\n", prefix, std::string(message).c_str());
}

void print_token(std::string_view token, void *) {
    std::cout << token;
    std::cout.flush();
}

void print_error(const agent::Error &e) {
    std::fprintf(stderr, "agent error [%d]: %s\n", static_cast<int>(e.code), e.message.c_str());
}

}

int main() {
    agent::AgentRuntime runtime;

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Mock;
    cfg.workspace_dir = ".workspace";
    cfg.logger = log_message;
    cfg.memory = {
        .provider = agent::MemoryProvider::Sqlite,
        .db_path = ":memory:",
    };

    auto sess_res = agent::init(cfg);
    if (!sess_res.ok) {
        print_error(sess_res.error);
        return 1;
    }
    agent::Session session = std::move(sess_res.value);

    {
        agent::MessageView msg;
        msg.role = "user";
        msg.content = "hello";

        agent::InferRequest req;
        req.messages = std::span<const agent::MessageView>(&msg, 1);

        auto res = agent::infer(session, req);
        if (!res.ok) {
            print_error(res.error);
            return 1;
        }
        std::cout << "response: " << res.value.message.content << "\n";
    }

    {
        agent::MessageView msg;
        msg.role = "user";
        msg.content = "hello streaming";

        agent::InferRequest req;
        req.messages = std::span<const agent::MessageView>(&msg, 1);
        req.stream = true;
        req.on_token = print_token;
        req.token_user_data = nullptr;

        auto res = agent::infer(session, req);
        if (!res.ok) {
            print_error(res.error);
            return 1;
        }
        std::cout << "\n";
    }

    {
        agent::FlowMemory state;
        state.id = "basic_demo";

        auto r1 = agent::run_turn(session, state, "What is 2 + 2?");
        if (!r1.ok) {
            print_error(r1.error);
            return 1;
        }
        std::cout << "turn 1: " << r1.value << "\n";

        auto r2 = agent::run_turn(session, state, "And what was my first question?");
        if (!r2.ok) {
            print_error(r2.error);
            return 1;
        }
        std::cout << "turn 2: " << r2.value << "\n";
    }

    return 0;
}
