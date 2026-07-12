#include <agent-cpp/agent.hpp>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

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

void print_error(const agent::Error &e) {
    std::fprintf(stderr, "agent error [%d]: %s\n", static_cast<int>(e.code), e.message.c_str());
}

std::string find_first_gguf(const std::filesystem::path &models_dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(models_dir, ec))
        return "";
    for (const auto &entry : std::filesystem::directory_iterator(models_dir, ec)) {
        if (entry.path().extension() == ".gguf")
            return entry.path().string();
    }
    return "";
}

std::string parse_flag(int argc, char **argv, std::string_view flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag == argv[i])
            return argv[i + 1];
    }
    return "";
}

struct ChatContext {
    agent::Session *session;
    agent::ConversationState *conv;
};

agent::Result<std::string_view> chat_transition(agent::Session &, void *context) {
    auto *ctx = static_cast<ChatContext *>(context);

    std::cout << "> ";
    std::string line;
    if (!std::getline(std::cin, line)) {
        return agent::ok(std::string_view("exit"));
    }
    if (line.empty()) {
        return agent::ok(std::string_view("chat"));
    }
    if (line == "/exit" || line == "/quit") {
        return agent::ok(std::string_view("exit"));
    }

    auto turn_res = agent::run_conversation_turn(*ctx->session, *ctx->conv, line);
    if (!turn_res.ok) {
        print_error(turn_res.error);
        return agent::ok(std::string_view("chat"));
    }

    std::cout << turn_res.value << "\n";
    return agent::ok(std::string_view("chat"));
}

} // namespace

int main(int argc, char **argv) {
    agent::AgentRuntime runtime;

    std::string workspace_dir = parse_flag(argc, argv, "--workspace");
    if (workspace_dir.empty())
        workspace_dir = ".workspace";

    std::string model = parse_flag(argc, argv, "--model");
    if (model.empty())
        model = find_first_gguf(std::filesystem::path(workspace_dir) / "models");

    if (model.empty()) {
        std::fprintf(stderr,
                     "No .gguf model found under %s/models. Pass one explicitly with --model <path>.\n",
                     workspace_dir.c_str());
        return 1;
    }

    agent::Config cfg;
    cfg.provider = agent::AiProvider::LlamaCpp;
    cfg.model = model;
    cfg.workspace_dir = workspace_dir;
    cfg.memory = {.provider = agent::MemoryProvider::RocksDB};
    cfg.context_window = 4096;
    cfg.max_tokens = 512;
    cfg.temperature = 0.7f;
    cfg.logger = log_message;

    auto sess_res = agent::init(cfg);
    if (!sess_res.ok) {
        print_error(sess_res.error);
        return 1;
    }
    agent::Session session = std::move(sess_res.value);

    std::cout << "agent_cli\n";
    std::cout << "  model:     " << model << "\n";
    std::cout << "  workspace: " << workspace_dir << "\n";
    std::cout << "  tools:    ";
    for (const auto &tool : session.config.tools)
        std::cout << " " << tool.name;
    std::cout << "\n";
    std::cout << "Type a message, or /exit to quit.\n\n";

    agent::ConversationState conv;
    conv.id = "agent_cli_session";

    ChatContext ctx{.session = &session, .conv = &conv};
    agent::FSMState chat_state{.state_id = "chat", .on_transition = chat_transition};
    std::vector<agent::FSMState> states = {chat_state};
    agent::FSMExecutor executor{.registered_states = states};

    auto fsm_res = agent::run_fsm(session, &ctx, executor);
    if (!fsm_res.ok) {
        print_error(fsm_res.error);
        return 1;
    }

    std::cout << "bye.\n";
    return 0;
}
