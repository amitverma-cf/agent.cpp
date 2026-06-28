#include <agent-cpp/agent.h>
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
    agent::init_backend();
    auto session_result =
        // agent::init({.provider = agent::Provider::OpenAICompatible,
        //              .base_url = "http://127.0.0.1",
        //              .api_key = "api_key",
        //              .model = "model-name",
        //              .workspace_dir = ".workspace",
        //              .logger = log_message});
        agent::init({.provider = agent::Provider::LlamaCpp,
                     .model = "models/LFM2.5-350M-Q4_K_M.gguf",
                     .workspace_dir = ".workspace",
                     .logger = log_message});
    if (!session_result.ok) {
        print_error(session_result.error);
        agent::free_backend();
        return 1;
    }

    agent::Session session = std::move(session_result.value);
    auto result = agent::generate_text(session, "hello");
    if (!result.ok) {
        print_error(result.error);
        return 1;
    }
    std::cout << result.value << "\n";

    auto stream_result = agent::stream_text(session, "hello", print_token);
    if (!stream_result.ok) {
        print_error(stream_result.error);
        return 1;
    }

    agent::free_backend();
    return 0;
}
