#include <agent-cpp/agent.h>
#include <iostream>

int main() {
    agent::Session session =
        agent::init({.provider = agent::Provider::OpenAICompatible,
                     .base_url = "http://127.0.0.1",
                     .api_key = "api_key",
                     .model = "model"});
    std::string result = agent::generate_text(session, "hello");
    std::cout << result << "\n";

    std::cout << "Stream\n";
    agent::stream_text(session, "hello", [](std::string_view token) { std::cout << token; });

    return 0;
}