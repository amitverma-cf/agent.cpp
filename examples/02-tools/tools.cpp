// A tool of your own. The model decides when to call it; you decide what is allowed.
// Needs models/qwen2.5-0.5b-instruct-q4_k_m.gguf and the llama engine.

#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <string>

// A tool is a function from a JSON argument string to a JSON result.
static agent::Result<std::string> get_weather(std::string_view arguments, void *) {
    std::printf("[tool] get_weather(%.*s)\n", static_cast<int>(arguments.size()), arguments.data());
    return agent::ok(std::string(R"({"city":"Paris","temperature_c":18,"sky":"light rain"})"));
}

// The permission gate runs before every tool call: allow it, deny it, or ask the user.
static agent::PermissionDecision gate(std::string_view tool, std::string_view, void *) {
    return tool == "run_command" ? agent::PermissionDecision::Deny : agent::PermissionDecision::Allow;
}

int main() {
    agent::AgentRuntime runtime;

    agent::Config config;
    config.workspace_dir = ".agent";
    config.permission_check = gate;
    config.tools.push_back(agent::Tool{
        .name = "get_weather",
        .description = "Current weather in a city.",
        .parameter_schema = R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
        .callback = get_weather,
    });

    auto session = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", config);
    if (!session.ok) return std::fprintf(stderr, "open: %s\n", session.error.message.c_str()), 1;

    agent::Conversation chat;
    auto reply = agent::run_turn(session.value, chat, "What is the weather in Paris right now? Use the get_weather tool.");
    if (!reply.ok) return std::fprintf(stderr, "turn: %s\n", reply.error.message.c_str()), 1;
    std::printf("%s\n", reply.value.c_str());
    std::printf("tool calls: %d\n", agent::get_stats(session.value).total_tool_calls);
    return 0;
}
