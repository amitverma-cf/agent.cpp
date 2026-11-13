// The smallest program: open a model, ask one question, stream the answer.
// Needs models/qwen2.5-0.5b-instruct-q4_k_m.gguf (examples/download-models.sh) and the llama engine.

#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    auto session = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if (!session.ok) return std::fprintf(stderr, "open: %s\n", session.error.message.c_str()), 1;

    agent::Conversation chat;
    auto reply = agent::run_turn(session.value, chat, "What is a KV cache, in two sentences?", /*stream=*/true,
                                 [](std::string_view token, void *) { std::fwrite(token.data(), 1, token.size(), stdout); });
    std::printf("\n");
    if (!reply.ok) return std::fprintf(stderr, "turn: %s\n", reply.error.message.c_str()), 1;

    auto stats = agent::get_stats(session.value);
    std::printf("prompt tokens %d, completion tokens %d\n", stats.total_prompt_tokens, stats.total_completion_tokens);
    return 0;
}
