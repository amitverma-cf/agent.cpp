// Two engines in one program: a chat model writes an image prompt, a stable-diffusion model draws it.
// Needs models/qwen2.5-0.5b-instruct-q4_k_m.gguf (llama engine) and models/sdxs-512-tinySDdistilled_Q8_0.gguf (sdcpp engine).

#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    auto writer = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if (!writer.ok) return std::fprintf(stderr, "writer: %s\n", writer.error.message.c_str()), 1;
    auto painter = agent::open_session(agent::engines::sdcpp, "models/sdxs-512-tinySDdistilled_Q8_0.gguf");
    if (!painter.ok) return std::fprintf(stderr, "painter: %s\n", painter.error.message.c_str()), 1;

    agent::Conversation chat;
    auto idea = agent::run_turn(writer.value, chat, "Write one vivid sentence describing a lighthouse in a storm, for an image generator.");
    if (!idea.ok) return std::fprintf(stderr, "prompt: %s\n", idea.error.message.c_str()), 1;
    std::printf("prompt: %s\n", idea.value.c_str());

    auto image = agent::draw(painter.value, idea.value, R"({"steps":1,"cfg_scale":1})");
    if (!image.ok || image.value.files.empty()) return std::fprintf(stderr, "draw: %s\n", image.error.message.c_str()), 1;
    std::printf("image: %s\n", image.value.files[0].c_str());
    return 0;
}
