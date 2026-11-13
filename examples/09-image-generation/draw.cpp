// Draw an image from a prompt with a stable-diffusion model.
// Needs models/sdxs-512-tinySDdistilled_Q8_0.gguf (examples/download-models.sh) and the sdcpp engine.

#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    // A single-file checkpoint (SD 1.x, SDXL, SDXS) is the "model". Split models (Flux, Wan) name "diffusion_model", "llm", "vae", ...
    auto session = agent::open_session(agent::engines::sdcpp, "models/sdxs-512-tinySDdistilled_Q8_0.gguf");
    if (!session.ok) return std::fprintf(stderr, "open: %s\n", session.error.message.c_str()), 1;

    // Engine parameters go in as JSON; a one-step distilled model wants steps 1 and cfg_scale 1.
    auto image = agent::draw(session.value, "a red lighthouse at dusk", R"({"steps":1,"cfg_scale":1})");
    if (!image.ok || image.value.files.empty()) return std::fprintf(stderr, "draw: %s\n", image.error.message.c_str()), 1;
    std::printf("%s\n", image.value.files[0].c_str());
    return 0;
}
