// Image critique loop: an LLM writes an image prompt, stable-diffusion.cpp draws it, a vision model
// critiques the picture, and the critique goes back to the LLM for the next prompt.
//
//   image_critique --llama <agent-engine-llama.so> --llm <chat.gguf>
//                  --sdcpp <agent-engine-sdcpp.so> --sd-model <sd.gguf>
//                  --vlm <vision.gguf> --mmproj <mmproj.gguf>
//                  [--goal "..."] [--rounds 3] [--sd-options '{"steps":1,...}'] [--workspace dir]
//
// Needs the llama engine (built by default) and the sdcpp engine (-DAGENT_ENGINE_SDCPP=ON). Without
// all of them it prints what is missing and exits with status 2. See examples/README.md.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string llama, llm, sdcpp, sd_model, vlm, mmproj;
    std::string goal = "a red lighthouse on a rocky coast at sunset";
    std::string sd_options = R"({"steps":1,"cfg_scale":1.0,"width":512,"height":512,"seed":42})"; // sdxs-512 defaults
    std::string workspace = ".workspace";
    int rounds = 3;
};

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --llama <agent-engine-llama.so> --llm <chat.gguf> --sdcpp <agent-engine-sdcpp.so> --sd-model <sd.gguf>\n"
                 "          --vlm <vision.gguf> --mmproj <mmproj.gguf> [--goal TEXT] [--rounds N] [--sd-options JSON] [--workspace DIR]\n",
                 argv0);
}

bool parse(int argc, char **argv, Options &o) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (i + 1 >= argc) return false;
        std::string *dst = a == "--llama"        ? &o.llama
                           : a == "--llm"        ? &o.llm
                           : a == "--sdcpp"      ? &o.sdcpp
                           : a == "--sd-model"   ? &o.sd_model
                           : a == "--vlm"        ? &o.vlm
                           : a == "--mmproj"     ? &o.mmproj
                           : a == "--goal"       ? &o.goal
                           : a == "--sd-options" ? &o.sd_options
                           : a == "--workspace"  ? &o.workspace
                                                 : nullptr;
        if (a == "--rounds") o.rounds = std::atoi(argv[i + 1]);
        else if (!dst) return false;
        else *dst = argv[i + 1];
        ++i;
    }
    return true;
}

// Every missing piece, as "--flag (what)" lines; empty when fully configured.
std::string missing_pieces(const Options &o) {
    std::string missing;
    const struct {
        const std::string &value;
        const char *flag;
        const char *what;
        bool is_file;
    } required[] = {
        {o.llama, "--llama", "llama engine plugin", true},
        {o.llm, "--llm", "chat model GGUF", true},
        {o.sdcpp, "--sdcpp", "sdcpp engine plugin (-DAGENT_ENGINE_SDCPP=ON)", true},
        {o.sd_model, "--sd-model", "stable-diffusion model", true},
        {o.vlm, "--vlm", "vision model GGUF", true},
        {o.mmproj, "--mmproj", "vision projector GGUF", true},
    };
    for (const auto &r : required) {
        if (r.value.empty()) missing += std::string("  ") + r.flag + " (" + r.what + ")\n";
        else if (r.is_file && !std::filesystem::exists(r.value))
            missing += std::string("  ") + r.flag + " " + r.value + ": file not found\n";
    }
    return missing;
}

void on_progress(uint64_t step, uint64_t total, void *) {
    std::fprintf(stderr, "\r  drawing: step %llu/%llu", static_cast<unsigned long long>(step), static_cast<unsigned long long>(total));
}

agent::Result<agent::Session> open_session(agent::Config cfg, const char *what) {
    auto s = agent::init(cfg);
    if (!s.ok) std::fprintf(stderr, "image_critique: cannot load the %s: %s\n", what, s.error.message.c_str());
    return s;
}

// The "SCORE: n/10" the critic was asked to end with, or -1.
int score_of(std::string_view critique) {
    const size_t at = critique.rfind("SCORE:");
    return at == std::string_view::npos ? -1 : std::atoi(std::string(critique.substr(at + 6, 4)).c_str());
}

} // namespace

int main(int argc, char **argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        usage(argv[0]);
        return 2;
    }
    if (const std::string missing = missing_pieces(o); !missing.empty()) {
        std::fprintf(stderr, "image_critique: not configured; missing:\n%s", missing.c_str());
        usage(argv[0]);
        return 2;
    }

    agent::AgentRuntime runtime;
    // Fail early with a clear message when the sdcpp plugin cannot be loaded (wrong ABI, not built, ...).
    if (auto name = agent::load_engine_plugin(o.sdcpp); !name.ok || !(agent::find_engine(name.value)->caps & AGENT_CAP_IMAGE_GEN)) {
        std::fprintf(stderr, "image_critique: %s is not an image generation engine: %s\n", o.sdcpp.c_str(),
                     name.ok ? "no AGENT_CAP_IMAGE_GEN" : name.error.message.c_str());
        return 1;
    }

    agent::Config base;
    base.provider = agent::AiProvider::Engine;
    base.workspace_dir = o.workspace;
    base.auto_default_tools = false;

    agent::Config llm_cfg = base; // prompt writer
    llm_cfg.engine = o.llama;
    llm_cfg.model = o.llm;
    llm_cfg.context_window = 4096;
    llm_cfg.max_tokens = 96;
    llm_cfg.temperature = 0.7f;

    agent::Config sd_cfg = base; // painter: every request is IMAGE_GEN
    sd_cfg.engine = o.sdcpp;
    sd_cfg.model = o.sd_model;
    sd_cfg.engine_options = o.sd_options;
    sd_cfg.default_task = AGENT_CAP_IMAGE_GEN;

    agent::Config vlm_cfg = base; // critic: llama + libmtmd projector
    vlm_cfg.engine = o.llama;
    vlm_cfg.model = o.vlm;
    vlm_cfg.engine_options = R"({"mmproj":")" + o.mmproj + R"("})";
    vlm_cfg.context_window = 4096;
    vlm_cfg.max_tokens = 128;
    vlm_cfg.temperature = 0.2f;

    auto llm = open_session(llm_cfg, "chat model");
    auto sd = open_session(sd_cfg, "stable-diffusion model");
    auto vlm = open_session(vlm_cfg, "vision model");
    if (!llm.ok || !sd.ok || !vlm.ok) return 1;

    agent::Conversation prompter; // the LLM keeps the whole exchange, so it sees every critique
    prompter.id = "image-critique-prompter";
    prompter.history.push_back({.role = "system",
                                .content = "You write prompts for a text-to-image model. Reply with the prompt only: one line, "
                                           "at most 40 words, concrete visual details, no quotes."});
    std::string request = "Goal: " + o.goal + "\nWrite the first prompt.";
    int best_score = -1;
    std::string best_image;

    for (int round = 1; round <= o.rounds; ++round) {
        auto prompt = agent::run_turn(llm.value, prompter, request);
        if (!prompt.ok) {
            std::fprintf(stderr, "image_critique: prompt writer failed: %s\n", prompt.error.message.c_str());
            return 1;
        }
        std::printf("round %d\n  prompt:   %s\n", round, prompt.value.c_str());

        agent::MessageView msg{.role = "user", .content = prompt.value};
        auto image = agent::infer(sd.value, {.messages = {&msg, 1}, .on_progress = on_progress});
        std::fprintf(stderr, "\n");
        if (!image.ok || image.value.message.media.empty()) {
            std::fprintf(stderr, "image_critique: image generation failed: %s\n",
                         image.ok ? "no image returned" : image.error.message.c_str());
            return 1;
        }
        const std::string path(image.value.message.media[0].path);
        std::printf("  image:    %s\n", path.c_str());

        agent::Conversation critic; // a fresh conversation per picture
        const agent::Media media[] = {{path, std::string(image.value.message.media[0].mime)}};
        auto critique = agent::run_turn_with_media(vlm.value, critic,
                                                   "The picture should show: " + o.goal +
                                                       "\nDescribe in two sentences what is wrong or missing, then end with "
                                                       "SCORE: <0-10> for how well it matches.",
                                                   media);
        if (!critique.ok) {
            std::fprintf(stderr, "image_critique: critic failed: %s\n", critique.error.message.c_str());
            return 1;
        }
        const int score = score_of(critique.value);
        std::printf("  critique: %s\n  score:    %d\n", critique.value.c_str(), score);
        if (best_image.empty() || score > best_score) { // score -1: the critic gave none
            best_score = score;
            best_image = path;
        }
        if (score >= 8) break;
        request = "Critique of the last image: " + critique.value + "\nWrite an improved prompt for the same goal.";
    }
    std::printf("best image: %s (score %d; -1 = the critic gave no score)\n", best_image.c_str(), best_score);
    return 0;
}
