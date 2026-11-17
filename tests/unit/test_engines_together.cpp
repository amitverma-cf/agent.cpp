// Several engines together: llama.cpp, stable-diffusion.cpp and audio.cpp each bring their own ggml (as shared libraries in a release),
// and the program runs each in its own agent-engine-host process. These tests load all of them, alternate requests, run them from several threads, and load
// and unload them repeatedly while watching the process's memory.
#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#if defined(AGENT_TEST_LLAMA_ENGINE_PATH) && defined(AGENT_TEST_SDCPP_ENGINE_PATH) && defined(AGENT_TEST_AUDIOCPP_ENGINE_PATH)

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

const char *env(const char *name) {
    const char *v = std::getenv(name);
    return v && *v ? v : nullptr;
}

struct Models {
    const char *llama = env("AGENT_TEST_LLAMA_MODEL");      // a small instruct GGUF
    const char *sd = env("AGENT_TEST_SDCPP_MODEL");         // a small single-file image model (SD-Turbo)
    const char *tts = env("AGENT_TEST_AUDIOCPP_TTS_MODEL"); // an audio.cpp text-to-speech GGUF (Pocket TTS)
    bool all() const { return llama && sd && tts; }
};

std::string fresh_dir(const char *name) {
    const fs::path p = fs::temp_directory_path() / (std::string("agent_test_together_") + name);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p);
    return p.string();
}

agent::Config llama_config(const Models &m) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_LLAMA_ENGINE_PATH;
    cfg.model = m.llama;
    cfg.workspace_dir = fresh_dir("llama");
    cfg.context_window = 512;
    cfg.max_tokens = 12;
    cfg.temperature = 0.0f;
    cfg.auto_default_tools = false;
    return cfg;
}

agent::Config sd_config(const Models &m) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_SDCPP_ENGINE_PATH;
    cfg.model = m.sd;
    cfg.workspace_dir = fresh_dir("sd");
    cfg.engine_options = R"({"threads":2,"width":64,"height":64,"steps":1,"cfg_scale":1.0,"seed":7})";
    return cfg;
}

agent::Config tts_config(const Models &m) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_AUDIOCPP_ENGINE_PATH;
    cfg.workspace_dir = fresh_dir("tts");
    cfg.engine_options = std::string(R"({"tts":{"model":")") + m.tts + R"(","language":"en"}})";
    return cfg;
}

// Each engine in its own process, as the CLI runs them: the ggml libraries of the three engines have the same file names, and one process keeps only
// one library of a name (in-process, audio.cpp would run on llama.cpp's ggml and fail on a missing symbol).
agent::Session open(agent::Config cfg) {
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_sandbox.enabled = false; // the sandbox has its own tests (test_engine_sandbox.cpp)
    auto res = agent::init(cfg);
    INFO(res.error.message);
    REQUIRE(res.ok);
    return std::move(res.value);
}

// One request of each kind. Every call checks its own result.
void chat(agent::Session &s, int round) {
    agent::Conversation memory;
    memory.id = "together_" + std::to_string(round);
    auto reply = agent::run_turn(s, memory, "Say hello.");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    CHECK_FALSE(reply.value.empty());
}

void draw(agent::Session &s, int round) {
    agent::MessageView msg{.role = "user", .content = "a red apple on a table"};
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    const std::string params = R"({"seed":)" + std::to_string(round) + "}";
    req.params_json = params;
    auto out = agent::run_task(s, AGENT_CAP_IMAGE_GEN, req);
    INFO(out.error.message);
    REQUIRE(out.ok);
    REQUIRE(out.value.message.media.size() == 1);
    const std::string path(out.value.message.media[0].path);
    CHECK(fs::file_size(path) > 100);
}

void speak(agent::Session &s) {
    agent::MessageView msg{.role = "user", .content = "Hello there."};
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    auto out = agent::run_task(s, AGENT_CAP_TTS, req);
    INFO(out.error.message);
    REQUIRE(out.ok);
    REQUIRE(out.value.message.media.size() == 1);
    CHECK(fs::file_size(std::string(out.value.message.media[0].path)) > 44);
}

} // namespace

TEST_CASE("llama, stable-diffusion and audio.cpp alternate in one process", "[engines][together][model]") {
    Models m;
    if (!m.all()) SKIP("AGENT_TEST_LLAMA_MODEL, AGENT_TEST_SDCPP_MODEL and AGENT_TEST_AUDIOCPP_TTS_MODEL not all set");
    agent::AgentRuntime runtime;
    agent::Session llama = open(llama_config(m));
    agent::Session sd = open(sd_config(m));
    agent::Session tts = open(tts_config(m));
    for (int round = 0; round < 3; ++round) {
        chat(llama, round);
        draw(sd, round);
        speak(tts);
    }
}

TEST_CASE("the three engines run from separate threads at once", "[engines][together][model]") {
    Models m;
    if (!m.all()) SKIP("AGENT_TEST_LLAMA_MODEL, AGENT_TEST_SDCPP_MODEL and AGENT_TEST_AUDIOCPP_TTS_MODEL not all set");
    agent::AgentRuntime runtime;
    agent::Session llama = open(llama_config(m));
    agent::Session sd = open(sd_config(m));
    agent::Session tts = open(tts_config(m));
    std::atomic<int> done{0};
    std::vector<std::thread> threads;
    threads.emplace_back([&] {
        for (int i = 0; i < 2; ++i) chat(llama, 100 + i);
        ++done;
    });
    threads.emplace_back([&] {
        for (int i = 0; i < 2; ++i) draw(sd, 100 + i);
        ++done;
    });
    threads.emplace_back([&] {
        for (int i = 0; i < 2; ++i) speak(tts);
        ++done;
    });
    for (auto &t : threads) t.join();
    CHECK(done == 3);
}

TEST_CASE("loading and unloading the engines repeatedly does not leak memory", "[engines][together][model][resources]") {
    Models m;
    if (!m.all()) SKIP("AGENT_TEST_LLAMA_MODEL, AGENT_TEST_SDCPP_MODEL and AGENT_TEST_AUDIOCPP_TTS_MODEL not all set");
    agent::AgentRuntime runtime;
    auto cycle = [&] {
        agent::Session llama = open(llama_config(m));
        chat(llama, 0);
        agent::Session sd = open(sd_config(m));
        draw(sd, 0);
        agent::Session tts = open(tts_config(m));
        speak(tts);
    };
    cycle(); // the first pass loads the libraries and warms the allocators
    const uint64_t after_first = agent::process_rss_bytes();
    for (int i = 0; i < 3; ++i) cycle();
    const uint64_t after_more = agent::process_rss_bytes();
    INFO("rss after the first cycle: " << after_first / (1024 * 1024) << " MiB, after three more: " << after_more / (1024 * 1024)
                                       << " MiB");
    if (after_first > 0) CHECK(after_more < after_first + 600ull * 1024 * 1024); // models are released; a leak of a whole model would show
}

#endif
