// Every engine capability, end to end, against the mock-all engine (deterministic stand-in output, no model files): through the
// public API in process and through the engine host process, through the command line, and the failure scenarios (engine error,
// empty answer, slow cancellable call).
#if defined(AGENT_TEST_MOCK_ALL_ENGINE_PATH)

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Ws {
    fs::path dir = fs::temp_directory_path() / "agent_test_mock_all";
    fs::path audio = dir / "in.wav";
    fs::path image = dir / "in.png";
    std::string audio_s = (dir / "in.wav").string(), image_s = (dir / "in.png").string(); // MediaView keeps string_views: these outlive the calls
    Ws() {
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir);
        std::ofstream(audio, std::ios::binary) << "RIFF....WAVEfmt ";
        std::ofstream(image, std::ios::binary) << "\x89PNG";
    }
    ~Ws() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

agent::Session open(const Ws &ws, agent::EngineIsolation isolation) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ALL_ENGINE_PATH;
    cfg.model = "mock.bin";
    cfg.workspace_dir = ws.dir.string();
    cfg.engine_isolation = isolation;
    auto s = agent::init(cfg);
    REQUIRE(s.ok);
    return std::move(s.value);
}

struct Call {
    std::string text;
    std::vector<agent::MediaView> media;
    uint32_t task = 0;
    std::string params;
    const volatile int32_t *cancel = nullptr;
};

agent::Result<agent::InferResponse> run(agent::Session &s, const Call &c) {
    agent::MessageView m;
    m.role = "user";
    m.content = c.text;
    m.media = c.media;
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&m, 1);
    req.task = c.task;
    req.params_json = c.params;
    req.cancel = c.cancel;
    return agent::run_task(s, c.task, req);
}

} // namespace

TEST_CASE("mock-all: every capability answers, in process and through the engine host", "[mock_all]") {
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        Ws ws;
        auto s = open(ws, isolation);
        const agent::MediaView audio{ws.audio_s, "audio/wav"};
        const agent::MediaView image{ws.image_s, "image/png"};

        auto asr = run(s, {.text = "", .media = {audio}, .task = AGENT_CAP_ASR});
        INFO(asr.error.message);
        REQUIRE(asr.ok);
        CHECK(asr.value.message.content == "mock transcript");

        auto ocr = run(s, {.text = "", .media = {image}, .task = AGENT_CAP_OCR});
        REQUIRE(ocr.ok);
        CHECK(ocr.value.message.content == "mock ocr text");

        auto tts = run(s, {.text = "say this", .task = AGENT_CAP_TTS});
        REQUIRE(tts.ok);
        REQUIRE_FALSE(tts.value.message.media.empty());
        CHECK(std::string(tts.value.message.media[0].mime) == "audio/wav");
        CHECK(fs::exists(std::string(tts.value.message.media[0].path)));

        for (auto [task, mime] : {std::pair<uint32_t, const char *>{AGENT_CAP_IMAGE_GEN, "image/png"}, std::pair<uint32_t, const char *>{AGENT_CAP_IMAGE_EDIT, "image/png"},
                                  std::pair<uint32_t, const char *>{AGENT_CAP_VIDEO_GEN, "video/mp4"}, std::pair<uint32_t, const char *>{AGENT_CAP_MESH_GEN, "model/gltf-binary"}}) {
            auto r = run(s, {.text = "a cat", .media = {image}, .task = task});
            INFO("task " << task);
            REQUIRE(r.ok);
            REQUIRE_FALSE(r.value.message.media.empty());
            CHECK(std::string(r.value.message.media[0].mime) == mime);
            CHECK(fs::exists(std::string(r.value.message.media[0].path)));
        }

        for (uint32_t task : {static_cast<uint32_t>(AGENT_CAP_VAD), static_cast<uint32_t>(AGENT_CAP_DIARIZE)}) {
            auto r = run(s, {.text = "", .media = {audio}, .task = task});
            REQUIRE(r.ok);
            CHECK(r.value.message.content.find("\"start\"") != std::string::npos);
            REQUIRE_FALSE(r.value.output_tensors.empty());
            CHECK(std::string(r.value.output_tensors[0].name) == "segments");
        }

        auto texts = std::vector<std::string_view>{"hello", "world"};
        auto emb = agent::embed(s, texts);
        REQUIRE(emb.ok);
        REQUIRE(emb.value.size() == 2);
        CHECK(emb.value[0].size() == 8);
        CHECK(emb.value[0] != emb.value[1]);

        auto docs = std::vector<std::string_view>{"a red fox", "blue sky", "red"};
        auto scores = agent::rerank(s, "red fox", docs);
        REQUIRE(scores.ok);
        REQUIRE(scores.value.size() == 3);
        CHECK(scores.value[0] > scores.value[1]);

        auto options = std::vector<std::string_view>{"billing", "shipping", "technical"};
        auto choice = agent::decide(s, "Where does this go?", options);
        REQUIRE(choice.ok);
        CHECK(choice.value.options.size() == 3);
        CHECK(choice.value.probs.size() == 3);
        auto score = agent::decide_score(s, "How good?", 1, 5);
        REQUIRE(score.ok);
        CHECK(score.value.probs.size() == 5);
        auto yes = agent::decide_yes_no(s, "Is it?");
        REQUIRE(yes.ok);
        CHECK(yes.value > 0.0f);
        CHECK(yes.value < 1.0f);
    }
}

TEST_CASE("mock-all: engine failures, empty answers and bad input are reported, not swallowed", "[mock_all]") {
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        Ws ws;
        auto s = open(ws, isolation);
        const agent::MediaView audio{ws.audio_s, "audio/wav"};

        auto failed = run(s, {.text = "mock-error", .task = AGENT_CAP_OCR});
        REQUIRE_FALSE(failed.ok);
        CHECK(failed.error.message.find("simulated engine failure") != std::string::npos);

        auto empty = run(s, {.text = "mock-empty", .task = AGENT_CAP_OCR});
        REQUIRE(empty.ok);
        CHECK(empty.value.message.content.empty());

        auto no_audio = run(s, {.text = "", .task = AGENT_CAP_ASR});
        CHECK_FALSE(no_audio.ok);
        auto no_text = run(s, {.text = "", .task = AGENT_CAP_TTS});
        CHECK_FALSE(no_text.ok);
        auto bad_task = run(s, {.text = "x", .task = AGENT_CAP_RERANK}); // one text part: no document to rank
        CHECK_FALSE(bad_task.ok);

        // The session still works after the failures.
        auto after = run(s, {.text = "", .media = {audio}, .task = AGENT_CAP_ASR});
        INFO(after.error.message);
        CHECK(after.ok);
    }
}

TEST_CASE("mock-all: a slow call finishes, and is cancelled when the flag is set", "[mock_all]") {
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        Ws ws;
        auto s = open(ws, isolation);
        auto done = run(s, {.text = "mock-slow", .task = AGENT_CAP_OCR});
        REQUIRE(done.ok);
        CHECK(done.value.message.content == "slow task finished");

        volatile int32_t cancel = 0;
        std::thread stopper([&cancel] {
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            cancel = 1;
        });
        const auto start = std::chrono::steady_clock::now();
        auto cancelled = run(s, {.text = "mock-slow", .task = AGENT_CAP_OCR, .cancel = &cancel});
        stopper.join();
        CHECK_FALSE(cancelled.ok);
        CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
    }
}

#endif

TEST_CASE("embed: inputs with media give one embedding each, text-only still works", "[mock_all][embed]") {
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        Ws ws;
        auto s = open(ws, isolation);
        std::vector<agent::Media> img{{ws.image_s, "image/png"}};
        std::vector<agent::EmbedInput> in{{"hello", {}}, {"", img}, {"caption", img}};
        auto r = agent::embed(s, in);
        INFO(r.error.message);
        REQUIRE(r.ok);
        REQUIRE(r.value.size() == 3);
        auto bad = agent::embed(s, std::vector<agent::EmbedInput>{{"", {}}});
        REQUIRE_FALSE(bad.ok);
    }
}

TEST_CASE("engine_config: models by role, nested sections and merged options", "[engine_config]") {
    const std::vector<agent::ModelSlot> slots = {{"model", "/m/chat.gguf"}, {"diffusion_model", "/m/flux.gguf"}, {"tts.model", "/m/k.gguf"}, {"tts.voice", "alba"}, {"asr.model", "/m/p.gguf"}};
    auto cfg = agent::engine_config("sdcpp", slots, R"({"threads": 4})");
    REQUIRE(cfg.ok);
    CHECK(cfg.value.provider == agent::AiProvider::Engine);
    CHECK(cfg.value.engine == "sdcpp");
    CHECK(cfg.value.model == "/m/chat.gguf");
    CHECK(cfg.value.engine_options ==
          R"({"diffusion_model":"/m/flux.gguf","tts":{"model":"/m/k.gguf","voice":"alba"},"asr":{"model":"/m/p.gguf"},"threads": 4})");
    CHECK_FALSE(agent::engine_config("", slots).ok);
    CHECK_FALSE(agent::engine_config("x", std::vector<agent::ModelSlot>{{"", "p"}}).ok);
    CHECK_FALSE(agent::engine_config("x", slots, "[1]").ok);
    auto only_model = agent::engine_config("llama", std::vector<agent::ModelSlot>{{"model", "a\"b"}});
    REQUIRE(only_model.ok);
    CHECK(only_model.value.engine_options.empty());
    CHECK(only_model.value.model == "a\"b");
}

TEST_CASE("infer: a response stays valid after later calls (it owns its memory)", "[mock_all][owning]") {
    Ws ws;
    auto s = open(ws, agent::EngineIsolation::InProcess);
    const agent::MediaView audio{ws.audio_s, "audio/wav"};
    auto first = run(s, {.text = "", .media = {audio}, .task = AGENT_CAP_ASR});
    REQUIRE(first.ok);
    agent::InferResponse kept = std::move(first.value);
    for (int i = 0; i < 3; ++i) {
        auto again = run(s, {.text = "a different request that reuses the session", .task = AGENT_CAP_OCR});
        REQUIRE(again.ok);
    }
    CHECK(kept.message.content == "mock transcript");
    CHECK(kept.storage != nullptr);
}
