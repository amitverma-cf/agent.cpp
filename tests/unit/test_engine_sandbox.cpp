// Real engines under the default engine sandbox (Config::engine_sandbox, docs/security-sandbox.md):
// each test runs one real request through agent-engine-host (EngineIsolation::Process) without
// widening the policy. Linux only. Gated like the per-engine tests:
//   AGENT_TEST_LLAMA_MODEL, AGENT_TEST_LLAMA_VISION_MODEL + AGENT_TEST_LLAMA_VISION_MMPROJ,
//   AGENT_TEST_EMBED_MODEL, AGENT_TEST_RERANK_MODEL, AGENT_TEST_AUDIOCPP_DIR,
//   AGENT_TEST_SDCPP_MODEL. The ONNX test builds
//   its own model; the TRELLIS test checks the create() error paths (no weights needed).

#include "hal/sandbox.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(__linux__)

namespace {

namespace fs = std::filesystem;

const std::string kRoot = "/tmp/agent_test_engine_sandbox";

[[maybe_unused]] const char *env(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

[[maybe_unused]] std::string fresh_dir(const std::string &name) {
    const std::string dir = kRoot + "/" + name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir;
}

[[maybe_unused]] void write_file(const fs::path &p, const std::string &content) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << content;
}

[[maybe_unused]] std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

[[maybe_unused]] bool landlock() { return agent::hal::sandbox_support().landlock_abi > 0; }

// Every message the session logs at Warning or above, e.g. "sandbox: allow-list path ... skipped".
struct Log {
    std::vector<std::string> warnings;
    static void fn(agent::LogLevel level, std::string_view msg, void *ud) {
        if (level >= agent::LogLevel::Warning) static_cast<Log *>(ud)->warnings.emplace_back(msg);
    }
    bool clean() const {
        for (const auto &w : warnings)
            if (w.find("sandbox:") != std::string::npos) return false; // hal/sandbox warnings
        return true;
    }
};

// Process isolation under the untouched default engine_sandbox.
[[maybe_unused]] agent::Config sandboxed(const char *plugin, std::string model, const std::string &workspace, Log &log) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = plugin;
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_host_path = AGENT_TEST_ENGINE_HOST_PATH;
    cfg.model = std::move(model);
    cfg.workspace_dir = workspace;
    cfg.auto_default_tools = false;
    cfg.logger = Log::fn;
    cfg.logger_user_data = &log;
    return cfg;
}

[[maybe_unused]] agent::Session open(const agent::Config &cfg, const Log &log) {
    REQUIRE(cfg.engine_sandbox.enabled); // the default: no network, no /tmp, workspace read-only
    REQUIRE_FALSE(cfg.engine_sandbox.allow_tmp);
    REQUIRE(cfg.engine_sandbox.network == agent::SandboxNetwork::Deny);
    REQUIRE(cfg.engine_sandbox.read_only_paths.empty());
    REQUIRE(cfg.engine_sandbox.read_write_paths.empty());
    auto res = agent::init(cfg);
    INFO(res.error.message);
    REQUIRE(res.ok);
    if (landlock()) {
        for (const auto &w : log.warnings) UNSCOPED_INFO("log: " << w);
        CHECK(log.clean());
    }
    return std::move(res.value);
}

[[maybe_unused]] std::string solid_bmp(uint32_t channel) {
    const uint32_t w = 64, h = 64, row = w * 3, data = row * h;
    std::string bmp(54 + data, '\0');
    auto put32 = [&](size_t at, uint32_t v) {
        for (size_t i = 0; i < 4; ++i) bmp[at + i] = static_cast<char>((v >> (8 * i)) & 0xff);
    };
    bmp[0] = 'B';
    bmp[1] = 'M';
    put32(2, 54 + data);
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, h);
    bmp[26] = 1;  // planes
    bmp[28] = 24; // bits per pixel
    put32(34, data);
    for (uint32_t i = 0; i < w * h; ++i) bmp[54 + i * 3 + channel] = static_cast<char>(0xff);
    return bmp;
}

} // namespace

// ------------------------------------------------------------------ llama

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH

TEST_CASE("sandboxed engine host: llama generates text", "[engine_sandbox][llama][model]") {
    const char *model = env("AGENT_TEST_LLAMA_MODEL");
    if (!model) SKIP("AGENT_TEST_LLAMA_MODEL not set");
    Log log;
    auto cfg = sandboxed(AGENT_TEST_LLAMA_ENGINE_PATH, model, fresh_dir("llama"), log);
    cfg.context_window = 512;
    cfg.max_tokens = 8;
    cfg.temperature = 0.0f;
    agent::Session session = open(cfg, log);
    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "Say hello.");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    CHECK(memory.stats.completion_tokens > 0);
}

TEST_CASE("sandboxed engine host: llama reads an image from the workspace but not a secret elsewhere",
          "[engine_sandbox][llama][model][media]") {
    const char *model = env("AGENT_TEST_LLAMA_VISION_MODEL");
    const char *mmproj = env("AGENT_TEST_LLAMA_VISION_MMPROJ");
    if (!model || !mmproj) SKIP("AGENT_TEST_LLAMA_VISION_MODEL / AGENT_TEST_LLAMA_VISION_MMPROJ not set");
    const std::string ws = fresh_dir("llama_vision/ws");
    const std::string image = ws + "/red.bmp";
    write_file(image, solid_bmp(2));
    // The same image outside every allow-list: a stand-in for any secret file in /tmp.
    const std::string secret = fresh_dir("llama_vision/secret") + "/secret.bmp";
    write_file(secret, solid_bmp(2));

    Log log;
    auto cfg = sandboxed(AGENT_TEST_LLAMA_ENGINE_PATH, model, ws, log);
    cfg.context_window = 4096;
    cfg.max_tokens = 16;
    cfg.temperature = 0.0f;
    cfg.engine_options = std::string("{\"mmproj\":\"") + mmproj + "\"}";
    agent::Session session = open(cfg, log);

    {
        const agent::Media media[] = {{image, "image/bmp"}};
        agent::Conversation memory;
        auto reply = agent::run_turn_with_media(session, memory, "What color is this image? Answer with one word.", media);
        INFO(reply.error.message);
        REQUIRE(reply.ok);
        CHECK(lower(reply.value).find("red") != std::string::npos);
    }
    if (!landlock()) SKIP("Landlock unavailable: the secret-file check needs it");
    {
        const agent::Media media[] = {{secret, "image/bmp"}};
        agent::Conversation memory;
        auto reply = agent::run_turn_with_media(session, memory, "What color is this image? Answer with one word.", media);
        INFO("reply: " << reply.value << " error: " << reply.error.message);
        REQUIRE_FALSE(reply.ok);
        CHECK(reply.error.code != agent::ErrorCode::EngineCrashed);
    }
    // The host is still usable after the denied read.
    const agent::Media media[] = {{image, "image/bmp"}};
    agent::Conversation memory;
    CHECK(agent::run_turn_with_media(session, memory, "What color is this image? Answer with one word.", media).ok);
}

TEST_CASE("sandboxed engine host: llama embeds and reranks", "[engine_sandbox][llama][rag][model]") {
    const char *embed_model = env("AGENT_TEST_EMBED_MODEL");
    const char *rerank_model = env("AGENT_TEST_RERANK_MODEL");
    if (!embed_model && !rerank_model) SKIP("AGENT_TEST_EMBED_MODEL / AGENT_TEST_RERANK_MODEL not set");
    const std::string ws = fresh_dir("llama_rag");
    if (embed_model) {
        Log log;
        auto cfg = sandboxed(AGENT_TEST_LLAMA_ENGINE_PATH, embed_model, ws, log);
        cfg.context_window = 512;
        cfg.engine_options = R"({"mode":"embed"})";
        agent::Session session = open(cfg, log);
        const std::string_view texts[] = {"The cat is sleeping on the sofa.", "A kitten naps on the couch."};
        auto e = agent::embed(session, texts);
        INFO(e.error.message);
        REQUIRE(e.ok);
        REQUIRE(e.value.size() == 2);
        CHECK(e.value[0].size() >= 64);
    }
    if (rerank_model) {
        Log log;
        auto cfg = sandboxed(AGENT_TEST_LLAMA_ENGINE_PATH, rerank_model, ws, log);
        cfg.context_window = 512;
        cfg.engine_options = R"({"mode":"rerank"})";
        agent::Session session = open(cfg, log);
        const std::string_view docs[] = {"Bananas are a good source of potassium.", "Paris is the capital of France."};
        auto scores = agent::rerank(session, "What is the capital of France?", docs);
        INFO(scores.error.message);
        REQUIRE(scores.ok);
        REQUIRE(scores.value.size() == 2);
        CHECK(scores.value[1] > scores.value[0]);
    }
}

#endif // AGENT_TEST_LLAMA_ENGINE_PATH

// ------------------------------------------------------------------ ONNX Runtime

#ifdef AGENT_TEST_ONNX_ENGINE_PATH

namespace {
// Same hand-encoded model as test_onnx.cpp: sum = Add(a, b), float32 [2, 2].
constexpr unsigned char kAddModel[] = {
    0x08, 0x08, 0x12, 0x09, 0x61, 0x67, 0x65, 0x6e, 0x74, 0x2e, 0x63, 0x70, 0x70, 0x3a, 0x63, 0x0a, 0x15, 0x0a, 0x01, 0x61,
    0x0a, 0x01, 0x62, 0x12, 0x03, 0x73, 0x75, 0x6d, 0x1a, 0x03, 0x61, 0x64, 0x64, 0x22, 0x03, 0x41, 0x64, 0x64, 0x12, 0x09,
    0x61, 0x64, 0x64, 0x5f, 0x67, 0x72, 0x61, 0x70, 0x68, 0x5a, 0x13, 0x0a, 0x01, 0x61, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01,
    0x12, 0x08, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x5a, 0x13, 0x0a, 0x01, 0x62, 0x12, 0x0e, 0x0a, 0x0c, 0x08,
    0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x62, 0x15, 0x0a, 0x03, 0x73, 0x75, 0x6d, 0x12, 0x0e,
    0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d,
};
constexpr int64_t kShape[] = {2, 2};
} // namespace

TEST_CASE("sandboxed engine host: onnx runs a model", "[engine_sandbox][onnx]") {
    const std::string root = fresh_dir("onnx");
    const std::string model = root + "/models/add.onnx";
    write_file(model, std::string(reinterpret_cast<const char *>(kAddModel), sizeof(kAddModel)));
    Log log;
    // Default options: ORT picks its own thread count (reads /proc and /sys for the CPU topology).
    agent::Session session = open(sandboxed(AGENT_TEST_ONNX_ENGINE_PATH, model, root + "/ws", log), log);

    const float a[4] = {0.5f, -1.0f, 2.0f, 8.0f};
    const float b[4] = {0.5f, 1.0f, -2.0f, 2.0f};
    const agent::TensorView inputs[] = {
        {.name = "a", .dtype = agent::DType::Float32, .shape = kShape, .data = {reinterpret_cast<const uint8_t *>(a), sizeof(a)}},
        {.name = "b", .dtype = agent::DType::Float32, .shape = kShape, .data = {reinterpret_cast<const uint8_t *>(b), sizeof(b)}},
    };
    agent::MessageView msg{};
    msg.role = "user";
    msg.tensors = inputs;
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    auto out = agent::infer(session, req);
    INFO(out.error.message);
    REQUIRE(out.ok);
    REQUIRE(out.value.output_tensors.size() == 1);
    float got[4];
    REQUIRE(out.value.output_tensors[0].data.size() == sizeof(got));
    std::memcpy(got, out.value.output_tensors[0].data.data(), sizeof(got));
    CHECK(got[0] == 1.0f);
    CHECK(got[3] == 10.0f);
}

#endif // AGENT_TEST_ONNX_ENGINE_PATH

// ------------------------------------------------------------------ audio.cpp

#ifdef AGENT_TEST_AUDIOCPP_ENGINE_PATH

// AGENT_TEST_AUDIOCPP_DIR: the audio.cpp model folder (python3 tools/model_manager_v2.py install ... --models-root DIR) holding
// Supertonic-3-GGUF, Parakeet-TDT-0.6B-v3-GGUF, FireRedVAD-GGUF and Nemotron-3-Diarization-GGUF.
TEST_CASE("sandboxed engine host: audiocpp speaks, transcribes, detects speech and diarizes", "[engine_sandbox][audiocpp][model]") {
    const char *dir = env("AGENT_TEST_AUDIOCPP_DIR");
    if (!dir) SKIP("AGENT_TEST_AUDIOCPP_DIR not set");
    const std::string d(dir);
    auto model = [&](const char *rel) { return fs::exists(d + "/" + rel) ? d + "/" + rel : std::string(); };
    const std::string tts = model("Supertonic-3-GGUF/supertonic-3-q8_0.gguf");
    if (tts.empty()) SKIP("Supertonic-3-GGUF/supertonic-3-q8_0.gguf not in AGENT_TEST_AUDIOCPP_DIR");
    const std::string asr = model("Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf");
    const std::string vad = model("FireRedVAD-GGUF/firered-vad-f32.gguf");
    const std::string diar = model("Nemotron-3-Diarization-GGUF/nemotron-3-diarization-q8_0.gguf");
    std::string options = "{\"tts\":{\"model\":\"" + tts + "\",\"language\":\"en\"}";
    if (!asr.empty()) options += ",\"asr\":{\"model\":\"" + asr + "\"}";
    if (!vad.empty()) options += ",\"vad\":{\"model\":\"" + vad + "\"}";
    if (!diar.empty()) options += ",\"diarize\":{\"model\":\"" + diar + "\"}";
    options += "}";

    Log log;
    auto cfg = sandboxed(AGENT_TEST_AUDIOCPP_ENGINE_PATH, "", fresh_dir("audiocpp"), log);
    cfg.engine_options = options;
    agent::Session session = open(cfg, log);

    auto task = [&](uint32_t cap, std::string_view text, std::span<const agent::MediaView> media) {
        agent::MessageView msg{.role = "user", .content = text};
        msg.media = media;
        agent::InferRequest req;
        req.messages = std::span<const agent::MessageView>(&msg, 1);
        return agent::run_task(session, cap, req);
    };

    // TTS writes its WAV into workspace/artifacts (the one read-write directory).
    auto speech = task(AGENT_CAP_TTS, "Hello world, this is a test.", {});
    INFO(speech.error.message);
    REQUIRE(speech.ok);
    REQUIRE(speech.value.message.media.size() == 1);
    const std::string wav(speech.value.message.media[0].path);
    REQUIRE(fs::file_size(wav) > 44);
    const agent::MediaView audio[] = {{wav, "audio/wav"}};

    if (!asr.empty()) {
        auto heard = task(AGENT_CAP_ASR, "", audio);
        INFO(heard.error.message);
        REQUIRE(heard.ok);
        INFO("transcript: " << heard.value.message.content);
        CHECK(lower(std::string(heard.value.message.content)).find("hello") != std::string::npos);
    }
    if (!vad.empty()) {
        auto segments = task(AGENT_CAP_VAD, "", audio);
        INFO(segments.error.message);
        REQUIRE(segments.ok);
        CHECK(std::string(segments.value.message.content).find("\"start\"") != std::string::npos);
    }
    if (!diar.empty()) {
        auto speakers = task(AGENT_CAP_DIARIZE, "", audio);
        INFO(speakers.error.message);
        REQUIRE(speakers.ok);
        CHECK(std::string(speakers.value.message.content).find("\"speaker\"") != std::string::npos);
    }
}

#endif // AGENT_TEST_AUDIOCPP_ENGINE_PATH

// ------------------------------------------------------------------ stable-diffusion.cpp

#ifdef AGENT_TEST_SDCPP_ENGINE_PATH

TEST_CASE("sandboxed engine host: sdcpp generates an image", "[engine_sandbox][sdcpp][model]") {
    const char *model = env("AGENT_TEST_SDCPP_MODEL");
    if (!model) SKIP("AGENT_TEST_SDCPP_MODEL not set");
    Log log;
    const std::string ws = fresh_dir("sdcpp");
    auto cfg = sandboxed(AGENT_TEST_SDCPP_ENGINE_PATH, model, ws, log);
    cfg.engine_options = R"({"threads":2,"width":64,"height":64,"steps":1,"cfg_scale":1.0,"seed":42})";
    agent::Session session = open(cfg, log);
    agent::MessageView msg{.role = "user", .content = "a red apple on a wooden table"};
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    auto image = agent::run_task(session, AGENT_CAP_IMAGE_GEN, req);
    INFO(image.error.message);
    REQUIRE(image.ok);
    REQUIRE_FALSE(image.value.message.media.empty());
    const std::string png(image.value.message.media[0].path);
    CHECK(png.rfind(ws + "/artifacts/", 0) == 0);
    CHECK(fs::file_size(png) > 8);
}

#endif // AGENT_TEST_SDCPP_ENGINE_PATH

// ------------------------------------------------------------------ TRELLIS.2

#ifdef AGENT_TEST_TRELLIS_ENGINE_PATH

TEST_CASE("sandboxed engine host: trellis loads and reports model errors", "[engine_sandbox][trellis]") {
    // No weights needed: the host must start, load the plugin and see the model directory. A
    // permission problem would surface as a different message (or a crash) instead of these.
    const std::string root = fresh_dir("trellis");
    const std::string models = root + "/models";
    fs::create_directories(models);
    Log log;
    auto cfg = sandboxed(AGENT_TEST_TRELLIS_ENGINE_PATH, models, root + "/ws", log);
    auto empty = agent::init(cfg);
    REQUIRE_FALSE(empty.ok);
    CHECK(empty.error.code == agent::ErrorCode::ModelLoadFailed);
    INFO(empty.error.message);
    CHECK(empty.error.message.find("dino_f16.gguf") != std::string::npos);

    for (const char *name : {"dino_f16.gguf", "ss_flow_f16.gguf", "ss_dec_f16.gguf"}) write_file(models + "/" + name, "this is not a gguf");
    auto garbage = agent::init(cfg);
    REQUIRE_FALSE(garbage.ok);
    CHECK(garbage.error.code == agent::ErrorCode::ModelLoadFailed);
    INFO(garbage.error.message);
    CHECK(garbage.error.message.find("dino") != std::string::npos);
    if (landlock()) CHECK(log.clean());
}

#endif // AGENT_TEST_TRELLIS_ENGINE_PATH

#endif // __linux__
