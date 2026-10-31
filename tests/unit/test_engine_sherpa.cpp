// agent-engine-sherpa tests. Built only with -DAGENT_ENGINE_SHERPA=ON.
//
// Model-free tests always run. Real-model tests are gated on:
//   AGENT_TEST_SHERPA_TTS_DIR  extracted Kokoro dir (model.onnx, voices.bin, tokens.txt, lexicon-en-us.txt)
//   AGENT_TEST_SHERPA_ASR_DIR  extracted Whisper dir (*-encoder[.int8].onnx, *-decoder[.int8].onnx, *-tokens.txt)
//                              or NeMo transducer dir (encoder*.onnx, decoder*.onnx, joiner*.onnx, tokens.txt)
//   AGENT_TEST_SHERPA_VAD_MODEL silero_vad.onnx
//   AGENT_TEST_SHERPA_SEGMENTATION_MODEL + AGENT_TEST_SHERPA_EMBEDDING_MODEL  speaker diarization models
#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

const char *kOutDir = "/tmp/agent_test_sherpa_out";

agent_str c_str(std::string_view s) { return agent_str{s.data(), s.size()}; }
std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

const agent_engine_v1 *sherpa_engine() {
    auto loaded = agent::load_engine_plugin(AGENT_TEST_SHERPA_ENGINE_PATH);
    REQUIRE(loaded.ok);
    REQUIRE(loaded.value == "sherpa");
    return agent::find_engine("sherpa");
}

int32_t create(const agent_engine_v1 *e, const std::string &cfg, void **inst, std::string *err_out = nullptr) {
    char err[512] = {0};
    int32_t rc = e->create(c_str(cfg), inst, err, sizeof(err));
    if (err_out) *err_out = err;
    return rc;
}

struct Captured {
    size_t audio_bytes = 0;
    size_t audio_chunks = 0;
    std::string text;
    uint64_t last_step = 0, total = 0;
    volatile int32_t cancel = 0;
    bool cancel_on_first_chunk = false;
};

void on_chunk(void *ud, uint32_t kind, const void *data, size_t size) {
    auto *c = static_cast<Captured *>(ud);
    if (kind == AGENT_CHUNK_AUDIO) {
        c->audio_bytes += size;
        ++c->audio_chunks;
        if (c->cancel_on_first_chunk) c->cancel = 1;
    } else if (kind == AGENT_CHUNK_TEXT) {
        c->text.append(static_cast<const char *>(data), size);
    }
}

void on_progress(void *ud, uint64_t step, uint64_t total) {
    auto *c = static_cast<Captured *>(ud);
    c->last_step = step;
    c->total = total;
}

// One user message with an optional text part and an optional audio FILE part.
struct OneMessage {
    agent_part parts[2]{};
    agent_message msg{};
    agent_request req{.struct_size = sizeof(agent_request)};

    OneMessage(uint32_t task, std::string_view text, std::string_view audio_path, Captured *cap) {
        size_t n = 0;
        if (!text.empty()) {
            parts[n].kind = AGENT_PART_TEXT;
            parts[n++].text = c_str(text);
        }
        if (!audio_path.empty()) {
            parts[n].kind = AGENT_PART_FILE;
            parts[n].path = c_str(audio_path);
            parts[n++].mime = c_str("audio/wav");
        }
        msg.role = c_str("user");
        msg.parts = parts;
        msg.n_parts = n;
        req.task = task;
        req.messages = &msg;
        req.n_messages = 1;
        req.output_dir = c_str(kOutDir);
        if (cap) {
            req.on_chunk = on_chunk;
            req.on_progress = on_progress;
            req.user_data = cap;
            req.cancel = &cap->cancel;
        }
    }
};

const char *env(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

std::string tts_options() {
    const char *dir = env("AGENT_TEST_SHERPA_TTS_DIR");
    if (!dir) return {};
    const std::string d(dir);
    std::string lexicon;
    for (const char *name : {"lexicon-en-us.txt", "lexicon-us-en.txt", "lexicon.txt"})
        if (fs::exists(d + "/" + name)) {
            lexicon = d + "/" + name;
            break;
        }
    std::string o = "\"tts\":{\"kokoro\":{\"model\":\"" + d + "/model.onnx\",\"voices\":\"" + d + "/voices.bin\",\"tokens\":\"" + d +
                    "/tokens.txt\"";
    if (!lexicon.empty()) o += ",\"lexicon\":\"" + lexicon + "\"";
    o += "},\"speaker_id\":0,\"speed\":1.0}";
    return o;
}

// First file in dir whose name contains every needle, preferring an int8 variant.
std::string find_model_file(const fs::path &dir, std::initializer_list<std::string_view> needles) {
    std::string best;
    for (const auto &entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        bool match = true;
        for (auto n : needles) match = match && name.find(n) != std::string::npos;
        if (!match) continue;
        if (best.empty() || name.find("int8") != std::string::npos) best = entry.path().string();
    }
    return best;
}

std::string asr_options() {
    const char *dir = env("AGENT_TEST_SHERPA_ASR_DIR");
    if (!dir) return {};
    const fs::path d(dir);
    const std::string enc = find_model_file(d, {"encoder", ".onnx"});
    const std::string dec = find_model_file(d, {"decoder", ".onnx"});
    const std::string joiner = find_model_file(d, {"joiner", ".onnx"});
    const std::string tokens = find_model_file(d, {"tokens.txt"});
    if (!joiner.empty())
        return "\"asr\":{\"nemo_transducer\":{\"encoder\":\"" + enc + "\",\"decoder\":\"" + dec + "\",\"joiner\":\"" + joiner +
               "\",\"tokens\":\"" + tokens + "\"}}";
    return "\"asr\":{\"whisper\":{\"encoder\":\"" + enc + "\",\"decoder\":\"" + dec + "\",\"tokens\":\"" + tokens + "\"}}";
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // namespace

TEST_CASE("sherpa plugin loads with TTS and ASR capabilities", "[engines][sherpa]") {
    const agent_engine_v1 *e = sherpa_engine();
    REQUIRE(e != nullptr);
    REQUIRE(e->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE(std::string_view(e->name) == "sherpa");
    REQUIRE((e->caps & AGENT_CAP_TTS) != 0);
    REQUIRE((e->caps & AGENT_CAP_ASR) != 0);
    REQUIRE((e->caps & AGENT_CAP_TEXT_GEN) == 0);
    REQUIRE(e->flags == 0);
}

TEST_CASE("sherpa without components creates but every task is UNSUPPORTED", "[engines][sherpa]") {
    const agent_engine_v1 *e = sherpa_engine();
    void *inst = nullptr;
    REQUIRE(create(e, R"({"model":"","options":{}})", &inst) == AGENT_OK);
    REQUIRE(inst != nullptr);

    agent_response resp{.struct_size = sizeof(agent_response)};
    OneMessage tts(AGENT_CAP_TTS, "hello", "", nullptr);
    REQUIRE(e->run(inst, &tts.req, &resp) == AGENT_ERR_UNSUPPORTED);
    REQUIRE(std::string_view(e->last_error(inst)).find("tts") != std::string_view::npos);

    OneMessage asr(AGENT_CAP_ASR, "", "/tmp/whatever.wav", nullptr);
    REQUIRE(e->run(inst, &asr.req, &resp) == AGENT_ERR_UNSUPPORTED);
    REQUIRE(std::string_view(e->last_error(inst)).find("asr") != std::string_view::npos);

    OneMessage embed(AGENT_CAP_EMBED, "hello", "", nullptr);
    REQUIRE(e->run(inst, &embed.req, &resp) == AGENT_ERR_UNSUPPORTED);
    e->destroy(inst);

    // No "options" key at all is also fine.
    inst = nullptr;
    REQUIRE(create(e, R"({"model":""})", &inst) == AGENT_OK);
    e->destroy(inst);
}

TEST_CASE("sherpa create fails cleanly on bad configs and model paths", "[engines][sherpa]") {
    const agent_engine_v1 *e = sherpa_engine();
    void *inst = nullptr;
    std::string err;

    REQUIRE(create(e, "not json", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(inst == nullptr);

    REQUIRE(create(e, R"({"options":{"tts":{"kokoro":{"model":"/nonexistent/model.onnx","voices":"/nonexistent/voices.bin","tokens":"/nonexistent/tokens.txt"}}}})",
                   &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);
    REQUIRE(err.find("/nonexistent/model.onnx") != std::string::npos);

    REQUIRE(create(e, R"({"options":{"tts":{"kokoro":{"model":"/nonexistent/model.onnx"}}}})", &inst, &err) ==
            AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(create(e, R"({"options":{"tts":{"kokoro":{"voices":"x"}}}})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(err.find("tts.kokoro.model") != std::string::npos);
    REQUIRE(create(e, R"({"options":{"tts":{"vits":{}}}})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);

    REQUIRE(create(e, R"({"options":{"asr":{"whisper":{"encoder":"/nonexistent/e.onnx","decoder":"/nonexistent/d.onnx","tokens":"/nonexistent/t.txt"}}}})",
                   &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(create(e, R"({"options":{"asr":{"nemo_transducer":{"encoder":"/nonexistent/e.onnx"}}}})", &inst, &err) ==
            AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(create(e, R"({"options":{"asr":{"paraformer":{}}}})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);

    // Note: files that exist but are not valid models are not tested here: sherpa-onnx calls exit()
    // on some of those (e.g. a malformed tokens file), which would take the test runner down.
    REQUIRE(inst == nullptr);
}

TEST_CASE("sherpa Kokoro TTS writes a WAV and Whisper transcribes it", "[engines][sherpa][model]") {
    const std::string tts = tts_options();
    if (tts.empty()) SKIP("AGENT_TEST_SHERPA_TTS_DIR not set");
    const std::string asr = asr_options();

    const agent_engine_v1 *e = sherpa_engine();
    fs::create_directories(kOutDir);
    void *inst = nullptr;
    std::string err;
    const std::string cfg = "{\"model\":\"\",\"options\":{\"threads\":2," + tts + (asr.empty() ? "" : "," + asr) + "}}";
    REQUIRE(create(e, cfg, &inst, &err) == AGENT_OK);
    INFO(err);

    // TTS through the explicit task.
    Captured cap;
    OneMessage tts_req(AGENT_CAP_TTS, "Hello world.", "", &cap);
    agent_response resp{.struct_size = sizeof(agent_response)};
    REQUIRE(e->run(inst, &tts_req.req, &resp) == AGENT_OK);
    REQUIRE(resp.message.n_parts == 2);
    REQUIRE(resp.message.parts[0].kind == AGENT_PART_TEXT);
    REQUIRE(resp.message.parts[1].kind == AGENT_PART_FILE);
    REQUIRE(sv(resp.message.parts[1].mime) == "audio/wav");
    const std::string wav(sv(resp.message.parts[1].path));
    e->free_response(inst, &resp);
    REQUIRE(wav.starts_with(kOutDir));
    REQUIRE(cap.audio_chunks > 0);
    REQUIRE(cap.audio_bytes > 0);
    REQUIRE(cap.total == 1000);

    std::ifstream f(wav, std::ios::binary);
    REQUIRE(f.good());
    char header[44] = {0};
    f.read(header, sizeof(header));
    REQUIRE(f.gcount() == 44);
    REQUIRE(std::memcmp(header, "RIFF", 4) == 0);
    REQUIRE(std::memcmp(header + 8, "WAVE", 4) == 0);
    uint32_t data_bytes = 0;
    std::memcpy(&data_bytes, header + 40, 4);
    REQUIRE(data_bytes > 0);

    // A second synthesis gets a different file name.
    OneMessage again(AGENT_CAP_TTS, "Hello again.", "", nullptr);
    REQUIRE(e->run(inst, &again.req, &resp) == AGENT_OK);
    REQUIRE(sv(resp.message.parts[1].path) != wav);
    e->free_response(inst, &resp);

    // Cancel from inside the chunk callback stops a multi-sentence synthesis.
    Captured cancel_cap;
    cancel_cap.cancel_on_first_chunk = true;
    OneMessage cancel_req(AGENT_CAP_TTS, "First sentence. Second sentence. Third sentence.", "", &cancel_cap);
    REQUIRE(e->run(inst, &cancel_req.req, &resp) == AGENT_ERR_CANCELLED);
    REQUIRE(cancel_cap.audio_chunks == 1);

    if (asr.empty()) {
        e->destroy(inst);
        SKIP("AGENT_TEST_SHERPA_ASR_DIR not set; TTS verified only");
    }

    // ASR task routed by content: the generated 24 kHz WAV is resampled by sherpa-onnx.
    Captured asr_cap;
    OneMessage asr_req(AGENT_CAP_ASR, "", wav, &asr_cap);
    REQUIRE(e->run(inst, &asr_req.req, &resp) == AGENT_OK);
    REQUIRE(resp.message.n_parts == 1);
    REQUIRE(resp.message.parts[0].kind == AGENT_PART_TEXT);
    const std::string transcript(sv(resp.message.parts[0].text));
    e->free_response(inst, &resp);
    INFO("transcript: " << transcript);
    REQUIRE_FALSE(transcript.empty());
    REQUIRE(lower(transcript).find("hello") != std::string::npos);
    REQUIRE(asr_cap.text == transcript);

    // Real recordings shipped with sherpa-onnx ASR model archives (16 kHz and 8 kHz), if present.
    for (const char *name : {"test_wavs/0.wav", "test_wavs/8k.wav"}) {
        const std::string path = std::string(env("AGENT_TEST_SHERPA_ASR_DIR")) + "/" + name;
        if (!fs::exists(path)) continue;
        OneMessage rec(AGENT_CAP_ASR, "", path, nullptr);
        REQUIRE(e->run(inst, &rec.req, &resp) == AGENT_OK);
        INFO(name << ": " << sv(resp.message.parts[0].text));
        REQUIRE(resp.message.parts[0].text.size > 10);
        e->free_response(inst, &resp);
    }

    // An automatic ASR task (the host's pick) with text and no audio falls back to TTS ...
    OneMessage fallback(AGENT_CAP_ASR, "Hello.", "", nullptr);
    fallback.req.flags = AGENT_REQUEST_FLAG_AUTO_TASK;
    REQUIRE(e->run(inst, &fallback.req, &resp) == AGENT_OK);
    REQUIRE(resp.message.n_parts == 2);
    e->free_response(inst, &resp);
    // ... an explicit one is an error.
    OneMessage explicit_asr(AGENT_CAP_ASR, "Hello.", "", nullptr);
    REQUIRE(e->run(inst, &explicit_asr.req, &resp) == AGENT_ERR_INVALID_CONFIG);

    // A non-WAV file is reported, not crashed on.
    const std::string bogus = std::string(kOutDir) + "/bogus.wav";
    std::ofstream(bogus) << "not a wave file";
    OneMessage bad(AGENT_CAP_ASR, "", bogus, nullptr);
    REQUIRE(e->run(inst, &bad.req, &resp) == AGENT_ERR_UNSUPPORTED);
    REQUIRE(std::string_view(e->last_error(inst)).find("WAV") != std::string_view::npos);
    e->destroy(inst);
}

TEST_CASE("sherpa engine speaks and listens through a Session", "[engines][sherpa][model]") {
    const std::string tts = tts_options();
    const std::string asr = asr_options();
    if (tts.empty() || asr.empty()) SKIP("AGENT_TEST_SHERPA_TTS_DIR and AGENT_TEST_SHERPA_ASR_DIR not both set");

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_SHERPA_ENGINE_PATH;
    cfg.workspace_dir = "/tmp/agent_test_sherpa_ws";
    cfg.engine_options = "{" + tts + "," + asr + "}";
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "sherpa_session";
    auto spoken = agent::run_turn(session, memory, "Hello world.");
    REQUIRE(spoken.ok);
    const agent::Message &out = memory.history.back();
    REQUIRE(out.media.size() == 1);
    REQUIRE(out.media[0].mime == "audio/wav");
    const std::string wav = out.media[0].path;
    REQUIRE(fs::file_size(wav) > 44);

    agent::FlowMemory listen;
    listen.id = "sherpa_session_asr";
    const agent::Media media[] = {{wav, "audio/wav"}};
    auto heard = agent::run_turn_with_media(session, listen, "", media);
    REQUIRE(heard.ok);
    INFO("transcript: " << heard.value);
    REQUIRE(lower(heard.value).find("hello") != std::string::npos);

    // Explicit tasks through the public API.
    agent::MessageView msg{.role = "user", .content = "Hello there."};
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    auto speech = agent::run_task(session, AGENT_CAP_TTS, req);
    REQUIRE(speech.ok);
    REQUIRE(speech.value.message.media.size() == 1);
    auto no_audio = agent::run_task(session, AGENT_CAP_ASR, req);
    REQUIRE_FALSE(no_audio.ok);
    REQUIRE(no_audio.error.message.find("audio part") != std::string::npos);
}

// ------------------------------------------------------------------ VAD / diarization
//
// Gated on AGENT_TEST_SHERPA_VAD_MODEL (silero_vad.onnx) and, for diarization,
// AGENT_TEST_SHERPA_SEGMENTATION_MODEL (pyannote segmentation-3.0 model.onnx) +
// AGENT_TEST_SHERPA_EMBEDDING_MODEL (a 3D-Speaker / NeMo speaker embedding .onnx). The speech comes
// from Kokoro (AGENT_TEST_SHERPA_TTS_DIR), so every model test also needs that.

namespace {

// One user message whose audio is an F32 tensor of 16 kHz samples, plus an optional text command.
struct TensorMessage {
    agent_part parts[2]{};
    int64_t shape[1]{};
    agent_message msg{};
    agent_request req{};

    TensorMessage(uint32_t task, const std::vector<float> &samples, std::string_view text = {}) {
        size_t n = 0;
        if (!text.empty()) {
            parts[n].kind = AGENT_PART_TEXT;
            parts[n++].text = c_str(text);
        }
        shape[0] = static_cast<int64_t>(samples.size());
        parts[n].kind = AGENT_PART_TENSOR;
        parts[n].name = c_str("audio");
        parts[n].dtype = AGENT_DTYPE_F32;
        parts[n].shape = shape;
        parts[n].ndim = 1;
        parts[n].data = samples.data();
        parts[n++].size = samples.size() * sizeof(float);
        msg.role = c_str("user");
        msg.parts = parts;
        msg.n_parts = n;
        req.task = task;
        req.messages = &msg;
        req.n_messages = 1;
        req.output_dir = c_str(kOutDir);
    }
};

// 16-bit PCM mono WAV as written by sherpa-onnx (44-byte header) -> samples resampled to 16 kHz.
std::vector<float> read_wav_16k(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    REQUIRE(bytes.size() > 44);
    uint32_t rate = 0;
    std::memcpy(&rate, bytes.data() + 24, 4);
    const size_t n = (bytes.size() - 44) / 2;
    std::vector<float> in(n);
    for (size_t i = 0; i < n; ++i) {
        int16_t v;
        std::memcpy(&v, bytes.data() + 44 + 2 * i, 2);
        in[i] = static_cast<float>(v) / 32768.0f;
    }
    const double step = static_cast<double>(rate) / 16000.0;
    std::vector<float> out(static_cast<size_t>(static_cast<double>(n) / step));
    for (size_t i = 0; i < out.size(); ++i) out[i] = in[std::min(n - 1, static_cast<size_t>(static_cast<double>(i) * step))];
    return out;
}

void append_silence(std::vector<float> &v, double seconds) { v.insert(v.end(), static_cast<size_t>(seconds * 16000), 0.0f); }

// Speaks text with the instance and returns it as 16 kHz samples.
std::vector<float> speak(const agent_engine_v1 *e, void *tts, std::string_view text) {
    OneMessage m(AGENT_CAP_TTS, text, "", nullptr);
    agent_response resp{};
    REQUIRE(e->run(tts, &m.req, &resp) == AGENT_OK);
    const std::string wav(sv(resp.message.parts[1].path));
    e->free_response(tts, &resp);
    return read_wav_16k(wav);
}

struct Seg {
    double start, end;
    int speaker;
    bool open;
};

// Minimal parser for the engine's [{"start":..,"end":..[,"speaker":..][,"open":true]}] output.
std::vector<Seg> parse_segments(std::string_view json) {
    std::vector<Seg> out;
    size_t at = 0;
    while ((at = json.find("{\"start\":", at)) != std::string_view::npos) {
        Seg s{0, 0, -1, false};
        s.start = std::strtod(std::string(json.substr(at + 9, 20)).c_str(), nullptr);
        const size_t e = json.find("\"end\":", at);
        s.end = std::strtod(std::string(json.substr(e + 6, 20)).c_str(), nullptr);
        const size_t close = json.find('}', at);
        const size_t sp = json.find("\"speaker\":", at);
        if (sp != std::string_view::npos && sp < close) s.speaker = std::atoi(std::string(json.substr(sp + 10, 8)).c_str());
        const size_t op = json.find("\"open\":true", at);
        s.open = op != std::string_view::npos && op < close;
        out.push_back(s);
        at = close;
    }
    return out;
}

std::string vad_options(bool streaming) {
    const char *model = env("AGENT_TEST_SHERPA_VAD_MODEL");
    if (!model) return {};
    return std::string("\"vad\":{\"silero\":{\"model\":\"") + model +
           "\"},\"min_silence\":0.4,\"streaming\":" + (streaming ? "true" : "false") + "}";
}

} // namespace

TEST_CASE("sherpa VAD and diarization configs are validated", "[engines][sherpa]") {
    const agent_engine_v1 *e = sherpa_engine();
    REQUIRE((e->caps & AGENT_CAP_VAD) != 0);
    REQUIRE((e->caps & AGENT_CAP_DIARIZE) != 0);
    void *inst = nullptr;
    std::string err;
    REQUIRE(create(e, R"({"options":{"vad":{"silero":{}}}})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(err.find("vad.silero.model") != std::string::npos);
    REQUIRE(create(e, R"({"options":{"vad":{"silero":{"model":"/nonexistent/silero_vad.onnx"}}}})", &inst, &err) ==
            AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(create(e, R"({"options":{"vad":{"webrtc":{}}}})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(create(e, R"({"options":{"diarize":{"pyannote":{"model":"x"}}}})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(
        create(e, R"({"options":{"diarize":{"pyannote":{"model":"/nonexistent/seg.onnx"},"embedding":{"model":"/nonexistent/emb.onnx"}}}})",
               &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);

    // No VAD configured: a VAD request is UNSUPPORTED; a VAD request without audio is a config error.
    REQUIRE(create(e, R"({"options":{}})", &inst) == AGENT_OK);
    agent_response resp{};
    const std::vector<float> samples(1600, 0.0f);
    TensorMessage vad(AGENT_CAP_VAD, samples);
    REQUIRE(e->run(inst, &vad.req, &resp) == AGENT_ERR_UNSUPPORTED);
    REQUIRE(std::string_view(e->last_error(inst)).find("vad") != std::string_view::npos);
    TensorMessage dia(AGENT_CAP_DIARIZE, samples);
    REQUIRE(e->run(inst, &dia.req, &resp) == AGENT_ERR_UNSUPPORTED);
    OneMessage no_audio(AGENT_CAP_VAD, "hello", "", nullptr);
    REQUIRE(e->run(inst, &no_audio.req, &resp) == AGENT_ERR_INVALID_CONFIG);
    // A non-F32 tensor is not audio.
    TensorMessage bad(AGENT_CAP_VAD, samples);
    bad.parts[0].dtype = AGENT_DTYPE_I32;
    REQUIRE(e->run(inst, &bad.req, &resp) == AGENT_ERR_INVALID_CONFIG);
    e->destroy(inst);
}

TEST_CASE("sherpa Silero VAD finds speech offline and streaming", "[engines][sherpa][model]") {
    const std::string tts = tts_options();
    const std::string vad = vad_options(false);
    if (tts.empty() || vad.empty()) SKIP("AGENT_TEST_SHERPA_TTS_DIR and AGENT_TEST_SHERPA_VAD_MODEL not both set");
    const agent_engine_v1 *e = sherpa_engine();
    fs::create_directories(kOutDir);
    void *speaker = nullptr;
    REQUIRE(create(e, "{\"options\":{" + tts + "}}", &speaker) == AGENT_OK);

    // 0.5 s silence | speech A | 1.2 s silence | speech B | 0.6 s silence
    std::vector<float> audio;
    append_silence(audio, 0.5);
    const std::vector<float> a = speak(e, speaker, "Hello world, this is the first sentence.");
    audio.insert(audio.end(), a.begin(), a.end());
    const double a_end = static_cast<double>(audio.size()) / 16000;
    append_silence(audio, 1.2);
    const double b_start = static_cast<double>(audio.size()) / 16000;
    const std::vector<float> b = speak(e, speaker, "And here comes the second one.");
    audio.insert(audio.end(), b.begin(), b.end());
    const double b_end = static_cast<double>(audio.size()) / 16000;
    append_silence(audio, 0.6);
    e->destroy(speaker);
    INFO("speech A 0.50-" << a_end << " s, speech B " << b_start << "-" << b_end << " s");

    // Offline, explicit task, tensor input: two segments + a [2, 2] tensor.
    void *inst = nullptr;
    std::string err;
    REQUIRE(create(e, "{\"options\":{" + vad + "}}", &inst, &err) == AGENT_OK);
    TensorMessage offline(AGENT_CAP_VAD, audio);
    agent_response resp{};
    REQUIRE(e->run(inst, &offline.req, &resp) == AGENT_OK);
    REQUIRE(resp.message.n_parts == 2);
    const std::string json(sv(resp.message.parts[0].text));
    INFO("offline: " << json);
    const std::vector<Seg> segs = parse_segments(json);
    REQUIRE(segs.size() == 2);
    REQUIRE(segs[0].start == Catch::Approx(0.5).margin(0.35));
    REQUIRE(segs[0].end == Catch::Approx(a_end).margin(0.5));
    REQUIRE(segs[1].start == Catch::Approx(b_start).margin(0.35));
    REQUIRE(segs[1].end == Catch::Approx(b_end).margin(0.5));
    const agent_part &tensor = resp.message.parts[1];
    REQUIRE(tensor.kind == AGENT_PART_TENSOR);
    REQUIRE(sv(tensor.name) == "segments");
    REQUIRE(tensor.ndim == 2);
    REQUIRE(tensor.shape[0] == 2);
    REQUIRE(tensor.shape[1] == 2);
    REQUIRE(static_cast<const float *>(tensor.data)[2] == Catch::Approx(segs[1].start).margin(0.001));
    e->free_response(inst, &resp);

    // A VAD-only instance answers the ASR task (what hosts without an explicit task send) with segments.
    TensorMessage routed(AGENT_CAP_ASR, audio);
    REQUIRE(e->run(inst, &routed.req, &resp) == AGENT_OK);
    REQUIRE(parse_segments(sv(resp.message.parts[0].text)).size() == 2);
    e->free_response(inst, &resp);
    e->destroy(inst);

    // Streaming: 100 ms chunks; segments come out as they finish, an "open" one while speech goes on.
    REQUIRE(create(e, "{\"options\":{" + vad_options(true) + "}}", &inst, &err) == AGENT_OK);
    std::vector<Seg> finished;
    bool saw_open = false;
    for (size_t off = 0; off < audio.size(); off += 1600) {
        const std::vector<float> chunk(audio.begin() + static_cast<std::ptrdiff_t>(off),
                                       audio.begin() + static_cast<std::ptrdiff_t>(std::min(audio.size(), off + 1600)));
        TensorMessage m(AGENT_CAP_VAD, chunk, off == 0 ? "reset" : "");
        REQUIRE(e->run(inst, &m.req, &resp) == AGENT_OK);
        for (const Seg &s : parse_segments(sv(resp.message.parts[0].text))) {
            if (s.open) saw_open = true;
            else finished.push_back(s);
        }
        e->free_response(inst, &resp);
    }
    TensorMessage flush(AGENT_CAP_VAD, std::vector<float>(160, 0.0f), "flush");
    REQUIRE(e->run(inst, &flush.req, &resp) == AGENT_OK);
    for (const Seg &s : parse_segments(sv(resp.message.parts[0].text))) finished.push_back(s);
    e->free_response(inst, &resp);
    REQUIRE(saw_open);
    REQUIRE(finished.size() == 2);
    REQUIRE(finished[0].start == Catch::Approx(segs[0].start).margin(0.1));
    REQUIRE(finished[1].end == Catch::Approx(segs[1].end).margin(0.15));

    // "reset" starts the clock again.
    TensorMessage again(AGENT_CAP_VAD, audio, "reset");
    REQUIRE(e->run(inst, &again.req, &resp) == AGENT_OK);
    const std::vector<Seg> again_segs = parse_segments(sv(resp.message.parts[0].text));
    REQUIRE_FALSE(again_segs.empty());
    REQUIRE(again_segs[0].start == Catch::Approx(segs[0].start).margin(0.1));
    e->free_response(inst, &resp);
    e->destroy(inst);
}

TEST_CASE("sherpa diarization tells two Kokoro voices apart", "[engines][sherpa][model]") {
    const std::string tts = tts_options();
    const char *seg_model = env("AGENT_TEST_SHERPA_SEGMENTATION_MODEL");
    const char *emb_model = env("AGENT_TEST_SHERPA_EMBEDDING_MODEL");
    if (tts.empty() || !seg_model || !emb_model)
        SKIP("AGENT_TEST_SHERPA_TTS_DIR, AGENT_TEST_SHERPA_SEGMENTATION_MODEL and AGENT_TEST_SHERPA_EMBEDDING_MODEL not all set");
    const agent_engine_v1 *e = sherpa_engine();
    fs::create_directories(kOutDir);

    // Speaker 0 (af) and speaker 9 (bm_george) of kokoro-en-v0_19 take turns.
    std::string tts_b = tts;
    const size_t sid = tts_b.find("\"speaker_id\":0");
    REQUIRE(sid != std::string::npos);
    tts_b.replace(sid, 14, "\"speaker_id\":9");
    void *voice_a = nullptr, *voice_b = nullptr;
    REQUIRE(create(e, "{\"options\":{" + tts + "}}", &voice_a) == AGENT_OK);
    REQUIRE(create(e, "{\"options\":{" + tts_b + "}}", &voice_b) == AGENT_OK);
    std::vector<float> audio;
    append_silence(audio, 0.3);
    for (const auto &[voice, text] : {std::pair{voice_a, "Good morning. Did you sleep well last night? I hope the noise did not wake you."},
                                      std::pair{voice_b, "I slept very well, thank you. The room was quiet and the bed was comfortable."},
                                      std::pair{voice_a, "That is great to hear. Shall we have breakfast together before the meeting?"}}) {
        const std::vector<float> s = speak(e, voice, text);
        audio.insert(audio.end(), s.begin(), s.end());
        append_silence(audio, 0.8);
    }
    e->destroy(voice_a);
    e->destroy(voice_b);

    void *inst = nullptr;
    std::string err;
    const std::string cfg = std::string("{\"options\":{\"diarize\":{\"pyannote\":{\"model\":\"") + seg_model +
                            "\"},\"embedding\":{\"model\":\"" + emb_model + "\"},\"num_speakers\":2}}}";
    REQUIRE(create(e, cfg, &inst, &err) == AGENT_OK);
    INFO(err);
    Captured cap;
    TensorMessage m(AGENT_CAP_DIARIZE, audio);
    m.req.on_progress = on_progress;
    m.req.user_data = &cap;
    agent_response resp{};
    REQUIRE(e->run(inst, &m.req, &resp) == AGENT_OK);
    const std::string json(sv(resp.message.parts[0].text));
    INFO("diarization: " << json);
    const std::vector<Seg> segs = parse_segments(json);
    REQUIRE(segs.size() >= 3);
    REQUIRE(resp.message.parts[1].shape[1] == 3);
    // The speaker in the middle of the audio differs from the one at the start and the end.
    const double total = static_cast<double>(audio.size()) / 16000;
    auto speaker_at = [&](double t) {
        for (const Seg &s : segs)
            if (s.start <= t && t <= s.end) return s.speaker;
        return -1;
    };
    const int first = segs.front().speaker, last = segs.back().speaker, middle = speaker_at(total / 2);
    REQUIRE(first >= 0);
    REQUIRE(first == last);
    REQUIRE(std::any_of(segs.begin(), segs.end(), [&](const Seg &s) { return s.speaker != first; }));
    if (middle >= 0) REQUIRE(middle != first); // total / 2 may fall into a pause
    REQUIRE(cap.total > 0);
    e->free_response(inst, &resp);
    e->destroy(inst);
}
