// agent-engine-sherpa tests. Built only with -DAGENT_ENGINE_SHERPA=ON.
//
// Model-free tests always run. Real-model tests are gated on:
//   AGENT_TEST_SHERPA_TTS_DIR  extracted Kokoro dir (model.onnx, voices.bin, tokens.txt, lexicon-en-us.txt)
//   AGENT_TEST_SHERPA_ASR_DIR  extracted Whisper dir (*-encoder[.int8].onnx, *-decoder[.int8].onnx, *-tokens.txt)
//                              or NeMo transducer dir (encoder*.onnx, decoder*.onnx, joiner*.onnx, tokens.txt)
#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

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
    agent_request req{};

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

    agent_response resp{};
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
    agent_response resp{};
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

    // ASR task with text and no audio falls back to TTS.
    OneMessage fallback(AGENT_CAP_ASR, "Hello.", "", nullptr);
    REQUIRE(e->run(inst, &fallback.req, &resp) == AGENT_OK);
    REQUIRE(resp.message.n_parts == 2);
    e->free_response(inst, &resp);

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
}
