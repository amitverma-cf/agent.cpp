// agent-engine-whisper tests. Built only with -DAGENT_ENGINE_WHISPER=ON.
//
// Model-free tests always run. Real-model tests are gated on:
//   AGENT_TEST_WHISPER_MODEL      ggml whisper model, e.g. ggml-tiny.en.bin
//   AGENT_TEST_WHISPER_VAD_MODEL  ggml Silero VAD model, e.g. ggml-silero-v5.1.2.bin
// The speech is whisper.cpp's own samples/jfk.wav (AGENT_TEST_WHISPER_SAMPLES_DIR).
#include "speech_engine_test_util.hpp"

#include <filesystem>

namespace fs = std::filesystem;
using namespace speech_test;

namespace {

const char *kOutDir = "/tmp/agent_test_whisper_out";

const agent_engine_v1 *whisper_engine() { return load(AGENT_TEST_WHISPER_ENGINE_PATH, "whisper"); }

// A static string: request parts only borrow the path.
const std::string &jfk_wav() {
    static const std::string path = std::string(AGENT_TEST_WHISPER_SAMPLES_DIR) + "/jfk.wav";
    return path;
}

} // namespace

TEST_CASE("whisper plugin loads with ASR and VAD capabilities", "[engines][whisper]") {
    const agent_engine_v1 *e = whisper_engine();
    REQUIRE(e != nullptr);
    REQUIRE(e->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE(e->struct_size >= AGENT_ENGINE_V2_SIZE);
    REQUIRE(std::string_view(e->name) == "whisper");
    REQUIRE(e->caps == (AGENT_CAP_ASR | AGENT_CAP_VAD));
    REQUIRE(e->flags == 0);
}

TEST_CASE("whisper create fails cleanly on bad configs and model paths", "[engines][whisper]") {
    const agent_engine_v1 *e = whisper_engine();
    void *inst = nullptr;
    std::string err;
    REQUIRE(create(e, "not json", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(create(e, R"({"model":""})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(err.find("config.model") != std::string::npos);
    REQUIRE(create(e, R"({"model":"/nonexistent/ggml-tiny.en.bin"})", &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(err.find("/nonexistent/ggml-tiny.en.bin") != std::string::npos);

    // A file that exists but is not a whisper model: whisper.cpp rejects it without crashing.
    fs::create_directories(kOutDir);
    const std::string bogus = std::string(kOutDir) + "/bogus.bin";
    std::ofstream(bogus) << "definitely not a ggml model";
    REQUIRE(create(e, "{\"model\":\"" + bogus + "\"}", &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);
}

TEST_CASE("whisper transcribes jfk.wav", "[engines][whisper][model]") {
    const char *model = env("AGENT_TEST_WHISPER_MODEL");
    if (!model) SKIP("AGENT_TEST_WHISPER_MODEL not set");
    const char *vad_model = env("AGENT_TEST_WHISPER_VAD_MODEL");

    const agent_engine_v1 *e = whisper_engine();
    fs::create_directories(kOutDir);
    void *inst = nullptr;
    std::string err;
    std::string cfg = std::string("{\"model\":\"") + model + "\",\"options\":{\"threads\":2";
    if (vad_model) cfg += std::string(",\"vad\":{\"model\":\"") + vad_model + "\"}";
    cfg += "}}";
    REQUIRE(create(e, cfg, &inst, &err) == AGENT_OK);
    INFO(err);

    // Explicit ASR task, FILE part; segments stream as TEXT chunks that add up to the transcript.
    Captured cap;
    OneMessage asr(AGENT_CAP_ASR, "", jfk_wav(), &cap, kOutDir);
    agent_response resp{.struct_size = sizeof(agent_response)};
    const int32_t asr_rc = e->run(inst, &asr.req, &resp);
    INFO("last_error: " << e->last_error(inst));
    REQUIRE(asr_rc == AGENT_OK);
    REQUIRE(sv(resp.message.role) == "assistant");
    REQUIRE(resp.message.n_parts == 1);
    REQUIRE(resp.message.parts[0].kind == AGENT_PART_TEXT);
    const std::string transcript(sv(resp.message.parts[0].text));
    e->free_response(inst, &resp);
    INFO("transcript: " << transcript);
    const std::string t = lower(transcript);
    REQUIRE(t.find("ask not what your country can do for you") != std::string::npos);
    REQUIRE(t.find("americans") != std::string::npos);
    REQUIRE(cap.text == transcript);
    REQUIRE(cap.total == 100);
    REQUIRE(cap.last_step == 100);

    // The same audio as BYTES and as a 16 kHz F32 tensor, via the automatic task.
    const std::vector<uint8_t> bytes = read_file(jfk_wav());
    OneMessage as_bytes(AGENT_CAP_ASR, "", "", nullptr, kOutDir);
    as_bytes.parts[0].kind = AGENT_PART_BYTES;
    as_bytes.parts[0].mime = c_str("audio/wav");
    as_bytes.parts[0].data = bytes.data();
    as_bytes.parts[0].size = bytes.size();
    as_bytes.msg.n_parts = 1;
    as_bytes.req.flags = AGENT_REQUEST_FLAG_AUTO_TASK;
    REQUIRE(e->run(inst, &as_bytes.req, &resp) == AGENT_OK);
    REQUIRE(lower(std::string(sv(resp.message.parts[0].text))).find("your country") != std::string::npos);
    e->free_response(inst, &resp);

    const Pcm pcm = read_pcm16_wav(jfk_wav());
    REQUIRE(pcm.rate == 16000);
    OneMessage as_tensor(AGENT_CAP_ASR, "", "", nullptr, kOutDir, &pcm.samples);
    REQUIRE(e->run(inst, &as_tensor.req, &resp) == AGENT_OK);
    REQUIRE(lower(std::string(sv(resp.message.parts[0].text))).find("your country") != std::string::npos);
    e->free_response(inst, &resp);

    // An 8 kHz copy is resampled to 16 kHz.
    {
        std::vector<int16_t> pcm8;
        for (size_t i = 0; i < pcm.samples.size(); i += 2) pcm8.push_back(static_cast<int16_t>(pcm.samples[i] * 32767.0f));
        const std::string wav8 = std::string(kOutDir) + "/jfk-8k.wav";
        std::ofstream f(wav8, std::ios::binary);
        auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
        auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };
        f.write("RIFF", 4);
        u32(static_cast<uint32_t>(36 + pcm8.size() * 2));
        f.write("WAVEfmt ", 8);
        u32(16);
        u16(1);
        u16(1);
        u32(8000);
        u32(16000);
        u16(2);
        u16(16);
        f.write("data", 4);
        u32(static_cast<uint32_t>(pcm8.size() * 2));
        f.write(reinterpret_cast<const char *>(pcm8.data()), static_cast<std::streamsize>(pcm8.size() * 2));
        f.close();
        OneMessage low(AGENT_CAP_ASR, "", wav8, nullptr, kOutDir);
        REQUIRE(e->run(inst, &low.req, &resp) == AGENT_OK);
        INFO("8 kHz transcript: " << sv(resp.message.parts[0].text));
        REQUIRE(lower(std::string(sv(resp.message.parts[0].text))).find("country") != std::string::npos);
        e->free_response(inst, &resp);
    }

    // Errors: no audio, unsupported task, unreadable audio, unknown language, cancel.
    OneMessage no_audio(AGENT_CAP_ASR, "hello", "", nullptr, kOutDir);
    REQUIRE(e->run(inst, &no_audio.req, &resp) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(std::string_view(e->last_error(inst)).find("audio part") != std::string_view::npos);
    OneMessage tts(AGENT_CAP_TTS, "hello", "", nullptr, kOutDir);
    REQUIRE(e->run(inst, &tts.req, &resp) == AGENT_ERR_UNSUPPORTED);
    const std::string bogus = std::string(kOutDir) + "/bogus.wav";
    std::ofstream(bogus) << "not a wave file";
    OneMessage bad(AGENT_CAP_ASR, "", bogus, nullptr, kOutDir);
    REQUIRE(e->run(inst, &bad.req, &resp) == AGENT_ERR_UNSUPPORTED);
    REQUIRE(std::string_view(e->last_error(inst)).find("WAV") != std::string_view::npos);
    OneMessage bad_lang(AGENT_CAP_ASR, "", jfk_wav(), nullptr, kOutDir);
    bad_lang.req.params_json = c_str(R"({"language":"klingon"})");
    REQUIRE(e->run(inst, &bad_lang.req, &resp) == AGENT_ERR_INVALID_CONFIG);

    Captured cancel_cap;
    cancel_cap.cancel_on_progress = true; // the first progress report (0/100) raises the flag
    OneMessage cancel(AGENT_CAP_ASR, "", jfk_wav(), &cancel_cap, kOutDir);
    REQUIRE(e->run(inst, &cancel.req, &resp) == AGENT_ERR_CANCELLED);

    // VAD: jfk.wav is one 11 s clip with a few pauses.
    OneMessage vad(AGENT_CAP_VAD, "", jfk_wav(), nullptr, kOutDir);
    if (!vad_model) {
        REQUIRE(e->run(inst, &vad.req, &resp) == AGENT_ERR_UNSUPPORTED);
        e->destroy(inst);
        SKIP("AGENT_TEST_WHISPER_VAD_MODEL not set; ASR verified only");
    }
    REQUIRE(e->run(inst, &vad.req, &resp) == AGENT_OK);
    REQUIRE(resp.message.n_parts == 2);
    const std::string json(sv(resp.message.parts[0].text));
    INFO("segments: " << json);
    REQUIRE(json.starts_with("[{\"start\":"));
    REQUIRE(resp.message.parts[1].kind == AGENT_PART_TENSOR);
    REQUIRE(sv(resp.message.parts[1].name) == "segments");
    REQUIRE(resp.message.parts[1].ndim == 2);
    const int64_t rows = resp.message.parts[1].shape[0];
    REQUIRE(rows >= 1);
    REQUIRE(resp.message.parts[1].shape[1] == 2);
    const auto *seg = static_cast<const float *>(resp.message.parts[1].data);
    REQUIRE(seg[0] < 1.5f);                 // speech starts near the beginning
    REQUIRE(seg[2 * rows - 1] > 9.0f);      // and ends near the end of the 11 s clip
    REQUIRE(seg[2 * rows - 1] < 11.5f);
    e->free_response(inst, &resp);
    e->destroy(inst);

    // vad.filter runs the VAD before transcription; the transcript is unchanged.
    void *filtered = nullptr;
    const std::string fcfg = std::string("{\"model\":\"") + model + "\",\"options\":{\"threads\":2,\"vad\":{\"model\":\"" + vad_model +
                             "\",\"filter\":true}}}";
    REQUIRE(create(e, fcfg, &filtered, &err) == AGENT_OK);
    OneMessage again(AGENT_CAP_ASR, "", jfk_wav(), nullptr, kOutDir);
    REQUIRE(e->run(filtered, &again.req, &resp) == AGENT_OK);
    INFO("filtered transcript: " << sv(resp.message.parts[0].text));
    REQUIRE(lower(std::string(sv(resp.message.parts[0].text))).find("your country") != std::string::npos);
    e->free_response(filtered, &resp);
    e->destroy(filtered);
}

TEST_CASE("whisper engine transcribes through a Session", "[engines][whisper][model]") {
    const char *model = env("AGENT_TEST_WHISPER_MODEL");
    if (!model) SKIP("AGENT_TEST_WHISPER_MODEL not set");

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_WHISPER_ENGINE_PATH;
    cfg.model = model;
    cfg.workspace_dir = "/tmp/agent_test_whisper_ws";
    cfg.engine_options = R"({"threads":2})";
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "whisper_session";
    const agent::Media media[] = {{jfk_wav(), "audio/wav"}};
    auto heard = agent::run_turn_with_media(session, memory, "", media);
    REQUIRE(heard.ok);
    INFO("transcript: " << heard.value);
    REQUIRE(lower(heard.value).find("ask not what your country can do for you") != std::string::npos);
}
