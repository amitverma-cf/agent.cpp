// agent-engine-parakeet tests. Built only with -DAGENT_ENGINE_PARAKEET=ON.
//
// Model-free tests always run. Real-model tests are gated on:
//   AGENT_TEST_PARAKEET_MODEL  a parakeet.cpp GGUF, e.g. tdt_ctc-110m-q8_0.gguf from mudler/parakeet-cpp-gguf
// The speech is parakeet.cpp's own tests/fixtures/speech.wav (AGENT_TEST_PARAKEET_FIXTURES_DIR).
#include "speech_engine_test_util.hpp"

#include <filesystem>

namespace fs = std::filesystem;
using namespace speech_test;

namespace {

const char *kOutDir = "/tmp/agent_test_parakeet_out";

const agent_engine_v1 *parakeet_engine() { return load(AGENT_TEST_PARAKEET_ENGINE_PATH, "parakeet"); }

// A static string: request parts only borrow the path.
const std::string &speech_wav() {
    static const std::string path = std::string(AGENT_TEST_PARAKEET_FIXTURES_DIR) + "/speech.wav";
    return path;
}

} // namespace

TEST_CASE("parakeet plugin loads with the ASR capability", "[engines][parakeet]") {
    const agent_engine_v1 *e = parakeet_engine();
    REQUIRE(e != nullptr);
    REQUIRE(e->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE(e->struct_size >= AGENT_ENGINE_V2_SIZE);
    REQUIRE(std::string_view(e->name) == "parakeet");
    REQUIRE(e->caps == AGENT_CAP_ASR);
    REQUIRE(e->flags == 0);
}

TEST_CASE("parakeet create fails cleanly on bad configs and model paths", "[engines][parakeet]") {
    const agent_engine_v1 *e = parakeet_engine();
    void *inst = nullptr;
    std::string err;
    REQUIRE(create(e, "not json", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(create(e, R"({"model":""})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(err.find("config.model") != std::string::npos);
    REQUIRE(create(e, R"({"model":"/nonexistent/tdt_ctc-110m-q8_0.gguf"})", &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(err.find("/nonexistent/tdt_ctc-110m-q8_0.gguf") != std::string::npos);

    fs::create_directories(kOutDir);
    const std::string bogus = std::string(kOutDir) + "/bogus.gguf";
    std::ofstream(bogus) << "definitely not a gguf file";
    REQUIRE(create(e, "{\"model\":\"" + bogus + "\",\"options\":{\"decoder\":\"beam\"}}", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(create(e, "{\"model\":\"" + bogus + "\"}", &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    INFO(err);
    REQUIRE(err.find("failed to load") != std::string::npos);
    REQUIRE(inst == nullptr);
}

TEST_CASE("parakeet transcribes speech.wav", "[engines][parakeet][model]") {
    const char *model = env("AGENT_TEST_PARAKEET_MODEL");
    if (!model) SKIP("AGENT_TEST_PARAKEET_MODEL not set");

    const agent_engine_v1 *e = parakeet_engine();
    fs::create_directories(kOutDir);
    void *inst = nullptr;
    std::string err;
    REQUIRE(create(e, std::string("{\"model\":\"") + model + "\",\"options\":{\"threads\":2}}", &inst, &err) == AGENT_OK);
    INFO(err);

    Captured cap;
    OneMessage asr(AGENT_CAP_ASR, "", speech_wav(), &cap, kOutDir);
    agent_response resp{.struct_size = sizeof(agent_response)};
    const int32_t rc = e->run(inst, &asr.req, &resp);
    INFO("last_error: " << e->last_error(inst));
    REQUIRE(rc == AGENT_OK);
    REQUIRE(sv(resp.message.role) == "assistant");
    REQUIRE(resp.message.n_parts == 1);
    REQUIRE(resp.message.parts[0].kind == AGENT_PART_TEXT);
    const std::string transcript(sv(resp.message.parts[0].text));
    e->free_response(inst, &resp);
    INFO("transcript: " << transcript);
    const std::string t = lower(transcript);
    REQUIRE(t.find("observed phoebe") != std::string::npos);
    REQUIRE(t.find("old portrait") != std::string::npos);
    REQUIRE(cap.text == transcript);
    REQUIRE(cap.total == 1);
    REQUIRE(cap.last_step == 1);

    // The same audio as a 16 kHz F32 tensor through the automatic task, and with the CTC head.
    const Pcm pcm = read_pcm16_wav(speech_wav());
    REQUIRE(pcm.rate == 16000);
    OneMessage as_tensor(AGENT_CAP_ASR, "", "", nullptr, kOutDir, &pcm.samples);
    as_tensor.req.flags = AGENT_REQUEST_FLAG_AUTO_TASK;
    REQUIRE(e->run(inst, &as_tensor.req, &resp) == AGENT_OK);
    REQUIRE(lower(std::string(sv(resp.message.parts[0].text))).find("old portrait") != std::string::npos);
    e->free_response(inst, &resp);

    OneMessage ctc(AGENT_CAP_ASR, "", speech_wav(), nullptr, kOutDir);
    ctc.req.params_json = c_str(R"({"decoder":"ctc"})");
    REQUIRE(e->run(inst, &ctc.req, &resp) == AGENT_OK);
    INFO("ctc transcript: " << sv(resp.message.parts[0].text));
    REQUIRE(lower(std::string(sv(resp.message.parts[0].text))).find("portrait") != std::string::npos);
    e->free_response(inst, &resp);

    // A 44.1 kHz stereo float copy is down-mixed and resampled.
    {
        const std::string wav = std::string(kOutDir) + "/speech-44k-stereo.wav";
        std::vector<float> st;
        const double ratio = 44100.0 / 16000.0;
        const size_t n = static_cast<size_t>(static_cast<double>(pcm.samples.size()) * ratio);
        for (size_t i = 0; i < n; ++i) {
            const float s = pcm.samples[std::min(pcm.samples.size() - 1, static_cast<size_t>(static_cast<double>(i) / ratio))];
            st.push_back(s);
            st.push_back(s);
        }
        std::ofstream f(wav, std::ios::binary);
        auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
        auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };
        f.write("RIFF", 4);
        u32(static_cast<uint32_t>(36 + st.size() * 4));
        f.write("WAVEfmt ", 8);
        u32(16);
        u16(3); // IEEE float
        u16(2);
        u32(44100);
        u32(44100 * 8);
        u16(8);
        u16(32);
        f.write("data", 4);
        u32(static_cast<uint32_t>(st.size() * 4));
        f.write(reinterpret_cast<const char *>(st.data()), static_cast<std::streamsize>(st.size() * 4));
        f.close();
        OneMessage hi(AGENT_CAP_ASR, "", wav, nullptr, kOutDir);
        REQUIRE(e->run(inst, &hi.req, &resp) == AGENT_OK);
        INFO("44.1 kHz stereo transcript: " << sv(resp.message.parts[0].text));
        REQUIRE(lower(std::string(sv(resp.message.parts[0].text))).find("portrait") != std::string::npos);
        e->free_response(inst, &resp);
    }

    // Errors: no audio, unsupported task, unreadable audio, cancel.
    OneMessage no_audio(AGENT_CAP_ASR, "hello", "", nullptr, kOutDir);
    REQUIRE(e->run(inst, &no_audio.req, &resp) == AGENT_ERR_INVALID_CONFIG);
    OneMessage tts(AGENT_CAP_TTS, "hello", "", nullptr, kOutDir);
    REQUIRE(e->run(inst, &tts.req, &resp) == AGENT_ERR_UNSUPPORTED);
    const std::string bogus = std::string(kOutDir) + "/bogus.wav";
    std::ofstream(bogus) << "not a wave file";
    OneMessage bad(AGENT_CAP_ASR, "", bogus, nullptr, kOutDir);
    REQUIRE(e->run(inst, &bad.req, &resp) == AGENT_ERR_UNSUPPORTED);
    REQUIRE(std::string_view(e->last_error(inst)).find("WAV") != std::string_view::npos);
    Captured cancel_cap;
    cancel_cap.cancel = 1;
    OneMessage cancel(AGENT_CAP_ASR, "", speech_wav(), &cancel_cap, kOutDir);
    REQUIRE(e->run(inst, &cancel.req, &resp) == AGENT_ERR_CANCELLED);
    e->destroy(inst);
}

TEST_CASE("parakeet engine transcribes through a Session", "[engines][parakeet][model]") {
    const char *model = env("AGENT_TEST_PARAKEET_MODEL");
    if (!model) SKIP("AGENT_TEST_PARAKEET_MODEL not set");

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_PARAKEET_ENGINE_PATH;
    cfg.model = model;
    cfg.workspace_dir = "/tmp/agent_test_parakeet_ws";
    cfg.engine_options = R"({"threads":2})";
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "parakeet_session";
    const agent::Media media[] = {{speech_wav(), "audio/wav"}};
    auto heard = agent::run_turn_with_media(session, memory, "", media);
    REQUIRE(heard.ok);
    INFO("transcript: " << heard.value);
    REQUIRE(lower(heard.value).find("old portrait") != std::string::npos);
}
