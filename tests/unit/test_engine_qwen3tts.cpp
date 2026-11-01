// agent-engine-qwen3tts tests. Built only with -DAGENT_ENGINE_QWEN3TTS=ON.
//
// Model-free tests always run. Real-model tests are gated on:
//   AGENT_TEST_QWEN3TTS_TALKER  talker GGUF, e.g. qwen-talker-0.6b-base-Q8_0.gguf (Serveurperso/Qwen3-TTS-GGUF)
//   AGENT_TEST_QWEN3TTS_CODEC   12 Hz tokenizer GGUF, e.g. qwen-tokenizer-12hz-Q8_0.gguf
// When the whisper engine is built too and AGENT_TEST_WHISPER_MODEL is set, the generated speech is
// transcribed back with whisper.cpp.
#include "speech_engine_test_util.hpp"

#include <filesystem>

namespace fs = std::filesystem;
using namespace speech_test;

namespace {

const char *kOutDir = "/tmp/agent_test_qwen3tts_out";

const agent_engine_v1 *qwen3tts_engine() { return load(AGENT_TEST_QWEN3TTS_ENGINE_PATH, "qwen3tts"); }

// Transcribes wav with the whisper plugin; empty when whisper is not available.
std::string transcribe_with_whisper(const std::string &wav) {
#ifdef AGENT_TEST_WHISPER_ENGINE_PATH
    const char *model = env("AGENT_TEST_WHISPER_MODEL");
    if (!model) return {};
    const agent_engine_v1 *w = load(AGENT_TEST_WHISPER_ENGINE_PATH, "whisper");
    void *inst = nullptr;
    REQUIRE(create(w, std::string("{\"model\":\"") + model + "\",\"options\":{\"threads\":2}}", &inst) == AGENT_OK);
    OneMessage asr(AGENT_CAP_ASR, "", wav, nullptr, kOutDir);
    agent_response resp{.struct_size = sizeof(agent_response)};
    REQUIRE(w->run(inst, &asr.req, &resp) == AGENT_OK);
    std::string text(sv(resp.message.parts[0].text));
    w->free_response(inst, &resp);
    w->destroy(inst);
    return text;
#else
    (void)wav;
    return {};
#endif
}

} // namespace

TEST_CASE("qwen3tts plugin loads with the TTS capability", "[engines][qwen3tts]") {
    const agent_engine_v1 *e = qwen3tts_engine();
    REQUIRE(e != nullptr);
    REQUIRE(e->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE(e->struct_size >= AGENT_ENGINE_V2_SIZE);
    REQUIRE(std::string_view(e->name) == "qwen3tts");
    REQUIRE(e->caps == AGENT_CAP_TTS);
    REQUIRE(e->flags == 0);
}

TEST_CASE("qwen3tts create fails cleanly on bad configs and model paths", "[engines][qwen3tts]") {
    const agent_engine_v1 *e = qwen3tts_engine();
    void *inst = nullptr;
    std::string err;
    REQUIRE(create(e, "not json", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(create(e, R"({"model":"/nonexistent/talker.gguf"})", &inst, &err) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(err.find("options.codec") != std::string::npos);
    REQUIRE(create(e, R"({"model":"/nonexistent/talker.gguf","options":{"codec":"/nonexistent/codec.gguf"}})", &inst, &err) ==
            AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(err.find("/nonexistent/talker.gguf") != std::string::npos);

    fs::create_directories(kOutDir);
    const std::string bogus = std::string(kOutDir) + "/bogus.gguf";
    std::ofstream(bogus) << "definitely not a gguf file";
    REQUIRE(create(e, "{\"model\":\"" + bogus + "\",\"options\":{\"codec\":\"" + bogus + "\"}}", &inst, &err) == AGENT_ERR_MODEL_LOAD_FAILED);
    INFO(err);
    REQUIRE(err.find("failed to load") != std::string::npos);
    REQUIRE(inst == nullptr);
}

TEST_CASE("qwen3tts says hello world, streams it and clones a voice", "[engines][qwen3tts][model]") {
    const char *talker = env("AGENT_TEST_QWEN3TTS_TALKER");
    const char *codec = env("AGENT_TEST_QWEN3TTS_CODEC");
    if (!talker || !codec) SKIP("AGENT_TEST_QWEN3TTS_TALKER and AGENT_TEST_QWEN3TTS_CODEC not both set");

    const agent_engine_v1 *e = qwen3tts_engine();
    fs::create_directories(kOutDir);
    void *inst = nullptr;
    std::string err;
    const std::string cfg = std::string("{\"model\":\"") + talker + "\",\"options\":{\"codec\":\"" + codec +
                            "\",\"language\":\"en\",\"seed\":42}}";
    REQUIRE(create(e, cfg, &inst, &err) == AGENT_OK);
    INFO(err);

    // Explicit TTS task: WAV file + summary, streamed audio chunks, progress.
    Captured cap;
    OneMessage tts(AGENT_CAP_TTS, "Hello world.", "", &cap, kOutDir);
    agent_response resp{.struct_size = sizeof(agent_response)};
    const int32_t rc = e->run(inst, &tts.req, &resp);
    INFO("last_error: " << e->last_error(inst));
    REQUIRE(rc == AGENT_OK);
    REQUIRE(sv(resp.message.role) == "assistant");
    REQUIRE(resp.message.n_parts == 2);
    REQUIRE(resp.message.parts[0].kind == AGENT_PART_TEXT);
    REQUIRE(sv(resp.message.parts[0].text).find("24000 Hz") != std::string_view::npos);
    REQUIRE(resp.message.parts[1].kind == AGENT_PART_FILE);
    REQUIRE(sv(resp.message.parts[1].mime) == "audio/wav");
    const std::string wav(sv(resp.message.parts[1].path));
    e->free_response(inst, &resp);
    REQUIRE(wav.starts_with(kOutDir));
    REQUIRE(cap.audio_chunks > 0);
    const Pcm pcm = read_pcm16_wav(wav);
    REQUIRE(pcm.rate == 24000);
    REQUIRE(pcm.samples.size() * sizeof(float) == cap.audio_bytes); // the file holds exactly the streamed audio
    const double seconds = static_cast<double>(pcm.samples.size()) / 24000.0;
    INFO("duration: " << seconds << " s");
    REQUIRE(seconds > 0.4);
    REQUIRE(seconds < 8.0);
    REQUIRE(cap.total > 0);
    REQUIRE(cap.last_step == cap.total);

    const std::string heard = transcribe_with_whisper(wav);
    if (!heard.empty()) {
        INFO("whisper heard: " << heard);
        REQUIRE(lower(heard).find("hello") != std::string::npos);
        REQUIRE(lower(heard).find("world") != std::string::npos);
    } else {
        WARN("whisper not available: the hello-world speech was not transcribed back");
    }

    // Cancel from inside the first audio chunk.
    Captured cancel_cap;
    cancel_cap.cancel_on_first_chunk = true;
    OneMessage cancel(AGENT_CAP_TTS, "This sentence is long enough to need several audio chunks before it ends.", "", &cancel_cap, kOutDir);
    REQUIRE(e->run(inst, &cancel.req, &resp) == AGENT_ERR_CANCELLED);
    REQUIRE(cancel_cap.audio_chunks >= 1);

    // Voice cloning: the hello-world clip as the reference audio part of a new request.
    OneMessage clone(AGENT_CAP_TTS, "Good morning.", wav, nullptr, kOutDir);
    REQUIRE(e->run(inst, &clone.req, &resp) == AGENT_OK);
    const std::string cloned(sv(resp.message.parts[1].path));
    e->free_response(inst, &resp);
    REQUIRE(cloned != wav);
    REQUIRE(read_pcm16_wav(cloned).samples.size() > 24000 / 4);
    const std::string heard_clone = transcribe_with_whisper(cloned);
    if (!heard_clone.empty()) {
        INFO("whisper heard (cloned voice): " << heard_clone);
        REQUIRE(lower(heard_clone).find("morning") != std::string::npos);
    }

    // Errors: no text, unsupported task, mode rules (speaker is for custom_voice models).
    OneMessage empty(AGENT_CAP_TTS, "", "", nullptr, kOutDir);
    REQUIRE(e->run(inst, &empty.req, &resp) == AGENT_ERR_INVALID_CONFIG);
    OneMessage asr(AGENT_CAP_ASR, "", wav, nullptr, kOutDir);
    REQUIRE(e->run(inst, &asr.req, &resp) == AGENT_ERR_UNSUPPORTED);
    OneMessage bad_lang(AGENT_CAP_TTS, "Hello.", "", nullptr, kOutDir);
    bad_lang.req.params_json = c_str(R"({"language":"klingon"})");
    REQUIRE(e->run(inst, &bad_lang.req, &resp) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(std::string_view(e->last_error(inst)).find("klingon") != std::string_view::npos);
    e->destroy(inst);
}
