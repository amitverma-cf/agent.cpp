// agent voice: speech detection, utterance segmentation, the loop with mock engines, WAV I/O, flags.
//
// The end-to-end test at the bottom runs the real `agent voice` binary on a WAV file (file audio
// backend) and is gated on AGENT_TEST_SHERPA_TTS_DIR (Kokoro), AGENT_TEST_SHERPA_ASR_DIR (Whisper)
// and AGENT_TEST_LLAMA_MODEL (a chat GGUF); AGENT_TEST_SHERPA_VAD_MODEL (silero_vad.onnx) switches
// it from the energy gate to the Silero VAD.
#include "audio.hpp"
#include "voice.hpp"

#include <array>
#include <atomic>
#include <catch_amalgamated.hpp>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using namespace agent::cli;

namespace {

constexpr int kRate = audio::kCaptureRate;

void add_silence(std::vector<float> &v, double seconds, float noise = 0.001f) {
    const size_t n = static_cast<size_t>(seconds * kRate);
    for (size_t i = 0; i < n; ++i)
        v.push_back(noise * static_cast<float>(static_cast<double>((i * 7919) % 200) / 100.0 - 1.0)); // faint deterministic hiss
}

void add_tone(std::vector<float> &v, double seconds, float amplitude = 0.3f, double hz = 440.0) {
    const size_t n = static_cast<size_t>(seconds * kRate);
    for (size_t i = 0; i < n; ++i)
        v.push_back(amplitude * static_cast<float>(std::sin(2.0 * 3.14159265358979 * hz * static_cast<double>(i) / kRate)));
}

double secs(int64_t samples) { return static_cast<double>(samples) / kRate; }

fs::path temp_dir(const char *name) {
    const fs::path d = fs::temp_directory_path() / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

// Mock engines for the loop: the "ASR" reports the utterance length, the "LLM" echoes, the "TTS"
// returns reply_seconds of a 24 kHz tone.
struct Mock {
    std::vector<size_t> utterance_samples;
    std::vector<std::string> events;
    double reply_seconds = 0.5;
    int fail_transcribe_at = -1; // index of the call that fails
    bool cancel_on_respond = false;
};

agent::Result<std::string> mock_transcribe(std::span<const float> utterance, void *ud) {
    auto *m = static_cast<Mock *>(ud);
    if (static_cast<int>(m->utterance_samples.size()) == m->fail_transcribe_at) {
        m->utterance_samples.push_back(utterance.size());
        return agent::fail<std::string>(agent::ErrorCode::DecodeFailed, "asr broke");
    }
    m->utterance_samples.push_back(utterance.size());
    return agent::ok("utterance " + std::to_string(m->utterance_samples.size()));
}

agent::Result<std::string> mock_respond(std::string_view text, void *ud) {
    auto *m = static_cast<Mock *>(ud);
    if (m->cancel_on_respond) return agent::fail<std::string>(agent::ErrorCode::Cancelled, "cancelled");
    return agent::ok("**You said** " + std::string(text) + ".");
}

agent::Result<audio::Wav> mock_speak(std::string_view, void *ud) {
    auto *m = static_cast<Mock *>(ud);
    audio::Wav w;
    w.rate = 24000;
    w.samples.assign(static_cast<size_t>(m->reply_seconds * w.rate), 0.1f);
    return agent::ok(std::move(w));
}

void mock_event(VoiceEvent e, std::string_view text, void *ud) {
    static constexpr std::array<const char *, 7> kNames = {"listening", "heard", "skipped", "reply", "spoken", "barge-in", "error"};
    static_cast<Mock *>(ud)->events.push_back(std::string(kNames[static_cast<size_t>(e)]) + (text.empty() ? "" : ":" + std::string(text)));
}

VoiceHooks mock_hooks(Mock &m) {
    VoiceHooks h;
    h.transcribe = mock_transcribe;
    h.respond = mock_respond;
    h.speak = mock_speak;
    h.on_event = mock_event;
    h.user_data = &m;
    return h;
}

bool has_event(const Mock &m, std::string_view prefix) {
    for (const auto &e : m.events)
        if (e.starts_with(prefix)) return true;
    return false;
}

} // namespace

// -------------------------- energy gate --------------------------

TEST_CASE("energy gate finds tones between silences", "[voice]") {
    std::vector<float> s;
    add_silence(s, 0.5);
    add_tone(s, 1.0);
    add_silence(s, 1.0);
    add_tone(s, 0.8, 0.05f, 300.0); // quieter, still well above the hiss
    add_silence(s, 0.7);

    const auto segs = detect_speech_energy(s);
    REQUIRE(segs.size() == 2);
    CHECK(secs(segs[0].start) == Catch::Approx(0.5).margin(0.04));
    CHECK(secs(segs[0].end) == Catch::Approx(1.5).margin(0.04));
    CHECK(secs(segs[1].start) == Catch::Approx(2.5).margin(0.04));
    CHECK(secs(segs[1].end) == Catch::Approx(3.3).margin(0.04));

    // Fed in odd-sized chunks, the gate reports the same segments as they finish.
    EnergyGate gate;
    std::vector<SpeechSegment> streamed;
    bool was_speaking = false;
    for (size_t off = 0; off < s.size(); off += 777) {
        const size_t n = std::min<size_t>(777, s.size() - off);
        was_speaking |= energy_gate_feed(gate, std::span<const float>(s.data() + off, n), streamed);
    }
    energy_gate_flush(gate, streamed);
    REQUIRE(was_speaking);
    REQUIRE(streamed.size() == 2);
    CHECK(streamed[0].start == segs[0].start);
    CHECK(streamed[1].end == segs[1].end);
}

TEST_CASE("energy gate ignores clicks and splits endless speech", "[voice]") {
    std::vector<float> s;
    add_silence(s, 0.5);
    add_tone(s, 0.06); // 60 ms click: shorter than min_speech_ms
    add_silence(s, 0.5);
    CHECK(detect_speech_energy(s).empty());

    std::vector<float> long_tone;
    add_tone(long_tone, 2.5);
    EnergyGateOptions o;
    o.max_speech_ms = 1000;
    const auto segs = detect_speech_energy(long_tone, o);
    REQUIRE(segs.size() == 3);
    CHECK(secs(segs[0].end - segs[0].start) == Catch::Approx(1.0).margin(0.04));

    // Open speech at the end of the input is closed by flush.
    std::vector<float> tail;
    add_silence(tail, 0.3);
    add_tone(tail, 0.5);
    const auto t = detect_speech_energy(tail);
    REQUIRE(t.size() == 1);
    CHECK(secs(t[0].end) == Catch::Approx(0.8).margin(0.04));
}

TEST_CASE("VAD engine output parses", "[voice]") {
    auto segs = parse_vad_segments(R"([{"start":0.512,"end":1.800},{"start":2.0,"end":2.4,"open":true}])");
    REQUIRE(segs.ok);
    REQUIRE(segs.value.size() == 2);
    CHECK(segs.value[0].start == Catch::Approx(0.512));
    CHECK_FALSE(segs.value[0].open);
    CHECK(segs.value[1].open);
    CHECK(parse_vad_segments("[]").value.empty());
    CHECK_FALSE(parse_vad_segments("Generated 1.2 s of speech").ok);
    CHECK_FALSE(parse_vad_segments(R"([{"start":1}])").ok);
}

// -------------------------- the loop --------------------------

TEST_CASE("voice loop segments utterances from a WAV file and answers each", "[voice]") {
    const fs::path dir = temp_dir("agent_test_voice_loop");
    std::vector<float> s;
    add_silence(s, 0.5);
    add_tone(s, 1.0);
    add_silence(s, 1.0);
    add_tone(s, 0.15); // a cough: the gate opens (>= 120 ms), but with the rolls it is shorter than min_utterance
    add_silence(s, 1.0);
    add_tone(s, 0.7);
    add_silence(s, 0.8);
    REQUIRE(audio::write_wav((dir / "in.wav").string(), s, kRate).ok);

    auto in = audio::open_file_input((dir / "in.wav").string());
    REQUIRE(in.ok);
    auto out = audio::open_file_output((dir / "out").string(), "reply", false);
    REQUIRE(out.ok);
    EnergyGate gate;
    SpeechDetector det = energy_detector(gate);
    Mock m;
    VoiceLoopOptions o;
    o.min_utterance_ms = 600;
    auto stats = run_voice_loop(in.value, out.value, det, mock_hooks(m), o);
    REQUIRE(stats.ok);
    CHECK(stats.value.turns == 2);
    CHECK(stats.value.skipped == 1);
    REQUIRE(m.utterance_samples.size() == 2);
    // Utterance = segment + 200 ms pre-roll + 150 ms post-roll.
    CHECK(secs(static_cast<int64_t>(m.utterance_samples[0])) == Catch::Approx(1.35).margin(0.06));
    CHECK(secs(static_cast<int64_t>(m.utterance_samples[1])) == Catch::Approx(1.05).margin(0.06));
    CHECK(has_event(m, "heard:utterance 1"));
    CHECK(has_event(m, "reply:**You said** utterance 2."));
    CHECK(has_event(m, "spoken:0.50 s of speech at 24000 Hz"));

    const audio::FileOutputInfo *info = audio::file_output_info(out.value);
    REQUIRE(info);
    REQUIRE(info->written.size() == 2);
    auto reply = audio::read_wav(info->written[0]);
    REQUIRE(reply.ok);
    CHECK(reply.value.rate == 24000);
    CHECK(reply.value.samples.size() == 12000);
    CHECK(fs::path(info->written[1]).filename() == "reply-002.wav");
}

TEST_CASE("voice loop stops after max_turns, on Cancelled, and survives hook errors", "[voice]") {
    std::vector<float> s;
    for (int i = 0; i < 3; ++i) {
        add_silence(s, 0.8);
        add_tone(s, 0.6);
    }
    add_silence(s, 0.8);
    const fs::path dir = temp_dir("agent_test_voice_turns");
    REQUIRE(audio::write_wav((dir / "in.wav").string(), s, kRate).ok);

    SECTION("max_turns") {
        auto in = audio::open_file_input((dir / "in.wav").string());
        auto out = audio::open_file_output((dir / "out").string(), "reply", false);
        EnergyGate gate;
        SpeechDetector det = energy_detector(gate);
        Mock m;
        VoiceLoopOptions o;
        o.max_turns = 1;
        auto stats = run_voice_loop(in.value, out.value, det, mock_hooks(m), o);
        REQUIRE(stats.ok);
        CHECK(stats.value.turns == 1);
        CHECK(m.utterance_samples.size() == 1);
    }
    SECTION("an ASR error is reported and the next utterance still runs") {
        auto in = audio::open_file_input((dir / "in.wav").string());
        auto out = audio::open_file_output((dir / "out").string(), "reply", false);
        EnergyGate gate;
        SpeechDetector det = energy_detector(gate);
        Mock m;
        m.fail_transcribe_at = 0;
        auto stats = run_voice_loop(in.value, out.value, det, mock_hooks(m), {});
        REQUIRE(stats.ok);
        CHECK(stats.value.errors == 1);
        CHECK(stats.value.turns == 2);
        CHECK(has_event(m, "error:asr broke"));
    }
    SECTION("Cancelled ends the loop") {
        auto in = audio::open_file_input((dir / "in.wav").string());
        auto out = audio::open_file_output((dir / "out").string(), "reply", false);
        EnergyGate gate;
        SpeechDetector det = energy_detector(gate);
        Mock m;
        m.cancel_on_respond = true;
        auto stats = run_voice_loop(in.value, out.value, det, mock_hooks(m), {});
        REQUIRE(stats.ok);
        CHECK(stats.value.turns == 0);
        CHECK(m.utterance_samples.size() == 1);
    }
    SECTION("the stop flag ends it before anything is heard") {
        auto in = audio::open_file_input((dir / "in.wav").string());
        auto out = audio::open_file_output((dir / "out").string(), "reply", false);
        EnergyGate gate;
        SpeechDetector det = energy_detector(gate);
        Mock m;
        std::atomic<bool> stop{true};
        VoiceLoopOptions o;
        o.stop = &stop;
        auto stats = run_voice_loop(in.value, out.value, det, mock_hooks(m), o);
        REQUIRE(stats.ok);
        CHECK(m.utterance_samples.empty());
    }
}

TEST_CASE("barge-in cuts the reply short when the user talks over it", "[voice]") {
    // Utterance 1 ends at 1.5 s; the gate closes it 0.6 s later, so a 3 s reply plays from ~2.1 s.
    // Utterance 2 starts at 3.0 s, during the reply.
    std::vector<float> s;
    add_silence(s, 0.5);
    add_tone(s, 1.0);
    add_silence(s, 1.5);
    add_tone(s, 1.0);
    add_silence(s, 4.0);
    const fs::path dir = temp_dir("agent_test_voice_barge");
    REQUIRE(audio::write_wav((dir / "in.wav").string(), s, kRate).ok);

    for (bool barge_in : {true, false}) {
        auto in = audio::open_file_input((dir / "in.wav").string());
        auto out = audio::open_file_output((dir / (barge_in ? "on" : "off")).string(), "reply", barge_in);
        REQUIRE(out.ok);
        EnergyGate gate;
        SpeechDetector det = energy_detector(gate);
        Mock m;
        m.reply_seconds = 3.0;
        VoiceLoopOptions o;
        o.barge_in = barge_in;
        auto stats = run_voice_loop(in.value, out.value, det, mock_hooks(m), o);
        REQUIRE(stats.ok);
        const audio::FileOutputInfo *info = audio::file_output_info(out.value);
        REQUIRE(info);
        if (barge_in) {
            CHECK(stats.value.barge_ins == 1);
            CHECK(stats.value.turns == 2); // the interrupting utterance is answered too
            CHECK(info->interrupted == 1);
            REQUIRE(info->written.size() == 2);
            auto first = audio::read_wav(info->written[0]);
            REQUIRE(first.ok);
            // Played from ~2.1 s until the barge-in at ~3.0 s + 120 ms (gate) + 200 ms.
            CHECK(static_cast<double>(first.value.samples.size()) / 24000 == Catch::Approx(1.2).margin(0.25));
            auto second = audio::read_wav(info->written[1]);
            CHECK(second.value.samples.size() == 72000);
            CHECK(has_event(m, "barge-in"));
        } else {
            // Without barge-in the reply plays out (file output finishes at once) and nothing is cut.
            CHECK(stats.value.barge_ins == 0);
            CHECK(info->interrupted == 0);
            CHECK(stats.value.turns == 2);
        }
    }
}

TEST_CASE("transcripts without words are skipped and replies are cleaned for TTS", "[voice]") {
    CHECK_FALSE(is_meaningful_transcript(""));
    CHECK_FALSE(is_meaningful_transcript("  [BLANK_AUDIO] "));
    CHECK_FALSE(is_meaningful_transcript("(music)"));
    CHECK_FALSE(is_meaningful_transcript("..."));
    CHECK(is_meaningful_transcript("What is two plus two?"));
    CHECK(is_meaningful_transcript("[noise] hello"));

    CHECK(speakable_text("**Four.** Two plus two is `4`.") == "Four. Two plus two is four.");
    // Kokoro's lexicon has no digits or symbols: they are spelled out.
    CHECK(speakable_text("2 + 2 = 4") == "two plus two equals four");
    CHECK(speakable_text("It costs 1,250.75 dollars, 15% off.") ==
          "It costs one thousand two hundred fifty point seven five dollars, fifteen percent off.");
    CHECK(speakable_text("10 - 3 is 7; 3x4 is 12; 9/3 is 3.") ==
          "ten minus three is seven; three times four is twelve; nine divided by three is three.");
    CHECK(speakable_text("In 2026 there were 1000000 users.") == "In two thousand twenty six there were one million users.");
    CHECK(speakable_text("well-known") == "well-known");
    CHECK(speakable_text("<think>hmm</think>Sure.\n\n<tool_call>{\"name\":\"x\"}</tool_call>") == "Sure.");
    CHECK(speakable_text("# Title\n- one\n- two") == "Title - one - two");
    CHECK(speakable_text("See [the docs](https://example.com) now.") == "See the docs now.");
    CHECK(speakable_text("Code:\n```cpp\nint x;\n```\nDone.") == "Code: Done.");
    CHECK(speakable_text("<tool_call>unterminated") == "");
}

// -------------------------- WAV I/O --------------------------

TEST_CASE("WAV files round-trip and other layouts parse", "[voice][audio]") {
    const fs::path dir = temp_dir("agent_test_voice_wav");
    std::vector<float> s;
    add_tone(s, 0.25, 0.5f);
    s.push_back(1.5f); // clamped
    s.push_back(-1.5f);
    const std::string path = (dir / "tone.wav").string();
    REQUIRE(audio::write_wav(path, s, kRate).ok);
    CHECK(fs::file_size(path) == 44 + s.size() * 2);
    auto back = audio::read_wav(path);
    REQUIRE(back.ok);
    CHECK(back.value.rate == kRate);
    REQUIRE(back.value.samples.size() == s.size());
    for (size_t i = 0; i + 2 < s.size(); i += 97) CHECK(back.value.samples[i] == Catch::Approx(s[i]).margin(1.0 / 16384));
    CHECK(back.value.samples[s.size() - 2] == Catch::Approx(1.0).margin(1e-3));
    CHECK(back.value.samples[s.size() - 1] == Catch::Approx(-1.0).margin(1e-3));

    // Stereo float32 with an extra chunk before "data": down-mixed to mono.
    std::vector<uint8_t> f;
    auto put = [&](const void *p, size_t n) { f.insert(f.end(), static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(p) + n); };
    auto put32 = [&](uint32_t v) { put(&v, 4); };
    auto put16 = [&](uint16_t v) { put(&v, 2); };
    const float frames[][2] = {{0.5f, 0.1f}, {-0.2f, -0.4f}, {1.0f, 0.0f}};
    put("RIFF", 4);
    put32(0);
    put("WAVEfmt ", 8);
    put32(16);
    put16(3);
    put16(2);
    put32(48000);
    put32(48000 * 8);
    put16(8);
    put16(32);
    put("LIST", 4);
    put32(3); // odd size: padded to 4
    put("abc\0", 4);
    put("data", 4);
    put32(sizeof(frames));
    put(frames, sizeof(frames));
    auto stereo = audio::parse_wav(f);
    REQUIRE(stereo.ok);
    CHECK(stereo.value.rate == 48000);
    REQUIRE(stereo.value.samples.size() == 3);
    CHECK(stereo.value.samples[0] == Catch::Approx(0.3f));
    CHECK(stereo.value.samples[1] == Catch::Approx(-0.3f));

    CHECK_FALSE(audio::parse_wav(std::vector<uint8_t>{'n', 'o', 'p', 'e'}).ok);
    CHECK_FALSE(audio::read_wav((dir / "missing.wav").string()).ok);

    // Resampling keeps the duration; a 24 kHz file input comes out at 16 kHz.
    std::vector<float> at24k(24000, 0.25f);
    CHECK(audio::resample(at24k, 24000, kRate).size() == 16000);
    REQUIRE(audio::write_wav((dir / "24k.wav").string(), at24k, 24000).ok);
    auto in = audio::open_file_input((dir / "24k.wav").string());
    REQUIRE(in.ok);
    std::vector<float> buf(40000);
    bool eof = false;
    CHECK(audio::read(in.value, buf.data(), buf.size(), 0, eof) == 16000);
    CHECK(eof);
}

TEST_CASE("sound devices fail cleanly when there are none", "[voice][audio]") {
    // This test box (CI, containers) usually has no sound card; with one, the devices simply open.
    auto mic = audio::open_device_input();
    if (!mic.ok) CHECK_FALSE(mic.error.message.empty());
    auto speaker = audio::open_device_output();
    if (!speaker.ok) CHECK_FALSE(speaker.error.message.empty());
}

// -------------------------- flags --------------------------

TEST_CASE("agent voice flags parse and validate", "[voice][cli]") {
    const std::vector<std::string_view> args = {
        "voice", "--asr-model",  "/m/whisper", "--tts-model",  "/m/kokoro", "--vad-model", "/m/vad.onnx", "-m",
        "qwen",  "--input-wav",  "q.wav",      "--output-dir", "out",       "--max-turns", "1",           "--barge-in",
        "-y",    "--speaker-id", "3"};
    auto cl = parse_command_line(args);
    REQUIRE(cl.ok);
    REQUIRE(cl.value.command == "voice");
    auto s = voice_settings_from(cl.value);
    REQUIRE(s.ok);
    CHECK(s.value.asr_model == "/m/whisper");
    CHECK(s.value.tts_model == "/m/kokoro");
    CHECK(s.value.vad_model == "/m/vad.onnx");
    CHECK(s.value.model == "qwen");
    CHECK(s.value.input_wav == "q.wav");
    CHECK(s.value.output_dir == "out");
    CHECK(s.value.max_turns == 1);
    CHECK(s.value.speaker_id == 3);
    CHECK(s.value.barge_in);
    CHECK(s.value.auto_approve);

    const std::vector<std::string_view> missing_tts = {"voice", "--asr-model", "a"};
    auto no_tts = voice_settings_from(parse_command_line(missing_tts).value);
    REQUIRE_FALSE(no_tts.ok);
    CHECK(no_tts.error.message.find("--tts-model") != std::string::npos);

    const std::vector<std::string_view> zero_turns = {"voice", "--asr-model", "a", "--tts-model", "b", "--max-turns", "0"};
    CHECK_FALSE(voice_settings_from(parse_command_line(zero_turns).value).ok);
    const std::vector<std::string_view> bad_number = {"voice", "--asr-model", "a", "--tts-model", "b", "--ctx", "lots"};
    CHECK_FALSE(voice_settings_from(parse_command_line(bad_number).value).ok);
    const std::vector<std::string_view> unknown = {"voice", "--asr-model", "a", "--tts-model", "b", "--list"};
    CHECK_FALSE(parse_command_line(unknown).ok);
    const std::vector<std::string_view> positional = {"voice", "extra"};
    CHECK_FALSE(parse_command_line(positional).ok);
    CHECK(usage_text().find("  voice ") != std::string::npos);
}

TEST_CASE("sherpa options come from model directories", "[voice][cli]") {
    const fs::path asr = temp_dir("agent_test_voice_asr");
    for (const char *n :
         {"tiny.en-encoder.onnx", "tiny.en-encoder.int8.onnx", "tiny.en-decoder.onnx", "tiny.en-decoder.int8.onnx", "tiny.en-tokens.txt"})
        std::ofstream(asr / n) << "x";
    auto whisper = sherpa_asr_options(asr.string());
    REQUIRE(whisper.ok);
    CHECK(whisper.value.find("\"whisper\"") != std::string::npos);
    CHECK(whisper.value.find("tiny.en-encoder.int8.onnx") != std::string::npos);
    CHECK(whisper.value.find("tiny.en-decoder.int8.onnx") != std::string::npos);

    std::ofstream(asr / "joiner.onnx") << "x";
    auto nemo = sherpa_asr_options(asr.string());
    REQUIRE(nemo.ok);
    CHECK(nemo.value.find("\"nemo_transducer\"") != std::string::npos);
    CHECK(sherpa_asr_options(R"({"whisper":{"encoder":"e"}})").value == R"({"asr":{"whisper":{"encoder":"e"}}})");
    CHECK_FALSE(sherpa_asr_options("/nonexistent/dir").ok);
    const fs::path empty = temp_dir("agent_test_voice_empty");
    CHECK_FALSE(sherpa_asr_options(empty.string()).ok);

    const fs::path tts = temp_dir("agent_test_voice_tts");
    for (const char *n : {"model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt", "lexicon-zh.txt"}) std::ofstream(tts / n) << "x";
    fs::create_directories(tts / "espeak-ng-data");
    auto kokoro = sherpa_tts_options(tts.string(), 4);
    REQUIRE(kokoro.ok);
    CHECK(kokoro.value.find("\"kokoro\"") != std::string::npos);
    CHECK(kokoro.value.find("lexicon-us-en.txt") != std::string::npos);
    CHECK(kokoro.value.find("espeak-ng-data") != std::string::npos);
    CHECK(kokoro.value.find("\"speaker_id\":4") != std::string::npos);
    CHECK_FALSE(sherpa_tts_options(empty.string()).ok);

    std::ofstream(asr / "silero_vad.onnx") << "x";
    auto vad = sherpa_vad_options((asr / "silero_vad.onnx").string());
    REQUIRE(vad.ok);
    CHECK(vad.value.find("\"silero\"") != std::string::npos);
    CHECK(vad.value.find("\"streaming\":true") != std::string::npos);
    CHECK_FALSE(sherpa_vad_options("/nonexistent/silero_vad.onnx").ok);
}

// -------------------------- end to end --------------------------

#if defined(AGENT_TEST_CLI_PATH) && defined(AGENT_TEST_SHERPA_ENGINE_PATH)

namespace {

const char *env_or_null(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

std::string run_capture(const std::string &cmd, int &status) {
    std::string out;
    std::FILE *p = popen(cmd.c_str(), "r");
    if (!p) {
        status = -1;
        return out;
    }
    char buf[4096];
    for (size_t n; (n = std::fread(buf, 1, sizeof(buf), p)) > 0;) out.append(buf, n);
    status = pclose(p);
    return out;
}

std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string line_starting(const std::string &text, std::string_view prefix) {
    for (size_t at = 0; at < text.size();) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        if (std::string_view(text).substr(at, end - at).starts_with(prefix))
            return text.substr(at + prefix.size(), end - at - prefix.size());
        at = end + 1;
    }
    return {};
}

} // namespace

TEST_CASE("agent voice answers a spoken question from a WAV file", "[voice][model]") {
    const char *tts_dir = env_or_null("AGENT_TEST_SHERPA_TTS_DIR");
    const char *asr_dir = env_or_null("AGENT_TEST_SHERPA_ASR_DIR");
    const char *llm = env_or_null("AGENT_TEST_LLAMA_MODEL");
    const char *vad = env_or_null("AGENT_TEST_SHERPA_VAD_MODEL");
    if (!tts_dir || !asr_dir || !llm) SKIP("AGENT_TEST_SHERPA_TTS_DIR, AGENT_TEST_SHERPA_ASR_DIR and AGENT_TEST_LLAMA_MODEL not all set");
    const fs::path dir = temp_dir("agent_test_voice_e2e");

    // The question, spoken by Kokoro through the sherpa engine, with some silence around it.
    auto tts_opts = sherpa_tts_options(tts_dir);
    REQUIRE(tts_opts.ok);
    std::string options = tts_opts.value.substr(1, tts_opts.value.size() - 2); // {"tts":{...}} -> "tts":{...}
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_SHERPA_ENGINE_PATH;
    cfg.workspace_dir = (dir / "tts-ws").string();
    cfg.engine_options = "{" + options + "}";
    auto session = agent::init(cfg);
    REQUIRE(session.ok);
    agent::FlowMemory memory;
    memory.id = "question";
    REQUIRE(agent::run_turn(session.value, memory, "What is two plus two?").ok);
    REQUIRE(memory.history.back().media.size() == 1);
    auto spoken = audio::read_wav(memory.history.back().media[0].path);
    REQUIRE(spoken.ok);
    std::vector<float> question(static_cast<size_t>(0.6 * spoken.value.rate), 0.0f);
    question.insert(question.end(), spoken.value.samples.begin(), spoken.value.samples.end());
    question.insert(question.end(), static_cast<size_t>(1.2 * spoken.value.rate), 0.0f);
    const std::string wav = (dir / "question.wav").string();
    REQUIRE(audio::write_wav(wav, question, spoken.value.rate).ok);

    std::string cmd = std::string(AGENT_TEST_CLI_PATH) + " voice --config-dir " + (dir / "config").string() + " --workspace " +
                      (dir / "ws").string() + " --model " + llm + " --asr-model " + asr_dir + " --tts-model " + tts_dir + " --input-wav " +
                      wav + " --output-dir " + (dir / "out").string() + " --max-turns 1 --max-tokens 48";
    if (vad) cmd += std::string(" --vad-model ") + vad;
    int status = 0;
    const std::string output = run_capture(cmd + " < /dev/null", status);
    INFO("command: " << cmd << "\noutput:\n" << output);
    REQUIRE(status == 0);
    const std::string heard = line_starting(output, "you: ");
    const std::string answer = line_starting(output, "agent: ");
    WARN("transcript: " << heard << "\nanswer: " << answer);
    // Whisper writes numbers and operators either way ("two plus two" or "2 + 2").
    CHECK((lower(heard).find("two") != std::string::npos || heard.find('2') != std::string::npos));
    CHECK((lower(heard).find("plus") != std::string::npos || heard.find('+') != std::string::npos));
    CHECK_FALSE(answer.empty());
    const fs::path reply = dir / "out" / "reply-001.wav";
    REQUIRE(fs::exists(reply));
    CHECK(fs::file_size(reply) > 44 + 24000); // at least half a second at 24 kHz
    CHECK(output.find("1 turn(s)") != std::string::npos);
}

#endif
