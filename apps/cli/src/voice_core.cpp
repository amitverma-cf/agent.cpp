#include "voice.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <simdjson.h>

namespace agent::cli {

namespace fs = std::filesystem;

// -------------------------- energy gate --------------------------

void energy_gate_reset(EnergyGate &gate) {
    const EnergyGateOptions options = gate.options;
    gate = EnergyGate{};
    gate.options = options;
}

namespace {

int64_t ms_to_samples(int ms) { return static_cast<int64_t>(ms) * audio::kCaptureRate / 1000; }

void gate_frame(EnergyGate &g, const float *frame, size_t n, std::vector<SpeechSegment> &finished) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += static_cast<double>(frame[i]) * frame[i];
    const float rms = static_cast<float>(std::sqrt(sum / static_cast<double>(n)));
    const EnergyGateOptions &o = g.options;
    if (g.noise_floor < 0.0f) g.noise_floor = o.threshold / o.noise_ratio;
    const bool voiced = rms > std::max(o.threshold, o.noise_ratio * g.noise_floor);
    const int64_t frame_start = g.pos;
    const int64_t frame_end = g.pos + static_cast<int64_t>(n);
    const int frames_for_speech = std::max(1, o.min_speech_ms / o.frame_ms);
    const int frames_for_silence = std::max(1, o.min_silence_ms / o.frame_ms);

    if (!g.speaking) {
        if (voiced) {
            if (g.voiced_run++ == 0) g.voiced_run_start = frame_start;
            if (g.voiced_run >= frames_for_speech) {
                g.speaking = true;
                g.speech_start = g.voiced_run_start;
                g.last_voiced_end = frame_end;
                g.quiet_run = 0;
            }
        } else {
            g.voiced_run = 0;
            // Track the noise floor: down quickly, up slowly.
            g.noise_floor += (rms < g.noise_floor ? 0.3f : 0.02f) * (rms - g.noise_floor);
        }
    } else {
        if (voiced) {
            g.last_voiced_end = frame_end;
            g.quiet_run = 0;
            g.noise_floor += 0.001f * (rms - g.noise_floor); // steady noise eventually stops counting as speech
        } else if (++g.quiet_run >= frames_for_silence) {
            finished.push_back({g.speech_start, g.last_voiced_end});
            g.speaking = false;
            g.voiced_run = 0;
        }
        if (g.speaking && frame_end - g.speech_start >= ms_to_samples(o.max_speech_ms)) {
            finished.push_back({g.speech_start, frame_end});
            g.speech_start = frame_end;
            g.last_voiced_end = frame_end;
        }
    }
    g.pos = frame_end;
}

} // namespace

bool energy_gate_feed(EnergyGate &gate, std::span<const float> samples, std::vector<SpeechSegment> &finished) {
    const size_t frame = static_cast<size_t>(ms_to_samples(std::max(1, gate.options.frame_ms)));
    size_t i = 0;
    if (!gate.partial.empty()) {
        const size_t take = std::min(frame - gate.partial.size(), samples.size());
        gate.partial.insert(gate.partial.end(), samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(take));
        i = take;
        if (gate.partial.size() == frame) {
            gate_frame(gate, gate.partial.data(), frame, finished);
            gate.partial.clear();
        }
    }
    for (; i + frame <= samples.size(); i += frame) gate_frame(gate, samples.data() + i, frame, finished);
    if (i < samples.size()) gate.partial.insert(gate.partial.end(), samples.begin() + static_cast<std::ptrdiff_t>(i), samples.end());
    return gate.speaking;
}

void energy_gate_flush(EnergyGate &gate, std::vector<SpeechSegment> &finished) {
    if (gate.speaking) finished.push_back({gate.speech_start, gate.last_voiced_end});
    gate.speaking = false;
    gate.voiced_run = 0;
    gate.partial.clear();
}

std::vector<SpeechSegment> detect_speech_energy(std::span<const float> samples, const EnergyGateOptions &options) {
    EnergyGate gate;
    gate.options = options;
    std::vector<SpeechSegment> out;
    energy_gate_feed(gate, samples, out);
    energy_gate_flush(gate, out);
    return out;
}

namespace {

Result<bool> energy_feed(void *state, std::span<const float> samples, std::vector<SpeechSegment> &finished) {
    return ok(energy_gate_feed(*static_cast<EnergyGate *>(state), samples, finished));
}
Result<void> energy_flush(void *state, std::vector<SpeechSegment> &finished) {
    energy_gate_flush(*static_cast<EnergyGate *>(state), finished);
    return ok();
}
void energy_reset(void *state) { energy_gate_reset(*static_cast<EnergyGate *>(state)); }

constexpr SpeechDetectorOps kEnergyOps = {energy_feed, energy_flush, energy_reset};

} // namespace

SpeechDetector energy_detector(EnergyGate &gate) { return SpeechDetector{&kEnergyOps, &gate, "energy"}; }

Result<std::vector<VadJsonSegment>> parse_vad_segments(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::array arr;
    if (parser.parse(json.data(), json.size()).get_array().get(arr))
        return fail<std::vector<VadJsonSegment>>(ErrorCode::ParseError,
                                                 "VAD output is not a JSON array: " + std::string(json.substr(0, 80)));
    std::vector<VadJsonSegment> out;
    for (simdjson::dom::element e : arr) {
        VadJsonSegment s;
        if (e["start"].get_double().get(s.start) || e["end"].get_double().get(s.end))
            return fail<std::vector<VadJsonSegment>>(ErrorCode::ParseError, "VAD segment without start/end");
        bool open = false;
        if (!e["open"].get_bool().get(open)) s.open = open;
        out.push_back(s);
    }
    return ok(std::move(out));
}

// -------------------------- the loop --------------------------

namespace {

struct LoopState {
    audio::AudioInput &input;
    audio::AudioOutput &output;
    SpeechDetector &detector;
    const VoiceHooks &hooks;
    const VoiceLoopOptions &options;
    VoiceLoopStats stats;
    std::vector<float> buf; // captured audio [buf_start, buf_start + buf.size())
    int64_t buf_start = 0;
    int64_t clock = 0;      // samples read so far
    int64_t det_origin = 0; // clock at the detector's last reset
    bool cancelled = false;

    void emit(VoiceEvent e, std::string_view text = {}) const {
        if (hooks.on_event) hooks.on_event(e, text, hooks.user_data);
    }

    // false = stop the loop (cancelled)
    bool hook_failed(const Error &e) {
        if (e.code == ErrorCode::Cancelled) {
            cancelled = true;
            return false;
        }
        ++stats.errors;
        emit(VoiceEvent::Error, e.message);
        return true;
    }

    void restart_listening() {
        audio::discard(input);
        detector.ops->reset(detector.state);
        det_origin = clock;
        buf.clear();
        buf_start = clock;
    }

    bool handle(const SpeechSegment &seg) {
        const int64_t buf_end = buf_start + static_cast<int64_t>(buf.size());
        const int64_t from = std::max(buf_start, det_origin + seg.start - ms_to_samples(options.pre_roll_ms));
        const int64_t to = std::min(buf_end, det_origin + seg.end + ms_to_samples(options.post_roll_ms));
        if (to - from < ms_to_samples(options.min_utterance_ms)) {
            ++stats.skipped;
            emit(VoiceEvent::Skipped, "too short");
            return true;
        }
        const std::span<const float> utterance(buf.data() + (from - buf_start), static_cast<size_t>(to - from));

        auto heard = hooks.transcribe(utterance, hooks.user_data);
        if (!heard.ok) return hook_failed(heard.error);
        if (!is_meaningful_transcript(heard.value)) {
            ++stats.skipped;
            emit(VoiceEvent::Skipped, heard.value);
            return true;
        }
        emit(VoiceEvent::Heard, heard.value);

        auto reply = hooks.respond(heard.value, hooks.user_data);
        if (!reply.ok) return hook_failed(reply.error);
        emit(VoiceEvent::Reply, reply.value);
        ++stats.turns;

        const std::string speech = speakable_text(reply.value);
        if (!speech.empty()) {
            auto wav = hooks.speak(speech, hooks.user_data);
            if (!wav.ok) return hook_failed(wav.error);
            if (auto r = audio::play(output, wav.value.samples, wav.value.rate); !r.ok) return hook_failed(r.error);
            char info[96];
            std::snprintf(info, sizeof(info), "%.2f s of speech at %d Hz", static_cast<double>(wav.value.samples.size()) / wav.value.rate,
                          wav.value.rate);
            emit(VoiceEvent::Spoken, info);
        }
        if (!options.barge_in) restart_listening(); // half-duplex: forget what the mic heard meanwhile
        return true;
    }

    bool done() const { return cancelled || (options.max_turns > 0 && stats.turns >= options.max_turns); }
};

} // namespace

Result<VoiceLoopStats> run_voice_loop(audio::AudioInput &input, audio::AudioOutput &output, SpeechDetector &detector,
                                      const VoiceHooks &hooks, const VoiceLoopOptions &options) {
    if (!hooks.transcribe || !hooks.respond || !hooks.speak || !detector.ops)
        return fail<VoiceLoopStats>(ErrorCode::InvalidConfig, "run_voice_loop: missing hooks or detector");
    LoopState st{input, output, detector, hooks, options, {}, {}, 0, 0, 0, false};
    std::vector<float> chunk(std::max<size_t>(options.chunk_samples, 160));
    std::vector<SpeechSegment> finished;
    bool was_muted = false;
    int64_t overlap = 0; // samples of speech while the reply plays (barge-in)
    const int64_t keep = static_cast<int64_t>(audio::kCaptureRate) * 30;

    st.emit(VoiceEvent::Listening);
    while (!st.done() && !(options.stop && options.stop->load())) {
        bool eof = false;
        const size_t n = audio::read(input, chunk.data(), chunk.size(), 200, eof);
        audio::tick(output, n);

        // Half-duplex: drop what the mic hears while the reply plays.
        if (!options.barge_in && audio::playing(output)) {
            st.clock += static_cast<int64_t>(n);
            st.buf.clear();
            st.buf_start = st.clock;
            was_muted = true;
            if (eof) break;
            continue;
        }
        if (was_muted) {
            was_muted = false;
            detector.ops->reset(detector.state);
            st.det_origin = st.clock;
        }

        if (n > 0) {
            st.buf.insert(st.buf.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(n));
            st.clock += static_cast<int64_t>(n);
            finished.clear();
            auto speaking = detector.ops->feed(detector.state, std::span<const float>(chunk.data(), n), finished);
            if (!speaking.ok) return fail<VoiceLoopStats>(speaking.error.code, speaking.error.message);
            if (options.barge_in && speaking.value && audio::playing(output)) {
                overlap += static_cast<int64_t>(n);
                if (overlap >= ms_to_samples(options.barge_in_ms)) {
                    audio::stop(output);
                    ++st.stats.barge_ins;
                    st.emit(VoiceEvent::BargeIn);
                    overlap = 0;
                }
            } else if (!speaking.value) {
                overlap = 0;
            }
            for (const SpeechSegment &seg : finished) {
                if (!st.handle(seg) || st.done()) break;
                st.emit(VoiceEvent::Listening);
            }
            if (st.buf_start < st.clock - keep) { // bound the buffer (segments are at most max_speech long)
                const int64_t drop = std::min<int64_t>(static_cast<int64_t>(st.buf.size()), st.clock - keep - st.buf_start);
                st.buf.erase(st.buf.begin(), st.buf.begin() + static_cast<std::ptrdiff_t>(drop));
                st.buf_start += drop;
            }
        }
        if (eof) {
            finished.clear();
            if (auto r = detector.ops->flush(detector.state, finished); !r.ok) return fail<VoiceLoopStats>(r.error.code, r.error.message);
            for (const SpeechSegment &seg : finished)
                if (st.done() || !st.handle(seg)) break;
            break;
        }
    }
    if (!st.cancelled) audio::drain(output);
    else audio::stop(output);
    return ok(st.stats);
}

bool is_meaningful_transcript(std::string_view t) {
    int depth = 0;
    for (char c : t) {
        if (c == '[' || c == '(' || c == '<') ++depth;
        else if ((c == ']' || c == ')' || c == '>') && depth > 0) --depth;
        else if (depth == 0 && std::isalnum(static_cast<unsigned char>(c))) return true;
    }
    return false;
}

namespace {

// Removes every <tag>...</tag> block (and an unterminated one up to the end).
void strip_blocks(std::string &s, std::string_view open, std::string_view close) {
    for (size_t at; (at = s.find(open)) != std::string::npos;) {
        const size_t end = s.find(close, at + open.size());
        s.erase(at, end == std::string::npos ? std::string::npos : end + close.size() - at);
    }
}

} // namespace

namespace {

// 0 .. 999'999'999'999 in English words.
std::string number_words(uint64_t n) {
    static constexpr const char *kSmall[] = {"zero",     "one",     "two",     "three",     "four",     "five",    "six",
                                             "seven",    "eight",   "nine",    "ten",       "eleven",   "twelve",  "thirteen",
                                             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    static constexpr const char *kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    if (n < 20) return kSmall[n];
    if (n < 100) return std::string(kTens[n / 10]) + (n % 10 ? std::string(" ") + kSmall[n % 10] : "");
    if (n < 1000) return std::string(kSmall[n / 100]) + " hundred" + (n % 100 ? " " + number_words(n % 100) : "");
    static constexpr struct {
        uint64_t value;
        const char *name;
    } kScales[] = {{1'000'000'000, "billion"}, {1'000'000, "million"}, {1'000, "thousand"}};
    for (const auto &sc : kScales)
        if (n >= sc.value) return number_words(n / sc.value) + " " + sc.name + (n % sc.value ? " " + number_words(n % sc.value) : "");
    return {};
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// TTS models speak words, not digits or symbols (many would skip them silently): spell them out.
std::string verbalize(std::string_view s) {
    std::string out;
    auto prev_non_space = [&](size_t i) {
        while (i > 0 && s[i - 1] == ' ') --i;
        return i > 0 ? s[i - 1] : '\0';
    };
    auto next_non_space = [&](size_t i) {
        while (i + 1 < s.size() && s[i + 1] == ' ') ++i;
        return i + 1 < s.size() ? s[i + 1] : '\0';
    };
    for (size_t i = 0; i < s.size();) {
        const char c = s[i];
        if (is_digit(c)) {
            std::string digits;
            for (; i < s.size() && (is_digit(s[i]) || (s[i] == ',' && i + 1 < s.size() && is_digit(s[i + 1]) && !digits.empty())); ++i)
                if (s[i] != ',') digits += s[i];
            out += ' ';
            if (digits.size() <= 12) {
                out += number_words(std::stoull(digits));
            } else {
                for (char d : digits) out += number_words(static_cast<uint64_t>(d - '0')) + " ";
            }
            if (i + 1 < s.size() && s[i] == '.' && is_digit(s[i + 1])) { // decimals digit by digit
                out += " point";
                for (++i; i < s.size() && is_digit(s[i]); ++i) out += " " + number_words(static_cast<uint64_t>(s[i] - '0'));
            }
            out += ' ';
            continue;
        }
        const bool between_numbers = is_digit(prev_non_space(i)) && is_digit(next_non_space(i));
        switch (c) {
        case '+': out += " plus "; break;
        case '=': out += " equals "; break;
        case '%': out += " percent "; break;
        case '&': out += " and "; break;
        case '-': out += between_numbers ? " minus " : "-"; break;
        case 'x':
        case '/': out += between_numbers ? (c == 'x' ? " times " : " divided by ") : std::string(1, c); break;
        default: out += c;
        }
        ++i;
    }
    return out;
}

} // namespace

std::string speakable_text(std::string_view reply) {
    std::string s(reply);
    strip_blocks(s, "<tool_call>", "</tool_call>");
    strip_blocks(s, "<think>", "</think>");
    strip_blocks(s, "```", "```");
    // [text](url) -> text
    for (size_t at; (at = s.find("](")) != std::string::npos;) {
        const size_t open = s.rfind('[', at);
        const size_t close = s.find(')', at);
        if (open == std::string::npos || close == std::string::npos) break;
        s = s.substr(0, open) + s.substr(open + 1, at - open - 1) + s.substr(close + 1);
    }
    std::string plain;
    for (char c : s)
        if (c != '*' && c != '#' && c != '`' && c != '_' && c != '|' && c != '>') plain += c;
    const std::string spoken = verbalize(plain);
    std::string out;
    out.reserve(spoken.size());
    bool space = false;
    for (char c : spoken) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            space = !out.empty();
            continue;
        }
        if (space && c != '.' && c != ',' && c != '?' && c != '!' && c != ';' && c != ':') out += ' ';
        space = false;
        out += c;
    }
    return out;
}

// -------------------------- settings --------------------------

namespace {

Result<int> parse_count(const CommandLine &cl, std::string_view flag, int min_value) {
    const std::string_view text = cl.get(flag);
    int v = 0;
    if (text.empty()) return fail<int>(ErrorCode::InvalidConfig, "--" + std::string(flag) + " expects a number");
    for (char c : text) {
        if (c < '0' || c > '9' || v > 10'000'000)
            return fail<int>(ErrorCode::InvalidConfig, "--" + std::string(flag) + " expects a number");
        v = v * 10 + (c - '0');
    }
    if (v < min_value)
        return fail<int>(ErrorCode::InvalidConfig, "--" + std::string(flag) + " must be at least " + std::to_string(min_value));
    return ok(v);
}

bool is_json_object(std::string_view s) {
    const size_t i = s.find_first_not_of(" \t\r\n");
    return i != std::string_view::npos && s[i] == '{';
}

std::vector<std::string> dir_entries(const fs::path &dir) {
    std::vector<std::string> names;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) names.push_back(it->path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

// First entry containing every needle; an int8 variant wins.
std::string pick(const std::vector<std::string> &names, std::initializer_list<std::string_view> needles) {
    std::string best;
    for (const auto &n : names) {
        bool match = true;
        for (auto needle : needles) match = match && n.find(needle) != std::string::npos;
        if (!match) continue;
        if (best.empty() || (n.find("int8") != std::string::npos && best.find("int8") == std::string::npos)) best = n;
    }
    return best;
}

std::string path_in(const fs::path &dir, const std::string &name) { return json_escape((dir / name).string()); }

} // namespace

Result<VoiceSettings> voice_settings_from(const CommandLine &cl) {
    VoiceSettings s;
    s.model = std::string(cl.get("model"));
    s.engine = std::string(cl.get("engine"));
    s.asr_model = std::string(cl.get("asr-model"));
    s.tts_model = std::string(cl.get("tts-model"));
    s.vad_model = std::string(cl.get("vad-model"));
    s.voice = std::string(cl.get("voice"));
    s.input_wav = std::string(cl.get("input-wav"));
    s.output_dir = std::string(cl.get("output-dir"));
    s.workspace = std::string(cl.get("workspace"));
    s.engine_options = std::string(cl.get("engine-options"));
    s.resume = std::string(cl.get("resume"));
    s.barge_in = cl.has("barge-in");
    s.auto_approve = cl.has("yes");
    s.verbose = cl.has("verbose");
    s.inprocess = cl.has("inprocess");
    struct IntFlag {
        std::string_view name;
        int *out;
        int min;
    };
    const IntFlag ints[] = {{"ctx", &s.context_window, 1}, {"max-tokens", &s.max_tokens, 1}, {"max-turns", &s.max_turns, 1}};
    for (const auto &f : ints) {
        if (!cl.has(f.name)) continue;
        auto v = parse_count(cl, f.name, f.min);
        if (!v.ok) return fail<VoiceSettings>(v.error.code, v.error.message);
        *f.out = v.value;
    }
    if (s.asr_model.empty())
        return fail<VoiceSettings>(ErrorCode::InvalidConfig,
                                   "agent voice needs --asr-model FILE (an audio.cpp speech-recognition GGUF, e.g. parakeet_tdt)");
    if (s.tts_model.empty())
        return fail<VoiceSettings>(ErrorCode::InvalidConfig,
                                   "agent voice needs --tts-model FILE (an audio.cpp text-to-speech GGUF, e.g. pocket_tts)");
    return ok(std::move(s));
}

// audio.cpp models are single GGUF files (the family is read from the file); a JSON object is passed through.
namespace {
Result<std::string> gguf_component(std::string_view arg, const char *key, const char *flag, std::string_view extra = {}) {
    if (is_json_object(arg)) return ok("{\"" + std::string(key) + "\":" + std::string(arg) + "}");
    if (!fs::is_regular_file(fs::path(std::string(arg))))
        return fail<std::string>(ErrorCode::FilesystemError,
                                 std::string(flag) + ": no such file: " + std::string(arg) + " (an audio.cpp GGUF model)");
    return ok("{\"" + std::string(key) + "\":{\"model\":\"" + json_escape(std::string(arg)) + "\"" + std::string(extra) + "}}");
}
} // namespace

Result<std::string> audiocpp_asr_options(std::string_view arg) { return gguf_component(arg, "asr", "--asr-model"); }

Result<std::string> audiocpp_tts_options(std::string_view arg, std::string_view voice) {
    std::string extra;
    if (!voice.empty()) extra = ",\"voice\":\"" + json_escape(std::string(voice)) + "\"";
    return gguf_component(arg, "tts", "--tts-model", extra);
}

Result<std::string> audiocpp_vad_options(std::string_view arg) { return gguf_component(arg, "vad", "--vad-model", ",\"streaming\":true"); }

} // namespace agent::cli
