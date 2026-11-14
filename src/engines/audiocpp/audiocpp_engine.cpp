// agent-engine-audiocpp: audio.cpp (https://github.com/0xShug0/audio.cpp, Apache-2.0) speech and audio models as an
// agent.cpp engine plugin. One engine for TTS, ASR, VAD and diarization; the model families come from audio.cpp.
//
// Wraps audio.cpp's C ABI (audiocpp.h). Components are created only when configured, one audio.cpp model per task:
//
//   options.tts      {"family": "pocket_tts", "model": "<gguf>", "voice": "<preset>", "language": "en", "speed": 1.0}
//   options.asr      {"family": "parakeet_tdt", "model": "<gguf>", "language": "en"}
//   options.vad      {"family": "firered_vad", "model": "<gguf>"}
//   options.diarize  {"family": "nemotron_3_diar", "model": "<gguf>"}
//   options.threads  worker threads (default 4)       options.backend  "cpu" | "cuda" | "vulkan" | "metal" | "best"
//
// Routing and response formats are the same as the engine this one replaced: TTS synthesizes the last user text and
// returns a WAV FILE part; ASR returns the transcript; VAD / DIARIZE return a JSON TEXT part plus an F32 "segments" tensor
// ([n, 2] start/end seconds for VAD, [n, 3] start/end/speaker for diarization).
//
// TTS audio is returned as a WAV file in the request output_dir (no sample data travels through the engine host pipe).
//
// Audio input is a FILE or BYTES WAV (PCM 8/16/24/32-bit or float32, any rate, any channel count) or an F32 TENSOR of mono
// samples at 16 kHz. It is mixed to mono and resampled to the rate the task needs.

#include <agent-cpp/engine_abi.h>
#include <atomic>
#include <audiocpp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX // windows.h would turn std::min and std::max into macros
#endif
#include <stdlib.h>
#include <windows.h>
#else
#include <stdlib.h>
#endif

namespace {

constexpr int kModelRate = 16000;

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }
agent_str to_c(const std::string &s) { return agent_str{s.data(), s.size()}; }

bool file_exists(const std::string &path) {
    struct stat st{};
    return !path.empty() && ::stat(path.c_str(), &st) == 0;
}

void set_env(const char *key, const std::string &value) {
#if defined(_WIN32)
    _putenv_s(key, value.c_str());
#else
    ::setenv(key, value.c_str(), 1);
#endif
}

// One audio.cpp model with the session created for its task.
struct Component {
    audiocpp_model *model = nullptr;
    audiocpp_session *session = nullptr;
    std::string voice, language;
    double speed = 0;

    ~Component() {
        if (session) audiocpp_session_free(session);
        if (model) audiocpp_model_free(model);
    }
    explicit operator bool() const { return session != nullptr; }
};

struct Instance {
    audiocpp_registry *registry = nullptr;
    Component tts, asr, vad, diar;
    bool vad_streaming = false; // options.vad.streaming: keep the audio since the last reset and report finished segments once
    std::vector<float> vad_buf; // 16 kHz mono, streaming only
    size_t vad_reported = 0;    // segments already returned as finished

    // Response storage: valid until free_response() or the next run().
    std::string text, path, error;
    agent_part parts[2]{};
    std::vector<float> tensor;
    int64_t tensor_shape[2]{};

    ~Instance() {
        // Sessions and models must go before the registry they were created from.
        tts = Component();
        asr = Component();
        vad = Component();
        diar = Component();
        if (registry) audiocpp_registry_free(registry);
    }
    int32_t fail(int32_t code, std::string msg) {
        error = std::move(msg);
        return code;
    }
    int32_t fail_audiocpp(const char *what) {
        error = std::string("audiocpp: ") + what + ": " + audiocpp_last_error();
        return AGENT_ERR_DECODE_FAILED;
    }
};

bool cancelled(const agent_request *req) { return req->cancel && *req->cancel != 0; }

// ------------------------------------------------------------------ config

std::string json_str(simdjson::dom::object obj, const char *key) {
    std::string_view v;
    if (obj[key].get_string().get(v)) return {};
    return std::string(v);
}
double json_num(simdjson::dom::object obj, const char *key, double fallback) {
    double v = 0;
    if (obj[key].get_double().get(v)) return fallback;
    return v;
}

int32_t create_component(Instance &inst, Component &c, simdjson::dom::object cfg, const char *section, const char *task,
                         const audiocpp_backend_config &backend, char *err, size_t err_cap) {
    const std::string model = json_str(cfg, "model");
    const std::string family = json_str(cfg, "family");
    if (model.empty()) {
        std::snprintf(err, err_cap, "audiocpp: options.%s.model is required", section);
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (!file_exists(model)) {
        std::snprintf(err, err_cap, "audiocpp: options.%s.model not found: %s", section, model.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    audiocpp_model_config mc{};
    mc.family_hint = family.empty() ? nullptr : family.c_str();
    const audiocpp_status loaded = audiocpp_model_load(inst.registry, model.c_str(), &mc, nullptr, &c.model);
    if (loaded != AUDIOCPP_OK) {
        std::snprintf(err, err_cap, "audiocpp: loading %s failed: %s", model.c_str(), audiocpp_last_error());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    if (!audiocpp_model_supports(c.model, task, "offline")) {
        std::snprintf(err, err_cap, "audiocpp: the %s model does not do task \"%s\"", audiocpp_model_family(c.model), task);
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (audiocpp_session_create(c.model, task, "offline", &backend, nullptr, &c.session) != AUDIOCPP_OK) {
        std::snprintf(err, err_cap, "audiocpp: creating a %s session failed: %s", task, audiocpp_last_error());
        return AGENT_ERR_INIT_FAILED;
    }
    c.voice = json_str(cfg, "voice");
    // PocketTTS has no default voice, but its model ships one: without a voice the first request would fail.
    if (c.voice.empty() && std::strcmp(task, "tts") == 0 && std::strcmp(audiocpp_model_family(c.model), "pocket_tts") == 0)
        c.voice = "alba";
    c.language = json_str(cfg, "language");
    c.speed = json_num(cfg, "speed", 0);
    return AGENT_OK;
}

// ------------------------------------------------------------------ request helpers

struct UserInput {
    std::string_view text;
    const agent_part *audio;
};

bool is_audio(const agent_part &p) {
    if (p.kind == AGENT_PART_TENSOR) return p.dtype == AGENT_DTYPE_F32;
    const std::string_view mime = sv(p.mime);
    if (mime.starts_with("audio/")) return true;
    if (p.kind == AGENT_PART_FILE && mime.empty()) return sv(p.path).ends_with(".wav");
    return false;
}

UserInput last_user_input(const agent_request *req) {
    UserInput in{{}, nullptr};
    for (size_t i = req->n_messages; i-- > 0;) {
        const agent_message &m = req->messages[i];
        if (sv(m.role) != "user") continue;
        for (size_t p = 0; p < m.n_parts; ++p) {
            const agent_part &part = m.parts[p];
            if (part.kind == AGENT_PART_TEXT && part.text.size > 0) in.text = sv(part.text);
            else if ((part.kind == AGENT_PART_FILE || part.kind == AGENT_PART_BYTES || part.kind == AGENT_PART_TENSOR) && is_audio(part))
                in.audio = &part;
        }
        break;
    }
    return in;
}

// ------------------------------------------------------------------ WAV

struct Audio {
    std::vector<float> samples; // mono
    int rate = 0;
};

uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint16_t rd16(const uint8_t *p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

// RIFF/WAVE: PCM 8/16/24/32-bit and IEEE float32, any channel count (mixed to mono).
bool decode_wav(const uint8_t *d, size_t n, Audio &out, std::string &why) {
    if (n < 44 || std::memcmp(d, "RIFF", 4) != 0 || std::memcmp(d + 8, "WAVE", 4) != 0) {
        why = "not a RIFF/WAVE file";
        return false;
    }
    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t *data = nullptr;
    size_t data_size = 0;
    for (size_t pos = 12; pos + 8 <= n;) {
        const uint32_t sz = rd32(d + pos + 4);
        const uint8_t *body = d + pos + 8;
        const size_t avail = n - (pos + 8);
        if (std::memcmp(d + pos, "fmt ", 4) == 0 && sz >= 16 && avail >= 16) {
            fmt = rd16(body);
            channels = rd16(body + 2);
            rate = rd32(body + 4);
            bits = rd16(body + 14);
            if (fmt == 0xFFFE && sz >= 26 && avail >= 26) fmt = rd16(body + 24); // WAVE_FORMAT_EXTENSIBLE sub-format
        } else if (std::memcmp(d + pos, "data", 4) == 0) {
            data = body;
            data_size = sz > avail ? avail : sz;
            break;
        }
        pos += 8 + static_cast<size_t>(sz) + (sz & 1);
    }
    if (!data || channels == 0 || rate == 0) {
        why = "missing fmt or data chunk";
        return false;
    }
    const size_t bytes = bits / 8;
    if (!((fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32))) {
        why = "unsupported sample format (need PCM 8/16/24/32-bit or float32)";
        return false;
    }
    const size_t frames = data_size / (bytes * channels);
    out.samples.assign(frames, 0.0f);
    for (size_t f = 0; f < frames; ++f) {
        float acc = 0;
        for (uint16_t c = 0; c < channels; ++c) {
            const uint8_t *s = data + (f * channels + c) * bytes;
            float v;
            if (fmt == 3) {
                std::memcpy(&v, s, 4);
            } else if (bits == 8) {
                v = (static_cast<int>(s[0]) - 128) / 128.0f;
            } else if (bits == 16) {
                v = static_cast<int16_t>(rd16(s)) / 32768.0f;
            } else if (bits == 24) {
                int32_t x = (s[0] << 8) | (s[1] << 16) | (static_cast<int32_t>(s[2]) << 24);
                v = static_cast<float>(x) / 2147483648.0f;
            } else {
                v = static_cast<float>(static_cast<int32_t>(rd32(s))) / 2147483648.0f;
            }
            acc += v;
        }
        out.samples[f] = acc / channels;
    }
    out.rate = static_cast<int>(rate);
    return true;
}

int32_t load_audio(Instance &inst, const agent_part &part, Audio &out) {
    if (part.kind == AGENT_PART_TENSOR) {
        if (part.dtype != AGENT_DTYPE_F32 || (part.size % sizeof(float)) != 0 || (part.size && !part.data))
            return inst.fail(AGENT_ERR_UNSUPPORTED, "audiocpp: an audio tensor must be F32 mono samples at 16 kHz");
        out.samples.resize(part.size / sizeof(float));
        if (part.size) std::memcpy(out.samples.data(), part.data, part.size);
        out.rate = kModelRate;
    } else {
        std::vector<uint8_t> file;
        const uint8_t *bytes = nullptr;
        size_t size = 0;
        std::string where = "<bytes>";
        if (part.kind == AGENT_PART_FILE) {
            where = std::string(sv(part.path));
            std::ifstream f(where, std::ios::binary);
            if (!f) return inst.fail(AGENT_ERR_FILESYSTEM, "audiocpp: cannot open " + where);
            file.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            bytes = file.data();
            size = file.size();
        } else {
            bytes = static_cast<const uint8_t *>(part.data);
            size = part.size;
        }
        std::string why;
        if (!decode_wav(bytes, size, out, why))
            return inst.fail(AGENT_ERR_UNSUPPORTED, "audiocpp: could not read audio " + where + ": " + why);
    }
    if (out.samples.empty() || out.rate <= 0) return inst.fail(AGENT_ERR_DECODE_FAILED, "audiocpp: audio input is empty");
    return AGENT_OK;
}

// Linear-interpolation resampler with a box low-pass when shrinking; good enough for speech models.
void resample_to(Audio &a, int rate) {
    if (a.rate == rate || a.samples.empty()) return;
    const double ratio = static_cast<double>(a.rate) / rate;
    const size_t n_out = static_cast<size_t>(static_cast<double>(a.samples.size()) / ratio);
    std::vector<float> out(n_out);
    const int taps = ratio > 1.0 ? static_cast<int>(std::ceil(ratio)) : 1;
    for (size_t i = 0; i < n_out; ++i) {
        const double x = static_cast<double>(i) * ratio;
        if (taps > 1) {
            // Average the input samples this output sample covers.
            const size_t b = static_cast<size_t>(x), e = std::min(a.samples.size(), static_cast<size_t>(x + ratio));
            double acc = 0;
            for (size_t k = b; k < e; ++k) acc += a.samples[k];
            out[i] = e > b ? static_cast<float>(acc / static_cast<double>(e - b)) : a.samples[std::min(b, a.samples.size() - 1)];
        } else {
            const size_t j = static_cast<size_t>(x);
            const double f = x - static_cast<double>(j);
            const float s0 = a.samples[j], s1 = a.samples[std::min(j + 1, a.samples.size() - 1)];
            out[i] = static_cast<float>(s0 + (s1 - s0) * f);
        }
    }
    a.samples = std::move(out);
    a.rate = rate;
}

std::string fmt_seconds(double s) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", s);
    return buf;
}

void set_text_response(Instance &inst, agent_response *resp) {
    inst.parts[0] = agent_part{};
    inst.parts[0].kind = AGENT_PART_TEXT;
    inst.parts[0].text = to_c(inst.text);
    resp->message.parts = inst.parts;
    resp->message.n_parts = 1;
}

void set_segments_response(Instance &inst, agent_response *resp, int64_t cols) {
    set_text_response(inst, resp);
    inst.tensor_shape[0] = static_cast<int64_t>(inst.tensor.size()) / cols;
    inst.tensor_shape[1] = cols;
    inst.parts[1] = agent_part{};
    inst.parts[1].kind = AGENT_PART_TENSOR;
    inst.parts[1].name = agent_str{"segments", 8};
    inst.parts[1].dtype = AGENT_DTYPE_F32;
    inst.parts[1].shape = inst.tensor_shape;
    inst.parts[1].ndim = 2;
    inst.parts[1].data = inst.tensor.data();
    inst.parts[1].size = inst.tensor.size() * sizeof(float);
    resp->message.n_parts = 2;
}

// Runs an audio-in task on a component; the caller reads the result and frees it.
int32_t run_audio_task(Instance &inst, Component &c, const agent_request *req, const agent_part &part, const char *what,
                       audiocpp_result **out, const std::vector<float> *whole = nullptr) {
    Audio audio;
    if (whole) {
        audio.samples = *whole;
        audio.rate = kModelRate;
    } else {
        if (int32_t rc = load_audio(inst, part, audio)) return rc;
        resample_to(audio, kModelRate);
    }
    if (cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "audiocpp: cancelled");
    audiocpp_request *r = audiocpp_request_create();
    if (!r) return inst.fail(AGENT_ERR_INIT_FAILED, "audiocpp: out of memory");
    int32_t rc = AGENT_OK;
    if (audiocpp_request_set_audio(r, audio.samples.data(), audio.samples.size(), audio.rate, 1) != AUDIOCPP_OK)
        rc = inst.fail_audiocpp("set_audio");
    if (rc == AGENT_OK && !c.language.empty()) audiocpp_request_set_option(r, "language", c.language.c_str());
    if (rc == AGENT_OK && req->on_progress) req->on_progress(req->user_data, 0, 1);
    if (rc == AGENT_OK && audiocpp_session_run(c.session, r, out) != AUDIOCPP_OK) rc = inst.fail_audiocpp(what);
    audiocpp_request_free(r);
    if (rc == AGENT_OK && req->on_progress) req->on_progress(req->user_data, 1, 1);
    return rc;
}

// ------------------------------------------------------------------ tasks

int32_t transcribe(Instance &inst, const agent_request *req, const agent_part &part, agent_response *resp) {
    audiocpp_result *res = nullptr;
    if (int32_t rc = run_audio_task(inst, inst.asr, req, part, "speech recognition", &res)) return rc;
    const char *text = nullptr;
    if (audiocpp_result_text(res, &text, nullptr) != AUDIOCPP_OK || !text) text = "";
    std::string_view t(text);
    while (!t.empty() && (t.front() == ' ' || t.front() == '\n')) t.remove_prefix(1);
    while (!t.empty() && (t.back() == ' ' || t.back() == '\n')) t.remove_suffix(1);
    inst.text.assign(t);
    audiocpp_result_free(res);
    if (req->on_chunk && !inst.text.empty()) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, inst.text.data(), inst.text.size());
    set_text_response(inst, resp);
    return AGENT_OK;
}

// Offline (default): the whole input is one stream; every speech segment is returned.
// Streaming (options.vad.streaming): the engine keeps the audio since the last reset and runs the model over all of it each
// call, so times are seconds since the stream start. A segment is returned once, when it has ended; while speech is ongoing
// (the last segment reaches the end of the audio) it is returned with "open":true and not counted as reported. A TEXT part
// "reset" on the request starts a new stream, "flush" ends the current one (an open segment is closed and returned).
int32_t detect_speech(Instance &inst, const agent_request *req, const agent_part &part, std::string_view command, agent_response *resp) {
    audiocpp_result *res = nullptr;
    const bool streaming = inst.vad_streaming;
    constexpr size_t kMaxStreamSamples = static_cast<size_t>(kModelRate) * 600; // 10 minutes since the last reset
    if (streaming) {
        if (command == "reset") {
            inst.vad_buf.clear();
            inst.vad_reported = 0;
        }
        Audio chunk;
        if (int32_t rc = load_audio(inst, part, chunk)) return rc;
        resample_to(chunk, kModelRate);
        if (inst.vad_buf.size() + chunk.samples.size() > kMaxStreamSamples)
            return inst.fail(AGENT_ERR_UNSUPPORTED, "audiocpp: VAD stream longer than 10 minutes without a reset");
        inst.vad_buf.insert(inst.vad_buf.end(), chunk.samples.begin(), chunk.samples.end());
        if (int32_t rc = run_audio_task(inst, inst.vad, req, part, "voice activity detection", &res, &inst.vad_buf)) return rc;
    } else if (int32_t rc = run_audio_task(inst, inst.vad, req, part, "voice activity detection", &res)) {
        return rc;
    }
    inst.tensor.clear();
    std::string json = "[";
    size_t count = 0;
    const double total = streaming ? static_cast<double>(inst.vad_buf.size()) / kModelRate : 0;
    const size_t n = audiocpp_result_segment_count(res);
    size_t reported = streaming ? inst.vad_reported : 0;
    for (size_t i = 0; i < n; ++i) {
        int64_t s = 0, e = 0;
        float conf = 0;
        if (audiocpp_result_segment(res, i, &s, &e, &conf, nullptr) != AUDIOCPP_OK) continue;
        const double start = static_cast<double>(s) / kModelRate, end = static_cast<double>(e) / kModelRate;
        const bool open = streaming && command != "flush" && i + 1 == n && end > total - 0.25;
        if (streaming && i < inst.vad_reported) continue;
        json += (count++ ? "," : "") + std::string("{\"start\":") + fmt_seconds(start) + ",\"end\":" + fmt_seconds(end) +
                (open ? ",\"open\":true" : "") + "}";
        inst.tensor.push_back(static_cast<float>(start));
        inst.tensor.push_back(static_cast<float>(end));
        if (!open) reported = i + 1;
    }
    audiocpp_result_free(res);
    if (streaming) {
        inst.vad_reported = reported;
        if (command == "flush") {
            inst.vad_buf.clear();
            inst.vad_reported = 0;
        }
    }
    inst.text = json + "]";
    set_segments_response(inst, resp, 2);
    return AGENT_OK;
}

int32_t diarize(Instance &inst, const agent_request *req, const agent_part &part, agent_response *resp) {
    audiocpp_result *res = nullptr;
    if (int32_t rc = run_audio_task(inst, inst.diar, req, part, "speaker diarization", &res)) return rc;
    inst.tensor.clear();
    std::string json = "[";
    std::vector<std::string> ids; // speaker label -> index in order of first appearance
    const size_t n = audiocpp_result_speaker_turn_count(res);
    for (size_t i = 0; i < n; ++i) {
        int64_t s = 0, e = 0;
        const char *label = nullptr;
        if (audiocpp_result_speaker_turn(res, i, &s, &e, &label, nullptr, nullptr) != AUDIOCPP_OK) continue;
        const std::string id = label ? label : "";
        size_t idx = 0;
        while (idx < ids.size() && ids[idx] != id) ++idx;
        if (idx == ids.size()) ids.push_back(id);
        const double start = static_cast<double>(s) / kModelRate, end = static_cast<double>(e) / kModelRate;
        json += (i ? "," : "") + std::string("{\"start\":") + fmt_seconds(start) + ",\"end\":" + fmt_seconds(end) +
                ",\"speaker\":" + std::to_string(idx) + "}";
        inst.tensor.push_back(static_cast<float>(start));
        inst.tensor.push_back(static_cast<float>(end));
        inst.tensor.push_back(static_cast<float>(idx));
    }
    audiocpp_result_free(res);
    inst.text = json + "]";
    set_segments_response(inst, resp, 3);
    return AGENT_OK;
}

std::string unique_wav_path(std::string_view dir) {
    static std::atomic<uint64_t> counter{0};
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string base(dir);
    if (!base.empty() && base.back() != '/' && base.back() != '\\') base += '/';
    for (;;) {
        std::string path = base + "audiocpp-tts-" + std::to_string(ms) + "-" + std::to_string(counter.fetch_add(1)) + ".wav";
        if (!file_exists(path)) return path;
    }
}

bool write_wav16(const std::string &path, const float *samples, size_t frames, int rate, int channels) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t data_bytes = static_cast<uint32_t>(frames * channels * 2);
    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
    auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };
    f.write("RIFF", 4);
    w32(36 + data_bytes);
    f.write("WAVEfmt ", 8);
    w32(16);
    w16(1);
    w16(static_cast<uint16_t>(channels));
    w32(static_cast<uint32_t>(rate));
    w32(static_cast<uint32_t>(rate * channels * 2));
    w16(static_cast<uint16_t>(channels * 2));
    w16(16);
    f.write("data", 4);
    w32(data_bytes);
    std::vector<int16_t> pcm(frames * channels);
    for (size_t i = 0; i < pcm.size(); ++i) {
        float v = samples[i] < -1.0f ? -1.0f : (samples[i] > 1.0f ? 1.0f : samples[i]);
        pcm[i] = static_cast<int16_t>(std::lrintf(v * 32767.0f));
    }
    f.write(reinterpret_cast<const char *>(pcm.data()), static_cast<std::streamsize>(pcm.size() * 2));
    return static_cast<bool>(f);
}

int32_t synthesize(Instance &inst, const agent_request *req, std::string_view text, agent_response *resp) {
    if (text.empty()) return inst.fail(AGENT_ERR_INVALID_CONFIG, "audiocpp: TTS needs text in the last user message");
    if (sv(req->output_dir).empty()) return inst.fail(AGENT_ERR_FILESYSTEM, "audiocpp: TTS needs request output_dir");
    if (cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "audiocpp: cancelled");
    audiocpp_request *r = audiocpp_request_create();
    if (!r) return inst.fail(AGENT_ERR_INIT_FAILED, "audiocpp: out of memory");
    const std::string input(text);
    Component &c = inst.tts;
    int32_t rc = AGENT_OK;
    if (audiocpp_request_set_text(r, input.c_str(), c.language.empty() ? nullptr : c.language.c_str()) != AUDIOCPP_OK)
        rc = inst.fail_audiocpp("set_text");
    if (rc == AGENT_OK && !c.voice.empty() && audiocpp_request_set_voice_id(r, c.voice.c_str()) != AUDIOCPP_OK)
        rc = inst.fail_audiocpp("set_voice");
    if (rc == AGENT_OK && c.speed > 0) audiocpp_request_set_speaking_rate(r, static_cast<float>(c.speed));
    if (rc == AGENT_OK && req->on_progress) req->on_progress(req->user_data, 0, 1);
    audiocpp_result *res = nullptr;
    if (rc == AGENT_OK && audiocpp_session_run(c.session, r, &res) != AUDIOCPP_OK) rc = inst.fail_audiocpp("speech synthesis");
    audiocpp_request_free(r);
    if (rc != AGENT_OK) return rc;

    const float *samples = nullptr;
    size_t frames = 0;
    int rate = 0, channels = 1;
    if (audiocpp_result_audio(res, &samples, &frames, &rate, &channels) != AUDIOCPP_OK || !samples || frames == 0) {
        audiocpp_result_free(res);
        return inst.fail(AGENT_ERR_DECODE_FAILED, "audiocpp: speech synthesis produced no audio");
    }
    inst.path = unique_wav_path(sv(req->output_dir));
    const bool written = write_wav16(inst.path, samples, frames, rate, channels);
    const double seconds = static_cast<double>(frames) / rate;
    audiocpp_result_free(res);
    if (!written) return inst.fail(AGENT_ERR_FILESYSTEM, "audiocpp: failed to write " + inst.path);
    if (req->on_progress) req->on_progress(req->user_data, 1, 1);

    char summary[128];
    std::snprintf(summary, sizeof(summary), "Generated %.2f s of speech (%d Hz).", seconds, rate);
    inst.text = summary;
    set_text_response(inst, resp);
    inst.parts[1] = agent_part{};
    inst.parts[1].kind = AGENT_PART_FILE;
    inst.parts[1].path = to_c(inst.path);
    inst.parts[1].mime = agent_str{"audio/wav", 9};
    resp->message.n_parts = 2;
    return AGENT_OK;
}

// ------------------------------------------------------------------ ABI functions

int32_t audiocpp_engine_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    *out_instance = nullptr;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object cfg;
    if (parser.parse(config_json.data, config_json.size).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "audiocpp: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    simdjson::dom::object options, tts, asr, vad, dia;
    bool has_tts = false, has_asr = false, has_vad = false, has_dia = false;
    int threads = 4;
    std::string backend = "cpu";
    if (!cfg["options"].get_object().get(options)) {
        has_tts = !options["tts"].get_object().get(tts);
        has_asr = !options["asr"].get_object().get(asr);
        has_vad = !options["vad"].get_object().get(vad);
        has_dia = !options["diarize"].get_object().get(dia);
        const double t = json_num(options, "threads", 0);
        if (t >= 1 && t <= 256) threads = static_cast<int>(t);
        std::string b = json_str(options, "backend");
        if (!b.empty()) backend = std::move(b);
    }
    if (!has_tts && !has_asr && !has_vad && !has_dia) {
        std::snprintf(err, err_cap, "audiocpp: configure at least one of options.tts, options.asr, options.vad, options.diarize");
        return AGENT_ERR_INVALID_CONFIG;
    }

    // audio.cpp unpacks files embedded in a GGUF and the eSpeak data into the temp directory: point it at the engine's scratch space
    // (the only writable place of a sandboxed engine besides artifacts).
    {
        const std::string scratch = json_str(cfg, "scratch_dir");
        if (!scratch.empty()) {
            set_env("TMPDIR", scratch);
            set_env("TEMP", scratch);
            set_env("TMP", scratch);
            set_env("XDG_CACHE_HOME", scratch);
        }
    }
    auto *inst = new (std::nothrow) Instance();
    if (!inst) {
        std::snprintf(err, err_cap, "audiocpp: out of memory");
        return AGENT_ERR_INIT_FAILED;
    }
    if (audiocpp_registry_create(nullptr, &inst->registry) != AUDIOCPP_OK) {
        std::snprintf(err, err_cap, "audiocpp: registry: %s", audiocpp_last_error());
        delete inst;
        return AGENT_ERR_INIT_FAILED;
    }
    audiocpp_backend_config bc{backend.c_str(), 0, threads};
    int32_t rc = AGENT_OK;
    if (has_tts) rc = create_component(*inst, inst->tts, tts, "tts", "tts", bc, err, err_cap);
    if (rc == AGENT_OK && has_asr) rc = create_component(*inst, inst->asr, asr, "asr", "asr", bc, err, err_cap);
    if (rc == AGENT_OK && has_vad) rc = create_component(*inst, inst->vad, vad, "vad", "vad", bc, err, err_cap);
    if (rc == AGENT_OK && has_vad) {
        bool streaming = false;
        if (!vad["streaming"].get_bool().get(streaming)) inst->vad_streaming = streaming;
    }
    if (rc == AGENT_OK && has_dia) rc = create_component(*inst, inst->diar, dia, "diarize", "diar", bc, err, err_cap);
    if (rc != AGENT_OK) {
        delete inst;
        return rc;
    }
    *out_instance = inst;
    return AGENT_OK;
}

void audiocpp_engine_destroy(void *instance) { delete static_cast<Instance *>(instance); }

int32_t audiocpp_engine_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    agent_response_reset(resp);
    inst->error.clear();
    if (cancelled(req)) return inst->fail(AGENT_ERR_CANCELLED, "audiocpp: cancelled");

    const UserInput in = last_user_input(req);
    uint32_t task = req->task;
    const bool auto_task = (req->flags & AGENT_REQUEST_FLAG_AUTO_TASK) != 0;
    switch (task) {
    case AGENT_CAP_TTS: break;
    case AGENT_CAP_VAD:
    case AGENT_CAP_DIARIZE:
        if (!in.audio)
            return inst->fail(AGENT_ERR_INVALID_CONFIG, "audiocpp: VAD / diarization needs an audio part in the last user message");
        break;
    case AGENT_CAP_ASR:
        if (!auto_task) {
            if (!in.audio)
                return inst->fail(AGENT_ERR_INVALID_CONFIG, "audiocpp: an ASR request needs an audio part on the last user message");
            break;
        }
        [[fallthrough]];
    case AGENT_CAP_TEXT_GEN: // routed by content
        if (!in.audio) task = AGENT_CAP_TTS;
        else if (inst->asr) task = AGENT_CAP_ASR;
        else if (inst->vad) task = AGENT_CAP_VAD;
        else if (inst->diar) task = AGENT_CAP_DIARIZE;
        else task = AGENT_CAP_ASR;
        break;
    default: return inst->fail(AGENT_ERR_UNSUPPORTED, "audiocpp: only TTS, ASR, VAD and DIARIZE tasks are supported");
    }

    int32_t rc = AGENT_OK;
    if (task == AGENT_CAP_ASR) {
        if (!inst->asr) return inst->fail(AGENT_ERR_UNSUPPORTED, "audiocpp: ASR requested but options.asr was not configured");
        rc = transcribe(*inst, req, *in.audio, resp);
    } else if (task == AGENT_CAP_VAD) {
        if (!inst->vad) return inst->fail(AGENT_ERR_UNSUPPORTED, "audiocpp: VAD requested but options.vad was not configured");
        rc = detect_speech(*inst, req, *in.audio, in.text, resp);
    } else if (task == AGENT_CAP_DIARIZE) {
        if (!inst->diar) return inst->fail(AGENT_ERR_UNSUPPORTED, "audiocpp: diarization requested but options.diarize was not configured");
        rc = diarize(*inst, req, *in.audio, resp);
    } else {
        if (!inst->tts) return inst->fail(AGENT_ERR_UNSUPPORTED, "audiocpp: TTS requested but options.tts was not configured");
        rc = synthesize(*inst, req, in.text, resp);
    }
    if (rc != AGENT_OK) {
        agent_response_reset(resp);
        return rc;
    }
    resp->message.role = agent_str{"assistant", 9};
    return AGENT_OK;
}

void audiocpp_engine_free_response(void *, agent_response *resp) { agent_response_reset(resp); }

const char *audiocpp_engine_last_error(void *instance) { return static_cast<Instance *>(instance)->error.c_str(); }

const agent_engine_v1 kEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "audiocpp",
    "0.1",
    AGENT_CAP_ASR | AGENT_CAP_TTS | AGENT_CAP_VAD | AGENT_CAP_DIARIZE,
    0,
    audiocpp_engine_create,
    audiocpp_engine_destroy,
    audiocpp_engine_run,
    audiocpp_engine_free_response,
    audiocpp_engine_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kEngine; }
