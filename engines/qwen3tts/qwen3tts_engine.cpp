// agent-engine-qwen3tts: Qwen3-TTS text-to-speech (with voice cloning) as an agent.cpp engine
// plugin, through qwentts.cpp's C API (qwen.h).
//
// qwen.h is a plain C99 API: every qt_* entry catches C++ exceptions internally (std::bad_alloc
// included) and reports failures as a qt_status plus qt_last_error(), so no exception boundary is
// needed here.
//
//   create(): config.model          -> talker GGUF (base / custom_voice / voice_design)
//             config.options.codec  -> 12 Hz audio tokenizer GGUF
//             config.options.voice  -> optional default clone reference (base models)
//   run():    task TTS (or automatic / TEXT_GEN) speaks the last user text. An audio part on that
//             message is a voice-clone reference for this request (base models).
//
// Output follows the sherpa conventions: a TEXT summary plus a FILE part (audio/wav, 24 kHz mono
// 16-bit PCM written to output_dir), audio streamed as AGENT_CHUNK_AUDIO f32 samples.

#include <agent-cpp/engine_abi.h>

#include "../common/speech_audio.hpp"

#include <cctype>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <vector>

#include <qwen.h>

namespace {

using agent_speech::Audio;
using agent_speech::sv;
using agent_speech::to_c;

constexpr int32_t kRate = 24000; // codec output and voice-reference rate
constexpr int32_t kTensorRate = 16000; // F32 tensors follow the agent.cpp audio-input convention

// qwentts' logger is process-wide. Warnings and errors always reach stderr; info only when some
// instance asked for "verbose".
std::atomic<int> g_verbose{0};
void log_cb(enum qt_log_level level, const char *msg, void *) {
    if (level >= QT_LOG_WARN || (level == QT_LOG_INFO && g_verbose.load() > 0)) std::fprintf(stderr, "qwen3tts: %s\n", msg);
}

// Works around a qwentts.cpp bug (at the pinned commit): gf_close() resets its GGUF handle with
// `*gf = {}`, which leaves fd == 0 rather than -1, and the zeroed handles are closed again later
// (qt_free, and qt_init's failure path), so qwentts calls close(0). That closes the process's stdin,
// and the next open() / pipe() in the host (a landlock ruleset, an IPC pipe, ...) silently gets fd
// 0. The guard keeps fd 0 alive across those calls: it saves a duplicate before, and puts it back
// if fd 0 was closed. If fd 0 was not open at all, /dev/null is put there first, so the stray
// close(0) never hits another descriptor. A thread that opens a file in the short window between
// the stray close and the restore could still be affected; qt_init / qt_free are rare.
class StdinGuard {
public:
    StdinGuard() {
        if (::fcntl(0, F_GETFD) == -1) {
            const int fd = ::open("/dev/null", O_RDONLY);
            if (fd > 0) ::close(fd); // fd 0 was taken meanwhile: nothing to protect
        }
        saved_ = ::fcntl(0, F_DUPFD_CLOEXEC, 3);
        if (saved_ >= 0) ::fstat(saved_, &st_);
    }
    ~StdinGuard() {
        if (saved_ < 0) return;
        struct stat now{};
        const bool intact = ::fstat(0, &now) == 0 && now.st_dev == st_.st_dev && now.st_ino == st_.st_ino;
        if (!intact) {
            // dup2 does not set FD_CLOEXEC, like the original stdin.
            ::dup2(saved_, 0);
        }
        ::close(saved_);
    }
    StdinGuard(const StdinGuard &) = delete;
    StdinGuard &operator=(const StdinGuard &) = delete;

private:
    int saved_ = -1;
    struct stat st_{};
};

// Sampling and voice settings: defaults from options, overridable per request in params_json.
struct Gen {
    std::string language = "auto";
    std::string speaker;   // custom_voice models
    std::string instruct;  // voice_design (required) / custom_voice (optional)
    std::string ref_text;  // transcript of a per-request reference clip (in-context cloning)
    int64_t seed = -1;
    int32_t max_new_tokens = 0; // 0 = qwentts default
    float temperature = -1.0f, top_p = -1.0f, repetition_penalty = -1.0f;
    int32_t top_k = -1;
};

struct Qwen3TtsInstance {
    qt_context *ctx = nullptr;
    std::string model_type;
    Gen defaults;
    qt_voice_ref voice{};    // options.voice, precomputed at create()
    std::string voice_text;  // its transcript (in-context cloning when set)
    bool verbose = false;

    // Response storage: valid until free_response() or the next run().
    std::string text;
    std::string path;
    agent_part parts[2]{};
    std::string error;

    ~Qwen3TtsInstance() {
        qt_voice_ref_free(&voice);
        if (ctx) {
            StdinGuard guard;
            qt_free(ctx);
        }
        if (verbose) g_verbose.fetch_sub(1);
    }

    int32_t fail(int32_t code, std::string msg) {
        error = std::move(msg);
        return code;
    }
};

std::string json_str(simdjson::dom::object obj, const char *key) {
    std::string_view v;
    if (obj[key].get_string().get(v)) return {};
    return std::string(v);
}

bool json_num(simdjson::dom::object obj, const char *key, double &out) {
    double v = 0;
    if (!obj[key].get_double().get(v)) {
        out = v;
        return true;
    }
    int64_t i = 0;
    if (!obj[key].get_int64().get(i)) {
        out = static_cast<double>(i);
        return true;
    }
    return false;
}

bool json_bool(simdjson::dom::object obj, const char *key, bool fallback) {
    bool v = false;
    if (obj[key].get_bool().get(v)) return fallback;
    return v;
}

// ISO 639-1 codes are accepted next to qwentts' own lowercase language names.
std::string language_name(std::string lang) {
    for (char &c : lang) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static const char *const kCodes[][2] = {{"en", "english"}, {"zh", "chinese"}, {"ja", "japanese"}, {"ko", "korean"},
                                            {"de", "german"},  {"fr", "french"},  {"ru", "russian"},  {"pt", "portuguese"},
                                            {"es", "spanish"}, {"it", "italian"}};
    for (const auto &c : kCodes)
        if (lang == c[0]) return c[1];
    return lang;
}

void read_gen(simdjson::dom::object obj, Gen &g) {
    std::string s;
    if (!(s = json_str(obj, "language")).empty()) g.language = language_name(s);
    if (!(s = json_str(obj, "speaker")).empty()) g.speaker = s;
    if (!(s = json_str(obj, "instruct")).empty()) g.instruct = s;
    if (!(s = json_str(obj, "ref_text")).empty()) g.ref_text = s;
    double v = 0;
    if (json_num(obj, "seed", v)) g.seed = static_cast<int64_t>(v);
    if (json_num(obj, "max_new_tokens", v) && v >= 1 && v <= 65536) g.max_new_tokens = static_cast<int32_t>(v);
    if (json_num(obj, "temperature", v) && v >= 0) g.temperature = static_cast<float>(v);
    if (json_num(obj, "top_k", v)) g.top_k = static_cast<int32_t>(v);
    if (json_num(obj, "top_p", v) && v > 0) g.top_p = static_cast<float>(v);
    if (json_num(obj, "repetition_penalty", v) && v > 0) g.repetition_penalty = static_cast<float>(v);
}

bool cancelled(const agent_request *req) { return req->cancel && *req->cancel != 0; }

// ------------------------------------------------------------------ create

int32_t engine_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    *out_instance = nullptr;
    static std::once_flag log_once;
    std::call_once(log_once, [] { qt_log_set(log_cb, nullptr); });

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object cfg;
    if (parser.parse(config_json.data, config_json.size).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "qwen3tts: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    const std::string model = json_str(cfg, "model");
    simdjson::dom::object options;
    const bool has_options = !cfg["options"].get_object().get(options);
    const std::string codec = has_options ? json_str(options, "codec") : std::string();
    if (model.empty() || codec.empty()) {
        std::snprintf(err, err_cap, "qwen3tts: config.model (talker GGUF) and options.codec (12 Hz tokenizer GGUF) are required");
        return AGENT_ERR_INVALID_CONFIG;
    }
    for (const std::string *p : {&model, &codec})
        if (!agent_speech::file_exists(*p)) {
            std::snprintf(err, err_cap, "qwen3tts: model file not found: %s", p->c_str());
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }

    auto *inst = new (std::nothrow) Qwen3TtsInstance();
    if (!inst) {
        std::snprintf(err, err_cap, "qwen3tts: out of memory");
        return AGENT_ERR_INIT_FAILED;
    }
    read_gen(options, inst->defaults);
    inst->defaults.ref_text.clear(); // only meaningful next to a per-request reference clip
    inst->verbose = json_bool(options, "verbose", false);
    if (inst->verbose) g_verbose.fetch_add(1);

    const std::string device = json_str(options, "device");
    qt_init_params ip;
    qt_init_default_params(&ip);
    ip.talker_path = model.c_str();
    ip.codec_path = codec.c_str();
    ip.device = device.empty() ? nullptr : device.c_str();
    ip.use_fa = json_bool(options, "flash_attn", ip.use_fa);
    {
        StdinGuard guard;
        inst->ctx = qt_init(&ip);
    }
    if (!inst->ctx) {
        std::snprintf(err, err_cap, "qwen3tts: failed to load %s + %s: %s", model.c_str(), codec.c_str(), qt_last_error());
        delete inst;
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    const char *mt = qt_model_type(inst->ctx);
    inst->model_type = mt ? mt : "";
    if (inst->model_type == "custom_voice" && inst->defaults.speaker.empty() && qt_n_speakers(inst->ctx) > 0) {
        const char *first = qt_speaker_name(inst->ctx, 0);
        if (first) inst->defaults.speaker = first;
    }

    // options.voice: {"wav": reference clip, "text": its transcript (optional)} for base models.
    simdjson::dom::object voice;
    if (!options["voice"].get_object().get(voice)) {
        const std::string wav = json_str(voice, "wav");
        Audio ref;
        std::string e = wav.empty() ? "options.voice.wav is required" : agent_speech::read_wav(wav, ref);
        if (e.empty()) agent_speech::resample(ref, kRate);
        if (!e.empty() || ref.samples.empty()) {
            std::snprintf(err, err_cap, "qwen3tts: options.voice: %s", e.empty() ? "empty reference clip" : e.c_str());
            delete inst;
            return wav.empty() ? AGENT_ERR_INVALID_CONFIG : AGENT_ERR_MODEL_LOAD_FAILED;
        }
        if (qt_extract_voice_ref(inst->ctx, ref.samples.data(), static_cast<int>(ref.samples.size()), &inst->voice) != QT_STATUS_OK) {
            std::snprintf(err, err_cap, "qwen3tts: voice reference failed (base models only): %s", qt_last_error());
            delete inst;
            return AGENT_ERR_INVALID_CONFIG;
        }
        inst->voice_text = json_str(voice, "text");
    }
    *out_instance = inst;
    return AGENT_OK;
}

void engine_destroy(void *instance) { delete static_cast<Qwen3TtsInstance *>(instance); }

// ------------------------------------------------------------------ TTS

struct RunCtx {
    const agent_request *req;
    std::vector<float> samples;
    uint64_t frames_total;
};

bool cancel_cb(void *ud) { return cancelled(static_cast<RunCtx *>(ud)->req); }

// Runs on qwentts' compute worker thread while run() blocks in qt_synthesize, so the host's
// callbacks are still invoked one at a time.
bool chunk_cb(const float *samples, int n, void *ud) {
    auto *rc = static_cast<RunCtx *>(ud);
    const agent_request *req = rc->req;
    if (samples && n > 0) {
        rc->samples.insert(rc->samples.end(), samples, samples + n);
        if (req->on_chunk) req->on_chunk(req->user_data, AGENT_CHUNK_AUDIO, samples, static_cast<size_t>(n) * sizeof(float));
    }
    if (req->on_progress) {
        const uint64_t frames = rc->samples.size() / 1920; // 12.5 Hz codec frames at 24 kHz
        req->on_progress(req->user_data, frames < rc->frames_total ? frames : rc->frames_total, rc->frames_total);
    }
    return !cancelled(req);
}

int32_t synthesize(Qwen3TtsInstance &inst, const agent_request *req, std::string_view text, const agent_part *ref_part,
                   agent_response *resp) {
    if (text.empty()) return inst.fail(AGENT_ERR_INVALID_CONFIG, "qwen3tts: TTS needs text in the last user message");
    if (sv(req->output_dir).empty()) return inst.fail(AGENT_ERR_FILESYSTEM, "qwen3tts: TTS needs request output_dir");

    Gen g = inst.defaults;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object params;
    if (req->params_json.size && !parser.parse(req->params_json.data, req->params_json.size).get(doc) && !doc.get_object().get(params))
        read_gen(params, g);

    Audio ref;
    if (ref_part) {
        if (std::string e = agent_speech::load_audio_part(*ref_part, kTensorRate, kRate, ref); !e.empty())
            return inst.fail(AGENT_ERR_UNSUPPORTED, "qwen3tts: could not read the voice reference: " + e);
    }

    const std::string input(text);
    qt_tts_params p;
    qt_tts_default_params(&p);
    p.text = input.c_str();
    p.lang = g.language.empty() ? nullptr : g.language.c_str();
    p.speaker = inst.model_type == "custom_voice" && !g.speaker.empty() ? g.speaker.c_str() : nullptr;
    p.instruct = inst.model_type != "base" && !g.instruct.empty() ? g.instruct.c_str() : nullptr;
    p.seed = g.seed;
    if (g.max_new_tokens > 0) p.max_new_tokens = g.max_new_tokens;
    if (g.temperature >= 0) p.temperature = g.temperature;
    if (g.top_k >= 0) p.top_k = g.top_k;
    if (g.top_p > 0) p.top_p = g.top_p;
    if (g.repetition_penalty > 0) p.repetition_penalty = g.repetition_penalty;
    if (ref_part) {
        p.ref_audio_24k = ref.samples.data();
        p.ref_n_samples = static_cast<int>(ref.samples.size());
        p.ref_text = g.ref_text.empty() ? nullptr : g.ref_text.c_str();
    } else if (inst.voice.ref_spk_emb) {
        p.ref_spk_emb = inst.voice.ref_spk_emb;
        p.ref_spk_dim = inst.voice.ref_spk_dim;
        if (!inst.voice_text.empty()) {
            p.ref_codes = inst.voice.ref_codes;
            p.ref_T = inst.voice.ref_T;
            p.ref_text = inst.voice_text.c_str();
        }
    }
    RunCtx rc{req, {}, static_cast<uint64_t>(p.max_new_tokens)};
    p.cancel = cancel_cb;
    p.cancel_user_data = &rc;
    p.on_chunk = chunk_cb; // streaming decode; the chunks are collected for the WAV file
    p.on_chunk_user_data = &rc;

    if (req->on_progress) req->on_progress(req->user_data, 0, rc.frames_total);
    const qt_status st = qt_synthesize(inst.ctx, &p, nullptr);
    if (st == QT_STATUS_CANCELLED || cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "qwen3tts: cancelled");
    if (st != QT_STATUS_OK) {
        const int32_t code = (st == QT_STATUS_INVALID_PARAMS || st == QT_STATUS_MODE_INVALID) ? AGENT_ERR_INVALID_CONFIG : AGENT_ERR_DECODE_FAILED;
        return inst.fail(code, std::string("qwen3tts: synthesis failed: ") + qt_last_error());
    }
    if (rc.samples.empty()) return inst.fail(AGENT_ERR_DECODE_FAILED, "qwen3tts: synthesis produced no audio");
    if (req->on_progress) req->on_progress(req->user_data, rc.frames_total, rc.frames_total);

    inst.path = agent_speech::unique_wav_path(sv(req->output_dir), "qwen3tts");
    if (!agent_speech::write_wav(inst.path, rc.samples.data(), rc.samples.size(), kRate))
        return inst.fail(AGENT_ERR_FILESYSTEM, "qwen3tts: failed to write " + inst.path);

    char summary[128];
    std::snprintf(summary, sizeof(summary), "Generated %.2f s of speech (%d Hz).", static_cast<double>(rc.samples.size()) / kRate, kRate);
    inst.text = summary;
    inst.parts[0] = agent_part{};
    inst.parts[0].kind = AGENT_PART_TEXT;
    inst.parts[0].text = to_c(inst.text);
    inst.parts[1] = agent_part{};
    inst.parts[1].kind = AGENT_PART_FILE;
    inst.parts[1].path = to_c(inst.path);
    inst.parts[1].mime = agent_str{"audio/wav", 9};
    resp->message.parts = inst.parts;
    resp->message.n_parts = 2;
    return AGENT_OK;
}

// ------------------------------------------------------------------ ABI functions

int32_t engine_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Qwen3TtsInstance *>(instance);
    agent_response_reset(resp);
    inst->error.clear();
    if (cancelled(req)) return inst->fail(AGENT_ERR_CANCELLED, "qwen3tts: cancelled");
    if (req->task != AGENT_CAP_TTS && req->task != AGENT_CAP_TEXT_GEN)
        return inst->fail(AGENT_ERR_UNSUPPORTED, "qwen3tts: only the TTS task is supported");
    const agent_speech::UserInput in = agent_speech::last_user_input(req);
    const int32_t rc = synthesize(*inst, req, in.text, in.audio, resp);
    if (rc != AGENT_OK) {
        agent_response_reset(resp);
        return rc;
    }
    resp->message.role = agent_str{"assistant", 9};
    return AGENT_OK;
}

void engine_free_response(void *, agent_response *resp) { agent_response_reset(resp); }

const char *engine_last_error(void *instance) { return static_cast<Qwen3TtsInstance *>(instance)->error.c_str(); }

const agent_engine_v1 kQwen3TtsEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "qwen3tts",
    "0.1",
    AGENT_CAP_TTS,
    0,
    engine_create,
    engine_destroy,
    engine_run,
    engine_free_response,
    engine_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kQwen3TtsEngine; }
