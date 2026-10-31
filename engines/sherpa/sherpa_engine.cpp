// agent-engine-sherpa: sherpa-onnx speech (TTS + ASR) as an agent.cpp engine plugin.
//
// Wraps the sherpa-onnx C API (sherpa-onnx/c-api/c-api.h), which reports failures through NULL
// returns, so no exception boundary is needed here. Components are created only when configured:
//
//   options.tts.kokoro            -> SherpaOnnxOfflineTts      (AGENT_CAP_TTS)
//   options.asr.whisper           -> SherpaOnnxOfflineRecognizer (AGENT_CAP_ASR)
//   options.asr.nemo_transducer / options.asr.transducer
//
// Routing: task TTS synthesizes the last user text. Task ASR (and TEXT_GEN, which older hosts send
// for every request) is routed by content: an audio part on the last user message is transcribed,
// otherwise its text is synthesized. A task whose component was not configured is UNSUPPORTED.

#include <agent-cpp/engine_abi.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <sys/stat.h>

#include <sherpa-onnx/c-api/c-api.h>

namespace {

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }
agent_str to_c(const std::string &s) { return agent_str{s.data(), s.size()}; }

bool file_exists(const std::string &path) {
    struct stat st{};
    return !path.empty() && ::stat(path.c_str(), &st) == 0;
}

struct SherpaInstance {
    const SherpaOnnxOfflineTts *tts = nullptr;
    int32_t tts_sid = 0;
    float tts_speed = 1.0f;
    const SherpaOnnxOfflineRecognizer *asr = nullptr;

    // Response storage: valid until free_response() or the next run().
    std::string text;
    std::string path;
    agent_part parts[2]{};
    std::string error;

    ~SherpaInstance() {
        if (tts) SherpaOnnxDestroyOfflineTts(tts);
        if (asr) SherpaOnnxDestroyOfflineRecognizer(asr);
    }

    int32_t fail(int32_t code, std::string msg) {
        error = std::move(msg);
        return code;
    }
};

// ------------------------------------------------------------------ config parsing

// Reads a string field from a JSON object; empty when absent or not a string.
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

// Strings handed to sherpa's config structs; they only need to outlive Create*().
struct ComponentPaths {
    std::string a, b, c, tokens, lexicon, data_dir, lang, language, task, model_type;
};

// Required path field: must be present and exist on disk.
int32_t require_path(simdjson::dom::object obj, const char *section, const char *key, std::string &out, char *err,
                     size_t err_cap) {
    out = json_str(obj, key);
    if (out.empty()) {
        std::snprintf(err, err_cap, "sherpa: options.%s.%s is required", section, key);
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (!file_exists(out)) {
        std::snprintf(err, err_cap, "sherpa: options.%s.%s not found: %s", section, key, out.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    return AGENT_OK;
}

// Optional path field: may be absent, but must exist when given.
int32_t optional_path(simdjson::dom::object obj, const char *section, const char *key, std::string &out, char *err,
                      size_t err_cap) {
    out = json_str(obj, key);
    if (!out.empty() && !file_exists(out)) {
        std::snprintf(err, err_cap, "sherpa: options.%s.%s not found: %s", section, key, out.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    return AGENT_OK;
}

const char *or_null(const std::string &s) { return s.empty() ? nullptr : s.c_str(); }

int32_t create_tts(SherpaInstance &inst, simdjson::dom::object tts, int32_t threads, const std::string &provider, char *err,
                   size_t err_cap) {
    simdjson::dom::object kokoro;
    if (tts["kokoro"].get_object().get(kokoro)) {
        std::snprintf(err, err_cap, "sherpa: options.tts needs a model family; supported: \"kokoro\"");
        return AGENT_ERR_INVALID_CONFIG;
    }
    ComponentPaths p;
    int32_t rc = AGENT_OK;
    if ((rc = require_path(kokoro, "tts.kokoro", "model", p.a, err, err_cap)) ||
        (rc = require_path(kokoro, "tts.kokoro", "voices", p.b, err, err_cap)) ||
        (rc = require_path(kokoro, "tts.kokoro", "tokens", p.tokens, err, err_cap)) ||
        (rc = optional_path(kokoro, "tts.kokoro", "lexicon", p.lexicon, err, err_cap)) ||
        (rc = optional_path(kokoro, "tts.kokoro", "data_dir", p.data_dir, err, err_cap)))
        return rc;
    p.lang = json_str(kokoro, "lang");

    SherpaOnnxOfflineTtsConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.model.kokoro.model = p.a.c_str();
    cfg.model.kokoro.voices = p.b.c_str();
    cfg.model.kokoro.tokens = p.tokens.c_str();
    cfg.model.kokoro.lexicon = or_null(p.lexicon);
    cfg.model.kokoro.data_dir = or_null(p.data_dir); // espeak-ng-data; ignored by sherpa-onnx >= 2.0
    cfg.model.kokoro.lang = or_null(p.lang);
    cfg.model.kokoro.length_scale = 1.0f;
    cfg.model.num_threads = threads;
    cfg.model.provider = provider.c_str();
    cfg.max_num_sentences = 1; // one callback per sentence -> incremental audio chunks
    cfg.silence_scale = 0.2f;

    inst.tts = SherpaOnnxCreateOfflineTts(&cfg);
    if (!inst.tts) {
        std::snprintf(err, err_cap, "sherpa: failed to create Kokoro TTS from %s", p.a.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    inst.tts_sid = static_cast<int32_t>(json_num(tts, "speaker_id", 0));
    inst.tts_speed = static_cast<float>(json_num(tts, "speed", 1.0));
    if (inst.tts_speed <= 0.0f) inst.tts_speed = 1.0f;
    return AGENT_OK;
}

int32_t create_asr(SherpaInstance &inst, simdjson::dom::object asr, int32_t threads, const std::string &provider, char *err,
                   size_t err_cap) {
    SherpaOnnxOfflineRecognizerConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.feat_config.sample_rate = 16000;
    cfg.feat_config.feature_dim = 80;
    cfg.decoding_method = "greedy_search";
    cfg.model_config.num_threads = threads;
    cfg.model_config.provider = provider.c_str();

    ComponentPaths p;
    int32_t rc = AGENT_OK;
    simdjson::dom::object model;
    const char *family = nullptr;
    bool nemo = false;
    if (!asr["whisper"].get_object().get(model)) {
        family = "asr.whisper";
        if ((rc = require_path(model, family, "encoder", p.a, err, err_cap)) ||
            (rc = require_path(model, family, "decoder", p.b, err, err_cap)) ||
            (rc = require_path(model, family, "tokens", p.tokens, err, err_cap)))
            return rc;
        p.language = json_str(model, "language");
        p.task = json_str(model, "task");
        cfg.model_config.whisper.encoder = p.a.c_str();
        cfg.model_config.whisper.decoder = p.b.c_str();
        cfg.model_config.whisper.language = or_null(p.language);
        cfg.model_config.whisper.task = or_null(p.task);
    } else if ((nemo = !asr["nemo_transducer"].get_object().get(model)) || !asr["transducer"].get_object().get(model)) {
        family = nemo ? "asr.nemo_transducer" : "asr.transducer";
        if ((rc = require_path(model, family, "encoder", p.a, err, err_cap)) ||
            (rc = require_path(model, family, "decoder", p.b, err, err_cap)) ||
            (rc = require_path(model, family, "joiner", p.c, err, err_cap)) ||
            (rc = require_path(model, family, "tokens", p.tokens, err, err_cap)))
            return rc;
        p.model_type = json_str(model, "model_type");
        if (p.model_type.empty() && nemo) p.model_type = "nemo_transducer";
        cfg.model_config.transducer.encoder = p.a.c_str();
        cfg.model_config.transducer.decoder = p.b.c_str();
        cfg.model_config.transducer.joiner = p.c.c_str();
        cfg.model_config.model_type = or_null(p.model_type);
    } else {
        std::snprintf(err, err_cap, "sherpa: options.asr needs a model family; supported: \"whisper\", \"nemo_transducer\", \"transducer\"");
        return AGENT_ERR_INVALID_CONFIG;
    }
    cfg.model_config.tokens = p.tokens.c_str();

    inst.asr = SherpaOnnxCreateOfflineRecognizer(&cfg);
    if (!inst.asr) {
        std::snprintf(err, err_cap, "sherpa: failed to create %s recognizer from %s", family, p.a.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    return AGENT_OK;
}

// ------------------------------------------------------------------ request helpers

struct UserInput {
    std::string_view text;   // last TEXT part of the last user message
    const agent_part *audio; // last audio FILE/BYTES part of the last user message
};

bool is_audio(const agent_part &p) {
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
            else if ((part.kind == AGENT_PART_FILE || part.kind == AGENT_PART_BYTES) && is_audio(part)) in.audio = &part;
        }
        break;
    }
    return in;
}

bool cancelled(const agent_request *req) { return req->cancel && *req->cancel != 0; }

void set_text_response(SherpaInstance &inst, agent_response *resp) {
    inst.parts[0] = agent_part{};
    inst.parts[0].kind = AGENT_PART_TEXT;
    inst.parts[0].text = to_c(inst.text);
    resp->message.parts = inst.parts;
    resp->message.n_parts = 1;
}

// ------------------------------------------------------------------ ASR

int32_t transcribe(SherpaInstance &inst, const agent_request *req, const agent_part &audio, agent_response *resp) {
    const SherpaOnnxWave *wave = nullptr;
    std::string where;
    if (audio.kind == AGENT_PART_FILE) {
        where = std::string(sv(audio.path));
        wave = SherpaOnnxReadWave(where.c_str());
    } else {
        where = "<bytes>";
        if (audio.size > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            return inst.fail(AGENT_ERR_UNSUPPORTED, "sherpa: audio buffer too large");
        wave = SherpaOnnxReadWaveFromBinaryData(static_cast<const char *>(audio.data), static_cast<int32_t>(audio.size));
    }
    if (!wave)
        return inst.fail(AGENT_ERR_UNSUPPORTED, "sherpa: could not read audio " + where + " (expected a 16-bit PCM mono WAV)");
    if (wave->num_samples <= 0 || wave->sample_rate <= 0) {
        SherpaOnnxFreeWave(wave);
        return inst.fail(AGENT_ERR_DECODE_FAILED, "sherpa: audio " + where + " is empty");
    }
    if (cancelled(req)) {
        SherpaOnnxFreeWave(wave);
        return inst.fail(AGENT_ERR_CANCELLED, "sherpa: cancelled");
    }

    const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(inst.asr);
    if (!stream) {
        SherpaOnnxFreeWave(wave);
        return inst.fail(AGENT_ERR_DECODE_FAILED, "sherpa: failed to create recognition stream");
    }
    if (req->on_progress) req->on_progress(req->user_data, 0, 1);
    // sherpa-onnx resamples to the model's 16 kHz feature rate when the WAV rate differs.
    SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples, wave->num_samples);
    SherpaOnnxDecodeOfflineStream(inst.asr, stream);
    const SherpaOnnxOfflineRecognizerResult *result = SherpaOnnxGetOfflineStreamResult(stream);
    if (req->on_progress) req->on_progress(req->user_data, 1, 1);

    int32_t rc = AGENT_OK;
    if (!result) {
        rc = inst.fail(AGENT_ERR_DECODE_FAILED, "sherpa: recognition returned no result");
    } else {
        std::string_view t = result->text ? std::string_view(result->text) : std::string_view();
        while (!t.empty() && (t.front() == ' ' || t.front() == '\n')) t.remove_prefix(1);
        while (!t.empty() && (t.back() == ' ' || t.back() == '\n')) t.remove_suffix(1);
        inst.text.assign(t);
        SherpaOnnxDestroyOfflineRecognizerResult(result);
    }
    SherpaOnnxDestroyOfflineStream(stream);
    SherpaOnnxFreeWave(wave);
    if (rc != AGENT_OK) return rc;

    if (req->on_chunk && !inst.text.empty()) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, inst.text.data(), inst.text.size());
    set_text_response(inst, resp);
    return AGENT_OK;
}

// ------------------------------------------------------------------ TTS

struct TtsCallbackCtx {
    const agent_request *req;
};

// Per sentence: forward the new samples (size in bytes) and the progress; returning 0 stops.
int32_t tts_callback(const float *samples, int32_t n, float progress, void *arg) {
    const agent_request *req = static_cast<TtsCallbackCtx *>(arg)->req;
    if (cancelled(req)) return 0;
    if (req->on_chunk && samples && n > 0)
        req->on_chunk(req->user_data, AGENT_CHUNK_AUDIO, samples, static_cast<size_t>(n) * sizeof(float));
    if (req->on_progress) {
        const double p = progress < 0.0f ? 0.0 : (progress > 1.0f ? 1.0 : static_cast<double>(progress));
        req->on_progress(req->user_data, static_cast<uint64_t>(p * 1000.0), 1000);
    }
    return cancelled(req) ? 0 : 1;
}

// <output_dir>/sherpa-tts-<ms>-<n>.wav, skipping names that already exist.
std::string unique_wav_path(std::string_view dir) {
    static std::atomic<uint64_t> counter{0};
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string base(dir);
    if (!base.empty() && base.back() != '/' && base.back() != '\\') base += '/';
    for (;;) {
        std::string path = base + "sherpa-tts-" + std::to_string(ms) + "-" + std::to_string(counter.fetch_add(1)) + ".wav";
        if (!file_exists(path)) return path;
    }
}

int32_t synthesize(SherpaInstance &inst, const agent_request *req, std::string_view text, agent_response *resp) {
    if (text.empty()) return inst.fail(AGENT_ERR_INVALID_CONFIG, "sherpa: TTS needs text in the last user message");
    if (sv(req->output_dir).empty()) return inst.fail(AGENT_ERR_FILESYSTEM, "sherpa: TTS needs request output_dir");

    const std::string input(text);
    SherpaOnnxGenerationConfig gen;
    std::memset(&gen, 0, sizeof(gen));
    gen.sid = inst.tts_sid;
    gen.speed = inst.tts_speed;
    gen.silence_scale = 0.2f;

    TtsCallbackCtx ctx{req};
    const SherpaOnnxGeneratedAudio *audio = SherpaOnnxOfflineTtsGenerateWithConfig(inst.tts, input.c_str(), &gen, tts_callback, &ctx);
    if (cancelled(req)) {
        if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        return inst.fail(AGENT_ERR_CANCELLED, "sherpa: cancelled");
    }
    if (!audio) return inst.fail(AGENT_ERR_DECODE_FAILED, "sherpa: speech synthesis failed");
    if (audio->n <= 0 || !audio->samples) {
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        return inst.fail(AGENT_ERR_DECODE_FAILED, "sherpa: speech synthesis produced no audio (missing lexicon entries?)");
    }

    inst.path = unique_wav_path(sv(req->output_dir));
    const int32_t n = audio->n;
    const int32_t rate = audio->sample_rate;
    const int32_t written = SherpaOnnxWriteWave(audio->samples, n, rate, inst.path.c_str());
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    if (!written) return inst.fail(AGENT_ERR_FILESYSTEM, "sherpa: failed to write " + inst.path);

    char summary[128];
    std::snprintf(summary, sizeof(summary), "Generated %.2f s of speech (%d Hz).", static_cast<double>(n) / rate, rate);
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

int32_t sherpa_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    *out_instance = nullptr;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object cfg;
    if (parser.parse(config_json.data, config_json.size).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "sherpa: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    simdjson::dom::object options;
    simdjson::dom::object tts, asr;
    bool has_tts = false, has_asr = false;
    int32_t threads = 2;
    std::string provider = "cpu";
    if (!cfg["options"].get_object().get(options)) {
        has_tts = !options["tts"].get_object().get(tts);
        has_asr = !options["asr"].get_object().get(asr);
        const double t = json_num(options, "threads", 0);
        if (t >= 1 && t <= 256) threads = static_cast<int32_t>(t);
        std::string prov = json_str(options, "provider");
        if (!prov.empty()) provider = std::move(prov);
    }

    auto *inst = new (std::nothrow) SherpaInstance();
    if (!inst) {
        std::snprintf(err, err_cap, "sherpa: out of memory");
        return AGENT_ERR_INIT_FAILED;
    }
    int32_t rc = AGENT_OK;
    if (has_tts) rc = create_tts(*inst, tts, threads, provider, err, err_cap);
    if (rc == AGENT_OK && has_asr) rc = create_asr(*inst, asr, threads, provider, err, err_cap);
    if (rc != AGENT_OK) {
        delete inst;
        return rc;
    }
    *out_instance = inst;
    return AGENT_OK;
}

void sherpa_destroy(void *instance) { delete static_cast<SherpaInstance *>(instance); }

int32_t sherpa_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<SherpaInstance *>(instance);
    *resp = agent_response{};
    inst->error.clear();
    if (cancelled(req)) return inst->fail(AGENT_ERR_CANCELLED, "sherpa: cancelled");

    const UserInput in = last_user_input(req);
    bool want_asr = false;
    switch (req->task) {
    case AGENT_CAP_TTS: break;
    case AGENT_CAP_ASR:
    case AGENT_CAP_TEXT_GEN: want_asr = in.audio != nullptr; break; // routed by content
    default: return inst->fail(AGENT_ERR_UNSUPPORTED, "sherpa: only TTS and ASR tasks are supported");
    }

    int32_t rc = AGENT_OK;
    if (want_asr) {
        if (!inst->asr) return inst->fail(AGENT_ERR_UNSUPPORTED, "sherpa: ASR requested but options.asr was not configured");
        rc = transcribe(*inst, req, *in.audio, resp);
    } else {
        if (!inst->tts) return inst->fail(AGENT_ERR_UNSUPPORTED, "sherpa: TTS requested but options.tts was not configured");
        rc = synthesize(*inst, req, in.text, resp);
    }
    if (rc != AGENT_OK) {
        *resp = agent_response{};
        return rc;
    }
    resp->message.role = agent_str{"assistant", 9};
    return AGENT_OK;
}

void sherpa_free_response(void *, agent_response *resp) { *resp = agent_response{}; }

const char *sherpa_last_error(void *instance) { return static_cast<SherpaInstance *>(instance)->error.c_str(); }

const agent_engine_v1 kSherpaEngine = {
    AGENT_ENGINE_ABI_VERSION,
    "sherpa",
    "0.1",
    AGENT_CAP_ASR | AGENT_CAP_TTS,
    0,
    sherpa_create,
    sherpa_destroy,
    sherpa_run,
    sherpa_free_response,
    sherpa_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kSherpaEngine; }
