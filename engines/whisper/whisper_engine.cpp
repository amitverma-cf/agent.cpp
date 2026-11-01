// agent-engine-whisper: whisper.cpp speech-to-text (and Silero VAD) as an agent.cpp engine plugin.
//
// Wraps whisper.cpp's C API (whisper.h), which reports failures through return codes and NULL
// pointers, so no exception boundary is needed here.
//
//   create(): config.model             -> whisper_context (ggml .bin)   (AGENT_CAP_ASR)
//             config.options.vad.model -> whisper_vad_context (Silero)  (AGENT_CAP_VAD)
//
// Routing: task ASR transcribes the last user audio part; task VAD returns its speech segments.
// When the host picked the task itself (AGENT_REQUEST_FLAG_AUTO_TASK; it sends ASR, the lowest
// capability bit) or sends TEXT_GEN, an audio part is transcribed. Anything else is UNSUPPORTED.
//
// Audio input is a FILE or BYTES WAV (any rate, resampled to 16 kHz) or an F32 TENSOR of mono
// samples at 16 kHz.

#include <agent-cpp/engine_abi.h>

#include "../common/speech_audio.hpp"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <vector>

#include <whisper.h>

namespace {

using agent_speech::Audio;
using agent_speech::sv;
using agent_speech::to_c;

constexpr int32_t kRate = WHISPER_SAMPLE_RATE; // 16 kHz

// whisper.cpp's logger is process-wide. Errors always reach stderr; everything else only when some
// instance asked for "verbose".
std::atomic<int> g_verbose{0};
void log_cb(enum ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || g_verbose.load() > 0) std::fputs(text, stderr);
}

struct WhisperInstance {
    whisper_context *ctx = nullptr;
    whisper_vad_context *vad = nullptr;
    bool verbose = false;

    // Defaults from options; params_json can override language / translate / beam_size per call.
    std::string language; // "auto" or a code like "en"
    bool translate = false;
    int32_t threads = 4;
    int32_t beam_size = 1; // 1 = greedy
    float temperature = 0.0f;
    std::string initial_prompt;
    bool vad_filter = false; // run whisper's VAD before transcription (needs vad.model)
    std::string vad_model_path;
    whisper_vad_params vad_params{};

    // Response storage: valid until free_response() or the next run().
    std::string text;
    agent_part parts[2]{};
    std::vector<float> tensor;
    int64_t tensor_shape[2]{};
    std::string error;

    ~WhisperInstance() {
        if (ctx) whisper_free(ctx);
        if (vad) whisper_vad_free(vad);
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

double json_num(simdjson::dom::object obj, const char *key, double fallback) {
    double v = 0;
    if (obj[key].get_double().get(v)) {
        int64_t i = 0;
        if (obj[key].get_int64().get(i)) return fallback;
        return static_cast<double>(i);
    }
    return v;
}

bool json_bool(simdjson::dom::object obj, const char *key, bool fallback) {
    bool v = false;
    if (obj[key].get_bool().get(v)) return fallback;
    return v;
}

bool cancelled(const agent_request *req) { return req->cancel && *req->cancel != 0; }

// ------------------------------------------------------------------ create

int32_t engine_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    *out_instance = nullptr;
    static std::once_flag log_once;
    std::call_once(log_once, [] { whisper_log_set(log_cb, nullptr); });

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object cfg;
    if (parser.parse(config_json.data, config_json.size).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "whisper: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    const std::string model = json_str(cfg, "model");
    if (model.empty()) {
        std::snprintf(err, err_cap, "whisper: config.model (a ggml .bin whisper model) is required");
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (!agent_speech::file_exists(model)) {
        std::snprintf(err, err_cap, "whisper: model not found: %s", model.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }

    auto *inst = new (std::nothrow) WhisperInstance();
    if (!inst) {
        std::snprintf(err, err_cap, "whisper: out of memory");
        return AGENT_ERR_INIT_FAILED;
    }
    whisper_context_params cparams = whisper_context_default_params();
    simdjson::dom::object options, vad;
    bool has_vad = false;
    if (!cfg["options"].get_object().get(options)) {
        inst->language = json_str(options, "language");
        inst->translate = json_bool(options, "translate", false);
        const double t = json_num(options, "threads", 0);
        if (t >= 1 && t <= 256) inst->threads = static_cast<int32_t>(t);
        const double b = json_num(options, "beam_size", 1);
        if (b >= 1 && b <= 16) inst->beam_size = static_cast<int32_t>(b);
        inst->temperature = static_cast<float>(json_num(options, "temperature", 0.0));
        inst->initial_prompt = json_str(options, "initial_prompt");
        inst->verbose = json_bool(options, "verbose", false);
        cparams.use_gpu = json_bool(options, "gpu", cparams.use_gpu);
        cparams.flash_attn = json_bool(options, "flash_attn", cparams.flash_attn);
        has_vad = !options["vad"].get_object().get(vad);
    }
    if (inst->verbose) g_verbose.fetch_add(1);

    inst->ctx = whisper_init_from_file_with_params(model.c_str(), cparams);
    if (!inst->ctx) {
        std::snprintf(err, err_cap, "whisper: failed to load model %s", model.c_str());
        delete inst;
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    // English-only models skip the language-detection pass; multilingual ones auto-detect by default.
    if (inst->language.empty()) inst->language = whisper_is_multilingual(inst->ctx) ? "auto" : "en";
    if (inst->language != "auto" && whisper_lang_id(inst->language.c_str()) < 0) {
        std::snprintf(err, err_cap, "whisper: unknown language \"%s\"", inst->language.c_str());
        delete inst;
        return AGENT_ERR_INVALID_CONFIG;
    }

    inst->vad_params = whisper_vad_default_params();
    if (has_vad) {
        inst->vad_model_path = json_str(vad, "model");
        if (inst->vad_model_path.empty()) {
            std::snprintf(err, err_cap, "whisper: options.vad.model (a ggml Silero VAD model) is required");
            delete inst;
            return AGENT_ERR_INVALID_CONFIG;
        }
        if (!agent_speech::file_exists(inst->vad_model_path)) {
            std::snprintf(err, err_cap, "whisper: options.vad.model not found: %s", inst->vad_model_path.c_str());
            delete inst;
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
        whisper_vad_params &vp = inst->vad_params;
        vp.threshold = static_cast<float>(json_num(vad, "threshold", vp.threshold));
        vp.min_speech_duration_ms = static_cast<int>(json_num(vad, "min_speech_ms", vp.min_speech_duration_ms));
        vp.min_silence_duration_ms = static_cast<int>(json_num(vad, "min_silence_ms", vp.min_silence_duration_ms));
        vp.max_speech_duration_s = static_cast<float>(json_num(vad, "max_speech_s", vp.max_speech_duration_s));
        vp.speech_pad_ms = static_cast<int>(json_num(vad, "speech_pad_ms", vp.speech_pad_ms));
        inst->vad_filter = json_bool(vad, "filter", false);

        whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = inst->threads;
        vcp.use_gpu = false; // Silero is tiny; the CPU is faster than a device round trip
        inst->vad = whisper_vad_init_from_file_with_params(inst->vad_model_path.c_str(), vcp);
        if (!inst->vad) {
            std::snprintf(err, err_cap, "whisper: failed to load VAD model %s", inst->vad_model_path.c_str());
            delete inst;
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
    }
    *out_instance = inst;
    return AGENT_OK;
}

void engine_destroy(void *instance) { delete static_cast<WhisperInstance *>(instance); }

// ------------------------------------------------------------------ ASR

struct RunCtx {
    WhisperInstance *inst;
    const agent_request *req;
    size_t streamed = 0; // segments forwarded through on_chunk
};

bool abort_cb(void *ud) { return cancelled(static_cast<RunCtx *>(ud)->req); }

bool encoder_begin_cb(whisper_context *, whisper_state *, void *ud) { return !cancelled(static_cast<RunCtx *>(ud)->req); }

void progress_cb(whisper_context *, whisper_state *, int progress, void *ud) {
    const agent_request *req = static_cast<RunCtx *>(ud)->req;
    if (req->on_progress) req->on_progress(req->user_data, static_cast<uint64_t>(progress < 0 ? 0 : progress), 100);
}

// Each finished segment goes out as one TEXT chunk; the chunks concatenate to the final transcript.
void new_segment_cb(whisper_context *ctx, whisper_state *, int n_new, void *ud) {
    auto *rc = static_cast<RunCtx *>(ud);
    const agent_request *req = rc->req;
    if (!req->on_chunk) return;
    const int n = whisper_full_n_segments(ctx);
    for (int i = n - n_new; i < n; ++i) {
        const char *t = whisper_full_get_segment_text(ctx, i);
        std::string_view seg = agent_speech::trim(t ? std::string_view(t) : std::string_view());
        if (seg.empty()) continue;
        std::string chunk = rc->streamed++ ? " " : "";
        chunk += seg;
        req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, chunk.data(), chunk.size());
    }
}

int32_t transcribe(WhisperInstance &inst, const agent_request *req, const agent_part &part, agent_response *resp) {
    Audio audio;
    if (std::string e = agent_speech::load_audio_part(part, kRate, kRate, audio); !e.empty())
        return inst.fail(AGENT_ERR_UNSUPPORTED, "whisper: could not read audio: " + e);
    if (cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "whisper: cancelled");

    // Per-request overrides.
    std::string language = inst.language;
    bool translate = inst.translate;
    int32_t beam_size = inst.beam_size;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object params;
    if (req->params_json.size && !parser.parse(req->params_json.data, req->params_json.size).get(doc) && !doc.get_object().get(params)) {
        std::string l = json_str(params, "language");
        if (!l.empty()) {
            if (l != "auto" && whisper_lang_id(l.c_str()) < 0) return inst.fail(AGENT_ERR_INVALID_CONFIG, "whisper: unknown language \"" + l + "\"");
            language = std::move(l);
        }
        translate = json_bool(params, "translate", translate);
        const double b = json_num(params, "beam_size", beam_size);
        if (b >= 1 && b <= 16) beam_size = static_cast<int32_t>(b);
    }

    whisper_full_params wp = whisper_full_default_params(beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    wp.n_threads = inst.threads;
    wp.language = language.c_str();
    wp.translate = translate;
    wp.beam_search.beam_size = beam_size;
    wp.temperature = inst.temperature;
    wp.initial_prompt = inst.initial_prompt.empty() ? nullptr : inst.initial_prompt.c_str();
    wp.print_progress = false;
    wp.print_realtime = false;
    wp.print_special = false;
    wp.print_timestamps = false;
    if (inst.vad_filter) {
        wp.vad = true;
        wp.vad_model_path = inst.vad_model_path.c_str();
        wp.vad_params = inst.vad_params;
    }
    RunCtx rc{&inst, req};
    wp.abort_callback = abort_cb;
    wp.abort_callback_user_data = &rc;
    wp.encoder_begin_callback = encoder_begin_cb;
    wp.encoder_begin_callback_user_data = &rc;
    wp.progress_callback = progress_cb;
    wp.progress_callback_user_data = &rc;
    wp.new_segment_callback = new_segment_cb;
    wp.new_segment_callback_user_data = &rc;

    if (req->on_progress) req->on_progress(req->user_data, 0, 100);
    const int r = whisper_full(inst.ctx, wp, audio.samples.data(), static_cast<int>(audio.samples.size()));
    if (cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "whisper: cancelled");
    if (r != 0) return inst.fail(AGENT_ERR_DECODE_FAILED, "whisper: transcription failed (whisper_full returned " + std::to_string(r) + ")");
    if (req->on_progress) req->on_progress(req->user_data, 100, 100);

    inst.text.clear();
    const int n = whisper_full_n_segments(inst.ctx);
    for (int i = 0; i < n; ++i) {
        const char *t = whisper_full_get_segment_text(inst.ctx, i);
        std::string_view seg = agent_speech::trim(t ? std::string_view(t) : std::string_view());
        if (seg.empty()) continue;
        if (!inst.text.empty()) inst.text += ' ';
        inst.text += seg;
    }
    inst.parts[0] = agent_part{};
    inst.parts[0].kind = AGENT_PART_TEXT;
    inst.parts[0].text = to_c(inst.text);
    resp->message.parts = inst.parts;
    resp->message.n_parts = 1;
    return AGENT_OK;
}

// ------------------------------------------------------------------ VAD

std::string fmt_seconds(double s) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", s);
    return buf;
}

int32_t detect_speech(WhisperInstance &inst, const agent_request *req, const agent_part &part, agent_response *resp) {
    Audio audio;
    if (std::string e = agent_speech::load_audio_part(part, kRate, kRate, audio); !e.empty())
        return inst.fail(AGENT_ERR_UNSUPPORTED, "whisper: could not read audio: " + e);
    if (cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "whisper: cancelled");
    if (req->on_progress) req->on_progress(req->user_data, 0, 1);
    whisper_vad_segments *segs =
        whisper_vad_segments_from_samples(inst.vad, inst.vad_params, audio.samples.data(), static_cast<int>(audio.samples.size()));
    if (!segs) return inst.fail(AGENT_ERR_DECODE_FAILED, "whisper: voice activity detection failed");
    const int n = whisper_vad_segments_n_segments(segs);
    inst.tensor.clear();
    std::string json = "[";
    for (int i = 0; i < n; ++i) {
        // whisper.cpp reports segment times in centiseconds.
        const double start = whisper_vad_segments_get_segment_t0(segs, i) / 100.0;
        const double end = whisper_vad_segments_get_segment_t1(segs, i) / 100.0;
        json += (i ? "," : "") + std::string("{\"start\":") + fmt_seconds(start) + ",\"end\":" + fmt_seconds(end) + "}";
        inst.tensor.push_back(static_cast<float>(start));
        inst.tensor.push_back(static_cast<float>(end));
    }
    whisper_vad_free_segments(segs);
    if (req->on_progress) req->on_progress(req->user_data, 1, 1);
    inst.text = json + "]";

    inst.parts[0] = agent_part{};
    inst.parts[0].kind = AGENT_PART_TEXT;
    inst.parts[0].text = to_c(inst.text);
    inst.tensor_shape[0] = static_cast<int64_t>(inst.tensor.size() / 2);
    inst.tensor_shape[1] = 2;
    inst.parts[1] = agent_part{};
    inst.parts[1].kind = AGENT_PART_TENSOR;
    inst.parts[1].name = agent_str{"segments", 8};
    inst.parts[1].dtype = AGENT_DTYPE_F32;
    inst.parts[1].shape = inst.tensor_shape;
    inst.parts[1].ndim = 2;
    inst.parts[1].data = inst.tensor.data();
    inst.parts[1].size = inst.tensor.size() * sizeof(float);
    resp->message.parts = inst.parts;
    resp->message.n_parts = 2;
    return AGENT_OK;
}

// ------------------------------------------------------------------ ABI functions

int32_t engine_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<WhisperInstance *>(instance);
    agent_response_reset(resp);
    inst->error.clear();
    if (cancelled(req)) return inst->fail(AGENT_ERR_CANCELLED, "whisper: cancelled");

    const agent_speech::UserInput in = agent_speech::last_user_input(req);
    int32_t rc = AGENT_OK;
    switch (req->task) {
    case AGENT_CAP_ASR:
    case AGENT_CAP_TEXT_GEN:
        if (!in.audio) return inst->fail(AGENT_ERR_INVALID_CONFIG, "whisper: ASR needs an audio part on the last user message");
        rc = transcribe(*inst, req, *in.audio, resp);
        break;
    case AGENT_CAP_VAD:
        if (!inst->vad) return inst->fail(AGENT_ERR_UNSUPPORTED, "whisper: VAD requested but options.vad was not configured");
        if (!in.audio) return inst->fail(AGENT_ERR_INVALID_CONFIG, "whisper: VAD needs an audio part on the last user message");
        rc = detect_speech(*inst, req, *in.audio, resp);
        break;
    default: return inst->fail(AGENT_ERR_UNSUPPORTED, "whisper: only ASR and VAD tasks are supported");
    }
    if (rc != AGENT_OK) {
        agent_response_reset(resp);
        return rc;
    }
    resp->message.role = agent_str{"assistant", 9};
    return AGENT_OK;
}

void engine_free_response(void *, agent_response *resp) { agent_response_reset(resp); }

const char *engine_last_error(void *instance) { return static_cast<WhisperInstance *>(instance)->error.c_str(); }

const agent_engine_v1 kWhisperEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "whisper",
    "0.1",
    AGENT_CAP_ASR | AGENT_CAP_VAD,
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

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kWhisperEngine; }
