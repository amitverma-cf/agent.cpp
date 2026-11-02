// agent-engine-parakeet: parakeet.cpp (NVIDIA Parakeet / Nemotron ASR on ggml) as an agent.cpp
// engine plugin.
//
// Wraps parakeet.cpp's C API (parakeet_capi.h). Every parakeet_capi_* entry catches C++ exceptions
// internally and reports failures through NULL returns plus parakeet_capi_last_error /
// parakeet_capi_load_error, so no exception boundary is needed here.
//
//   create(): config.model -> parakeet_ctx (a parakeet.cpp GGUF, or a bundle's ASR component)
//   run():    task ASR (or automatic / TEXT_GEN) transcribes the last user audio part.
//
// Audio input is a FILE or BYTES WAV (any rate, resampled to 16 kHz) or an F32 TENSOR of mono
// samples at 16 kHz.

#include <agent-cpp/engine_abi.h>

#include "../common/speech_audio.hpp"

#include <cstdio>
#include <cstring>
#include <new>
#include <simdjson.h>
#include <string>
#include <string_view>

#include <parakeet_capi.h>

// parakeet.cpp's process-wide compute-thread count (src/ggml_graph.hpp). The C API has no thread
// setting for the default single backend, so the plugin calls this internal function of the
// statically linked library directly: a relaxed atomic store, it cannot throw.
namespace pk {
void set_num_threads(int n);
}

namespace {

using agent_speech::Audio;
using agent_speech::sv;
using agent_speech::to_c;

constexpr int32_t kRate = 16000;

struct ParakeetInstance {
    parakeet_ctx *ctx = nullptr;
    int decoder = 0;      // 0 = model default, 1 = CTC head, 2 = TDT/RNNT head
    std::string language; // prompt-conditioned (Nemotron) models only; empty = model default

    // Response storage: valid until free_response() or the next run().
    std::string text;
    agent_part part{};
    std::string error;

    ~ParakeetInstance() {
        if (ctx) parakeet_capi_free(ctx);
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
    if (!obj[key].get_double().get(v)) return v;
    int64_t i = 0;
    if (!obj[key].get_int64().get(i)) return static_cast<double>(i);
    return fallback;
}

bool cancelled(const agent_request *req) { return req->cancel && *req->cancel != 0; }

int decoder_from_name(const std::string &name) {
    if (name.empty() || name == "default") return 0;
    if (name == "ctc") return 1;
    if (name == "tdt" || name == "rnnt" || name == "transducer") return 2;
    return -1;
}

int32_t engine_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    *out_instance = nullptr;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object cfg;
    if (parser.parse(config_json.data, config_json.size).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "parakeet: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    const std::string model = json_str(cfg, "model");
    if (model.empty()) {
        std::snprintf(err, err_cap, "parakeet: config.model (a parakeet.cpp GGUF) is required");
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (!agent_speech::file_exists(model)) {
        std::snprintf(err, err_cap, "parakeet: model not found: %s", model.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }

    int decoder = 0;
    int32_t threads = 4;
    std::string language, component;
    simdjson::dom::object options;
    if (!cfg["options"].get_object().get(options)) {
        const std::string d = json_str(options, "decoder");
        decoder = decoder_from_name(d);
        if (decoder < 0) {
            std::snprintf(err, err_cap, "parakeet: options.decoder must be \"default\", \"ctc\" or \"tdt\" (got \"%s\")", d.c_str());
            return AGENT_ERR_INVALID_CONFIG;
        }
        const double t = json_num(options, "threads", 0);
        if (t >= 1 && t <= 256) threads = static_cast<int32_t>(t);
        language = json_str(options, "language");
        component = json_str(options, "component");
    }

    auto *inst = new (std::nothrow) ParakeetInstance();
    if (!inst) {
        std::snprintf(err, err_cap, "parakeet: out of memory");
        return AGENT_ERR_INIT_FAILED;
    }
    inst->decoder = decoder;
    inst->language = std::move(language);
    pk::set_num_threads(threads);
    inst->ctx = component.empty() ? parakeet_capi_load(model.c_str()) : parakeet_capi_load_component(model.c_str(), component.c_str());
    if (!inst->ctx) {
        std::snprintf(err, err_cap, "parakeet: failed to load %s: %s", model.c_str(), parakeet_capi_load_error());
        delete inst;
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    if (parakeet_capi_model_kind(inst->ctx) != PARAKEET_MODEL_KIND_ASR) {
        std::snprintf(err, err_cap, "parakeet: %s is not a speech recognition model (use options.component for bundles)", model.c_str());
        delete inst;
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    *out_instance = inst;
    return AGENT_OK;
}

void engine_destroy(void *instance) { delete static_cast<ParakeetInstance *>(instance); }

int32_t transcribe(ParakeetInstance &inst, const agent_request *req, const agent_part &part, agent_response *resp) {
    Audio audio;
    if (std::string e = agent_speech::load_audio_part(part, kRate, kRate, audio); !e.empty())
        return inst.fail(AGENT_ERR_UNSUPPORTED, "parakeet: could not read audio: " + e);

    std::string language = inst.language;
    int decoder = inst.decoder;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object params;
    if (req->params_json.size && !parser.parse(req->params_json.data, req->params_json.size).get(doc) && !doc.get_object().get(params)) {
        std::string l = json_str(params, "language");
        if (!l.empty()) language = std::move(l);
        const std::string d = json_str(params, "decoder");
        if (!d.empty()) {
            decoder = decoder_from_name(d);
            if (decoder < 0) return inst.fail(AGENT_ERR_INVALID_CONFIG, "parakeet: decoder must be \"default\", \"ctc\" or \"tdt\"");
        }
    }
    // parakeet.cpp has no progress or abort hook inside a transcription: cancel is honoured before it.
    if (cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "parakeet: cancelled");
    if (req->on_progress) req->on_progress(req->user_data, 0, 1);
    char *out = parakeet_capi_transcribe_pcm_lang(inst.ctx, audio.samples.data(), static_cast<int>(audio.samples.size()), kRate, decoder,
                                                  language.empty() ? nullptr : language.c_str());
    if (!out) {
        const char *why = parakeet_capi_last_error(inst.ctx);
        return inst.fail(AGENT_ERR_DECODE_FAILED, std::string("parakeet: transcription failed: ") + (why && *why ? why : "unknown error"));
    }
    inst.text.assign(agent_speech::trim(out));
    parakeet_capi_free_string(out);
    if (req->on_progress) req->on_progress(req->user_data, 1, 1);
    if (cancelled(req)) return inst.fail(AGENT_ERR_CANCELLED, "parakeet: cancelled");
    if (req->on_chunk && !inst.text.empty()) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, inst.text.data(), inst.text.size());

    inst.part = agent_part{};
    inst.part.kind = AGENT_PART_TEXT;
    inst.part.text = to_c(inst.text);
    resp->message.parts = &inst.part;
    resp->message.n_parts = 1;
    return AGENT_OK;
}

int32_t engine_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<ParakeetInstance *>(instance);
    agent_response_reset(resp);
    inst->error.clear();
    if (cancelled(req)) return inst->fail(AGENT_ERR_CANCELLED, "parakeet: cancelled");
    if (req->task != AGENT_CAP_ASR && req->task != AGENT_CAP_TEXT_GEN)
        return inst->fail(AGENT_ERR_UNSUPPORTED, "parakeet: only the ASR task is supported");
    const agent_speech::UserInput in = agent_speech::last_user_input(req);
    if (!in.audio) return inst->fail(AGENT_ERR_INVALID_CONFIG, "parakeet: ASR needs an audio part on the last user message");
    const int32_t rc = transcribe(*inst, req, *in.audio, resp);
    if (rc != AGENT_OK) {
        agent_response_reset(resp);
        return rc;
    }
    resp->message.role = agent_str{"assistant", 9};
    return AGENT_OK;
}

void engine_free_response(void *, agent_response *resp) { agent_response_reset(resp); }

const char *engine_last_error(void *instance) { return static_cast<ParakeetInstance *>(instance)->error.c_str(); }

const agent_engine_v1 kParakeetEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "parakeet",
    "0.1",
    AGENT_CAP_ASR,
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

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kParakeetEngine; }
