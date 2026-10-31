// agent-engine-sdcpp: stable-diffusion.cpp image generation as an agent.cpp engine plugin.
//
// One instance = one loaded sd_ctx_t (SD 1.x/2.x/XL, SD3, FLUX, Qwen-Image, Z-Image, ...).
//   - AGENT_CAP_IMAGE_GEN: txt2img; the prompt is the text of the last user message.
//   - AGENT_CAP_IMAGE_EDIT: img2img; additionally takes the last image FILE (or BYTES) part of the
//     last user message as the init image.
// The result is written as a PNG into agent_request::output_dir and returned as a FILE part
// (image/png) plus a short text part.
//
// sd.cpp's log and progress callbacks are process-global, so every create()/run() across all
// instances holds g_sd_mu and publishes its state through g_active. Cancellation is checked at
// every sampler step (and during VAE decode) via sd_cancel_generation().

#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <simdjson.h>
#include <stable-diffusion.h>
#include <string>
#include <string_view>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <stb_image.h>

namespace {

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }
agent_str to_c(const std::string &s) { return agent_str{s.data(), s.size()}; }

// ------------------------------------------------------------------ generation settings

// Defaults come from Config::engine_options at create(); agent_request::params_json overrides them
// per call.
struct GenSettings {
    int width = 512;
    int height = 512;
    int steps = 20;
    int64_t seed = -1; // random
    float cfg_scale = 7.0f;
    float guidance = 3.5f;  // distilled guidance (FLUX)
    float strength = 0.75f; // img2img denoising strength
    int clip_skip = -1;
    sample_method_t sampler = SAMPLE_METHOD_COUNT; // model default
    scheduler_t scheduler = SCHEDULER_COUNT;       // model default
    bool vae_tiling = false;
    std::string negative_prompt;
};

template <typename T> T json_number(simdjson::ondemand::object &obj, const char *key, T fallback) {
    double d = 0;
    if (obj[key].get_double().get(d)) return fallback;
    return static_cast<T>(d);
}

bool json_string(simdjson::ondemand::object &obj, const char *key, std::string &out) {
    std::string_view s;
    if (obj[key].get_string().get(s)) return false;
    out = std::string(s);
    return true;
}

bool json_bool(simdjson::ondemand::object &obj, const char *key, bool fallback) {
    bool b = false;
    if (obj[key].get_bool().get(b)) return fallback;
    return b;
}

// Applies the generation keys present in obj on top of g. Returns false (with error) on bad values.
bool apply_gen_settings(simdjson::ondemand::object &obj, GenSettings &g, std::string &error) {
    g.width = json_number<int>(obj, "width", g.width);
    g.height = json_number<int>(obj, "height", g.height);
    g.steps = json_number<int>(obj, "steps", g.steps);
    int64_t seed = 0;
    if (!obj["seed"].get_int64().get(seed)) g.seed = seed;
    g.cfg_scale = json_number<float>(obj, "cfg_scale", g.cfg_scale);
    g.guidance = json_number<float>(obj, "guidance", g.guidance);
    g.strength = json_number<float>(obj, "strength", g.strength);
    g.clip_skip = json_number<int>(obj, "clip_skip", g.clip_skip);
    g.vae_tiling = json_bool(obj, "vae_tiling", g.vae_tiling);
    json_string(obj, "negative_prompt", g.negative_prompt);

    std::string name;
    if (json_string(obj, "sampler", name) && !name.empty()) {
        g.sampler = str_to_sample_method(name.c_str());
        if (g.sampler == SAMPLE_METHOD_COUNT) {
            error = "sdcpp: unknown sampler '" + name + "'";
            return false;
        }
    }
    if (json_string(obj, "scheduler", name) && !name.empty()) {
        g.scheduler = str_to_scheduler(name.c_str());
        if (g.scheduler == SCHEDULER_COUNT) {
            error = "sdcpp: unknown scheduler '" + name + "'";
            return false;
        }
    }
    if (g.width <= 0 || g.height <= 0 || g.width > 8192 || g.height > 8192) {
        error = "sdcpp: width/height must be in 1..8192";
        return false;
    }
    if (g.steps <= 0 || g.steps > 1000) {
        error = "sdcpp: steps must be in 1..1000";
        return false;
    }
    if (!(g.strength >= 0.0f && g.strength <= 1.0f)) {
        error = "sdcpp: strength must be in 0..1";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ instance + global callbacks

struct Instance {
    sd_ctx_t *ctx = nullptr;
    GenSettings defaults;
    bool verbose = false;
    uint64_t counter = 0;

    // Response storage, valid until free_response() / the next run().
    std::string path;
    std::string text;
    agent_part parts[2]{};
    std::string error;

    ~Instance() {
        if (ctx) free_sd_ctx(ctx);
    }
};

// State of the create()/run() currently inside sd.cpp. Only touched while g_sd_mu is held by the
// owning call; the callbacks read it from the generating thread.
struct Active {
    sd_ctx_t *ctx = nullptr;
    const agent_request *req = nullptr;
    bool verbose = false;
    bool cancelled = false;
    std::string last_error;
};

std::mutex g_sd_mu;
std::atomic<Active *> g_active{nullptr};
std::once_flag g_callbacks_once;

void on_sd_log(sd_log_level_t level, const char *text, void *) {
    Active *a = g_active.load(std::memory_order_acquire);
    if (a && level == SD_LOG_ERROR && text) {
        a->last_error = text;
        while (!a->last_error.empty() && (a->last_error.back() == '\n' || a->last_error.back() == '\r')) a->last_error.pop_back();
    }
    if (level >= SD_LOG_WARN || (a && a->verbose)) std::fputs(text, stderr);
}

void on_sd_progress(int step, int steps, float, void *) {
    Active *a = g_active.load(std::memory_order_acquire);
    if (!a || !a->req) return; // model loading inside create()
    const agent_request *req = a->req;
    if (req->cancel && *req->cancel && !a->cancelled) {
        a->cancelled = true;
        sd_cancel_generation(a->ctx, SD_CANCEL_ALL);
    }
    if (req->on_progress && step >= 0 && steps > 0)
        req->on_progress(req->user_data, static_cast<uint64_t>(step), static_cast<uint64_t>(steps));
}

// Publishes `a` to the global callbacks for the lifetime of the guard (g_sd_mu must be held).
struct ActiveScope {
    explicit ActiveScope(Active *a) { g_active.store(a, std::memory_order_release); }
    ~ActiveScope() { g_active.store(nullptr, std::memory_order_release); }
    ActiveScope(const ActiveScope &) = delete;
    ActiveScope &operator=(const ActiveScope &) = delete;
};

void install_callbacks() {
    std::call_once(g_callbacks_once, [] {
        sd_set_log_callback(on_sd_log, nullptr);
        sd_set_progress_callback(on_sd_progress, nullptr);
    });
}

bool file_exists(const std::string &path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

// ------------------------------------------------------------------ ABI functions

int32_t sd_engine_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    install_callbacks();

    simdjson::ondemand::parser parser;
    simdjson::padded_string json(sv(config_json));
    simdjson::ondemand::document doc;
    simdjson::ondemand::object cfg;
    if (parser.iterate(json).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "sdcpp: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }

    // Component paths, keyed by the option name. "model" is the top-level config field.
    std::string model, diffusion_model, vae, taesd, clip_l, clip_g, t5xxl, llm, prediction;
    json_string(cfg, "model", model);
    auto inst_defaults = GenSettings{};
    int threads = 0;
    bool flash_attn = false, verbose = false, eager_load = false;
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        json_string(options, "diffusion_model", diffusion_model);
        json_string(options, "vae", vae);
        json_string(options, "taesd", taesd);
        json_string(options, "clip_l", clip_l);
        json_string(options, "clip_g", clip_g);
        json_string(options, "t5xxl", t5xxl);
        json_string(options, "llm", llm);
        json_string(options, "prediction", prediction);
        threads = json_number<int>(options, "threads", 0);
        flash_attn = json_bool(options, "flash_attn", false);
        eager_load = json_bool(options, "eager_load", false);
        verbose = json_bool(options, "verbose", false);
        std::string error;
        if (!apply_gen_settings(options, inst_defaults, error)) {
            std::snprintf(err, err_cap, "%s", error.c_str());
            return AGENT_ERR_INVALID_CONFIG;
        }
    }
    if (model.empty() && diffusion_model.empty()) {
        std::snprintf(err, err_cap, "sdcpp: config.model (or options.diffusion_model) is required");
        return AGENT_ERR_INVALID_CONFIG;
    }
    for (const std::string *p : {&model, &diffusion_model, &vae, &taesd, &clip_l, &clip_g, &t5xxl, &llm}) {
        if (!p->empty() && !file_exists(*p)) {
            std::snprintf(err, err_cap, "sdcpp: model file not found: %s", p->c_str());
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    auto c_or_null = [](const std::string &s) { return s.empty() ? nullptr : s.c_str(); };
    params.model_path = c_or_null(model);
    params.diffusion_model_path = c_or_null(diffusion_model);
    params.vae_path = c_or_null(vae);
    params.taesd_path = c_or_null(taesd);
    params.clip_l_path = c_or_null(clip_l);
    params.clip_g_path = c_or_null(clip_g);
    params.t5xxl_path = c_or_null(t5xxl);
    params.llm_path = c_or_null(llm);
    if (threads > 0) params.n_threads = threads;
    params.flash_attn = flash_attn;
    params.diffusion_flash_attn = flash_attn;
    params.eager_load = eager_load;
    if (!prediction.empty()) {
        params.prediction = str_to_prediction(prediction.c_str());
        if (params.prediction == PREDICTION_COUNT) {
            std::snprintf(err, err_cap, "sdcpp: unknown prediction '%s'", prediction.c_str());
            return AGENT_ERR_INVALID_CONFIG;
        }
    }

    Active active;
    active.verbose = verbose;
    sd_ctx_t *ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_sd_mu);
        ActiveScope scope(&active);
        ctx = new_sd_ctx(&params);
    }
    if (!ctx) {
        std::snprintf(err, err_cap, "sdcpp: failed to load model%s%s", active.last_error.empty() ? "" : ": ", active.last_error.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    if (!sd_ctx_supports_image_generation(ctx)) {
        free_sd_ctx(ctx);
        std::snprintf(err, err_cap, "sdcpp: the loaded model does not support image generation");
        return AGENT_ERR_UNSUPPORTED;
    }

    auto *inst = new Instance();
    inst->ctx = ctx;
    inst->defaults = std::move(inst_defaults);
    inst->verbose = verbose;
    *out_instance = inst;
    return AGENT_OK;
}

void sd_engine_destroy(void *instance) { delete static_cast<Instance *>(instance); }

int32_t fail(Instance *inst, int32_t code, std::string msg) {
    inst->error = std::move(msg);
    return code;
}

// The last user message: its concatenated text parts, and (for IMAGE_EDIT) its last image part.
const agent_message *last_user_message(const agent_request *req) {
    for (size_t i = req->n_messages; i-- > 0;)
        if (sv(req->messages[i].role) == "user") return &req->messages[i];
    return nullptr;
}

// Loads the init image for img2img as 8-bit RGB. The pixels are freed with stbi_image_free.
bool load_init_image(const agent_message &m, sd_image_t &out, std::string &error) {
    const agent_part *src = nullptr;
    for (size_t p = 0; p < m.n_parts; ++p) {
        const agent_part &part = m.parts[p];
        const std::string_view mime = sv(part.mime);
        if ((part.kind == AGENT_PART_FILE || part.kind == AGENT_PART_BYTES) && (mime.empty() || mime.starts_with("image/"))) src = &part;
    }
    if (!src) {
        error = "sdcpp: IMAGE_EDIT needs an image FILE or BYTES part in the last user message";
        return false;
    }
    int w = 0, h = 0, c = 0;
    stbi_uc *pixels = nullptr;
    if (src->kind == AGENT_PART_FILE) {
        const std::string path(sv(src->path));
        pixels = stbi_load(path.c_str(), &w, &h, &c, 3);
        if (!pixels) error = "sdcpp: could not read image " + path;
    } else {
        pixels = stbi_load_from_memory(static_cast<const stbi_uc *>(src->data), static_cast<int>(src->size), &w, &h, &c, 3);
        if (!pixels) error = "sdcpp: could not decode the input image bytes";
    }
    if (!pixels) return false;
    out = sd_image_t{static_cast<uint32_t>(w), static_cast<uint32_t>(h), 3, pixels};
    return true;
}

// <output_dir>/sdcpp-<unix ms>-<n>.png, skipping names that already exist.
std::string unique_png_path(Instance *inst, const std::string &dir) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::filesystem::path base(dir);
    for (;;) {
        std::filesystem::path p = base / ("sdcpp-" + std::to_string(ms) + "-" + std::to_string(inst->counter++) + ".png");
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) return p.string();
    }
}

int32_t sd_engine_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    inst->path.clear();
    inst->text.clear();
    if (req->task != AGENT_CAP_IMAGE_GEN && req->task != AGENT_CAP_IMAGE_EDIT)
        return fail(inst, AGENT_ERR_UNSUPPORTED, "sdcpp: only IMAGE_GEN and IMAGE_EDIT are supported");

    const agent_message *user = last_user_message(req);
    std::string prompt;
    if (user)
        for (size_t p = 0; p < user->n_parts; ++p)
            if (user->parts[p].kind == AGENT_PART_TEXT) prompt += sv(user->parts[p].text);
    if (prompt.empty()) return fail(inst, AGENT_ERR_INVALID_CONFIG, "sdcpp: the last user message has no prompt text");

    const std::string output_dir(sv(req->output_dir));
    if (output_dir.empty()) return fail(inst, AGENT_ERR_INVALID_CONFIG, "sdcpp: request output_dir is required");

    GenSettings g = inst->defaults;
    sd_image_t init_image{};
    if (req->task == AGENT_CAP_IMAGE_EDIT) {
        std::string error;
        if (!load_init_image(*user, init_image, error)) return fail(inst, AGENT_ERR_INVALID_CONFIG, error);
        // img2img keeps the input size (rounded down to a multiple of 8) unless params_json sets one.
        g.width = std::max(8, static_cast<int>(init_image.width) / 8 * 8);
        g.height = std::max(8, static_cast<int>(init_image.height) / 8 * 8);
    }
    if (req->params_json.size) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(sv(req->params_json));
        simdjson::ondemand::document doc;
        simdjson::ondemand::object params;
        std::string error;
        if (parser.iterate(json).get(doc) || doc.get_object().get(params)) {
            if (init_image.data) stbi_image_free(init_image.data);
            return fail(inst, AGENT_ERR_PARSE, "sdcpp: params_json is not a JSON object");
        }
        if (!apply_gen_settings(params, g, error)) {
            if (init_image.data) stbi_image_free(init_image.data);
            return fail(inst, AGENT_ERR_INVALID_CONFIG, error);
        }
    }

    sd_img_gen_params_t gen;
    sd_img_gen_params_init(&gen);
    gen.prompt = prompt.c_str();
    gen.negative_prompt = g.negative_prompt.c_str();
    gen.clip_skip = g.clip_skip;
    gen.width = g.width;
    gen.height = g.height;
    gen.seed = g.seed;
    gen.batch_count = 1;
    gen.strength = g.strength;
    gen.init_image = init_image;
    gen.sample_params.sample_steps = g.steps;
    gen.sample_params.sample_method = g.sampler;
    gen.sample_params.scheduler = g.scheduler;
    gen.sample_params.guidance.txt_cfg = g.cfg_scale;
    gen.sample_params.guidance.distilled_guidance = g.guidance;
    gen.vae_tiling_params.enabled = g.vae_tiling;

    Active active;
    active.ctx = inst->ctx;
    active.req = req;
    active.verbose = inst->verbose;
    sd_image_generation_result_t *result = nullptr;
    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(g_sd_mu);
        if (req->cancel && *req->cancel) {
            active.cancelled = true;
        } else {
            ActiveScope scope(&active);
            ok = generate_image(inst->ctx, &gen, &result);
        }
    }
    if (init_image.data) stbi_image_free(init_image.data);

    const bool cancelled = active.cancelled || (req->cancel && *req->cancel);
    if (cancelled || !ok || !result || result->count < 1 || !result->items[0].image.data) {
        if (result) free_sd_image_generation_result(result);
        if (cancelled) return fail(inst, AGENT_ERR_CANCELLED, "sdcpp: cancelled");
        return fail(inst, AGENT_ERR_DECODE_FAILED,
                    "sdcpp: image generation failed" + (active.last_error.empty() ? std::string() : ": " + active.last_error));
    }

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    const sd_image_t &img = result->items[0].image;
    std::string path = unique_png_path(inst, output_dir);
    const int w = static_cast<int>(img.width), h = static_cast<int>(img.height), c = static_cast<int>(img.channel);
    const int written = stbi_write_png(path.c_str(), w, h, c, img.data, w * c);
    free_sd_image_generation_result(result);
    if (!written) return fail(inst, AGENT_ERR_FILESYSTEM, "sdcpp: could not write " + path);

    inst->path = std::move(path);
    inst->text = "generated image: " + inst->path;
    inst->parts[0] = agent_part{};
    inst->parts[0].kind = AGENT_PART_FILE;
    inst->parts[0].path = to_c(inst->path);
    inst->parts[0].mime = agent_str{"image/png", 9};
    inst->parts[1] = agent_part{};
    inst->parts[1].kind = AGENT_PART_TEXT;
    inst->parts[1].text = to_c(inst->text);

    *resp = agent_response{};
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = inst->parts;
    resp->message.n_parts = 2;
    return AGENT_OK;
}

void sd_engine_free_response(void *instance, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    inst->path.clear();
    inst->text.clear();
    inst->parts[0] = agent_part{};
    inst->parts[1] = agent_part{};
    *resp = agent_response{};
}

const char *sd_engine_last_error(void *instance) { return static_cast<Instance *>(instance)->error.c_str(); }

const agent_engine_v1 kSdcppEngine = {
    AGENT_ENGINE_ABI_VERSION,
    "sdcpp",
    "0.1",
    AGENT_CAP_IMAGE_GEN | AGENT_CAP_IMAGE_EDIT,
    0, // run() is serialized by the host; sd.cpp's global callbacks serialize across instances too
    sd_engine_create,
    sd_engine_destroy,
    sd_engine_run,
    sd_engine_free_response,
    sd_engine_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kSdcppEngine; }
