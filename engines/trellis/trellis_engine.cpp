// agent-engine-trellis: TRELLIS.2 image -> 3D (GLB) as an agent.cpp engine plugin.
//
// Wraps trellis2.cpp (vendor/trellis2cpp) through its flat C API (trellis2_capi.h), so no C++
// exception or STL type of the port crosses into this plugin's code paths.
//
// One instance = one loaded t2_pipeline (DINOv3 encoder, sparse-structure flow + decoder, and
// optionally the shape-SLAT flow(s) + shape decoder for the 512 / 1024 paths and the texture models
// for PBR). AGENT_CAP_MESH_GEN takes the last image FILE (or BYTES) part of the last user message,
// writes <output_dir>/trellis-<unix ms>-<n>.glb and returns it as a FILE part (model/gltf-binary)
// plus a short text part.
//
// trellis2.cpp picks its CPU thread count from the TRELLIS2_N_THREADS environment variable when a
// backend is created, so create()/run() hold g_t2_mu while they temporarily set it.

#include <agent-cpp/engine_abi.h>
#include <trellis2_capi.h>

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <random>
#include <simdjson.h>
#include <string>
#include <string_view>

namespace {

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }
agent_str to_c(const std::string &s) { return agent_str{s.data(), s.size()}; }

// ------------------------------------------------------------------ generation settings

// Defaults come from Config::engine_options at create(); agent_request::params_json overrides them
// per call.
struct GenSettings {
    int64_t seed = -1;        // -1 = random
    int steps = 0;            // flow sampler steps; 0 = pipeline default (12)
    float guidance = -1.0f;   // CFG strength; < 0 = pipeline default (7.5)
    int texture_steps = 0;    // texture flow steps; 0 = pipeline default (12)
    int pipeline = T2_PIPE_AUTO;
    int background = T2_BACKGROUND_AUTO;
    int texture_size = 0;     // GLB atlas size (only with T2GLB_XATLAS); 0 = default (2048)
    int component_filter = 0; // 0 remove tiny islands, 1 keep largest, 2 keep all
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

// Applies the generation keys present in obj on top of g. Returns false (with error) on bad values.
bool apply_gen_settings(simdjson::ondemand::object &obj, GenSettings &g, std::string &error) {
    int64_t seed = 0;
    if (!obj["seed"].get_int64().get(seed)) g.seed = seed;
    g.steps = json_number<int>(obj, "steps", g.steps);
    g.guidance = json_number<float>(obj, "guidance", g.guidance);
    g.texture_steps = json_number<int>(obj, "texture_steps", g.texture_steps);
    g.texture_size = json_number<int>(obj, "texture_size", g.texture_size);

    std::string name;
    if (json_string(obj, "pipeline", name)) {
        if (name == "auto") g.pipeline = T2_PIPE_AUTO;
        else if (name == "coarse") g.pipeline = T2_PIPE_COARSE;
        else if (name == "512") g.pipeline = T2_PIPE_512;
        else if (name == "1024") g.pipeline = T2_PIPE_1024;
        else {
            error = "trellis: pipeline must be auto, coarse, 512 or 1024 (got '" + name + "')";
            return false;
        }
    }
    if (json_string(obj, "background", name)) {
        if (name == "auto") g.background = T2_BACKGROUND_AUTO;
        else if (name == "keep") g.background = T2_BACKGROUND_KEEP;
        else if (name == "black") g.background = T2_BACKGROUND_BLACK;
        else if (name == "white") g.background = T2_BACKGROUND_WHITE;
        else {
            error = "trellis: background must be auto, keep, black or white (got '" + name + "')";
            return false;
        }
    }
    if (json_string(obj, "components", name)) {
        if (name == "remove_tiny") g.component_filter = 0;
        else if (name == "largest") g.component_filter = 1;
        else if (name == "all") g.component_filter = 2;
        else {
            error = "trellis: components must be remove_tiny, largest or all (got '" + name + "')";
            return false;
        }
    }
    if (g.steps < 0 || g.steps > 1000) {
        error = "trellis: steps must be in 0..1000 (0 = model default)";
        return false;
    }
    if (g.texture_steps < 0 || g.texture_steps > 1000) {
        error = "trellis: texture_steps must be in 0..1000 (0 = model default)";
        return false;
    }
    if (g.texture_size < 0 || g.texture_size > 8192) {
        error = "trellis: texture_size must be in 0..8192 (0 = default)";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ model files

// The pipeline's GGUF slots, in t2_pipeline_load() order. `base` is the file name the published
// GGUF repos use (LocalAI-io/TRELLIS.2-4B-GGUF etc.) without the "_f16.gguf" suffix.
struct Slot {
    const char *option;
    const char *base;
    bool required;
};
constexpr Slot kSlots[] = {
    {"dino", "dino", true},
    {"ss_flow", "ss_flow", true},
    {"ss_dec", "ss_dec", true},
    {"slat_flow", "slat_flow", false},
    {"slat_flow_1024", "slat_flow_1024", false},
    {"shape_dec", "shape_dec", false},
    {"shape_enc", "shape_enc", false},
    {"tex_dec", "tex_dec", false},
    {"tex_flow", "tex_slat_flow_512", false},
    {"tex_flow_1024", "tex_slat_flow_1024", false},
};
constexpr size_t kNumSlots = sizeof(kSlots) / sizeof(kSlots[0]);

bool is_file(const std::filesystem::path &p) {
    std::error_code ec;
    return std::filesystem::is_regular_file(p, ec);
}

// <dir>/<base>_f16.gguf, <base>_f32.gguf or <base>.gguf, whichever exists first.
std::string discover(const std::filesystem::path &dir, const char *base) {
    for (const char *suffix : {"_f16.gguf", "_f32.gguf", ".gguf"}) {
        std::filesystem::path p = dir / (std::string(base) + suffix);
        if (is_file(p)) return p.string();
    }
    return {};
}

void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// Sets TRELLIS2_N_THREADS for the lifetime of the guard (g_t2_mu must be held), restoring the
// previous value afterwards. threads <= 0 leaves the environment alone.
struct ThreadsEnv {
    explicit ThreadsEnv(int threads) {
        if (threads <= 0) return;
        active = true;
        if (const char *old = std::getenv("TRELLIS2_N_THREADS")) {
            had_old = true;
            old_value = old;
        }
        set_env("TRELLIS2_N_THREADS", std::to_string(threads).c_str());
    }
    ~ThreadsEnv() {
        if (!active) return;
        set_env("TRELLIS2_N_THREADS", had_old ? old_value.c_str() : nullptr);
    }
    ThreadsEnv(const ThreadsEnv &) = delete;
    ThreadsEnv &operator=(const ThreadsEnv &) = delete;

    bool active = false;
    bool had_old = false;
    std::string old_value;
};

std::mutex g_t2_mu;

// ------------------------------------------------------------------ instance

struct Instance {
    t2_pipeline *pipeline = nullptr;
    GenSettings defaults;
    int threads = 0;
    uint64_t counter = 0;

    // Response storage, valid until free_response() / the next run().
    std::string path;
    std::string text;
    agent_part parts[2]{};
    std::string error;

    ~Instance() {
        if (pipeline) t2_pipeline_free(pipeline);
    }
};

// ------------------------------------------------------------------ ABI functions

int32_t t2_engine_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(sv(config_json));
    simdjson::ondemand::document doc;
    simdjson::ondemand::object cfg;
    if (parser.iterate(json).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "trellis: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }

    // "model" is the directory holding the GGUFs; options.<slot> overrides (or, with "", disables)
    // individual files. Relative option paths are resolved against the model directory.
    std::string model;
    json_string(cfg, "model", model);
    std::string paths[kNumSlots];
    bool explicit_slot[kNumSlots] = {};
    GenSettings defaults;
    int threads = 0;
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        for (size_t i = 0; i < kNumSlots; ++i) explicit_slot[i] = json_string(options, kSlots[i].option, paths[i]);
        threads = json_number<int>(options, "threads", 0);
        std::string error;
        if (!apply_gen_settings(options, defaults, error)) {
            std::snprintf(err, err_cap, "%s", error.c_str());
            return AGENT_ERR_INVALID_CONFIG;
        }
    }

    const std::filesystem::path dir(model);
    if (!model.empty()) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            std::snprintf(err, err_cap, "trellis: model directory not found: %s", model.c_str());
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
    }
    for (size_t i = 0; i < kNumSlots; ++i) {
        if (explicit_slot[i]) {
            if (!paths[i].empty() && std::filesystem::path(paths[i]).is_relative() && !model.empty())
                paths[i] = (dir / paths[i]).string();
        } else if (!model.empty()) {
            paths[i] = discover(dir, kSlots[i].base);
        }
        if (kSlots[i].required && paths[i].empty()) {
            if (model.empty() && !explicit_slot[i]) {
                std::snprintf(err, err_cap, "trellis: config.model (a directory with the TRELLIS.2 GGUFs) or options.%s is required",
                              kSlots[i].option);
                return AGENT_ERR_INVALID_CONFIG;
            }
            if (model.empty()) std::snprintf(err, err_cap, "trellis: options.%s is empty", kSlots[i].option);
            else std::snprintf(err, err_cap, "trellis: model file not found: %s", (dir / (std::string(kSlots[i].base) + "_f16.gguf")).string().c_str());
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
        if (!paths[i].empty() && !is_file(paths[i])) {
            std::snprintf(err, err_cap, "trellis: model file not found: %s", paths[i].c_str());
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
    }

    auto c_or_null = [](const std::string &s) { return s.empty() ? nullptr : s.c_str(); };
    char load_err[512] = {0};
    t2_pipeline *pipeline = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_t2_mu);
        ThreadsEnv env(threads);
        pipeline = t2_pipeline_load(c_or_null(paths[0]), c_or_null(paths[1]), c_or_null(paths[2]), c_or_null(paths[3]),
                                    c_or_null(paths[4]), c_or_null(paths[5]), c_or_null(paths[6]), c_or_null(paths[7]),
                                    c_or_null(paths[8]), c_or_null(paths[9]), 0, load_err, static_cast<int>(sizeof(load_err)));
    }
    if (!pipeline) {
        std::snprintf(err, err_cap, "trellis: failed to load the pipeline%s%s", load_err[0] ? ": " : "", load_err);
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }

    auto *inst = new Instance();
    inst->pipeline = pipeline;
    inst->defaults = defaults;
    inst->threads = threads;
    *out_instance = inst;
    return AGENT_OK;
}

void t2_engine_destroy(void *instance) { delete static_cast<Instance *>(instance); }

int32_t fail(Instance *inst, int32_t code, std::string msg) {
    inst->error = std::move(msg);
    return code;
}

const agent_message *last_user_message(const agent_request *req) {
    for (size_t i = req->n_messages; i-- > 0;)
        if (sv(req->messages[i].role) == "user") return &req->messages[i];
    return nullptr;
}

const agent_part *last_image_part(const agent_message *m) {
    if (!m) return nullptr;
    const agent_part *src = nullptr;
    for (size_t p = 0; p < m->n_parts; ++p) {
        const agent_part &part = m->parts[p];
        const std::string_view mime = sv(part.mime);
        if ((part.kind == AGENT_PART_FILE || part.kind == AGENT_PART_BYTES) && (mime.empty() || mime.starts_with("image/"))) src = &part;
    }
    return src;
}

bool read_file(const std::string &path, std::string &out) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[1 << 16];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    const bool ok = !std::ferror(f);
    std::fclose(f);
    return ok;
}

// <output_dir>/trellis-<unix ms>-<n>.glb, skipping names that already exist.
std::string unique_glb_path(Instance *inst, const std::string &dir) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::filesystem::path base(dir);
    for (;;) {
        std::filesystem::path p = base / ("trellis-" + std::to_string(ms) + "-" + std::to_string(inst->counter++) + ".glb");
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) return p.string();
    }
}

// Forwards the flow samplers' per-step progress (stages with a step count) to on_progress; the
// counter restarts for every sampling phase (sparse structure, shape SLAT, HR SLAT, texture).
struct ProgressCtx {
    const agent_request *req;
};

void on_t2_progress(void *user, int /*stage*/, int step, int total) {
    const agent_request *req = static_cast<ProgressCtx *>(user)->req;
    if (req->on_progress && total > 0 && step >= 0)
        req->on_progress(req->user_data, static_cast<uint64_t>(step), static_cast<uint64_t>(total));
}

bool cancelled(const agent_request *req) { return req->cancel && *req->cancel; }

int32_t t2_engine_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    inst->path.clear();
    inst->text.clear();
    if (req->task != AGENT_CAP_MESH_GEN) return fail(inst, AGENT_ERR_UNSUPPORTED, "trellis: only MESH_GEN is supported");

    const agent_part *image = last_image_part(last_user_message(req));
    if (!image) return fail(inst, AGENT_ERR_INVALID_CONFIG, "trellis: MESH_GEN needs an image FILE or BYTES part in the last user message");

    const std::string output_dir(sv(req->output_dir));
    if (output_dir.empty()) return fail(inst, AGENT_ERR_INVALID_CONFIG, "trellis: request output_dir is required");

    GenSettings g = inst->defaults;
    if (req->params_json.size) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(sv(req->params_json));
        simdjson::ondemand::document doc;
        simdjson::ondemand::object params;
        if (parser.iterate(json).get(doc) || doc.get_object().get(params))
            return fail(inst, AGENT_ERR_PARSE, "trellis: params_json is not a JSON object");
        std::string error;
        if (!apply_gen_settings(params, g, error)) return fail(inst, AGENT_ERR_INVALID_CONFIG, error);
    }

    std::string file_bytes;
    const void *bytes = image->data;
    size_t size = image->size;
    if (image->kind == AGENT_PART_FILE) {
        const std::string path(sv(image->path));
        if (!read_file(path, file_bytes)) return fail(inst, AGENT_ERR_FILESYSTEM, "trellis: could not read image " + path);
        bytes = file_bytes.data();
        size = file_bytes.size();
    }
    if (!bytes || size == 0) return fail(inst, AGENT_ERR_INVALID_CONFIG, "trellis: the input image is empty");
    if (size > static_cast<size_t>(INT_MAX)) return fail(inst, AGENT_ERR_INVALID_CONFIG, "trellis: the input image is too large");

    uint64_t seed = 0;
    if (g.seed < 0) {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    } else {
        seed = static_cast<uint64_t>(g.seed);
    }

    if (cancelled(req)) return fail(inst, AGENT_ERR_CANCELLED, "trellis: cancelled");

    ProgressCtx progress{req};
    char gen_err[512] = {0};
    t2_mesh_result *mesh = nullptr;
    uint8_t *glb = nullptr;
    int glb_len = 0;
    int n_verts = 0, n_tris = 0;
    bool was_cancelled = false;
    {
        std::lock_guard<std::mutex> lock(g_t2_mu);
        ThreadsEnv env(inst->threads);
        // trellis2.cpp has no cancellation hook inside generation: the flag is honoured before
        // starting, between generation and GLB export, and after export.
        mesh = t2_generate(inst->pipeline, bytes, static_cast<int>(size), g.pipeline, g.background, seed, g.steps, g.guidance,
                           g.texture_steps, on_t2_progress, &progress, nullptr, nullptr, gen_err, static_cast<int>(sizeof(gen_err)));
        if (mesh && cancelled(req)) {
            was_cancelled = true;
        } else if (mesh) {
            n_verts = t2_mesh_n_verts(mesh);
            n_tris = t2_mesh_n_tris(mesh);
            glb = t2_bake_glb(t2_mesh_verts(mesh), n_verts, t2_mesh_tris(mesh), n_tris, t2_mesh_pbr(mesh), g.texture_size,
                              g.component_filter, &glb_len, gen_err, static_cast<int>(sizeof(gen_err)));
        }
    }
    if (mesh) t2_mesh_free(mesh);
    if (was_cancelled || (glb && cancelled(req))) {
        if (glb) t2_free_buffer(glb);
        return fail(inst, AGENT_ERR_CANCELLED, "trellis: cancelled");
    }
    if (!mesh) return fail(inst, AGENT_ERR_DECODE_FAILED, std::string("trellis: generation failed") + (gen_err[0] ? ": " : "") + gen_err);
    if (!glb || glb_len <= 0) {
        if (glb) t2_free_buffer(glb);
        return fail(inst, AGENT_ERR_DECODE_FAILED, std::string("trellis: GLB export failed") + (gen_err[0] ? ": " : "") + gen_err);
    }

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::string path = unique_glb_path(inst, output_dir);
    std::FILE *f = std::fopen(path.c_str(), "wb");
    bool written = false;
    if (f) {
        written = std::fwrite(glb, 1, static_cast<size_t>(glb_len), f) == static_cast<size_t>(glb_len);
        written = (std::fclose(f) == 0) && written;
    }
    t2_free_buffer(glb);
    if (!written) {
        std::filesystem::remove(path, ec);
        return fail(inst, AGENT_ERR_FILESYSTEM, "trellis: could not write " + path);
    }
    if (req->on_progress) req->on_progress(req->user_data, 1, 1);

    inst->path = std::move(path);
    inst->text = "generated 3D model: " + inst->path + " (" + std::to_string(n_verts) + " vertices, " + std::to_string(n_tris) +
                 " triangles)";
    inst->parts[0] = agent_part{};
    inst->parts[0].kind = AGENT_PART_FILE;
    inst->parts[0].path = to_c(inst->path);
    inst->parts[0].mime = agent_str{"model/gltf-binary", 17};
    inst->parts[1] = agent_part{};
    inst->parts[1].kind = AGENT_PART_TEXT;
    inst->parts[1].text = to_c(inst->text);

    *resp = agent_response{};
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = inst->parts;
    resp->message.n_parts = 2;
    return AGENT_OK;
}

void t2_engine_free_response(void *instance, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    inst->path.clear();
    inst->text.clear();
    inst->parts[0] = agent_part{};
    inst->parts[1] = agent_part{};
    *resp = agent_response{};
}

const char *t2_engine_last_error(void *instance) { return static_cast<Instance *>(instance)->error.c_str(); }

const agent_engine_v1 kTrellisEngine = {
    AGENT_ENGINE_ABI_VERSION,
    "trellis",
    "0.1",
    AGENT_CAP_MESH_GEN,
    0, // t2_generate is not thread-safe per pipeline; the host serializes run()
    t2_engine_create,
    t2_engine_destroy,
    t2_engine_run,
    t2_engine_free_response,
    t2_engine_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kTrellisEngine; }
