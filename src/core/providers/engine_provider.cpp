#include "../engines/engine_registry.hpp"
#include "../engines/engine_remote.hpp"
#include "../executor/ambient_turn.hpp"
#include "../sandbox/sandbox_policy.hpp"
#include "../tools/tools_ops.hpp"
#include "../utils/utils.hpp"
#include "provider_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

namespace agent::providers {

namespace {

// One of two shapes: an in-process engine (engine + instance) or an engine running in an
// agent-engine-host child process (remote). name / caps / flags are valid for both.
struct EngineState {
    const agent_engine_v1 *engine = nullptr;
    void *instance = nullptr;
    std::shared_ptr<engines::RemoteEngine> remote;
    std::mutex run_mutex; // serializes run() unless the engine sets AGENT_ENGINE_FLAG_CONCURRENT
    std::string output_dir;

    std::string name;
    uint64_t caps = 0;
    uint32_t flags = 0;
    // Set once a request carrying tool-produced media failed; see reject_tool_media.
    std::atomic<bool> tool_media_rejected{false};

    bool ready() const { return remote || (engine && instance); }

    ~EngineState() {
        if (engine && instance) engine->destroy(instance);
    }
};

agent_str to_c(std::string_view s) { return agent_str{s.data(), s.size()}; }
std::string_view from_c(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

bool looks_like_path(std::string_view s) {
    return s.find('/') != std::string_view::npos || s.find('\\') != std::string_view::npos || s.ends_with(".so") ||
           s.ends_with(".dll") || s.ends_with(".dylib");
}

uint32_t to_c_dtype(DType d) {
    switch (d) {
    case DType::Float32: return AGENT_DTYPE_F32;
    case DType::Float16: return AGENT_DTYPE_F16;
    case DType::Int8: return AGENT_DTYPE_I8;
    case DType::Int32: return AGENT_DTYPE_I32;
    case DType::Int64: return AGENT_DTYPE_I64;
    case DType::UInt8: return AGENT_DTYPE_U8;
    }
    return AGENT_DTYPE_F32;
}

DType from_c_dtype(uint32_t d) {
    switch (d) {
    case AGENT_DTYPE_F16: return DType::Float16;
    case AGENT_DTYPE_I8: return DType::Int8;
    case AGENT_DTYPE_I32: return DType::Int32;
    case AGENT_DTYPE_I64: return DType::Int64;
    case AGENT_DTYPE_U8: return DType::UInt8;
    default: return DType::Float32;
    }
}

bool tool_allowed(std::string_view name) {
    if (ambient::allowed_tools.empty()) return true;
    for (auto a : ambient::allowed_tools)
        if (a == name) return true;
    return false;
}

// Tool definitions as a JSON array. Rebuilt per call into a thread-local buffer whose capacity is
// reused, so steady-state turns don't allocate.
std::string_view build_tools_json(const Session &session) {
    thread_local std::string buf;
    buf.clear();
    buf += '[';
    bool first = true;
    for (const Tool &t : session.config.tools) {
        if (!tools::tool_visible(session, ambient::allowed_tools, t.name)) continue;
        if (!first) buf += ',';
        first = false;
        buf += "{\"name\":\"";
        utils::escape_json_string(t.name, buf);
        buf += "\",\"description\":\"";
        utils::escape_json_string(t.description, buf);
        buf += "\",\"parameters\":";
        buf += t.parameter_schema.empty() ? std::string_view("{}") : t.parameter_schema;
        buf += '}';
    }
    buf += ']';
    return first ? std::string_view() : std::string_view(buf);
}

struct CallbackCtx {
    TokenStreamFn on_token;
    void *token_user_data;
    ProgressFn on_progress;
    void *progress_user_data;
};

void forward_chunk(void *user_data, uint32_t kind, const void *data, size_t size) {
    auto *ctx = static_cast<CallbackCtx *>(user_data);
    if (kind == AGENT_CHUNK_TEXT && ctx->on_token)
        ctx->on_token(std::string_view(static_cast<const char *>(data), size), ctx->token_user_data);
}

void forward_progress(void *user_data, uint64_t step, uint64_t total) {
    auto *ctx = static_cast<CallbackCtx *>(user_data);
    if (ctx->on_progress) ctx->on_progress(step, total, ctx->progress_user_data);
}

// The task sent with every request: text generation when the engine supports it, otherwise the
// engine's lowest capability bit (e.g. AGENT_CAP_GENERIC for the onnx engine).
uint32_t engine_task(uint64_t caps) {
    if ((caps & AGENT_CAP_TEXT_GEN) || caps == 0) return AGENT_CAP_TEXT_GEN;
    return static_cast<uint32_t>(caps & (~caps + 1));
}

// An explicit task (InferRequest::task, Config::default_task, infer_task) must be one capability bit
// the engine declares.
Result<void> check_task(const EngineState &state, uint32_t task) {
    if (task == 0 || (task & (task - 1)) != 0)
        return fail(ErrorCode::InvalidConfig, "Engine task must be exactly one AGENT_CAP_* bit (got " + std::to_string(task) + ").");
    if ((state.caps & task) == 0)
        return fail(ErrorCode::UnsupportedProvider,
                    "Engine '" + state.name + "' does not declare the requested capability (task " + std::to_string(task) + ").");
    return ok();
}

Result<InferResponse> arena_full() {
    return fail<InferResponse>(ErrorCode::DecodeFailed, "Out of memory (arena growth failed).");
}

} // namespace

Result<void> init_engine(Session &session) {
    const Config &cfg = session.config;
    const std::string &spec = cfg.engine;
    if (spec.empty()) return fail(ErrorCode::InvalidConfig, "AiProvider::Engine requires Config::engine (engine name or plugin path).");

    // Which plugin file, and where it runs.
    const bool spec_is_path = looks_like_path(spec);
    std::string plugin_path = spec_is_path ? spec : std::string();
    EngineIsolation manifest_isolation = EngineIsolation::Default;
    if (!spec_is_path) engines::find_plugin_record(spec, plugin_path, manifest_isolation);
    const EngineIsolation isolation = cfg.engine_isolation != EngineIsolation::Default ? cfg.engine_isolation
                                      : manifest_isolation == EngineIsolation::Process ? EngineIsolation::Process
                                                                                       : EngineIsolation::InProcess;

    auto state = std::make_shared<EngineState>();
    state->output_dir = (std::filesystem::path(cfg.workspace_dir) / "artifacts").string();
    std::error_code ec;
    std::filesystem::create_directories(state->output_dir, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "Failed to create " + state->output_dir + ": " + ec.message());

    std::string config = "{\"model\":\"";
    utils::escape_json_string(cfg.model, config);
    config += "\",\"context_window\":" + std::to_string(cfg.context_window);
    config += ",\"max_tokens\":" + std::to_string(cfg.max_tokens);
    config += ",\"temperature\":" + std::to_string(cfg.temperature);
    config += ",\"parallel\":" + std::to_string(cfg.max_parallel_subagents);
    config += ",\"options\":";
    config += cfg.engine_options.empty() ? std::string("{}") : cfg.engine_options;
    config += '}';

    if (isolation == EngineIsolation::Process) {
        if (plugin_path.empty()) {
            if (find_engine(spec))
                return fail(ErrorCode::InvalidConfig, "Engine '" + spec +
                                                          "' is linked into this process (register_engine); process isolation needs a "
                                                          "plugin file. Use EngineIsolation::InProcess or a plugin path.");
            return fail(ErrorCode::UnsupportedProvider, "No engine named '" + spec + "' is registered.");
        }
        engines::RemoteOptions opts;
        opts.host_path = engines::resolve_engine_host_path(cfg.engine_host_path);
        opts.plugin_path = plugin_path;
        opts.create_config = std::move(config);
        opts.timeout_ms = cfg.engine_timeout_ms;
        opts.cancel_grace_ms = cfg.engine_cancel_grace_ms;
        opts.max_restarts = cfg.engine_max_restarts;
        opts.logger = cfg.logger;
        opts.logger_user_data = cfg.logger_user_data;
        opts.sandbox = sandbox::engine_policy(cfg.engine_sandbox, {.host_path = opts.host_path,
                                                                   .plugin_path = plugin_path,
                                                                   .model = cfg.model,
                                                                   .engine_options = cfg.engine_options,
                                                                   .workspace_dir = cfg.workspace_dir,
                                                                   .artifacts_dir = state->output_dir});
#ifdef AGENT_COVERAGE_RW_DIR
        // Coverage builds only: the host and its plugin write their .gcda files into the build tree when they exit.
        if (opts.sandbox.enabled) opts.sandbox.read_write_paths.push_back(AGENT_COVERAGE_RW_DIR);
#endif
        opts.env["MALLOC_ARENA_MAX"] = "2";                    // glibc: at most 2 heaps instead of 8 x cores
        opts.env["MALLOC_TRIM_THRESHOLD_"] = "131072";         // give freed memory back to the OS at 128 KiB
        opts.env["MALLOC_MMAP_THRESHOLD_"] = "262144";         // large blocks come straight from mmap and are returned on free
        if (opts.sandbox.enabled && cfg.engine_sandbox.gpu) {
            // Driver caches (compiled kernels / shaders) live under $HOME by default, which the sandbox hides;
            // without a writable folder every start recompiles. Untested on real GPUs: see docs/gpu.md.
            std::error_code gpu_ec;
            const std::filesystem::path cache = cfg.gpu_cache_dir.empty() ? std::filesystem::path(cfg.workspace_dir) / "cache" / "gpu"
                                                                          : std::filesystem::path(cfg.gpu_cache_dir);
            std::filesystem::create_directories(cache / "cuda", gpu_ec);
            std::filesystem::create_directories(cache / "mesa", gpu_ec);
            std::filesystem::create_directories(cache / "nv", gpu_ec);
            const std::string abs = std::filesystem::absolute(cache, gpu_ec).string();
            opts.sandbox.read_write_paths.push_back(abs);
            opts.env["CUDA_CACHE_PATH"] = abs + "/cuda";                // CUDA JIT (PTX -> SASS) cache
            opts.env["CUDA_CACHE_MAXSIZE"] = "1073741824";              // 1 GiB; the default is 256 MiB
            opts.env["MESA_SHADER_CACHE_DIR"] = abs + "/mesa";          // Mesa: radv, anv, llvmpipe, ...
            opts.env["MESA_SHADER_CACHE_MAX_SIZE"] = "1G";
            opts.env["__GL_SHADER_DISK_CACHE_PATH"] = abs + "/nv";      // NVIDIA OpenGL / Vulkan shader cache
            opts.env["__GL_SHADER_DISK_CACHE_SKIP_CLEANUP"] = "1";
        }
        auto remote = engines::start_remote_engine(std::move(opts));
        if (!remote.ok) return fail(remote.error.code, remote.error.message);
        state->remote = std::move(remote.value);
        const engines::RemoteInfo &info = engines::remote_info(*state->remote);
        state->name = info.name;
        state->caps = info.caps;
        state->flags = info.flags;
        if (cfg.default_task)
            if (auto t = check_task(*state, cfg.default_task); !t.ok) return fail(t.error.code, "Config::default_task: " + t.error.message);
        session.provider_state = state;
        utils::log(session, LogLevel::Info, "Engine '" + info.name + "' " + info.version + " loaded in agent-engine-host.");
        return ok();
    }

    const agent_engine_v1 *engine = spec_is_path ? nullptr : find_engine(spec);
    if (!engine && !plugin_path.empty()) {
        // A path, or a manifest entry marked "process" that this session runs in-process anyway.
        auto loaded = load_engine_plugin(plugin_path);
        if (!loaded.ok) return fail(loaded.error.code, loaded.error.message);
        engine = find_engine(loaded.value);
    }
    if (!engine) return fail(ErrorCode::UnsupportedProvider, "No engine named '" + spec + "' is registered.");

    state->engine = engine;
    state->name = engine->name;
    state->caps = engine->caps;
    state->flags = engine->flags;
    if (cfg.default_task)
        if (auto t = check_task(*state, cfg.default_task); !t.ok) return fail(t.error.code, "Config::default_task: " + t.error.message);

    char err[512] = {0};
    if (engine->create(to_c(config), &state->instance, err, sizeof(err)) != AGENT_OK || !state->instance) {
        state->instance = nullptr;
        return fail(ErrorCode::ModelLoadFailed, std::string("Engine '") + engine->name + "' failed to load: " + (err[0] ? err : "unknown error"));
    }

    session.provider_state = state;
    utils::log(session, LogLevel::Info, std::string("Engine '") + engine->name + "' " + (engine->version ? engine->version : "") + " loaded.");
    return ok();
}

namespace {

// Copies a response (engine-owned, or views into the remote engine's receive buffer) into arena.
// Text parts are concatenated into content; file parts become media; tensor parts become tensors.
Result<InferResponse> to_infer_response(const agent_response &resp, Arena &arena) {
    size_t text_bytes = 0, n_tensors = 0, n_media = 0;
    for (size_t i = 0; i < resp.message.n_parts; ++i) {
        const agent_part &pt = resp.message.parts[i];
        if (pt.kind == AGENT_PART_TEXT) text_bytes += pt.text.size;
        else if (pt.kind == AGENT_PART_FILE) ++n_media;
        else if (pt.kind == AGENT_PART_TENSOR) ++n_tensors;
    }

    MessageView out{};
    out.role = "assistant";
    char *text = static_cast<char *>(arena.allocate(text_bytes + 1, 1));
    auto tensors = arena.allocate_span<TensorView>(n_tensors);
    auto media = arena.allocate_span<MediaView>(n_media);
    auto calls = arena.allocate_span<ToolCallView>(resp.message.n_tool_calls);
    bool overflow = !text || tensors.size() != n_tensors || media.size() != n_media || calls.size() != resp.message.n_tool_calls;

    size_t off = 0, t = 0, m = 0;
    for (size_t i = 0; !overflow && i < resp.message.n_parts; ++i) {
        const agent_part &pt = resp.message.parts[i];
        if (pt.kind == AGENT_PART_TEXT) {
            if (pt.text.size) std::memcpy(text + off, pt.text.data, pt.text.size);
            off += pt.text.size;
        } else if (pt.kind == AGENT_PART_FILE) {
            media[m++] = MediaView{arena.allocate_string(from_c(pt.path)), arena.allocate_string(from_c(pt.mime))};
        } else if (pt.kind == AGENT_PART_TENSOR) {
            auto shape = arena.allocate_span<int64_t>(pt.ndim);
            auto *bytes = static_cast<uint8_t *>(arena.allocate(pt.size ? pt.size : 1, alignof(std::max_align_t)));
            if (shape.size() != pt.ndim || !bytes) {
                overflow = true;
                break;
            }
            if (pt.ndim) std::memcpy(shape.data(), pt.shape, pt.ndim * sizeof(int64_t));
            if (pt.size) std::memcpy(bytes, pt.data, pt.size);
            tensors[t++] = TensorView{.name = arena.allocate_string(from_c(pt.name)),
                                      .dtype = from_c_dtype(pt.dtype),
                                      .shape = shape,
                                      .data = std::span<const uint8_t>(bytes, pt.size)};
        }
    }
    for (size_t j = 0; !overflow && j < resp.message.n_tool_calls; ++j) {
        const agent_tool_call &c = resp.message.tool_calls[j];
        calls[j] = ToolCallView{arena.allocate_string(from_c(c.id)), arena.allocate_string(from_c(c.name)),
                                arena.allocate_string(from_c(c.arguments))};
    }
    const agent_usage usage = resp.usage;
    const agent_run_stats stats = AGENT_HAS_FIELD(&resp, agent_response, stats) ? resp.stats : agent_run_stats{};
    if (overflow) return arena_full();

    out.content = std::string_view(text, off);
    out.tool_calls = calls;
    out.tensors = tensors;
    out.media = media;
    return ok(InferResponse{.message = out,
                            .usage = TokenUsage{.prompt_tokens = usage.prompt_tokens,
                                                .completion_tokens = usage.completion_tokens,
                                                .total_tokens = usage.prompt_tokens + usage.completion_tokens,
                                                .cached_tokens = stats.cached_tokens,
                                                .kv_tokens = stats.kv_tokens,
                                                .kv_max_tokens = stats.kv_max_tokens,
                                                .kv_type = stats.kv_type,
                                                .kv_bytes = stats.kv_bytes,
                                                .prefill_ms = stats.prefill_ms,
                                                .decode_ms = stats.decode_ms},
                            .output_tensors = tensors});
}

// task == 0: automatic (engine_task()); otherwise an already checked capability bit. The request and
// response live in arena.
Result<InferResponse> run_engine(Session &session, uint32_t task, const InferRequest &request, Arena &arena) {
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || !state->ready()) return fail<InferResponse>(ErrorCode::ProviderInitFailed, "Engine not initialized.");
    const bool auto_task = task == 0;
    if (auto_task) task = engine_task(state->caps);

    // --- request: MessageView -> agent_message (arena-backed, zero-copy for strings/tensors) ---
    auto msgs = arena.allocate_span<agent_message>(request.messages.size());
    if (msgs.size() != request.messages.size()) return arena_full();
    for (size_t i = 0; i < request.messages.size(); ++i) {
        const MessageView &mv = request.messages[i];
        const size_t n_parts = (mv.content.empty() ? 0 : 1) + mv.media.size() + mv.tensors.size();
        auto parts = arena.allocate_span<agent_part>(n_parts);
        if (parts.size() != n_parts) return arena_full();
        size_t p = 0;
        if (!mv.content.empty()) parts[p++] = agent_part{.kind = AGENT_PART_TEXT, .text = to_c(mv.content)};
        for (const MediaView &media : mv.media)
            parts[p++] = agent_part{.kind = AGENT_PART_FILE, .path = to_c(media.path), .mime = to_c(media.mime)};
        for (const TensorView &tv : mv.tensors) {
            parts[p++] = agent_part{.kind = AGENT_PART_TENSOR,
                                    .name = to_c(tv.name),
                                    .dtype = to_c_dtype(tv.dtype),
                                    .shape = tv.shape.data(),
                                    .ndim = tv.shape.size(),
                                    .data = tv.data.data(),
                                    .size = tv.data.size()};
        }
        auto calls = arena.allocate_span<agent_tool_call>(mv.tool_calls.size());
        if (calls.size() != mv.tool_calls.size()) return arena_full();
        for (size_t j = 0; j < mv.tool_calls.size(); ++j)
            calls[j] = {to_c(mv.tool_calls[j].id), to_c(mv.tool_calls[j].name), to_c(mv.tool_calls[j].arguments)};

        msgs[i] = agent_message{.role = to_c(mv.role),
                                .parts = parts.data(),
                                .n_parts = n_parts,
                                .tool_calls = calls.data(),
                                .n_tool_calls = calls.size(),
                                .tool_call_id = to_c(mv.tool_call_id)};
    }

    const int max_tokens = request.max_tokens > 0 ? request.max_tokens : session.config.max_tokens;
    const float temperature = request.temperature >= 0.0f ? request.temperature : session.config.temperature;
    // InferRequest::params_json members go after the core's own (engines read the first occurrence of a key).
    auto trim = [](std::string_view v) {
        while (!v.empty() && std::isspace(static_cast<unsigned char>(v.front()))) v.remove_prefix(1);
        while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.remove_suffix(1);
        return v;
    };
    std::string_view extra = trim(request.params_json);
    if (!extra.empty()) {
        if (extra.size() < 2 || extra.front() != '{' || extra.back() != '}')
            return fail<InferResponse>(ErrorCode::InvalidConfig, "InferRequest::params_json must be a JSON object.");
        extra = trim(extra.substr(1, extra.size() - 2));
    }
    const size_t params_cap = 130 + extra.size();
    char *params = static_cast<char *>(arena.allocate(params_cap, 1));
    if (!params) return arena_full();
    int params_len = std::snprintf(params, 128, "{\"max_tokens\":%d,\"temperature\":%.4f,\"stream\":%s", max_tokens,
                                   static_cast<double>(temperature), request.stream ? "true" : "false");
    params_len = std::clamp(params_len, 0, 127);
    if (!extra.empty()) {
        params[params_len++] = ',';
        std::memcpy(params + params_len, extra.data(), extra.size());
        params_len += static_cast<int>(extra.size());
    }
    params[params_len++] = '}';

    CallbackCtx cb_ctx{request.on_token, request.token_user_data, request.on_progress, request.progress_user_data};
    agent_request req{.struct_size = sizeof(agent_request),
                      .task = task,
                      .flags = auto_task ? static_cast<uint32_t>(AGENT_REQUEST_FLAG_AUTO_TASK) : 0u,
                      .messages = msgs.data(),
                      .n_messages = msgs.size(),
                      // Tool definitions only matter for chat; embed / rerank / ... requests carry none.
                      .tools_json = to_c(task == AGENT_CAP_TEXT_GEN ? build_tools_json(session) : std::string_view()),
                      .params_json = agent_str{params, static_cast<size_t>(params_len)},
                      .output_dir = to_c(state->output_dir),
                      .on_chunk = (request.stream && request.on_token) ? forward_chunk : nullptr,
                      .on_progress = request.on_progress ? forward_progress : nullptr,
                      .user_data = &cb_ctx,
                      .cancel = request.cancel};

    // --- process isolation: the same request over IPC; the response views live in the call's buffers
    // until remote_release. Concurrent engines get several calls in flight through one host. ---
    if (state->remote) {
        agent_response resp{};
        resp.struct_size = sizeof(resp);
        engines::RemoteCall call;
        auto r = engines::remote_run(*state->remote, req, arena, resp, call);
        Result<InferResponse> out = r.ok ? to_infer_response(resp, arena) : fail<InferResponse>(r.error.code, r.error.message);
        engines::remote_release(*state->remote, call);
        return out;
    }

    // --- run, then copy the engine-owned response into the arena before freeing it ---
    const agent_engine_v1 *engine = state->engine;
    std::unique_lock<std::mutex> lk(state->run_mutex, std::defer_lock);
    if (!(engine->flags & AGENT_ENGINE_FLAG_CONCURRENT)) lk.lock();

    agent_response resp{};
    resp.struct_size = sizeof(resp);
    if (const int32_t rc = engine->run(state->instance, &req, &resp); rc != AGENT_OK) {
        const char *e = engine->last_error(state->instance);
        return fail<InferResponse>(rc == AGENT_ERR_CANCELLED ? ErrorCode::Cancelled : ErrorCode::DecodeFailed,
                                   std::string("Engine '") + engine->name + "': " + (e ? e : "run failed"));
    }
    auto out = to_infer_response(resp, arena);
    engine->free_response(state->instance, &resp);
    return out;
}

} // namespace

Result<InferResponse> infer_engine(Session &session, const InferRequest &request) {
    const uint32_t task = request.task ? request.task : session.config.default_task;
    if (task == 0) return run_engine(session, 0, request, ambient::current_arena(session));
    return infer_task(session, task, request, ambient::current_arena(session));
}

Result<InferResponse> infer_task(Session &session, uint32_t task, const InferRequest &request, Arena &arena) {
    if (session.config.provider != AiProvider::Engine)
        return fail<InferResponse>(ErrorCode::UnsupportedProvider,
                                   "This task needs an engine provider (Config::provider = AiProvider::Engine).");
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || !state->ready()) return fail<InferResponse>(ErrorCode::ProviderInitFailed, "Engine not initialized.");
    if (auto t = check_task(*state, task); !t.ok) return fail<InferResponse>(t.error.code, t.error.message);
    return run_engine(session, task, request, arena);
}

bool supports_parallel(const Session &session) {
    if (session.config.provider != AiProvider::Engine) return false;
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    // In process or out of process alike: the engine host runs concurrent engines' requests in parallel.
    return state && (state->flags & AGENT_ENGINE_FLAG_CONCURRENT);
}

std::string_view injected_system_text(const Session &session) {
    if (session.config.provider != AiProvider::Engine) return {};
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state) return {};
    const bool native = (state->flags & AGENT_ENGINE_FLAG_NATIVE_TOOLS) != 0;
    if (ambient::allowed_tools.empty() && !session.config.lazy_tools)
        return native ? std::string_view(session.skills_system_prompt) : std::string_view(session.tools_system_prompt);

    // A graph Agent node with allowed_tools: advertise only its tools (and the skills only when it
    // may call use_skill). Rebuilt per call into a reused buffer, like build_tools_json.
    const bool skills = !session.skills_system_prompt.empty() && tool_allowed("use_skill");
    if (native) return skills ? std::string_view(session.skills_system_prompt) : std::string_view();
    thread_local std::string buf;
    buf.clear();
    tools::append_tool_list(session, ambient::allowed_tools, buf);
    if (skills) {
        if (!buf.empty()) buf += "\n\n";
        buf += session.skills_system_prompt;
    }
    return buf;
}

bool accepts_tool_media(const Session &session, std::string_view mime) {
    if (session.config.provider != AiProvider::Engine) return true;
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || state->tool_media_rejected.load(std::memory_order_relaxed)) return false;
    if (mime.starts_with("image/")) return (state->caps & AGENT_CAP_VISION_IN) != 0;
    if (mime.starts_with("audio/")) return (state->caps & AGENT_CAP_AUDIO_IN) != 0;
    if (mime.starts_with("video/")) return (state->caps & AGENT_CAP_VIDEO_IN) != 0;
    return false;
}

void reject_tool_media(Session &session) {
    if (session.config.provider != AiProvider::Engine) return;
    if (auto *state = static_cast<EngineState *>(session.provider_state.get()))
        state->tool_media_rejected.store(true, std::memory_order_relaxed);
}

int engine_host_starts(const Session &session) {
    if (session.config.provider != AiProvider::Engine) return -1;
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || !state->remote) return -1;
    return engines::remote_stats(*state->remote).starts;
}

EngineHostStats engine_host_stats(const Session &session) {
    if (session.config.provider != AiProvider::Engine) return {};
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || !state->remote) return {};
    const engines::RemoteStats s = engines::remote_stats(*state->remote);
    return EngineHostStats{.starts = s.starts,
                           .shm_requests = s.shm_requests,
                           .shm_responses = s.shm_responses,
                           .max_in_flight = s.max_in_flight,
                           .shm = engines::remote_info(*state->remote).shm};
}

Result<int> count_tokens_engine(Session &session, std::string_view text) {
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (state && state->remote) return engines::remote_count_tokens(*state->remote, text);
    if (!state || !state->engine || !state->engine->count_tokens)
        return fail<int>(ErrorCode::UnsupportedProvider, "Engine does not count tokens.");
    int32_t n = 0;
    if (state->engine->count_tokens(state->instance, to_c(text), &n) != AGENT_OK)
        return fail<int>(ErrorCode::DecodeFailed, "Engine token count failed.");
    return ok(static_cast<int>(n));
}

Result<void> clear_kv_cache_engine(Session &session) {
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (state && state->remote) return engines::remote_reset_cache(*state->remote);
    if (!state || !state->engine || !state->engine->reset_cache) return ok();
    if (state->engine->reset_cache(state->instance) != AGENT_OK) return fail(ErrorCode::DecodeFailed, "Engine cache reset failed.");
    return ok();
}

} // namespace agent::providers
