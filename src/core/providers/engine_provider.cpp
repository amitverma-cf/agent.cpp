#include "../executor/ambient_turn.hpp"
#include "../utils/utils.hpp"
#include "provider_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

namespace agent::providers {

namespace {

struct EngineState {
    const agent_engine_v1 *engine = nullptr;
    void *instance = nullptr;
    std::mutex run_mutex; // serializes run() unless the engine sets AGENT_ENGINE_FLAG_CONCURRENT
    std::string output_dir;

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
        if (!tool_allowed(t.name)) continue;
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

struct StreamCtx {
    TokenStreamFn on_token;
    void *user_data;
};

void forward_chunk(void *user_data, uint32_t kind, const void *data, size_t size) {
    auto *ctx = static_cast<StreamCtx *>(user_data);
    if (kind == AGENT_CHUNK_TEXT && ctx->on_token) ctx->on_token(std::string_view(static_cast<const char *>(data), size), ctx->user_data);
}

Result<InferResponse> arena_full() {
    return fail<InferResponse>(ErrorCode::DecodeFailed, "Arena capacity exhausted. Increase Config::arena_capacity.");
}

} // namespace

Result<void> init_engine(Session &session) {
    const std::string &spec = session.config.engine;
    if (spec.empty()) return fail(ErrorCode::InvalidConfig, "AiProvider::Engine requires Config::engine (engine name or plugin path).");

    const agent_engine_v1 *engine = nullptr;
    if (looks_like_path(spec)) {
        auto loaded = load_engine_plugin(spec);
        if (!loaded.ok) return fail(loaded.error.code, loaded.error.message);
        engine = find_engine(loaded.value);
    } else {
        engine = find_engine(spec);
        if (!engine) return fail(ErrorCode::UnsupportedProvider, "No engine named '" + spec + "' is registered.");
    }

    auto state = std::make_shared<EngineState>();
    state->engine = engine;
    state->output_dir = (std::filesystem::path(session.config.workspace_dir) / "artifacts").string();
    std::error_code ec;
    std::filesystem::create_directories(state->output_dir, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "Failed to create " + state->output_dir + ": " + ec.message());

    std::string config = "{\"model\":\"";
    utils::escape_json_string(session.config.model, config);
    config += "\",\"context_window\":" + std::to_string(session.config.context_window);
    config += ",\"max_tokens\":" + std::to_string(session.config.max_tokens);
    config += ",\"temperature\":" + std::to_string(session.config.temperature);
    config += ",\"parallel\":" + std::to_string(session.config.max_parallel_subagents);
    config += ",\"options\":";
    config += session.config.engine_options.empty() ? std::string("{}") : session.config.engine_options;
    config += '}';

    char err[512] = {0};
    if (engine->create(to_c(config), &state->instance, err, sizeof(err)) != AGENT_OK || !state->instance) {
        state->instance = nullptr;
        return fail(ErrorCode::ModelLoadFailed, std::string("Engine '") + engine->name + "' failed to load: " + (err[0] ? err : "unknown error"));
    }

    session.provider_state = state;
    utils::log(session, LogLevel::Info, std::string("Engine '") + engine->name + "' " + (engine->version ? engine->version : "") + " loaded.");
    return ok();
}

Result<InferResponse> infer_engine(Session &session, const InferRequest &request) {
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || !state->instance) return fail<InferResponse>(ErrorCode::ProviderInitFailed, "Engine not initialized.");
    const agent_engine_v1 *engine = state->engine;
    Arena &arena = ambient::current_arena(session);

    // --- request: MessageView -> agent_message (arena-backed, zero-copy for strings/tensors) ---
    auto msgs = arena.allocate_span<agent_message>(request.messages.size());
    if (msgs.size() != request.messages.size()) return arena_full();
    for (size_t i = 0; i < request.messages.size(); ++i) {
        const MessageView &mv = request.messages[i];
        const size_t n_parts = (mv.content.empty() ? 0 : 1) + mv.tensors.size();
        auto parts = arena.allocate_span<agent_part>(n_parts);
        if (parts.size() != n_parts) return arena_full();
        size_t p = 0;
        if (!mv.content.empty()) parts[p++] = agent_part{.kind = AGENT_PART_TEXT, .text = to_c(mv.content)};
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
    char *params = static_cast<char *>(arena.allocate(96, 1));
    if (!params) return arena_full();
    const int params_len = std::snprintf(params, 96, "{\"max_tokens\":%d,\"temperature\":%.4f,\"stream\":%s}", max_tokens,
                                         static_cast<double>(temperature), request.stream ? "true" : "false");

    StreamCtx stream_ctx{request.on_token, request.token_user_data};
    agent_request req{.task = AGENT_CAP_TEXT_GEN,
                      .messages = msgs.data(),
                      .n_messages = msgs.size(),
                      .tools_json = to_c(build_tools_json(session)),
                      .params_json = agent_str{params, static_cast<size_t>(params_len)},
                      .output_dir = to_c(state->output_dir),
                      .on_chunk = (request.stream && request.on_token) ? forward_chunk : nullptr,
                      .on_progress = nullptr,
                      .user_data = &stream_ctx,
                      .cancel = nullptr};

    // --- run, then copy the engine-owned response into the arena before freeing it ---
    std::unique_lock<std::mutex> lk(state->run_mutex, std::defer_lock);
    if (!(engine->flags & AGENT_ENGINE_FLAG_CONCURRENT)) lk.lock();

    agent_response resp{};
    if (engine->run(state->instance, &req, &resp) != AGENT_OK) {
        const char *e = engine->last_error(state->instance);
        return fail<InferResponse>(ErrorCode::DecodeFailed, std::string("Engine '") + engine->name + "': " + (e ? e : "run failed"));
    }

    // Text parts are concatenated into content; file parts become "[artifact] <mime> <path>" lines
    // until the message model carries media natively.
    size_t text_bytes = 0, n_tensors = 0;
    for (size_t i = 0; i < resp.message.n_parts; ++i) {
        const agent_part &pt = resp.message.parts[i];
        if (pt.kind == AGENT_PART_TEXT) text_bytes += pt.text.size;
        else if (pt.kind == AGENT_PART_FILE) text_bytes += pt.mime.size + pt.path.size + 14;
        else if (pt.kind == AGENT_PART_TENSOR) ++n_tensors;
    }

    MessageView out{};
    out.role = "assistant";
    char *text = static_cast<char *>(arena.allocate(text_bytes + 1, 1));
    auto tensors = arena.allocate_span<TensorView>(n_tensors);
    auto calls = arena.allocate_span<ToolCallView>(resp.message.n_tool_calls);
    bool overflow = !text || tensors.size() != n_tensors || calls.size() != resp.message.n_tool_calls;

    size_t off = 0, t = 0;
    for (size_t i = 0; !overflow && i < resp.message.n_parts; ++i) {
        const agent_part &pt = resp.message.parts[i];
        if (pt.kind == AGENT_PART_TEXT) {
            if (pt.text.size) std::memcpy(text + off, pt.text.data, pt.text.size);
            off += pt.text.size;
        } else if (pt.kind == AGENT_PART_FILE) {
            off += static_cast<size_t>(std::snprintf(text + off, text_bytes + 1 - off, "%s[artifact] %.*s %.*s", off ? "\n" : "",
                                                     static_cast<int>(pt.mime.size), pt.mime.data ? pt.mime.data : "",
                                                     static_cast<int>(pt.path.size), pt.path.data ? pt.path.data : ""));
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
    engine->free_response(state->instance, &resp);
    lk = {};
    if (overflow) return arena_full();

    out.content = std::string_view(text, off);
    out.tool_calls = calls;
    out.tensors = tensors;
    return ok(InferResponse{.message = out,
                            .usage = TokenUsage{.prompt_tokens = usage.prompt_tokens,
                                                .completion_tokens = usage.completion_tokens,
                                                .total_tokens = usage.prompt_tokens + usage.completion_tokens},
                            .output_tensors = tensors});
}

bool supports_parallel(const Session &session) {
    if (session.config.provider != AiProvider::Engine) return false;
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    return state && (state->engine->flags & AGENT_ENGINE_FLAG_CONCURRENT);
}

bool needs_text_tools(const Session &session) {
    if (session.config.provider != AiProvider::Engine) return false;
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    return state && !(state->engine->flags & AGENT_ENGINE_FLAG_NATIVE_TOOLS);
}

Result<int> count_tokens_engine(Session &session, std::string_view text) {
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || !state->engine->count_tokens) return fail<int>(ErrorCode::UnsupportedProvider, "Engine does not count tokens.");
    int32_t n = 0;
    if (state->engine->count_tokens(state->instance, to_c(text), &n) != AGENT_OK)
        return fail<int>(ErrorCode::DecodeFailed, "Engine token count failed.");
    return ok(static_cast<int>(n));
}

Result<void> clear_kv_cache_engine(Session &session) {
    auto *state = static_cast<EngineState *>(session.provider_state.get());
    if (!state || !state->engine->reset_cache) return ok();
    if (state->engine->reset_cache(state->instance) != AGENT_OK) return fail(ErrorCode::DecodeFailed, "Engine cache reset failed.");
    return ok();
}

} // namespace agent::providers
