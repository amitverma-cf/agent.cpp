#include "hal/log_file.hpp"
#include "hal/platform.hpp"
#include "memory/memory_store.hpp"
#include "providers/provider_ops.hpp"
#include "tools/tools_ops.hpp"
#include "utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <system_error>
#include <vector>

namespace agent {

void init_backend() { providers::init_global_backends(); }
void free_backend() { providers::free_global_backends(); }

AgentRuntime::AgentRuntime() { init_backend(); }
AgentRuntime::~AgentRuntime() { free_backend(); }

Result<Config> engine_config(std::string_view engine, std::span<const ModelSlot> models, std::string_view extra_options_json,
                             const Config &base) {
    if (engine.empty()) return fail<Config>(ErrorCode::InvalidConfig, "engine_config: the engine name or plugin path is required.");
    Config cfg = base;
    cfg.provider = AiProvider::Engine;
    cfg.engine = std::string(engine);

    // Members of the options object, in the order given; "section.key" roles are grouped into one nested object per section.
    struct Section {
        std::string name;
        std::string members; // already-escaped JSON members, comma separated
    };
    std::vector<Section> sections;
    std::string top;
    auto member = [](std::string_view key, std::string_view value) {
        std::string m = "\"";
        utils::escape_json_string(key, m);
        m += "\":\"";
        utils::escape_json_string(value, m);
        m += '"';
        return m;
    };
    auto add = [](std::string &list, const std::string &m) {
        if (!list.empty()) list += ',';
        list += m;
    };
    for (const ModelSlot &slot : models) {
        if (slot.role.empty() || slot.path.empty())
            return fail<Config>(ErrorCode::InvalidConfig, "engine_config: every model needs a role and a path.");
        if (slot.role == "model") {
            cfg.model = std::string(slot.path);
            continue;
        }
        const size_t dot = slot.role.find('.');
        if (dot == std::string_view::npos) {
            add(top, member(slot.role, slot.path));
            continue;
        }
        const std::string_view name = slot.role.substr(0, dot);
        auto it = std::find_if(sections.begin(), sections.end(), [&](const Section &s) { return s.name == name; });
        if (it == sections.end()) it = sections.insert(sections.end(), Section{std::string(name), {}});
        add(it->members, member(slot.role.substr(dot + 1), slot.path));
    }
    for (const Section &s : sections) {
        std::string obj = "\"";
        utils::escape_json_string(s.name, obj);
        obj += "\":{" + s.members + "}";
        add(top, obj);
    }
    if (!extra_options_json.empty()) {
        size_t b = extra_options_json.find_first_not_of(" \t\r\n"), e = extra_options_json.find_last_not_of(" \t\r\n");
        if (b == std::string_view::npos || extra_options_json[b] != '{' || extra_options_json[e] != '}')
            return fail<Config>(ErrorCode::InvalidConfig, "engine_config: extra_options_json must be a JSON object.");
        std::string_view inner = extra_options_json.substr(b + 1, e - b - 1);
        const size_t ib = inner.find_first_not_of(" \t\r\n");
        if (ib != std::string_view::npos) add(top, std::string(inner.substr(ib)));
    }
    cfg.engine_options = top.empty() && base.engine_options.empty() ? std::string() : "{" + top + "}";
    if (!top.empty() && !base.engine_options.empty())
        return fail<Config>(ErrorCode::InvalidConfig,
                            "engine_config: base.engine_options is already set; pass the options through engine_config instead.");
    if (top.empty()) cfg.engine_options = base.engine_options;
    return ok(std::move(cfg));
}

Result<Session> open_session(std::string_view engine, std::string_view model_path, const Config &base) {
    Config seed = base;
    if (seed.workspace_dir.empty()) seed.workspace_dir = ".agent";
    const ModelSlot slot[] = {{"model", model_path}};
    auto cfg = engine_config(engine, slot, {}, seed);
    if (!cfg.ok) return fail<Session>(cfg.error.code, cfg.error.message);
    return init(cfg.value);
}

Result<Session> open_session(std::string_view engine, std::initializer_list<ModelSlot> models, const Config &base) {
    Config seed = base;
    if (seed.workspace_dir.empty()) seed.workspace_dir = ".agent";
    auto cfg = engine_config(engine, std::span<const ModelSlot>(models.begin(), models.size()), {}, seed);
    if (!cfg.ok) return fail<Session>(cfg.error.code, cfg.error.message);
    return init(cfg.value);
}

Result<TaskOutput> perform(Session &session, uint32_t task, std::string_view text, std::string_view media_path,
                           std::string_view params_json) {
    MessageView message{.role = "user", .content = text};
    MediaView media[1];
    if (!media_path.empty()) {
        media[0] = MediaView{media_path, ""};
        message.media = std::span<const MediaView>(media, 1);
    }
    InferRequest request;
    request.messages = std::span<const MessageView>(&message, 1);
    request.task = task;
    request.params_json = params_json;
    auto r = run_task(session, task, request);
    if (!r.ok) return fail<TaskOutput>(r.error.code, r.error.message);
    TaskOutput out;
    out.text = std::string(r.value.message.content);
    for (const MediaView &m : r.value.message.media) out.files.emplace_back(m.path);
    return ok(std::move(out));
}

Result<TaskOutput> transcribe(Session &session, std::string_view audio_path) { return perform(session, AGENT_CAP_ASR, {}, audio_path); }
Result<TaskOutput> speak(Session &session, std::string_view text) { return perform(session, AGENT_CAP_TTS, text); }
Result<TaskOutput> draw(Session &session, std::string_view prompt, std::string_view params_json) {
    return perform(session, AGENT_CAP_IMAGE_GEN, prompt, {}, params_json);
}

Result<Session> init(const Config &config) {
    if (config.workspace_dir.empty()) return fail<Session>(ErrorCode::InvalidConfig, "workspace_dir is required.");

    if (config.context_window <= 0) return fail<Session>(ErrorCode::InvalidConfig, "context_window must be > 0.");
    if (config.max_tokens <= 0) return fail<Session>(ErrorCode::InvalidConfig, "max_tokens must be > 0.");
    if (!std::isfinite(config.temperature) || config.temperature < 0.0f)
        return fail<Session>(ErrorCode::InvalidConfig, "temperature must be a non-negative finite number.");

    for (char c : config.workspace_dir) {
        if (c == '"' || c == '\'' || c == '&' || c == '|' || c == ';' || c == '`' || c == '$' || c == '\n' || c == '\r') {
            return fail<Session>(ErrorCode::InvalidConfig, "workspace_dir contains shell metacharacters. "
                                                           "Use a plain filesystem path without quotes, $, &, |, or ;.");
        }
    }

    std::error_code ec;
    for (const char *sub : {"models", "memory", "logs", "tmp"}) {
        auto p = std::filesystem::path(config.workspace_dir) / sub;
        std::filesystem::create_directories(p, ec);
        if (ec) return fail<Session>(ErrorCode::FilesystemError, "Failed to create " + p.string() + ": " + ec.message());
    }

    Session session;
    session.config = config;

    const size_t arena_size =
        config.arena_capacity > 0 ? config.arena_capacity : static_cast<size_t>(config.context_window) * 8 + (2UL * 1024 * 1024);
    session.arena = Arena(arena_size);

    session.workspace_path = std::make_shared<std::string>(config.workspace_dir);

    if (config.enable_file_logging) {
        session.log_file = hal::open_session_log(config.workspace_dir);
        if (!session.log_file) {
            return fail<Session>(ErrorCode::FilesystemError, "enable_file_logging is set but the log file could not be opened "
                                                             "under workspace logs/.");
        }
    }

    const providers::ProviderOps *ops = providers::find_provider_ops(config.provider);
    if (!ops || !ops->init || !ops->infer) return fail<Session>(ErrorCode::UnsupportedProvider, "Provider not compiled into this build.");

    auto init_res = ops->init(session);
    if (!init_res.ok) return fail<Session>(init_res.error.code, init_res.error.message);

    if (providers::supports_parallel(session) && config.max_parallel_subagents > 0) {
        session.worker_arenas.reserve(static_cast<size_t>(config.max_parallel_subagents));
        for (int i = 0; i < config.max_parallel_subagents; ++i) session.worker_arenas.emplace_back(arena_size);
    }

    auto mem_res = init_memory(session);
    if (!mem_res.ok) return fail<Session>(mem_res.error.code, mem_res.error.message);

    auto tools_res = init_tools(session);
    if (!tools_res.ok) return fail<Session>(tools_res.error.code, tools_res.error.message);

    utils::log(session, LogLevel::Info,
               std::string("Session initialised. platform=") + hal::platform_name() + " workspace=" + config.workspace_dir);

    return ok(std::move(session));
}

namespace {

// Copies a response whose views live in the session's per-call arena into memory the response owns.
InferResponse make_owning(const InferResponse &src) {
    auto arena = std::make_shared<Arena>(4096);
    InferResponse out;
    out.usage = src.usage;
    out.message.role = arena->allocate_string(src.message.role);
    out.message.content = arena->allocate_string(src.message.content);
    out.message.tool_call_id = arena->allocate_string(src.message.tool_call_id);
    auto calls = arena->allocate_span<ToolCallView>(src.message.tool_calls.size());
    for (size_t i = 0; i < calls.size(); ++i)
        calls[i] =
            ToolCallView{arena->allocate_string(src.message.tool_calls[i].id), arena->allocate_string(src.message.tool_calls[i].name),
                         arena->allocate_string(src.message.tool_calls[i].arguments)};
    out.message.tool_calls = calls;
    auto media = arena->allocate_span<MediaView>(src.message.media.size());
    for (size_t i = 0; i < media.size(); ++i)
        media[i] = MediaView{arena->allocate_string(src.message.media[i].path), arena->allocate_string(src.message.media[i].mime)};
    out.message.media = media;
    auto copy_tensors = [&](std::span<const TensorView> in) {
        auto tv = arena->allocate_span<TensorView>(in.size());
        for (size_t i = 0; i < tv.size(); ++i) {
            auto shape = arena->allocate_span<int64_t>(in[i].shape.size());
            std::copy(in[i].shape.begin(), in[i].shape.end(), shape.begin());
            auto *bytes = static_cast<uint8_t *>(arena->allocate(in[i].data.size() ? in[i].data.size() : 1, alignof(std::max_align_t)));
            if (!in[i].data.empty()) std::copy(in[i].data.begin(), in[i].data.end(), bytes);
            tv[i] = TensorView{arena->allocate_string(in[i].name), in[i].dtype, shape, std::span<const uint8_t>(bytes, in[i].data.size())};
        }
        return std::span<const TensorView>(tv);
    };
    out.message.tensors = copy_tensors(src.message.tensors);
    out.output_tensors = copy_tensors(src.output_tensors);
    out.storage = std::move(arena);
    return out;
}

} // namespace

Result<InferResponse> infer(Session &session, const InferRequest &request) {
    utils::log(session, LogLevel::Debug, "infer: submitting to provider.");
    const providers::ProviderOps *ops = providers::find_provider_ops(session.config.provider);
    if (!ops || !ops->infer) return fail<InferResponse>(ErrorCode::UnsupportedProvider, "Provider not compiled into this build.");
    auto res = ops->infer(session, request);
    if (!res.ok) return res;
    return ok(make_owning(res.value));
}

Result<InferResponse> run_task(Session &session, uint32_t task, const InferRequest &request) {
    if (task == 0) return fail<InferResponse>(ErrorCode::InvalidConfig, "run_task: task must be an AGENT_CAP_* bit, not 0.");
    InferRequest req = request;
    req.task = task;
    return infer(session, req);
}

Result<void> init_memory(Session &session) {
    const StoreOps *ops = memory::active_store(session);
    if (!ops->open || !ops->put || !ops->get || !ops->clear)
        return fail(ErrorCode::InvalidConfig, "MemoryConfig::store is missing one of open/put/get/clear.");
    auto res = ops->open(session.config.memory, session.config.workspace_dir);
    if (!res.ok) return fail(res.error.code, res.error.message);
    session.memory_state = std::move(res.value);
    return ok();
}

Result<void> store_memory(Session &session, std::string_view key, std::string_view value) {
    if (!session.memory_state) return fail(ErrorCode::ProviderInitFailed, "Memory store not initialized.");
    return memory::active_store(session)->put(session.memory_state.get(), key, value);
}

Result<std::string> retrieve_memory(Session &session, std::string_view key) {
    if (!session.memory_state) return fail<std::string>(ErrorCode::ProviderInitFailed, "Memory store not initialized.");
    return memory::active_store(session)->get(session.memory_state.get(), key);
}

Result<void> clear_memory(Session &session) {
    if (!session.memory_state) return fail(ErrorCode::ProviderInitFailed, "Memory store not initialized.");
    return memory::active_store(session)->clear(session.memory_state.get());
}

Result<void> init_tools(Session &session) {
    const tools::ToolsOps *ops = tools::find_tools_ops();
    if (!ops || !ops->init) return fail(ErrorCode::UnsupportedProvider, "Tools registry not compiled into this build.");
    return ops->init(session);
}

Result<ToolResult> execute_tool_parts(Session &session, std::string_view name, std::string_view arguments) {
    const tools::ToolsOps *ops = tools::find_tools_ops();
    if (!ops || !ops->execute) return fail<ToolResult>(ErrorCode::UnsupportedProvider, "Tools registry not compiled into this build.");
    return ops->execute(session, name, arguments);
}

Result<std::string> execute_tool(Session &session, std::string_view name, std::string_view arguments) {
    auto res = execute_tool_parts(session, name, arguments);
    if (!res.ok) return fail<std::string>(res.error.code, res.error.message);
    std::string text = tools::render_tool_text(res.value);
    if (res.value.is_error) return fail<std::string>(ErrorCode::ToolError, text.empty() ? std::string("tool reported an error") : text);
    return ok(std::move(text));
}

void rebuild_tool_index(Session &session) {
    session.tool_by_name.clear();
    session.tool_by_name.reserve(session.config.tools.size());
    for (size_t i = 0; i < session.config.tools.size(); ++i) session.tool_by_name.emplace(std::string(session.config.tools[i].name), i);

    session.tools_system_prompt.clear();
    session.skills_system_prompt.clear();
    tools::append_tool_list(session, {}, session.tools_system_prompt);

    if (!session.skills.empty()) {
        session.skills_system_prompt += "You have access to the following skills. Call the "
                                        "use_skill tool with a skill's name to load its full "
                                        "instructions when relevant:\n";

        // session.skills is an unordered_map (for O(1) lookup in read_skill_body); iterate in a
        // sorted, deterministic order here so the system prompt text -- and therefore any
        // provider-side prompt-prefix caching -- doesn't shuffle between runs for the same set
        // of skills on disk.
        std::vector<const std::string *> names;
        names.reserve(session.skills.size());
        for (const auto &[name, skill] : session.skills) names.push_back(&name);
        std::sort(names.begin(), names.end(), [](const std::string *a, const std::string *b) { return *a < *b; });

        for (const std::string *name : names) {
            const Skill &skill = session.skills.at(*name);
            session.skills_system_prompt += "- ";
            session.skills_system_prompt += *name;
            session.skills_system_prompt += ": ";
            session.skills_system_prompt += skill.description;
            session.skills_system_prompt += "\n";
        }
    }
    if (!session.skills_system_prompt.empty()) {
        if (!session.tools_system_prompt.empty()) session.tools_system_prompt += "\n\n";
        session.tools_system_prompt += session.skills_system_prompt;
    }
}

SessionStats get_stats(const Session &session) {
    std::lock_guard<std::mutex> lock(*session.stats_mutex);
    return session.stats;
}

void reset_stats(Session &session) {
    std::lock_guard<std::mutex> lock(*session.stats_mutex);
    session.stats = {};
}

Result<void> clear_provider_kv_cache(Session &session) {
    const providers::ProviderOps *ops = providers::find_provider_ops(session.config.provider);
    if (!ops) return fail(ErrorCode::UnsupportedProvider, "Provider not compiled into this build.");
    if (!ops->clear_kv_cache) return ok();
    return ops->clear_kv_cache(session);
}

} // namespace agent
