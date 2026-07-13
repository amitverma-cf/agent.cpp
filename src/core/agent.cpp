#include "memory/memory_ops.hpp"
#include "providers/provider_ops.hpp"
#include "tools/tools_ops.hpp"
#include "utils/utils.hpp"

#include "hal/log_file.hpp"
#include "hal/platform.hpp"

#include <agent-cpp/agent.hpp>
#include <cmath>
#include <filesystem>
#include <system_error>

namespace agent {

void init_backend() { providers::init_global_backends(); }
void free_backend() { providers::free_global_backends(); }

AgentRuntime::AgentRuntime() { init_backend(); }
AgentRuntime::~AgentRuntime() { free_backend(); }

Result<Session> init(const Config &config) {
    if (config.workspace_dir.empty())
        return fail<Session>(ErrorCode::InvalidConfig, "workspace_dir is required.");

    if (config.context_window <= 0)
        return fail<Session>(ErrorCode::InvalidConfig, "context_window must be > 0.");
    if (config.max_tokens <= 0)
        return fail<Session>(ErrorCode::InvalidConfig, "max_tokens must be > 0.");
    if (!std::isfinite(config.temperature) || config.temperature < 0.0f)
        return fail<Session>(ErrorCode::InvalidConfig,
                             "temperature must be a non-negative finite number.");

    for (char c : config.workspace_dir) {
        if (c == '"' || c == '\'' || c == '&' || c == '|' || c == ';' || c == '`' ||
            c == '$' || c == '\n' || c == '\r') {
            return fail<Session>(ErrorCode::InvalidConfig,
                                 "workspace_dir contains shell metacharacters. "
                                 "Use a plain filesystem path without quotes, $, &, |, or ;.");
        }
    }

    std::error_code ec;
    for (const char *sub : {"models", "memory", "logs", "tmp"}) {
        auto p = std::filesystem::path(config.workspace_dir) / sub;
        std::filesystem::create_directories(p, ec);
        if (ec)
            return fail<Session>(ErrorCode::FilesystemError,
                                 "Failed to create " + p.string() + ": " + ec.message());
    }

    Session session;
    session.config = config;

    const size_t arena_size =
        config.arena_capacity > 0
            ? config.arena_capacity
            : static_cast<size_t>(config.context_window) * 8 + (2UL * 1024 * 1024);
    session.arena = Arena(arena_size);

    const bool provider_can_run_parallel =
        config.provider == AiProvider::OpenAICompatible || config.provider == AiProvider::LlamaCpp;
    if (provider_can_run_parallel && config.max_parallel_subagents > 0) {
        session.worker_arenas.reserve(static_cast<size_t>(config.max_parallel_subagents));
        for (int i = 0; i < config.max_parallel_subagents; ++i)
            session.worker_arenas.emplace_back(arena_size);
    }

    session.workspace_path = std::make_shared<std::string>(config.workspace_dir);

    if (config.enable_file_logging) {
        session.log_file = hal::open_session_log(config.workspace_dir);
        if (!session.log_file) {
            return fail<Session>(ErrorCode::FilesystemError,
                                 "enable_file_logging is set but the log file could not be opened "
                                 "under workspace logs/.");
        }
    }

    const providers::ProviderOps *ops = providers::find_provider_ops(config.provider);
    if (!ops || !ops->init || !ops->infer)
        return fail<Session>(ErrorCode::UnsupportedProvider,
                             "Provider not compiled into this build.");

    auto init_res = ops->init(session);
    if (!init_res.ok)
        return fail<Session>(init_res.error.code, init_res.error.message);

    if (session.config.memory.db_path == "memory/db.sqlite3") {
        session.config.memory.db_path =
            (std::filesystem::path(config.workspace_dir) / "memory" / "db.sqlite3").string();
    }

    auto mem_res = init_memory(session);
    if (!mem_res.ok)
        return fail<Session>(mem_res.error.code, mem_res.error.message);

    auto tools_res = init_tools(session);
    if (!tools_res.ok)
        return fail<Session>(tools_res.error.code, tools_res.error.message);

    utils::log(session, LogLevel::Info,
               std::string("Session initialised. platform=") + hal::platform_name() +
                   " workspace=" + config.workspace_dir);

    return ok(std::move(session));
}

Result<InferResponse> infer(Session &session, const InferRequest &request) {
    utils::log(session, LogLevel::Debug, "infer: submitting to provider.");
    const providers::ProviderOps *ops = providers::find_provider_ops(session.config.provider);
    if (!ops || !ops->infer)
        return fail<InferResponse>(ErrorCode::UnsupportedProvider,
                                   "Provider not compiled into this build.");
    return ops->infer(session, request);
}

Result<void> init_memory(Session &session) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->init)
        return fail(ErrorCode::UnsupportedProvider,
                    "Memory provider not compiled into this build.");
    return ops->init(session);
}

Result<void> store_memory(Session &session, std::string_view key, std::string_view value) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->store)
        return fail(ErrorCode::UnsupportedProvider,
                    "Memory provider not compiled into this build.");
    return ops->store(session, key, value);
}

Result<std::string> retrieve_memory(Session &session, std::string_view key) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->retrieve)
        return fail<std::string>(ErrorCode::UnsupportedProvider,
                                 "Memory provider not compiled into this build.");
    return ops->retrieve(session, key);
}

Result<void> clear_memory(Session &session) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->clear)
        return fail(ErrorCode::UnsupportedProvider,
                    "Memory provider not compiled into this build.");
    return ops->clear(session);
}

Result<void> init_tools(Session &session) {
    const tools::ToolsOps *ops = tools::find_tools_ops();
    if (!ops || !ops->init)
        return fail(ErrorCode::UnsupportedProvider,
                    "Tools registry not compiled into this build.");
    return ops->init(session);
}

Result<std::string> execute_tool(Session &session, std::string_view name,
                                 std::string_view arguments) {
    const tools::ToolsOps *ops = tools::find_tools_ops();
    if (!ops || !ops->execute)
        return fail<std::string>(ErrorCode::UnsupportedProvider,
                                 "Tools registry not compiled into this build.");
    return ops->execute(session, name, arguments);
}

void rebuild_tool_index(Session &session) {
    session.tool_by_name.clear();
    session.tool_by_name.reserve(session.config.tools.size());
    for (size_t i = 0; i < session.config.tools.size(); ++i)
        session.tool_by_name.emplace(std::string(session.config.tools[i].name), i);

    session.tools_system_prompt.clear();
    if (!session.config.tools.empty()) {
        session.tools_system_prompt += "You have access to the following tools:\n";
        for (const auto &tool : session.config.tools) {
            session.tools_system_prompt += "- ";
            session.tools_system_prompt += tool.name;
            session.tools_system_prompt += ": ";
            session.tools_system_prompt += tool.description;
            session.tools_system_prompt += "\n  Schema: ";
            session.tools_system_prompt += tool.parameter_schema;
            session.tools_system_prompt += "\n";
        }
        session.tools_system_prompt +=
            "When you need a tool, respond with ONLY a single JSON object of "
            "the exact form {\"name\": \"tool_name\", \"arguments\": {...}} and "
            "stop. Wait for the tool result before continuing.";
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
    if (!ops)
        return fail(ErrorCode::UnsupportedProvider, "Provider not compiled into this build.");
    if (!ops->clear_kv_cache)
        return ok();
    return ops->clear_kv_cache(session);
}

} // namespace agent
