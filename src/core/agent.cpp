#include "memory/memory_ops.hpp"
#include "providers/provider_ops.hpp"
#include "tools/tools_ops.hpp"
#include "utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <cmath>
#include <filesystem>
#include <system_error>

namespace agent {

void init_backend() {
    providers::init_global_backends();
}
void free_backend() {
    providers::free_global_backends();
}

AgentRuntime::AgentRuntime() {
    init_backend();
}
AgentRuntime::~AgentRuntime() {
    free_backend();
}

Result<Session> init(const Config &config) {
    if (config.workspace_dir.empty())
        return fail<Session>(ErrorCode::InvalidConfig, "workspace_dir is required");
    if (config.context_window <= 0)
        return fail<Session>(ErrorCode::InvalidConfig, "context_window must be > 0");
    if (config.max_tokens <= 0)
        return fail<Session>(ErrorCode::InvalidConfig, "max_tokens must be > 0");
    if (!std::isfinite(config.temperature) || config.temperature < 0.0f)
        return fail<Session>(ErrorCode::InvalidConfig, "temperature must be a non-negative finite number");

    std::error_code ec;
    for (const char *sub : {"models", "memory", "logs", "tmp"}) {
        auto p = std::filesystem::path(config.workspace_dir) / sub;
        std::filesystem::create_directories(p, ec);
        if (ec)
            return fail<Session>(ErrorCode::FilesystemError, "Failed to create " + p.string() + ": " + ec.message());
    }

    Session session;
    session.config = config;
    session.provider_state = nullptr;
    session.memory_state = nullptr;

    session.workspace_dir_ref = std::make_shared<std::string>(config.workspace_dir);

    const size_t arena_size = config.arena_capacity > 0
                                  ? config.arena_capacity
                                  : static_cast<size_t>(config.context_window) * 8 + (2UL * 1024 * 1024);
    session.arena = Arena(arena_size);

    const providers::ProviderOps *ops = providers::find_provider_ops(config.provider);
    if (!ops || !ops->init || !ops->execute_turn)
        return fail<Session>(ErrorCode::UnsupportedProvider, "Provider not compiled into this build");

    auto init_res = ops->init(session);
    if (!init_res.ok)
        return fail<Session>(init_res.error.code, init_res.error.message);

    if (session.config.memory.db_path == "memory/db") {
        session.config.memory.db_path = (std::filesystem::path(config.workspace_dir) / "memory" / "db").string();
    }

    auto mem_res = init_memory(session);
    if (!mem_res.ok)
        return fail<Session>(mem_res.error.code, mem_res.error.message);

    auto tools_res = init_tools(session);
    if (!tools_res.ok)
        return fail<Session>(tools_res.error.code, tools_res.error.message);

    return ok(std::move(session));
}

Result<ChatResponse> execute_turn(Session &session, const ChatRequest &request) {
    utils::log(session, LogLevel::Info, "Executing inference turn...");
    const providers::ProviderOps *ops = providers::find_provider_ops(session.config.provider);
    if (!ops || !ops->execute_turn)
        return fail<ChatResponse>(ErrorCode::UnsupportedProvider, "Provider not compiled into this build");
    return ops->execute_turn(session, request);
}

Result<void> init_memory(Session &session) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->init)
        return fail(ErrorCode::UnsupportedProvider, "Memory provider not compiled into this build");
    return ops->init(session);
}

Result<void> store_memory(Session &session, std::string_view key, std::string_view value) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->store)
        return fail(ErrorCode::UnsupportedProvider, "Memory provider not compiled into this build");
    return ops->store(session, key, value);
}

Result<std::string> retrieve_memory(Session &session, std::string_view key) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->retrieve)
        return fail<std::string>(ErrorCode::UnsupportedProvider, "Memory provider not compiled into this build");
    return ops->retrieve(session, key);
}

Result<void> clear_memory(Session &session) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->clear)
        return fail(ErrorCode::UnsupportedProvider, "Memory provider not compiled into this build");
    return ops->clear(session);
}

Result<void> init_tools(Session &session) {
    const tools::ToolsOps *ops = tools::find_tools_ops();
    if (!ops || !ops->init)
        return fail(ErrorCode::UnsupportedProvider, "Tools registry not compiled into this build");
    return ops->init(session);
}

Result<std::string> execute_tool(Session &session, std::string_view name, std::string_view arguments) {
    const tools::ToolsOps *ops = tools::find_tools_ops();
    if (!ops || !ops->execute)
        return fail<std::string>(ErrorCode::UnsupportedProvider, "Tools registry not compiled into this build");
    return ops->execute(session, name, arguments);
}

} // namespace agent
