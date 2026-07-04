#include "providers/provider_ops.hpp"
#include "memory/memory_ops.hpp"
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

Result<Session> init(const Config &config) {
    if (config.context_window <= 0) {
        return fail<Session>(ErrorCode::InvalidConfig, "Config error: context_window must be greater than 0.");
    }
    if (config.max_tokens <= 0) {
        return fail<Session>(ErrorCode::InvalidConfig, "Config error: max_tokens must be greater than 0.");
    }
    if (!std::isfinite(config.temperature) || config.temperature < 0.0f) {
        return fail<Session>(ErrorCode::InvalidConfig, "Config error: temperature must be a non-negative finite number.");
    }

    if (!config.workspace_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(config.workspace_dir), ec);
        if (ec) {
            return fail<Session>(ErrorCode::FilesystemError, "Failed to create workspace directory: " + ec.message());
        }
    }

    Session session;
    session.config = config;
    session.provider_state = nullptr;
    session.memory_state = nullptr;

    const providers::ProviderOps *ops = providers::find_provider_ops(config.provider);
    if (!ops || !ops->init || !ops->generate || !ops->stream) {
        return fail<Session>(ErrorCode::UnsupportedProvider, "Requested provider is not compiled into this build.");
    }

    Result<void> init_result = ops->init(session);
    if (!init_result.ok) {
        return fail<Session>(init_result.error.code, init_result.error.message);
    }

    Result<void> mem_init_result = init_memory(session);
    if (!mem_init_result.ok) {
        return fail<Session>(mem_init_result.error.code, mem_init_result.error.message);
    }

    return ok(std::move(session));
}

Result<GenerationResult> generate_text(Session &session, std::string_view prompt) {
    const providers::ProviderOps *ops = providers::find_provider_ops(session.config.provider);
    if (!ops || !ops->generate) {
        return fail<GenerationResult>(ErrorCode::UnsupportedProvider, "Requested provider is not compiled into this build.");
    }
    return ops->generate(session, prompt);
}

Result<void> stream_text(Session &session, std::string_view prompt, TokenCallback on_token, void *user_data) {
    if (!on_token) {
        return fail(ErrorCode::InvalidConfig, "stream_text requires a token callback.");
    }
    const providers::ProviderOps *ops = providers::find_provider_ops(session.config.provider);
    if (!ops || !ops->stream) {
        return fail(ErrorCode::UnsupportedProvider, "Requested provider is not compiled into this build.");
    }
    utils::log(session, LogLevel::Debug, "Starting provider stream.");
    return ops->stream(session, prompt, on_token, user_data);
}

Result<void> init_memory(Session &session) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->init) {
        return fail(ErrorCode::UnsupportedProvider, "Requested memory provider is not compiled into this build.");
    }
    return ops->init(session);
}

Result<void> store_memory(Session &session, std::string_view key, std::string_view value) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->store) {
        return fail(ErrorCode::UnsupportedProvider, "Requested memory provider is not compiled into this build.");
    }
    return ops->store(session, key, value);
}

Result<std::string> retrieve_memory(Session &session, std::string_view key) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->retrieve) {
        return fail<std::string>(ErrorCode::UnsupportedProvider, "Requested memory provider is not compiled into this build.");
    }
    return ops->retrieve(session, key);
}

Result<void> clear_memory(Session &session) {
    const memory::MemoryOps *ops = memory::find_memory_ops(session.config.memory.provider);
    if (!ops || !ops->clear) {
        return fail(ErrorCode::UnsupportedProvider, "Requested memory provider is not compiled into this build.");
    }
    return ops->clear(session);
}

} // namespace agent