#include "providers/provider_ops.h"
#include "utils/utils.h"

#include <agent-cpp/agent.h>
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
    if (!config.workspace_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(config.workspace_dir), ec);
        if (ec) {
            return fail<Session>(ErrorCode::FilesystemError, "Failed to create workspace directory: " + ec.message());
        }
    }

    Session session{.config = config, .state = nullptr};
    const providers::ProviderOps *ops = providers::find_provider_ops(config.provider);
    if (!ops || !ops->init || !ops->generate || !ops->stream) {
        return fail<Session>(ErrorCode::UnsupportedProvider, "Requested provider is not compiled into this build.");
    }

    Result<void> init_result = ops->init(session);
    if (!init_result.ok) {
        return fail<Session>(init_result.error.code, init_result.error.message);
    }
    return ok(session);
}

Result<std::string> generate_text(Session &session, std::string_view prompt) {
    const providers::ProviderOps *ops = providers::find_provider_ops(session.config.provider);
    if (!ops || !ops->generate) {
        return fail<std::string>(ErrorCode::UnsupportedProvider, "Requested provider is not compiled into this build.");
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

} // namespace agent
