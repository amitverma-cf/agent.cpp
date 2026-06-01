#include <agent-cpp/agent.h>

namespace agent {

Session init(const Config &config) {
    return {.config = config};
}

std::string run(Session &session, std::string_view prompt) {
    switch (session.config.provider) {
    case Provider::Mock:
        return "Mock Response";

    case Provider::LlamaCpp:
        return "not implemented";

    case Provider::OpenAICompatible:
        return "not implemented";

    default:
        return std::string(prompt);
    }
}

} // namespace agent