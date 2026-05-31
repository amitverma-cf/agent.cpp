#include <agent-cpp/agent.h>

namespace agent {

Session init() {
    return {};
}

std::string run(Session &session, std::string_view prompt) {
    (void) session;

    return std::string(prompt);
}

} // namespace agent