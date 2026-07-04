#pragma once

#include <agent-cpp/agent.hpp>
#include <string_view>

namespace agent::utils {

void log(const Session &session, LogLevel level, std::string_view message);

} // namespace agent::utils