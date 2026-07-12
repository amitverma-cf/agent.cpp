#pragma once

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>

namespace agent::utils {

void log(const Session &session, LogLevel level, std::string_view message);

void escape_json_string(std::string_view src, std::string &dst);

std::string escape_json_string(std::string_view src);

} // namespace agent::utils
