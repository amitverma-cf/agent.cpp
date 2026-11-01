#pragma once

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>

namespace agent::utils {

void log(const Session &session, LogLevel level, std::string_view message);

void escape_json_string(std::string_view src, std::string &dst);
std::string escape_json_string(std::string_view src);

// Length of the longest prefix of s (at most max bytes) that does not end inside a UTF-8 sequence.
size_t utf8_prefix_length(std::string_view s, size_t max);
// Replaces every byte that is not part of a well-formed UTF-8 sequence with U+FFFD, so tool output
// (file contents, command output) is valid text for JSON and chat templates.
void sanitize_utf8(std::string &s);

} // namespace agent::utils
