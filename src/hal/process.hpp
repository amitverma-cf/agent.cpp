#pragma once

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>

namespace agent::hal {

struct ProcessResult {
    int exit_code = -1;
    std::string output;
    bool timed_out = false;
};

Result<ProcessResult> run_shell_command(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                        size_t max_output_bytes = 32 * 1024);

} // namespace agent::hal
