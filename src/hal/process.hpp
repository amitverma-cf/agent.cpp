#pragma once

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace agent::hal {

struct ProcessResult {
    int exit_code = -1;
    std::string output;
    bool timed_out = false;
    std::vector<std::string> sandbox_warnings; // what the sandbox could not enforce here
};

// sandbox: optional OS sandbox (hal/sandbox.hpp) applied to the shell before exec; with one, stdin
// is /dev/null (NUL) and the command runs in its own session / Job Object.
Result<ProcessResult> run_shell_command(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                        size_t max_output_bytes = 32 * 1024, const SandboxPolicy *sandbox = nullptr);

} // namespace agent::hal
