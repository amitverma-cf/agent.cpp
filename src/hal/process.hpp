#pragma once

#include <agent-cpp/agent.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace agent::hal {

struct ProcessResult {
    int exit_code = -1;
    std::string output;
    bool timed_out = false;
    bool cancelled = false;                    // stopped because *cancel became non-zero; output is what was captured so far
    std::vector<std::string> sandbox_warnings; // what the sandbox could not enforce here
};

// sandbox: optional OS sandbox (hal/sandbox.hpp) applied to the shell before exec; with one, stdin
// is /dev/null (NUL).
// The command always runs in its own session (POSIX) / Job Object (Windows), so a timeout or a cancel
// stops the whole process tree, grandchildren included. cancel: optional flag polled (~20 ms) while
// waiting; once it is non-zero the tree is stopped (POSIX: SIGTERM to the process group, SIGKILL after
// 500 ms; Windows: the Job Object is terminated) and the result has cancelled = true with the output
// captured so far. If it is already non-zero the command is not started: ErrorCode::Cancelled.
Result<ProcessResult> run_shell_command(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                        size_t max_output_bytes = 32 * 1024, const SandboxPolicy *sandbox = nullptr,
                                        const volatile int32_t *cancel = nullptr);

} // namespace agent::hal
