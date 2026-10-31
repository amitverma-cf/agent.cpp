#pragma once

#include <agent-cpp/agent.hpp>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace agent::hal {

// A long-running child process with its stdin, stdout and stderr connected to pipes (used by the
// MCP stdio client). Not a shell: program is looked up on PATH and args are passed verbatim.
struct SpawnOptions {
    std::string program;
    std::vector<std::string> args;
    std::map<std::string, std::string> env; // added to / overriding the parent's environment
    std::string working_directory;          // empty = inherit
};

enum class PipeRead {
    Data,    // at least one byte was appended
    Timeout, // nothing arrived within the timeout
    Closed,  // the child closed its end (usually: it exited)
    Error,
};

enum class ChildStream { Stdout, Stderr };

class ChildProcess {
  public:
    ~ChildProcess(); // terminate(0) if still running
    ChildProcess(const ChildProcess &) = delete;
    ChildProcess &operator=(const ChildProcess &) = delete;

    // Writes all of data to the child's stdin. Fails (instead of raising SIGPIPE) if the child is gone.
    Result<void> write_stdin(std::string_view data);
    void close_stdin();

    // Appends whatever is available on the stream (up to 64 KiB) to out, waiting at most timeout_ms
    // (0 = poll, < 0 = wait forever). stdout and stderr may be read from two different threads.
    PipeRead read(ChildStream stream, std::string &out, int timeout_ms);

    // Non-blocking check; true once the child has exited (exit_code is then valid).
    bool exited();
    int exit_code() const { return exit_code_; }
    // POSIX: the signal that terminated the child (exit_code is then 128 + signal), 0 otherwise.
    int term_signal() const { return term_signal_; }

    // Clean shutdown: closes stdin, waits up to grace_ms for the child to exit, then asks it to stop
    // (SIGTERM; TerminateProcess on Windows), waits again, and finally kills it. Returns the exit code.
    int terminate(int grace_ms);

    struct Impl;

  private:
    friend Result<std::unique_ptr<ChildProcess>> spawn_child(const SpawnOptions &options);
    explicit ChildProcess(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    int exit_code_ = -1;
    int term_signal_ = 0;
};

Result<std::unique_ptr<ChildProcess>> spawn_child(const SpawnOptions &options);

} // namespace agent::hal
