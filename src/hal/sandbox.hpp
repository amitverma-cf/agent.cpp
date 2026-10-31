#pragma once

// OS-level sandboxing for child processes (engine host, run_command, opt-in MCP servers). See
// docs/security-sandbox.md.
//
// Two halves, because the child side runs between fork() and exec() where only async-signal-safe
// calls are allowed:
//   prepare_sandbox()        parent: resolves paths, builds the Landlock ruleset fd and the seccomp
//                            program, collects warnings (allocates, may fail with a message).
//   apply_prepared_sandbox() child: setsid, rlimits, NO_NEW_PRIVS, landlock_restrict_self, seccomp.
//                            Raw syscalls only.
//
// Linux: Landlock (filesystem allow-list, TCP deny on ABI >= 4, signal / abstract-socket scoping on
// ABI >= 6), seccomp-bpf (socket() other than AF_UNIX and io_uring fail when the network is denied),
// PR_SET_NO_NEW_PRIVS, RLIMIT_CORE = 0, RLIMIT_AS, RLIMIT_NPROC, a new session. A kernel without
// Landlock or seccomp degrades to a warning, never to a failed spawn.
// macOS: a sandbox_init() SBPL profile (written, untested). Windows: a Job Object (written, untested).

#include <agent-cpp/agent.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace agent::hal {

struct SandboxSupport {
    int landlock_abi = 0; // 0 = Landlock unavailable (kernel too old, disabled, or not Linux)
    bool seccomp = false; // seccomp-bpf filters can be installed (and this arch is supported)
};

// What this machine can enforce. Probed once, then cached.
SandboxSupport sandbox_support();

// Same layout as struct sock_filter (linux/filter.h), kept here so the header stays portable.
struct BpfInsn {
    uint16_t code;
    uint8_t jt;
    uint8_t jf;
    uint32_t k;
};

class PreparedSandbox {
  public:
    PreparedSandbox() = default;
    ~PreparedSandbox();
    PreparedSandbox(PreparedSandbox &&other) noexcept;
    PreparedSandbox &operator=(PreparedSandbox &&other) noexcept;
    PreparedSandbox(const PreparedSandbox &) = delete;
    PreparedSandbox &operator=(const PreparedSandbox &) = delete;

    bool active = false;          // policy.enabled
    bool new_session = true;      // setsid() in the child
    int landlock_fd = -1;         // Linux: ruleset fd (O_CLOEXEC), -1 = no Landlock
    std::vector<BpfInsn> seccomp; // Linux: empty = no filter
    uint64_t memory_limit_bytes = 0;
    int max_processes = 0;
    std::string macos_profile;         // macOS: SBPL text for sandbox_init(), empty = none
    std::vector<std::string> warnings; // what could not be enforced; callers log these
};

// Resolves policy for the current machine. Fails only on invalid input; missing kernel features
// become warnings. The result is inert until apply_prepared_sandbox() runs in a child.
Result<PreparedSandbox> prepare_sandbox(const SandboxPolicy &policy);

// Child side, after fork() and before exec(). Async-signal-safe; returns 0 or an errno value (the
// caller should report it and _exit). Irreversible for the calling process and its descendants.
int apply_prepared_sandbox(const PreparedSandbox &prepared) noexcept;

// prepare + apply on the calling process (e.g. a host restricting itself at startup). Call it while
// the process is still single-threaded: Landlock and seccomp apply to the calling thread only.
Result<void> apply_sandbox(const SandboxPolicy &policy);

// Read-only system locations a sandboxed program typically needs (/usr, /lib*, selected /etc files,
// /proc, /sys, ...) plus device nodes it may open read-write (/dev/null, GPUs). Exposed for docs/tests.
std::vector<std::string> sandbox_system_read_only_paths(bool network);
std::vector<std::string> sandbox_system_read_write_paths();

#if defined(_WIN32)
// Windows: creates a Job Object carrying the limits (kill-on-close, memory, process count, no
// breakaway) and assigns process (a HANDLE, ideally created suspended) to it. Returns the job HANDLE
// (close it to kill the tree) or nullptr with a warning appended to prepared.warnings.
void *attach_job_object(PreparedSandbox &prepared, void *process);
#endif

} // namespace agent::hal
