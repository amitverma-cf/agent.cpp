#include "hal/process.hpp"

#include "hal/platform.hpp"
#include "hal/sandbox.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>

#if defined(AGENT_HAL_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace agent::hal {
namespace {

bool cancel_requested(const volatile int32_t *cancel) {
    // Written with std::atomic_ref by request_cancel() (executor/turn_executor.cpp).
    return cancel != nullptr &&
           std::atomic_ref<int32_t>(*const_cast<int32_t *>(cancel)).load(std::memory_order_relaxed) != 0;
}

#if defined(AGENT_HAL_WINDOWS)

std::wstring utf8_to_wide(std::string_view s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return {};
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

// Kill-on-close Job Object for a command run without a sandbox, so a timeout / cancel / early return
// also stops the grandchildren. Null if it cannot be set up (the caller then terminates the shell only).
HANDLE make_kill_job(HANDLE process) {
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job) return nullptr;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info)) ||
        !AssignProcessToJobObject(job, process)) {
        CloseHandle(job);
        return nullptr;
    }
    return job;
}

Result<ProcessResult> run_shell_windows(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                        size_t max_output_bytes, PreparedSandbox &sandbox, const volatile int32_t *cancel) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) return fail<ProcessResult>(ErrorCode::FilesystemError, "CreatePipe failed.");
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    std::string cmdline = "cmd.exe /C ";
    cmdline.append(command);
    std::wstring wcmd = utf8_to_wide(cmdline);
    std::wstring wdir = utf8_to_wide(working_directory);

    STARTUPINFOEXW sie{};
    STARTUPINFOW &si = sie.StartupInfo;
    si.cb = sizeof(sie);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE nul = nullptr;
    if (sandbox.active) {
        HANDLE in_w = nullptr; // an empty pipe: stdin that is at its end (a NUL handle is refused by an AppContainer)
        if (CreatePipe(&nul, &in_w, &sa, 0)) CloseHandle(in_w); else nul = nullptr;
        si.hStdInput = nul;
    }

    std::string why;
    // cmd.exe and the system folders are readable by every AppContainer; the working directory is the folder it works in.
    auto container = appcontainer_launch_for(sandbox, {std::string(working_directory)}, why);
    HANDLE low_token = container || !why.empty() ? nullptr : static_cast<HANDLE>(low_integrity_token_for(sandbox, why));
    if (container) sie.lpAttributeList = static_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(container->attributes());
    if (!why.empty()) {
        CloseHandle(read_pipe);
        CloseHandle(write_pipe);
        if (nul && nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
        return fail<ProcessResult>(ErrorCode::SandboxViolation, "cannot confine the command to its folders: " + why);
    }
    PROCESS_INFORMATION pi{};
    BOOL created = low_token ? CreateProcessAsUserW(low_token, nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                                                    wdir.empty() ? nullptr : wdir.c_str(), &si, &pi)
                             : CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE,
                                              CREATE_NO_WINDOW | CREATE_SUSPENDED | (container ? EXTENDED_STARTUPINFO_PRESENT : 0), nullptr,
                                              wdir.empty() ? nullptr : wdir.c_str(), &si, &pi);
    if (low_token) CloseHandle(low_token);
    CloseHandle(write_pipe);
    if (nul && nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!created) {
        CloseHandle(read_pipe);
        return fail<ProcessResult>(ErrorCode::FilesystemError, "CreateProcess failed.");
    }
    // Kill-on-close Job Object: closing it at the end also stops anything the command left running.
    // The shell starts suspended so that no grandchild can escape the job before it is assigned.
    HANDLE job = sandbox.active ? static_cast<HANDLE>(attach_job_object(sandbox, pi.hProcess)) : make_kill_job(pi.hProcess);
    ResumeThread(pi.hThread);

    ProcessResult result;
    std::string output;
    output.reserve(4096);
    char buf[512];
    bool timed_out = false;
    bool cancelled = false;

    const ULONGLONG start = GetTickCount64();
    const ULONGLONG limit_ms = timeout_seconds > 0 ? static_cast<ULONGLONG>(timeout_seconds) * 1000ull : 0;

    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
            DWORD got = 0;
            DWORD chunk = avail > sizeof(buf) ? static_cast<DWORD>(sizeof(buf)) : avail;
            if (ReadFile(read_pipe, buf, chunk, &got, nullptr) && got > 0) {
                if (output.size() < max_output_bytes) {
                    size_t room = max_output_bytes - output.size();
                    size_t take = static_cast<size_t>(got) < room ? static_cast<size_t>(got) : room;
                    output.append(buf, take);
                    if (take < static_cast<size_t>(got)) output += "\n[output truncated]";
                }
            }
        }

        DWORD wait = WaitForSingleObject(pi.hProcess, 20);
        if (wait == WAIT_OBJECT_0) break;

        const bool cancel_now = cancel_requested(cancel);
        if (cancel_now || (limit_ms > 0 && (GetTickCount64() - start) >= limit_ms)) {
            // The whole tree (the Job Object) if there is one, else only the shell.
            if (job) TerminateJobObject(job, 1);
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            cancelled = cancel_now;
            timed_out = !cancel_now;
            break;
        }
    }

    DWORD got = 0;
    for (;;) {
        // After a forced stop only take what is buffered: a process that escaped the job could hold the pipe open.
        if (cancelled || timed_out) {
            DWORD avail = 0;
            if (!PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) break;
        }
        if (!ReadFile(read_pipe, buf, sizeof(buf), &got, nullptr) || got == 0) break;
        if (output.size() < max_output_bytes) {
            size_t room = max_output_bytes - output.size();
            size_t take = static_cast<size_t>(got) < room ? static_cast<size_t>(got) : room;
            output.append(buf, take);
        }
    }

    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(read_pipe);
    if (job) CloseHandle(job);

    result.exit_code = timed_out ? -1 : static_cast<int>(exit_code);
    result.output = std::move(output);
    result.timed_out = timed_out;
    result.cancelled = cancelled;
    if (timed_out) result.output += "\n[timed out]";
    if (cancelled) result.output += "\n[cancelled]";
    return ok(std::move(result));
}

#else // POSIX

using SteadyClock = std::chrono::steady_clock;

// How long a stopped (cancelled / timed out) command gets to react to SIGTERM before SIGKILL.
constexpr long long kKillGraceMs = 500;

// Signals the command's process group; falls back to the shell alone in the instant between fork()
// and the child's setsid(), when the group does not exist yet.
void signal_tree(pid_t pid, int sig) {
    if (kill(-pid, sig) != 0) (void)kill(pid, sig);
}

long long ms_since(SteadyClock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(SteadyClock::now() - t).count();
}

Result<ProcessResult> run_shell_posix(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                      size_t max_output_bytes, const PreparedSandbox &sandbox, const volatile int32_t *cancel) {
    // Built before fork(): the child of a multi-threaded process may only make async-signal-safe calls.
    const std::string wd(working_directory);
    const std::string cmd(command);

    int pipefd[2];
    if (pipe(pipefd) != 0) return fail<ProcessResult>(ErrorCode::FilesystemError, "pipe() failed.");

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return fail<ProcessResult>(ErrorCode::FilesystemError, "fork() failed.");
    }

    if (pid == 0) {
        // Own session and process group (pgid == pid): a timeout or cancel signals the whole tree with
        // kill(-pid, ...). Also detaches from the terminal, so the command cannot steal its input.
        (void)setsid();
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        // Never run the command somewhere other than the requested directory.
        if (!wd.empty() && chdir(wd.c_str()) != 0) {
            const char msg[] = "agent: cannot enter working directory\n";
            (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
            _exit(126);
        }
        if (sandbox.active) {
            // No inherited terminal: nothing to read from, nothing to inject keystrokes into.
            int devnull = open("/dev/null", O_RDONLY | O_CLOEXEC);
            if (devnull >= 0) {
                dup2(devnull, STDIN_FILENO);
                close(devnull);
            }
            if (apply_prepared_sandbox(sandbox) != 0) {
                const char msg[] = "agent: cannot apply the sandbox\n";
                (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
                _exit(126);
            }
        }

        // The timeout is enforced by the wait loop below, not by timeout(1): that would put the command
        // in yet another process group, out of reach of kill(-pid).
        execlp("sh", "sh", "-c", cmd.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }

    ProcessResult result;
    std::string output;
    output.reserve(4096);
    char buf[512];

    close(pipefd[1]);
    const int fd = pipefd[0];

    enum class Stop { None, Cancelled, TimedOut };
    Stop stop = Stop::None;
    bool eof = false;
    bool sigkilled = false;
    bool reaped = false;
    int status = 0;
    const auto start = SteadyClock::now();
    SteadyClock::time_point stop_at{};
    SteadyClock::time_point kill_at{};

    // The leader is only reaped once the pipe is at EOF (or we gave up on it), so the process group id
    // cannot be recycled while we still signal it.
    for (;;) {
        if (stop == Stop::None) {
            if (cancel_requested(cancel))
                stop = Stop::Cancelled;
            else if (timeout_seconds > 0 && ms_since(start) >= static_cast<long long>(timeout_seconds) * 1000)
                stop = Stop::TimedOut;
            if (stop != Stop::None) {
                signal_tree(pid, SIGTERM);
                stop_at = SteadyClock::now();
            }
        }
        // Escalate once the grace period is over, or at once if the output already ended (the shell
        // is gone, but a descendant that closed its stdout may not be).
        if (stop != Stop::None && !sigkilled && (eof || ms_since(stop_at) >= kKillGraceMs)) {
            signal_tree(pid, SIGKILL);
            sigkilled = true;
            kill_at = SteadyClock::now();
        }

        if (!eof) {
            struct pollfd pfd{fd, POLLIN, 0};
            int pr = poll(&pfd, 1, 20);
            if (pr > 0) {
                ssize_t n = read(fd, buf, sizeof(buf));
                if (n > 0) {
                    if (output.size() < max_output_bytes) {
                        size_t room = max_output_bytes - output.size();
                        size_t take = static_cast<size_t>(n) < room ? static_cast<size_t>(n) : room;
                        output.append(buf, take);
                        if (take < static_cast<size_t>(n)) output += "\n[output truncated]";
                    }
                } else if (n == 0 || (errno != EINTR && errno != EAGAIN)) {
                    eof = true;
                }
            } else if (pr < 0 && errno != EINTR) {
                eof = true;
            }
        } else {
            if (waitpid(pid, &status, WNOHANG) == pid) {
                reaped = true;
                break;
            }
            // Not exited yet (it closed its stdout and keeps running): keep polling the flag and deadline.
            struct pollfd none{-1, 0, 0};
            (void)poll(&none, 0, 20);
        }

        // SIGKILL cannot be ignored, but a descendant that left the session may still hold the pipe:
        // stop waiting for it after a moment instead of hanging.
        if (sigkilled && ms_since(kill_at) >= 1000) break;
    }
    close(fd);

    if (!reaped && waitpid(pid, &status, 0) < 0)
        return fail<ProcessResult>(ErrorCode::FilesystemError, "waitpid() failed.");

    int exit_code = -1;
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exit_code = 128 + WTERMSIG(status);
    }

    result.exit_code = stop == Stop::TimedOut ? 124 : exit_code; // 124: what timeout(1) reports
    result.output = std::move(output);
    result.timed_out = stop == Stop::TimedOut;
    result.cancelled = stop == Stop::Cancelled;
    if (result.timed_out) result.output += "\n[timed out]";
    if (result.cancelled) result.output += "\n[cancelled]";
    return ok(std::move(result));
}

#endif

} // namespace

Result<ProcessResult> run_shell_command(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                        size_t max_output_bytes, const SandboxPolicy *sandbox, const volatile int32_t *cancel) {
    if (command.empty()) return fail<ProcessResult>(ErrorCode::InvalidConfig, "Missing command.");
    if (cancel_requested(cancel)) return fail<ProcessResult>(ErrorCode::Cancelled, "run_command: cancelled.");
    auto prepared = prepare_sandbox(sandbox ? *sandbox : SandboxPolicy{});
    if (!prepared.ok) return fail<ProcessResult>(prepared.error.code, prepared.error.message);
#if defined(AGENT_HAL_WINDOWS)
    auto result = run_shell_windows(working_directory, command, timeout_seconds, max_output_bytes, prepared.value, cancel);
#else
    auto result = run_shell_posix(working_directory, command, timeout_seconds, max_output_bytes, prepared.value, cancel);
#endif
    if (result.ok) result.value.sandbox_warnings = std::move(prepared.value.warnings);
    return result;
}

} // namespace agent::hal
