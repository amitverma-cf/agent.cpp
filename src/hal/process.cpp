#include "hal/process.hpp"

#include "hal/platform.hpp"

#include <cstdio>
#include <string>

#if defined(AGENT_HAL_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace agent::hal {
namespace {

#if defined(AGENT_HAL_WINDOWS)

std::wstring utf8_to_wide(std::string_view s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return {};
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

Result<ProcessResult> run_shell_windows(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                        size_t max_output_bytes) {
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

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    BOOL created = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                                  wdir.empty() ? nullptr : wdir.c_str(), &si, &pi);
    CloseHandle(write_pipe);
    if (!created) {
        CloseHandle(read_pipe);
        return fail<ProcessResult>(ErrorCode::FilesystemError, "CreateProcess failed.");
    }

    ProcessResult result;
    std::string output;
    output.reserve(4096);
    char buf[512];
    bool timed_out = false;

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

        if (limit_ms > 0 && (GetTickCount64() - start) >= limit_ms) {
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            timed_out = true;
            break;
        }
    }

    DWORD got = 0;
    while (ReadFile(read_pipe, buf, sizeof(buf), &got, nullptr) && got > 0) {
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

    result.exit_code = timed_out ? -1 : static_cast<int>(exit_code);
    result.output = std::move(output);
    result.timed_out = timed_out;
    if (timed_out) result.output += "\n[timed out]";
    return ok(std::move(result));
}

#else // POSIX

Result<ProcessResult> run_shell_posix(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                      size_t max_output_bytes) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return fail<ProcessResult>(ErrorCode::FilesystemError, "pipe() failed.");

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return fail<ProcessResult>(ErrorCode::FilesystemError, "fork() failed.");
    }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        std::string wd(working_directory);
        if (!wd.empty()) (void)chdir(wd.c_str());

        std::string cmd(command);
        if (timeout_seconds > 0) {
            std::string t = std::to_string(timeout_seconds);
            execlp("timeout", "timeout", t.c_str(), "sh", "-c", cmd.c_str(), static_cast<char *>(nullptr));
            // Fallback if `timeout` is not installed:
            execlp("sh", "sh", "-c", cmd.c_str(), static_cast<char *>(nullptr));
        } else {
            execlp("sh", "sh", "-c", cmd.c_str(), static_cast<char *>(nullptr));
        }
        _exit(127);
    }

    close(pipefd[1]);
    ProcessResult result;
    std::string output;
    output.reserve(4096);
    char buf[512];
    ssize_t n = 0;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        if (output.size() < max_output_bytes) {
            size_t room = max_output_bytes - output.size();
            size_t take = static_cast<size_t>(n) < room ? static_cast<size_t>(n) : room;
            output.append(buf, take);
            if (take < static_cast<size_t>(n)) output += "\n[output truncated]";
        }
    }
    close(pipefd[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return fail<ProcessResult>(ErrorCode::FilesystemError, "waitpid() failed.");

    bool timed_out = false;
    int exit_code = -1;
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
        if (timeout_seconds > 0 && exit_code == 124) timed_out = true;
    } else if (WIFSIGNALED(status)) {
        exit_code = 128 + WTERMSIG(status);
    }

    result.exit_code = exit_code;
    result.output = std::move(output);
    result.timed_out = timed_out;
    if (timed_out) result.output += "\n[timed out]";
    return ok(std::move(result));
}

#endif

} // namespace

Result<ProcessResult> run_shell_command(std::string_view working_directory, std::string_view command, int timeout_seconds,
                                        size_t max_output_bytes) {
    if (command.empty()) return fail<ProcessResult>(ErrorCode::InvalidConfig, "Missing command.");
#if defined(AGENT_HAL_WINDOWS)
    return run_shell_windows(working_directory, command, timeout_seconds, max_output_bytes);
#else
    return run_shell_posix(working_directory, command, timeout_seconds, max_output_bytes);
#endif
}

} // namespace agent::hal
