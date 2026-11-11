#include "hal/child_process.hpp"

#include "hal/platform.hpp"
#include "hal/sandbox.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#if defined(AGENT_HAL_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <filesystem>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace agent::hal {

namespace {

constexpr size_t kReadChunk = 64 * 1024;

using Clock = std::chrono::steady_clock;

[[maybe_unused]] int remaining_ms(Clock::time_point deadline) {
    auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    return left > 0 ? static_cast<int>(left) : 0;
}

} // namespace

#if defined(AGENT_HAL_WINDOWS)

// ------------------------------------------------------------------ Windows

struct ChildProcess::Impl {
    HANDLE process = nullptr;
    HANDLE in = nullptr;  // write end of the child's stdin
    HANDLE out = nullptr; // read end of the child's stdout
    HANDLE err = nullptr; // read end of the child's stderr
    HANDLE job = nullptr; // sandbox Job Object (kill-on-close), if any
    bool reaped = false;
};

namespace {

std::wstring widen(std::string_view s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return {};
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

// Quotes one argument following the MSVC runtime's CommandLineToArgvW rules.
void append_quoted(std::wstring &cmd, const std::wstring &arg) {
    if (!cmd.empty()) cmd.push_back(L' ');
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        cmd += arg;
        return;
    }
    cmd.push_back(L'"');
    for (size_t i = 0;; ++i) {
        size_t backslashes = 0;
        while (i < arg.size() && arg[i] == L'\\') {
            ++i;
            ++backslashes;
        }
        if (i == arg.size()) {
            cmd.append(backslashes * 2, L'\\');
            break;
        }
        if (arg[i] == L'"') {
            cmd.append(backslashes * 2 + 1, L'\\');
            cmd.push_back(L'"');
        } else {
            cmd.append(backslashes, L'\\');
            cmd.push_back(arg[i]);
        }
    }
    cmd.push_back(L'"');
}

bool name_overridden(const std::wstring &entry, const std::map<std::string, std::string> &env) {
    size_t eq = entry.find(L'=', 1); // entries like "=C:=C:\\" start with '='
    if (eq == std::wstring::npos) return false;
    std::wstring name = entry.substr(0, eq);
    for (const auto &[k, v] : env) {
        std::wstring wk = widen(k);
        if (wk.size() == name.size() &&
            CompareStringOrdinal(wk.c_str(), static_cast<int>(wk.size()), name.c_str(), static_cast<int>(name.size()), TRUE) == CSTR_EQUAL)
            return true;
    }
    return false;
}

std::wstring build_environment(const std::map<std::string, std::string> &env) {
    std::wstring block;
    if (LPWCH current = GetEnvironmentStringsW()) {
        for (LPWCH p = current; *p; p += wcslen(p) + 1) {
            std::wstring entry(p);
            if (name_overridden(entry, env)) continue;
            block += entry;
            block.push_back(L'\0');
        }
        FreeEnvironmentStringsW(current);
    }
    for (const auto &[k, v] : env) {
        block += widen(k);
        block.push_back(L'=');
        block += widen(v);
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return block;
}

void close_handle(HANDLE &h) {
    if (h) CloseHandle(h);
    h = nullptr;
}

} // namespace

Result<std::unique_ptr<ChildProcess>> spawn_child(const SpawnOptions &options) {
    using R = Result<std::unique_ptr<ChildProcess>>;
    if (options.program.empty()) return R{.ok = false, .value = {}, .error = make_error(ErrorCode::InvalidConfig, "Missing program.")};

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr, err_r = nullptr, err_w = nullptr;
    auto cleanup = [&]() {
        close_handle(in_r);
        close_handle(in_w);
        close_handle(out_r);
        close_handle(out_w);
        close_handle(err_r);
        close_handle(err_w);
    };
    if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0) || !CreatePipe(&err_r, &err_w, &sa, 0)) {
        cleanup();
        return R{.ok = false, .value = {}, .error = make_error(ErrorCode::FilesystemError, "CreatePipe failed.")};
    }
    // The parent's ends must not leak into the child.
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);

    std::wstring cmd;
    append_quoted(cmd, widen(options.program));
    for (const auto &a : options.args) append_quoted(cmd, widen(a));
    std::wstring env_block = options.env.empty() ? std::wstring() : build_environment(options.env);
    std::wstring wdir = widen(options.working_directory);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = err_w;

    auto prepared = prepare_sandbox(options.sandbox);
    if (!prepared.ok) {
        cleanup();
        return R{.ok = false, .value = {}, .error = prepared.error};
    }

    // Filesystem confinement: the child runs at Low integrity and only the allowed folders accept its writes. If that cannot be
    // set up the process is not started (a sandbox that silently does nothing is worse than none).
    std::string why;
    HANDLE low_token = static_cast<HANDLE>(agent::hal::low_integrity_token_for(prepared.value, why));
    if (!why.empty()) {
        cleanup();
        return R{.ok = false, .value = {}, .error = make_error(ErrorCode::SandboxViolation, "cannot confine '" + options.program + "' to its folders: " + why)};
    }

    PROCESS_INFORMATION pi{};
    DWORD flags = CREATE_NO_WINDOW | (env_block.empty() ? 0 : CREATE_UNICODE_ENVIRONMENT) | (prepared.value.active ? CREATE_SUSPENDED : 0);
    BOOL created = low_token ? CreateProcessAsUserW(low_token, nullptr, cmd.data(), nullptr, nullptr, TRUE, flags, env_block.empty() ? nullptr : env_block.data(),
                                                    wdir.empty() ? nullptr : wdir.c_str(), &si, &pi)
                             : CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, flags, env_block.empty() ? nullptr : env_block.data(),
                                              wdir.empty() ? nullptr : wdir.c_str(), &si, &pi);
    if (low_token) CloseHandle(low_token);
    DWORD last_error = GetLastError();
    close_handle(in_r);
    close_handle(out_w);
    close_handle(err_w);
    if (!created) {
        cleanup();
        return R{.ok = false,
                 .value = {},
                 .error = make_error(ErrorCode::FilesystemError,
                                     "CreateProcess failed for '" + options.program + "' (error " + std::to_string(last_error) + ").")};
    }
    HANDLE job = prepared.value.active ? static_cast<HANDLE>(attach_job_object(prepared.value, pi.hProcess)) : nullptr;
    if (prepared.value.active) ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);

    auto impl = std::make_unique<ChildProcess::Impl>();
    impl->job = job;
    impl->process = pi.hProcess;
    impl->in = in_w;
    impl->out = out_r;
    impl->err = err_r;
    auto child = std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
    child->sandbox_warnings_ = std::move(prepared.value.warnings);
    return R{.ok = true, .value = std::move(child), .error = {}};
}

ChildProcess::ChildProcess(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ChildProcess::~ChildProcess() {
    terminate(0);
    close_handle(impl_->out);
    close_handle(impl_->err);
    close_handle(impl_->job); // kill-on-close: takes down anything the child left behind
    close_handle(impl_->process);
}

Result<void> ChildProcess::write_stdin(std::string_view data) {
    // Anonymous pipes are synchronous on Windows: this blocks while the pipe is full.
    while (!data.empty()) {
        if (!impl_->in) return fail(ErrorCode::NetworkError, "stdin is closed");
        DWORD chunk = data.size() > 1u << 20 ? 1u << 20 : static_cast<DWORD>(data.size());
        DWORD written = 0;
        if (!WriteFile(impl_->in, data.data(), chunk, &written, nullptr))
            return fail(ErrorCode::NetworkError, "write to child stdin failed (error " + std::to_string(GetLastError()) + ")");
        data.remove_prefix(written);
    }
    return ok();
}

void ChildProcess::close_stdin() { close_handle(impl_->in); }

PipeRead ChildProcess::read(ChildStream stream, std::string &out, int timeout_ms) {
    HANDLE h = stream == ChildStream::Stdout ? impl_->out : impl_->err;
    if (!h) return PipeRead::Closed;
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 0);
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr))
            return GetLastError() == ERROR_BROKEN_PIPE ? PipeRead::Closed : PipeRead::Error;
        if (avail > 0) {
            char buf[kReadChunk];
            DWORD want = avail > kReadChunk ? static_cast<DWORD>(kReadChunk) : avail;
            DWORD got = 0;
            if (!ReadFile(h, buf, want, &got, nullptr)) return GetLastError() == ERROR_BROKEN_PIPE ? PipeRead::Closed : PipeRead::Error;
            out.append(buf, got);
            return PipeRead::Data;
        }
        if (timeout_ms >= 0 && Clock::now() >= deadline) return PipeRead::Timeout;
        Sleep(2);
    }
}

bool ChildProcess::exited() {
    if (impl_->reaped) return true;
    if (WaitForSingleObject(impl_->process, 0) != WAIT_OBJECT_0) return false;
    DWORD code = 1;
    GetExitCodeProcess(impl_->process, &code);
    exit_code_ = static_cast<int>(code);
    impl_->reaped = true;
    return true;
}

int ChildProcess::terminate(int grace_ms) {
    close_stdin();
    if (exited()) return exit_code_;
    if (grace_ms > 0) WaitForSingleObject(impl_->process, static_cast<DWORD>(grace_ms));
    if (!exited()) {
        TerminateProcess(impl_->process, 1);
        WaitForSingleObject(impl_->process, 5000);
        exited();
    }
    return exit_code_;
}

#else

// ------------------------------------------------------------------ POSIX

struct ChildProcess::Impl {
    pid_t pid = -1;
    int in = -1;  // write end of the child's stdin (non-blocking)
    int out = -1; // read end of the child's stdout
    int err = -1; // read end of the child's stderr
    bool own_group = false; // the child leads its own session / process group (sandboxed)
    bool reaped = false;
};

namespace {

void close_fd(int &fd) {
    if (fd >= 0) ::close(fd);
    fd = -1;
}

bool make_pipe(int fds[2]) {
#if defined(__linux__)
    return ::pipe2(fds, O_CLOEXEC) == 0;
#else
    if (::pipe(fds) != 0) return false;
    ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
    ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
    return true;
#endif
}

bool is_executable_file(const std::string &path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && ::access(path.c_str(), X_OK) == 0;
}

// execvp's lookup, done in the parent so the child only has to call execve (async-signal-safe).
std::string resolve_program(const std::string &program, const std::map<std::string, std::string> &env) {
    if (program.find('/') != std::string::npos) return program;
    std::string path;
    if (auto it = env.find("PATH"); it != env.end()) path = it->second;
    else if (const char *p = std::getenv("PATH")) path = p;
    else path = "/usr/local/bin:/usr/bin:/bin";
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find(':', start);
        if (end == std::string::npos) end = path.size();
        std::string dir = path.substr(start, end - start);
        if (dir.empty()) dir = ".";
        std::string candidate = dir + "/" + program;
        if (is_executable_file(candidate)) return candidate;
        start = end + 1;
    }
    return {};
}

std::vector<std::string> build_environment(const std::map<std::string, std::string> &env) {
    std::vector<std::string> out;
    for (char **e = environ; e && *e; ++e) {
        std::string_view entry(*e);
        size_t eq = entry.find('=');
        if (eq != std::string_view::npos && env.count(std::string(entry.substr(0, eq)))) continue;
        out.emplace_back(entry);
    }
    for (const auto &[k, v] : env) out.push_back(k + "=" + v);
    return out;
}

} // namespace

Result<std::unique_ptr<ChildProcess>> spawn_child(const SpawnOptions &options) {
    using R = Result<std::unique_ptr<ChildProcess>>;
    auto failure = [](ErrorCode code, std::string msg) { return R{.ok = false, .value = {}, .error = make_error(code, std::move(msg))}; };
    if (options.program.empty()) return failure(ErrorCode::InvalidConfig, "Missing program.");

    const std::string path = resolve_program(options.program, options.env);
    if (path.empty()) return failure(ErrorCode::FilesystemError, "Program not found on PATH: " + options.program);

    // Everything the child needs is prepared before fork(): after it, only async-signal-safe calls.
    std::vector<std::string> env_strings = build_environment(options.env);
    std::vector<char *> envp;
    envp.reserve(env_strings.size() + 1);
    for (auto &s : env_strings) envp.push_back(s.data());
    envp.push_back(nullptr);

    std::vector<std::string> arg_strings;
    arg_strings.reserve(options.args.size() + 1);
    arg_strings.push_back(options.program);
    for (const auto &a : options.args) arg_strings.push_back(a);
    std::vector<char *> argv;
    argv.reserve(arg_strings.size() + 1);
    for (auto &s : arg_strings) argv.push_back(s.data());
    argv.push_back(nullptr);

    const std::string wd = options.working_directory;

    auto prepared = prepare_sandbox(options.sandbox);
    if (!prepared.ok) return failure(prepared.error.code, prepared.error.message);
    const PreparedSandbox &sandbox = prepared.value;

    int in[2] = {-1, -1}, out[2] = {-1, -1}, err[2] = {-1, -1}, status_pipe[2] = {-1, -1};
    auto close_all = [&]() {
        for (int *p : {in, out, err, status_pipe}) {
            close_fd(p[0]);
            close_fd(p[1]);
        }
    };
    if (!make_pipe(in) || !make_pipe(out) || !make_pipe(err) || !make_pipe(status_pipe)) {
        close_all();
        return failure(ErrorCode::FilesystemError, std::string("pipe() failed: ") + std::strerror(errno));
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        close_all();
        return failure(ErrorCode::FilesystemError, std::string("fork() failed: ") + std::strerror(errno));
    }

    if (pid == 0) {
        // dup2 clears FD_CLOEXEC on the targets; every other pipe end closes on exec.
        ::dup2(in[0], STDIN_FILENO);
        ::dup2(out[1], STDOUT_FILENO);
        ::dup2(err[1], STDERR_FILENO);
        sigset_t none;
        sigemptyset(&none);
        ::sigprocmask(SIG_SETMASK, &none, nullptr);
        ::signal(SIGPIPE, SIG_DFL);
        int code = 0;
        if (!wd.empty() && ::chdir(wd.c_str()) != 0) code = errno;
        else if (int e = apply_prepared_sandbox(sandbox); e != 0) code = -e; // negative: the sandbox failed
        else {
            ::execve(path.c_str(), argv.data(), envp.data());
            code = errno;
        }
        (void)!::write(status_pipe[1], &code, sizeof(code));
        ::_exit(127);
    }

    close_fd(in[0]);
    close_fd(out[1]);
    close_fd(err[1]);
    close_fd(status_pipe[1]);

    // The status pipe closes on a successful exec (CLOEXEC) or carries errno on failure.
    int child_errno = 0;
    ssize_t n = 0;
    do { n = ::read(status_pipe[0], &child_errno, sizeof(child_errno)); } while (n < 0 && errno == EINTR);
    close_fd(status_pipe[0]);
    if (n == static_cast<ssize_t>(sizeof(child_errno))) {
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        close_all();
        if (child_errno < 0)
            return failure(ErrorCode::SandboxViolation, "cannot sandbox '" + options.program + "': " + std::strerror(-child_errno));
        std::string what = (!wd.empty() ? "cannot start '" + options.program + "' in '" + wd + "': " : "cannot exec '" + path + "': ");
        return failure(ErrorCode::FilesystemError, what + std::strerror(child_errno));
    }

    ::fcntl(in[1], F_SETFL, ::fcntl(in[1], F_GETFL) | O_NONBLOCK);

    auto impl = std::make_unique<ChildProcess::Impl>();
    impl->pid = pid;
    impl->in = in[1];
    impl->out = out[0];
    impl->err = err[0];
    impl->own_group = sandbox.active && sandbox.new_session;
    auto child = std::unique_ptr<ChildProcess>(new ChildProcess(std::move(impl)));
    child->sandbox_warnings_ = std::move(prepared.value.warnings);
    return R{.ok = true, .value = std::move(child), .error = {}};
}

ChildProcess::ChildProcess(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ChildProcess::~ChildProcess() {
    terminate(0);
    close_fd(impl_->out);
    close_fd(impl_->err);
}

Result<void> ChildProcess::write_stdin(std::string_view data) {
    if (impl_->in < 0) return fail(ErrorCode::NetworkError, "stdin is closed");

    // Writing to a pipe whose reader is gone raises SIGPIPE, which would kill the host. Block it on
    // this thread for the write and swallow the one our write generated (pthread_sigmask is per thread).
    sigset_t pipe_set, old_set, pending;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    ::pthread_sigmask(SIG_BLOCK, &pipe_set, &old_set);
    sigpending(&pending);
    const bool was_pending = sigismember(&pending, SIGPIPE) == 1;

    Result<void> result = ok();
    while (!data.empty()) {
        ssize_t n = ::write(impl_->in, data.data(), data.size());
        if (n > 0) {
            data.remove_prefix(static_cast<size_t>(n));
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // The child is not reading. Wait for room, but not forever.
            pollfd pfd{.fd = impl_->in, .events = POLLOUT, .revents = 0};
            int ready = ::poll(&pfd, 1, 30000);
            if (ready == 0) {
                result = fail(ErrorCode::NetworkError, "child is not reading its stdin (write timed out)");
                break;
            }
            if (ready < 0 && errno != EINTR) {
                result = fail(ErrorCode::NetworkError, std::string("poll on child stdin failed: ") + std::strerror(errno));
                break;
            }
            continue;
        }
        result = fail(ErrorCode::NetworkError, std::string("write to child stdin failed: ") + std::strerror(errno));
        if (errno == EPIPE && !was_pending) {
            sigpending(&pending);
            if (sigismember(&pending, SIGPIPE) == 1) {
                int sig = 0;
                sigwait(&pipe_set, &sig);
            }
        }
        break;
    }
    ::pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
    return result;
}

void ChildProcess::close_stdin() { close_fd(impl_->in); }

PipeRead ChildProcess::read(ChildStream stream, std::string &out, int timeout_ms) {
    const int fd = stream == ChildStream::Stdout ? impl_->out : impl_->err;
    if (fd < 0) return PipeRead::Closed;
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 0);
    for (;;) {
        pollfd pfd{.fd = fd, .events = POLLIN, .revents = 0};
        int wait = timeout_ms < 0 ? -1 : remaining_ms(deadline);
        int ready = ::poll(&pfd, 1, wait);
        if (ready < 0) {
            if (errno == EINTR) continue;
            return PipeRead::Error;
        }
        if (ready == 0) return PipeRead::Timeout;
        char buf[kReadChunk];
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, static_cast<size_t>(n));
            return PipeRead::Data;
        }
        if (n == 0) return PipeRead::Closed;
        if (errno == EINTR || errno == EAGAIN) continue;
        return PipeRead::Error;
    }
}

bool ChildProcess::exited() {
    if (impl_->reaped) return true;
    int status = 0;
    pid_t r = ::waitpid(impl_->pid, &status, WNOHANG);
    if (r == 0) return false;
    if (r < 0 && errno == EINTR) return false;
    impl_->reaped = true;
    if (r == impl_->pid) {
        if (WIFEXITED(status)) exit_code_ = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) {
            term_signal_ = WTERMSIG(status);
            exit_code_ = 128 + term_signal_;
        }
    }
    return true;
}

int ChildProcess::terminate(int grace_ms) {
    close_stdin();
    auto wait_for_exit = [this](int ms) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(ms);
        while (!exited()) {
            if (Clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    };
    if (exited()) return exit_code_;
    if (grace_ms > 0 && wait_for_exit(grace_ms)) return exit_code_;
    // A sandboxed child leads its own process group: stop whatever it started along with it.
    const pid_t target = impl_->own_group ? -impl_->pid : impl_->pid;
    ::kill(target, SIGTERM);
    if (wait_for_exit(grace_ms > 0 ? grace_ms : 200)) return exit_code_;
    ::kill(target, SIGKILL);
    int status = 0;
    while (::waitpid(impl_->pid, &status, 0) < 0 && errno == EINTR) {}
    impl_->reaped = true;
    term_signal_ = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
    exit_code_ = WIFSIGNALED(status) ? 128 + WTERMSIG(status) : (WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return exit_code_;
}

#endif

} // namespace agent::hal
