// OS helpers for the agent CLI: config/data directories, executable location, hardware detection,
// TTY checks, interruptible line reads and "open with the default app".

#include "cli_core.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
// after windows.h
#include <shellapi.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
extern char **environ;
#endif

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string env(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

bool on_path(std::string_view exe) {
    const std::string path = env("PATH");
#if defined(_WIN32)
    constexpr char sep = ';';
#else
    constexpr char sep = ':';
#endif
    size_t i = 0;
    while (i <= path.size()) {
        size_t j = path.find(sep, i);
        if (j == std::string::npos) j = path.size();
        if (j > i) {
            std::error_code ec;
            if (fs::is_regular_file(fs::path(path.substr(i, j - i)) / exe, ec)) return true;
        }
        i = j + 1;
    }
    return false;
}

} // namespace

std::string default_config_dir() {
    if (std::string d = env("AGENT_CONFIG_DIR"); !d.empty()) return d;
#if defined(_WIN32)
    if (std::string d = env("APPDATA"); !d.empty()) return (fs::path(d) / "agent").string();
    return (fs::path(env("USERPROFILE")) / "AppData" / "Roaming" / "agent").string();
#else
    if (std::string d = env("XDG_CONFIG_HOME"); !d.empty()) return (fs::path(d) / "agent").string();
    return (fs::path(env("HOME")) / ".config" / "agent").string();
#endif
}

std::string default_workspace_dir() {
#if defined(_WIN32)
    if (std::string d = env("LOCALAPPDATA"); !d.empty()) return (fs::path(d) / "agent" / "workspace").string();
    return (fs::path(env("USERPROFILE")) / "agent" / "workspace").string();
#else
    if (std::string d = env("XDG_DATA_HOME"); !d.empty()) return (fs::path(d) / "agent" / "workspace").string();
    return (fs::path(env("HOME")) / ".local" / "share" / "agent" / "workspace").string();
#endif
}

std::string executable_dir() {
#if defined(_WIN32)
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) return {};
    return fs::path(std::string(buf, n)).parent_path().string();
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};
    std::error_code ec;
    return fs::canonical(buf, ec).parent_path().string();
#else
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (ec) return {};
    return p.parent_path().string();
#endif
}

std::vector<std::string> engine_search_dirs(std::string_view extra_dir) {
    std::vector<std::string> dirs;
    auto add = [&](const fs::path &p) {
        std::error_code ec;
        if (p.empty() || !fs::is_directory(p, ec)) return;
        std::string s = p.lexically_normal().string();
        if (std::find(dirs.begin(), dirs.end(), s) == dirs.end()) dirs.push_back(std::move(s));
    };
    if (!extra_dir.empty()) add(fs::path(extra_dir));
    const std::string exe = executable_dir();
    if (!exe.empty()) {
        const fs::path e(exe);
        add(e);
        add(e / "engines");
        std::error_code ec;
        // Build tree: <build>/engines/<name>/agent-engine-<name>.so
        if (fs::is_directory(e / "engines", ec))
            for (const auto &sub : fs::directory_iterator(e / "engines", ec))
                if (sub.is_directory(ec)) add(sub.path());
        add(e.parent_path() / "lib" / "agent" / "engines");
    }
    return dirs;
}

std::string find_plugin(std::string_view engine_name, std::span<const std::string> dirs) {
    const std::string file = plugin_file_name(engine_name);
    for (const auto &d : dirs) {
        std::error_code ec;
        const fs::path p = fs::path(d) / file;
        if (fs::is_regular_file(p, ec)) return p.string();
    }
    return {};
}

HardwareInfo detect_hardware() {
    HardwareInfo hw;
    hw.cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
#if defined(_WIN32)
    hw.nvidia_smi = on_path("nvidia-smi.exe");
#else
    hw.nvidia_smi = on_path("nvidia-smi");
    std::error_code ec;
    hw.nvidia_device = fs::exists("/dev/nvidia0", ec);
    if (fs::is_directory("/dev/dri", ec))
        for (const auto &e : fs::directory_iterator("/dev/dri", ec)) {
            const std::string name = e.path().filename().string();
            if (name.rfind("renderD", 0) == 0) hw.dri_nodes.push_back(e.path().string());
        }
    std::sort(hw.dri_nodes.begin(), hw.dri_nodes.end());
#endif
    return hw;
}

bool file_exists(std::string_view path) {
    std::error_code ec;
    return !path.empty() && fs::is_regular_file(fs::path(path), ec);
}

bool stdin_is_tty() {
#if defined(_WIN32)
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(STDIN_FILENO) != 0;
#endif
}

bool stream_is_tty(std::FILE *stream) {
#if defined(_WIN32)
    return _isatty(_fileno(stream)) != 0;
#else
    return isatty(fileno(stream)) != 0;
#endif
}

bool enable_ansi(std::FILE *stream) {
#if defined(_WIN32)
    // Windows 10+ understands ANSI escapes once ENABLE_VIRTUAL_TERMINAL_PROCESSING is set on the console.
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream)));
    DWORD mode = 0;
    if (h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) return false;
    return (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0 || SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
    (void)stream;
    return true;
#endif
}

int terminal_width(std::FILE *stream) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info;
    const HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream)));
    if (h != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(h, &info)) return info.srWindow.Right - info.srWindow.Left + 1;
#else
    struct winsize ws;
    if (ioctl(fileno(stream), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
#endif
    return 0;
}

ReadStatus read_line(std::FILE *in, std::string &line) {
    line.clear();
    char buf[1024];
    for (;;) {
        errno = 0;
        if (!std::fgets(buf, sizeof(buf), in)) {
            const bool interrupted = errno == EINTR;
            std::clearerr(in);
            if (interrupted) return ReadStatus::Interrupted;
            if (!line.empty()) return ReadStatus::Line;
            return ReadStatus::Eof;
        }
        line += buf;
        if (!line.empty() && line.back() == '\n') break;
    }
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    return ReadStatus::Line;
}

std::FILE *redirect_stderr(const std::string &path) {
    std::fflush(stderr);
#if defined(_WIN32)
    const int saved = _dup(2);
    if (saved < 0) return nullptr;
    const int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_APPEND, _S_IREAD | _S_IWRITE);
    if (fd < 0 || _dup2(fd, 2) != 0) {
        if (fd >= 0) _close(fd);
        _close(saved);
        return nullptr;
    }
    _close(fd);
    std::FILE *original = _fdopen(saved, "w");
    if (!original) {
        _dup2(saved, 2);
        _close(saved);
        return nullptr;
    }
#else
    const int saved = dup(STDERR_FILENO);
    if (saved < 0) return nullptr;
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0 || dup2(fd, STDERR_FILENO) < 0) {
        if (fd >= 0) close(fd);
        close(saved);
        return nullptr;
    }
    close(fd);
    std::FILE *original = fdopen(saved, "w");
    if (!original) {
        dup2(saved, STDERR_FILENO);
        close(saved);
        return nullptr;
    }
#endif
    std::setvbuf(original, nullptr, _IONBF, 0);
    return original;
}

void restore_stderr(std::FILE *original) {
    if (!original) return;
    std::fflush(stderr);
#if defined(_WIN32)
    _dup2(_fileno(original), 2);
#else
    dup2(fileno(original), STDERR_FILENO);
#endif
    std::fclose(original);
}

Result<void> open_with_os(const std::string &path) {
    std::error_code ec;
    if (!fs::exists(path, ec)) return fail(ErrorCode::FilesystemError, "no such file: " + path);
#if defined(_WIN32)
    const auto rc = reinterpret_cast<intptr_t>(ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (rc <= 32) return fail(ErrorCode::FilesystemError, "ShellExecute failed (" + std::to_string(rc) + ") for " + path);
    return ok();
#else
#if defined(__APPLE__)
    const char *opener = "open";
#else
    const char *opener = "xdg-open";
#endif
    // argv, not a shell command line: the path is never interpreted by a shell.
    std::string arg0 = opener;
    std::string arg1 = path;
    char *argv[] = {arg0.data(), arg1.data(), nullptr};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    const int rc = posix_spawnp(&pid, opener, &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) return fail(ErrorCode::FilesystemError, std::string("cannot run ") + opener + ": " + std::strerror(rc));
    // xdg-open / open return quickly after handing the file over; reap them so no zombie is left.
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        return fail(ErrorCode::FilesystemError, std::string(opener) + " exited with status " + std::to_string(WEXITSTATUS(status)));
    return ok();
#endif
}

} // namespace agent::cli
