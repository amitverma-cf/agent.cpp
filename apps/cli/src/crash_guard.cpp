#include "crash_guard.hpp"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace agent::cli {

namespace {

std::string g_dir;       // <home>/cache/crashed
char g_marker[512] = ""; // the file the handler writes; "" = no engine call is running
char g_model[400] = "";  // what is written in it

#if !defined(_WIN32)
constexpr int kSignals[] = {SIGABRT, SIGSEGV, SIGFPE, SIGILL, SIGBUS};
constexpr int kSignalCount = sizeof(kSignals) / sizeof(kSignals[0]);
// The handlers that were there when ours went in (the full-screen chat installs its own, which put the terminal back).
struct sigaction g_previous[kSignalCount];
#endif

// Only async-signal-safe calls here: open, write, close, sigaction, raise.
void on_crash(int sig) {
    if (g_marker[0]) {
#if defined(_WIN32)
        const int fd = _open(g_marker, _O_WRONLY | _O_CREAT | _O_TRUNC, _S_IREAD | _S_IWRITE);
        if (fd >= 0) {
            _write(fd, g_model, static_cast<unsigned>(std::strlen(g_model)));
            _close(fd);
        }
#else
        const int fd = open(g_marker, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0) {
            const ssize_t n = write(fd, g_model, std::strlen(g_model));
            (void)n;
            close(fd);
        }
#endif
    }
#if defined(_WIN32)
    std::signal(sig, SIG_DFL);
    std::raise(sig);
#else
    struct sigaction dfl{};
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    sigaction(sig, &dfl, nullptr);
    void (*previous)(int) = SIG_DFL;
    for (int i = 0; i < kSignalCount; ++i)
        if (kSignals[i] == sig) previous = g_previous[i].sa_handler;
    // The signal is blocked while this handler runs, so a raise() in the previous handler lands on return, with the default action.
    if (previous != SIG_DFL && previous != SIG_IGN && previous != on_crash) {
        previous(sig);
    } else {
        static const char restore[] = "\033[?1049l\033[?25h\033[0m";
        const ssize_t m = write(2, restore, sizeof(restore) - 1);
        (void)m;
    }
    if (g_marker[0]) {
        static const char note[] = "\nagent: an engine crashed; the model that was running will get its own process next time. "
                                   "Start agent again.\n";
        const ssize_t m = write(2, note, sizeof(note) - 1);
        (void)m;
    }
    std::raise(sig);
#endif
}

// Puts on_crash in front of whatever handles the crash signals now. Called again at every engine call, because the full-screen chat
// installs its handlers after the program has started.
void hook_signals() {
#if defined(_WIN32)
    for (const int sig : {SIGABRT, SIGSEGV, SIGFPE, SIGILL}) std::signal(sig, on_crash);
#else
    for (int i = 0; i < kSignalCount; ++i) {
        struct sigaction now{};
        if (sigaction(kSignals[i], nullptr, &now) != 0 || now.sa_handler == on_crash) continue;
        struct sigaction sa{};
        sa.sa_handler = on_crash;
        sigemptyset(&sa.sa_mask);
        if (sigaction(kSignals[i], &sa, nullptr) == 0) g_previous[i] = now;
    }
#endif
}

} // namespace

std::string crash_marker_path(std::string_view model_path) {
    if (g_dir.empty() || model_path.empty()) return {};
    return (std::filesystem::path(g_dir) / (std::to_string(std::hash<std::string_view>{}(model_path)) + ".txt")).string();
}

void crash_guard_install(const std::string &home_cache) {
    std::error_code ec;
    g_dir = (std::filesystem::path(home_cache) / "crashed").string();
    std::filesystem::create_directories(g_dir, ec);
    hook_signals();
}

CrashScope::CrashScope(std::string_view model_path) {
    std::memcpy(previous_, g_marker, sizeof(previous_));
    const std::string marker = crash_marker_path(model_path);
    if (marker.empty() || marker.size() >= sizeof(g_marker)) return;
    hook_signals();
    std::snprintf(g_model, sizeof(g_model), "%.*s", static_cast<int>(model_path.size()), model_path.data());
    std::memcpy(g_marker, marker.c_str(), marker.size() + 1);
}

CrashScope::~CrashScope() { std::memcpy(g_marker, previous_, sizeof(previous_)); }

bool crashed_before(std::string_view model_path) {
    const std::string marker = crash_marker_path(model_path);
    std::error_code ec;
    return !marker.empty() && std::filesystem::exists(marker, ec);
}

void crash_forget(std::string_view model_path) {
    std::error_code ec;
    if (const std::string marker = crash_marker_path(model_path); !marker.empty()) std::filesystem::remove(marker, ec);
}

} // namespace agent::cli
