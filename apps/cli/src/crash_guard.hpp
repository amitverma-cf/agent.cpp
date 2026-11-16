#pragma once

// Self-healing for engines that run inside the program. An engine that aborts or segfaults takes the program with it; the crash handler
// leaves a small marker file naming the model that was running. The next time that model is loaded it gets its own agent-engine-host
// process instead, so the same fault fails one answer, not the chat.

#include <string>
#include <string_view>

namespace agent::cli {

// Installs the handlers for SIGABRT, SIGSEGV, SIGFPE, SIGILL (and SIGBUS). Call once at start; `home_cache` is <home>/cache.
void crash_guard_install(const std::string &home_cache);

// While one of these lives, a crash is recorded against `model_path`. They nest: the inner one wins until it ends.
class CrashScope {
  public:
    explicit CrashScope(std::string_view model_path);
    ~CrashScope();
    CrashScope(const CrashScope &) = delete;
    CrashScope &operator=(const CrashScope &) = delete;

  private:
    char previous_[512];
};

// True when a crash was recorded against this model (in this or an earlier run of the program).
bool crashed_before(std::string_view model_path);
// Forgets the record (the user asked to try in-process again).
void crash_forget(std::string_view model_path);
// The marker file for a model: <home>/cache/crashed/<hash>.txt. Empty before crash_guard_install.
std::string crash_marker_path(std::string_view model_path);

} // namespace agent::cli
