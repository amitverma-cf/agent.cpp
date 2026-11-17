// Everything that installs, changes or removes agent.cpp is a script, not code in this program: scripts/install.sh and scripts/install.ps1
// (see them for what each command does). The program only hands over to the copy of the script that was installed with it
// (<home>/scripts/), so the script and the list of components it uses always belong to the installed release and stay pinned until
// `agent update`. These commands are that hand-over:
//   agent setup | update | uninstall                   -> the script's setup, update, uninstall
//   agent engine pull NAME | agent runtime install NAME -> add;  agent engine rm NAME | agent runtime rm NAME -> rm
// Also `agent home`: where everything is kept (the environment variable AGENT_HOME, set by the installer).

#include "commands.hpp"
#include "home.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#endif

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

// A word for a shell command line: single quotes on Linux, double quotes on Windows.
std::string shell_quote(const std::string &s) {
#if defined(_WIN32)
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"') out += '\\';
        out += c;
    }
    return out + "\"";
#else
    std::string out = "'";
    for (const char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out + "'";
#endif
}

void set_env(const char *name, const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// The options of the command line again, for the script: --name or --name VALUE, then the positionals.
std::vector<std::string> forwarded_args(const CommandLine &cl) {
    std::vector<std::string> args;
    for (const Flag &f : cl.flags) {
        if (f.name == "home" || f.name == "config-dir") continue; // the program's own, not the script's
        args.push_back("--" + f.name);
        if (!f.value.empty()) args.push_back(f.value);
    }
    for (const std::string &p : cl.positionals) args.push_back(p);
    return args;
}

std::string join_names(const std::vector<std::string> &names) {
    std::string out;
    for (const std::string &n : names) out += (out.empty() ? "" : ",") + n;
    return out;
}

} // namespace

// Runs `command` of the installed script with `args`. `detach`: on Windows the script waits in a window of its own for this program to
// exit (a running program cannot be replaced or deleted); elsewhere it just runs.
int run_install_script(const Context &ctx, const std::string &command, const std::vector<std::string> &args, bool detach) {
    if (!ctx.has_home) {
        std::fprintf(stderr,
                     "agent %s: this copy of agent was not installed by the installer, so there is no folder to work in. Install with:\n"
#if defined(_WIN32)
                     "  irm https://agent.qozu.fyi/install.ps1 | iex\n",
#else
                     "  curl -fsSL https://agent.qozu.fyi/install.sh | sh\n",
#endif
                     command.c_str());
        return 1;
    }
#if defined(_WIN32)
    const char *ext = ".ps1";
#else
    const char *ext = ".sh";
#endif
    fs::path script = fs::path(ctx.home.root) / "scripts" / (std::string("install") + ext);
    if (const char *own = std::getenv("AGENT_SCRIPTS"); own && *own && std::string(own).rfind("http", 0) != 0)
        if (std::error_code ec; !fs::exists(script, ec))
            script = fs::path(own) / (std::string("install") + ext); // tests and offline copies
    std::error_code ec;
    if (!fs::exists(script, ec)) {
        std::fprintf(stderr, "agent %s: %s is missing; reinstall with the installer one-liner (agent.qozu.fyi)\n", command.c_str(),
                     script.string().c_str());
        return 1;
    }
    set_env("AGENT_HOME", ctx.home.root);
    std::string line;
#if defined(_WIN32)
    line = "powershell -NoProfile -ExecutionPolicy Bypass -File " + shell_quote(script.string()) + " " + command;
    for (const std::string &a : args) line += " " + shell_quote(a);
    if (detach) {
        set_env("AGENT_WAIT_PID", std::to_string(_getpid()));
        std::printf("continuing in a new window; this program exits now\n");
        std::fflush(stdout);
        return std::system(("start \"agent " + command + "\" " + line).c_str()) == 0 ? 0 : 1;
    }
#else
    (void)detach;
    line = "sh " + shell_quote(script.string()) + " " + command;
    for (const std::string &a : args) line += " " + shell_quote(a);
#endif
    std::fflush(stdout);
    return std::system(line.c_str()) == 0 ? 0 : 1;
}

int cmd_setup(const CommandLine &cl, Context &ctx) { return run_install_script(ctx, "setup", forwarded_args(cl), false); }

int cmd_update(const CommandLine &cl, Context &ctx) {
    std::vector<std::string> args = forwarded_args(cl);
    return run_install_script(ctx, "update", args, !cl.has("check"));
}

int cmd_uninstall(const CommandLine &cl, Context &ctx) { return run_install_script(ctx, "uninstall", forwarded_args(cl), true); }

// `agent engine pull NAME`, `agent runtime install NAME`: add components of the installed release.
int add_components(const Context &ctx, const std::vector<std::string> &names, const CommandLine &cl) {
    std::vector<std::string> args = {join_names(names)};
    if (cl.has("backend")) {
        args.push_back("--backend");
        args.push_back(std::string(cl.get("backend")));
    }
    return run_install_script(ctx, "add", args, false);
}

// `agent engine rm NAME`, `agent runtime rm NAME`.
int remove_components(const Context &ctx, const std::vector<std::string> &names) {
    return run_install_script(ctx, "rm", {join_names(names)}, false);
}

int cmd_home(const CommandLine &, Context &ctx) {
    if (!ctx.has_home) {
        std::fprintf(stderr, "no home: AGENT_HOME is not set. Install with the installer one-liner (agent.qozu.fyi)\n");
        return 1;
    }
    std::printf("%s\n", ctx.home.root.c_str());
    return 0;
}

} // namespace agent::cli
