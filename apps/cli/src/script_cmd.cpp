// `agent setup` and `agent update` are scripts, not code in this program: they are fetched from the repository (scripts/setup.sh,
// scripts/setup.ps1, scripts/install.sh, scripts/install.ps1) and run, so they can be fixed and improved without a new release.
// Also `agent home [DIR]`: where agent.cpp keeps its files (the one thing the setup script needs from the program itself).

#include "commands.hpp"
#include "home.hpp"

#include <agent-cpp/hub.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#if defined(_WIN32)
#include <process.h>
#endif

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

constexpr std::string_view kDefaultScripts = "https://raw.githubusercontent.com/amitverma-cf/agent.cpp/main/scripts";

std::string scripts_base() {
    const char *env = std::getenv("AGENT_SCRIPTS");
    return env && *env ? std::string(env) : std::string(kDefaultScripts);
}

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

#if defined(_WIN32)
constexpr const char *kExtension = ".ps1";
#else
constexpr const char *kExtension = ".sh";
#endif

// Downloads scripts/<name>.<sh|ps1> and returns the file it was saved to.
Result<std::string> fetch_script(const std::string &name) {
    const std::string url = scripts_base() + "/" + name + kExtension;
    if (scripts_base().rfind("http", 0) != 0) {
        // AGENT_SCRIPTS can be a folder: for tests and for people who keep their own copy.
        const fs::path local = fs::path(scripts_base()) / (name + kExtension);
        if (!fs::exists(local)) return fail<std::string>(ErrorCode::FilesystemError, "no script at " + local.string());
        return ok(local.string());
    }
    auto text = agent::hub::http_get_text(url, agent::hub::http_options_from_env(), 1u << 20);
    if (!text.ok) return fail<std::string>(text.error.code, "cannot fetch " + url + ": " + text.error.message);
    std::error_code ec;
    const fs::path file = fs::temp_directory_path(ec) /
                          ("agent-" + name + "-" + std::to_string(static_cast<unsigned long long>(std::time(nullptr))) + kExtension);
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << text.value;
    if (!out) return fail<std::string>(ErrorCode::FilesystemError, "cannot write " + file.string());
    return ok(file.string());
}

// The options of the command line again, for the script: --name or --name VALUE, then the positionals.
std::vector<std::string> forwarded_args(const CommandLine &cl) {
    std::vector<std::string> args;
    for (const Flag &f : cl.flags) {
        args.push_back("--" + f.name);
        if (!f.value.empty()) args.push_back(f.value);
    }
    for (const std::string &p : cl.positionals) args.push_back(p);
    return args;
}

void set_env(const char *name, const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

} // namespace

int run_remote_script(const std::string &name, const std::vector<std::string> &args, bool detach) {
    auto file = fetch_script(name);
    if (!file.ok) {
        std::fprintf(stderr,
                     "error: %s\n(no network? the scripts are in the repository under scripts/; AGENT_SCRIPTS can point at a copy)\n",
                     file.error.message.c_str());
        return 1;
    }
    std::string command;
#if defined(_WIN32)
    command = "powershell -NoProfile -ExecutionPolicy Bypass -File " + shell_quote(file.value);
    for (const std::string &a : args) command += " " + shell_quote(a);
    if (detach) {
        // Windows will not replace a program that is running: the script waits for this process to exit, in a window of its own.
        set_env("AGENT_WAIT_PID", std::to_string(_getpid()));
        std::printf("continuing in a new window; this program exits now\n");
        std::fflush(stdout);
        return std::system(("start \"agent " + name + "\" " + command).c_str()) == 0 ? 0 : 1;
    }
#else
    (void)detach;
    command = "sh " + shell_quote(file.value);
    for (const std::string &a : args) command += " " + shell_quote(a);
#endif
    const std::string exe = executable_dir();
    if (!exe.empty()) {
#if defined(_WIN32)
        set_env("AGENT_BIN", (fs::path(exe) / "agent.exe").string());
#else
        set_env("AGENT_BIN", (fs::path(exe) / "agent").string());
#endif
    }
    std::fflush(stdout);
    const int rc = std::system(command.c_str());
    std::error_code ec;
    if (scripts_base().rfind("http", 0) == 0) fs::remove(file.value, ec);
    return rc == 0 ? 0 : 1;
}

namespace {
// Whether the folder of the running program was made by the installer (so it is the installer's to remove, not a build tree).
bool installed_program(const fs::path &dir) {
    std::error_code ec;
    return fs::is_regular_file(dir / ".version", ec) || !install_root().empty();
}
std::string lexical(const fs::path &p) { return p.lexically_normal().string(); }
bool within(const fs::path &path, const fs::path &folder) {
    const fs::path rel = path.lexically_normal().lexically_relative(folder.lexically_normal());
    return !rel.empty() && rel.native().rfind(fs::path("..").native(), 0) != 0;
}
} // namespace

int cmd_uninstall(const CommandLine &cl, Context &ctx) {
    const std::string exe_dir = executable_dir();
    const fs::path program = exe_dir.empty() ? fs::path() : fs::path(exe_dir);
    const bool remove_program = !program.empty() && installed_program(program);
    const std::string pointer = home_pointer_path();
    std::error_code ec;
    const bool has_data = ctx.has_home && !cl.has("keep-data") && fs::is_directory(ctx.home.root, ec);
    std::printf("agent uninstall removes:\n");
    if (has_data) std::printf("  the data folder   %s  (models, engines, conversations, notes, logs, caches)\n", ctx.home.root.c_str());
    else if (ctx.has_home) std::printf("  nothing of the data folder %s (--keep-data)\n", ctx.home.root.c_str());
    if (remove_program) std::printf("  the program       %s\n", lexical(program).c_str());
    else std::printf("  nothing of the program: %s does not look like an installed copy (a build or a package manager)\n", exe_dir.c_str());
    const bool pointer_outside =
        has_data && fs::exists(pointer, ec) && !within(pointer, ctx.home.root) && !(remove_program && within(pointer, program));
    if (pointer_outside) std::printf("  the pointer file  %s\n", pointer.c_str());
    if (remove_program) std::printf("  the PATH entry of the program\n");
    if (!has_data && !remove_program) {
        std::printf("nothing to remove\n");
        return 0;
    }
    if (!cl.has("yes")) {
        if (!stdin_is_tty()) {
            std::fprintf(stderr, "agent uninstall: removing files needs a yes: run it in a terminal, or pass --yes\n");
            return 1;
        }
        std::printf("This cannot be undone. Type yes to remove it: ");
        std::fflush(stdout);
        char line[32] = {0};
        if (!std::fgets(line, sizeof(line), stdin) || std::string(line).rfind("yes", 0) != 0) {
            std::printf("nothing was removed\n");
            return 1;
        }
    }
    set_env("AGENT_UNINSTALL_HOME", has_data ? lexical(fs::path(ctx.home.root)) : std::string());
    set_env("AGENT_UNINSTALL_PROGRAM", remove_program ? lexical(program) : std::string());
    set_env("AGENT_UNINSTALL_POINTER", pointer_outside ? pointer : std::string());
    return run_remote_script("uninstall", {}, true);
}

int cmd_setup(const CommandLine &cl, Context &) { return run_remote_script("setup", forwarded_args(cl), false); }

int cmd_update(const CommandLine &cl, Context &) {
    // The installer does the update: the same version is left alone, a newer one replaces this one in place. It does not set anything up.
    set_env("AGENT_NO_START", "1");
    if (cl.has("check")) set_env("AGENT_CHECK", "1");
    if (cl.has("tag")) set_env("AGENT_VERSION", std::string(cl.get("tag")));
    return run_remote_script("install", {}, true);
}

int cmd_home(const CommandLine &cl, Context &ctx) {
    if (cl.positionals.empty()) {
        std::printf("%s\n", ctx.has_home ? ctx.home.root.c_str() : default_home_suggestion().c_str());
        return 0;
    }
    ctx.home = home_layout(fs::absolute(cl.positionals[0]).string());
    ctx.has_home = true;
    if (auto r = create_home_layout(ctx.home); !r.ok) return report_error(r.error);
    if (auto r = save_home_pointer(home_pointer_path(), ctx.home.root); !r.ok) return report_error(r.error);
    std::printf("%s\n", ctx.home.root.c_str());
    return 0;
}

} // namespace agent::cli
