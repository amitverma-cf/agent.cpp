// agent wake: resume the graph runs that are paused at a wait node and due. See waiting.hpp.
//
// agent run records a run that stops at a wait node in <workspace>/waiting.json; `agent wake` reads that list and runs
// `agent run GRAPH <the original flags> --resume THREAD` for every entry whose time has come (each resume updates the list itself:
// finished runs leave it, a run that waits again gets its new time). Start it from `agent schedule add wake --every 15m -- wake --yes`.

#include "commands.hpp"
#include "schedule.hpp"
#include "waiting.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>

namespace agent::cli {

namespace {

std::string waiting_path(const std::string &workspace) { return (std::filesystem::path(workspace) / "waiting.json").string(); }

std::string read_all(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string local_text(int64_t epoch) {
    const std::time_t t = static_cast<std::time_t>(epoch);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

std::string shell_quote(const std::string &word) {
#if defined(_WIN32)
    return quote_windows(word);
#else
    return quote_posix(word);
#endif
}

} // namespace

std::vector<WaitingRun> load_waiting(const std::string &workspace) {
    auto parsed = waiting_from_json(read_all(waiting_path(workspace)));
    return parsed.ok ? std::move(parsed.value) : std::vector<WaitingRun>{};
}

void save_waiting(const std::string &workspace, const std::vector<WaitingRun> &runs) {
    std::error_code ec;
    std::filesystem::create_directories(workspace, ec);
    const std::string path = waiting_path(workspace);
    if (runs.empty()) {
        std::filesystem::remove(path, ec);
        return;
    }
    std::ofstream(path, std::ios::binary | std::ios::trunc) << waiting_to_json(runs);
}

int cmd_wake(const CommandLine &cl, Context &ctx) {
    const std::string workspace = workspace_dir(ctx, cl.get("workspace"));
    const auto runs = load_waiting(workspace);
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto due = waiting_due(runs, now);

    if (runs.empty()) {
        std::printf("no runs are waiting in %s\n", workspace.c_str());
        return 0;
    }
    for (const WaitingRun &r : runs)
        std::printf("%s  %-24s %s%s\n", local_text(r.until).c_str(), r.thread.c_str(), r.graph.c_str(), r.until <= now ? "   (due)" : "");
    if (due.empty()) {
        std::printf("none is due yet\n");
        return 0;
    }
    if (cl.has("dry-run")) {
        std::printf("%zu due (dry run: nothing resumed)\n", due.size());
        return 0;
    }

    std::string exe = executable_dir();
#if defined(_WIN32)
    exe += "\\agent.exe";
#else
    exe += "/agent";
#endif
    int failed = 0;
    for (const WaitingRun &r : due) {
        std::string command = shell_quote(exe);
        if (ctx.has_home) command += " --home " + shell_quote(ctx.home.root);
        for (const std::string &w : waiting_resume_args(r)) command += " " + shell_quote(w);
        // From the directory the run started in: relative paths among its flags (a model file, say) must still point at the same files.
        if (!r.cwd.empty()) {
#if defined(_WIN32)
            command = "cd /d " + shell_quote(r.cwd) + " && " + command;
#else
            command = "cd " + shell_quote(r.cwd) + " && " + command;
#endif
        }
        std::printf("resuming %s ...\n", r.thread.c_str());
        std::fflush(stdout);
        if (std::system(command.c_str()) != 0) {
            std::printf("resuming %s failed\n", r.thread.c_str());
            ++failed;
        }
    }
    std::printf("%zu resumed, %d failed\n", due.size(), failed);
    return failed ? 1 : 0;
}

} // namespace agent::cli
