// agent schedule add / ls / rm: scheduled runs through cron (Linux) or Task Scheduler (Windows). See schedule.hpp.

#include "commands.hpp"
#include "schedule.hpp"

#include <cstdio>
#include <string>

namespace agent::cli {

namespace {

constexpr std::string_view kTaskFolder = "agent.cpp";

std::string run_capture(const std::string &command) {
#if defined(_WIN32)
    std::FILE *pipe = _popen(command.c_str(), "r");
#else
    std::FILE *pipe = popen(command.c_str(), "r");
#endif
    std::string out;
    if (!pipe) return out;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return out;
}

int run_status(const std::string &command) { return std::system(command.c_str()); }

std::string agent_executable() {
    std::string dir = executable_dir();
#if defined(_WIN32)
    return dir + "\\agent.exe";
#else
    return dir + "/agent";
#endif
}

#if !defined(_WIN32)

std::string read_crontab() { return run_capture("crontab -l 2>/dev/null"); }

bool write_crontab(const std::string &text) {
    std::FILE *pipe = popen("crontab - 2>&1", "w");
    if (!pipe) return false;
    std::fwrite(text.data(), 1, text.size(), pipe);
    return pclose(pipe) == 0;
}

#endif

} // namespace

int cmd_schedule(const CommandLine &cl, Context &ctx) {
    const std::string &sub = cl.subcommand;

    if (sub == "ls") {
#if defined(_WIN32)
        const std::string out = run_capture("schtasks /Query /FO CSV /NH 2>nul");
        bool any = false;
        size_t at = 0;
        while (at < out.size()) {
            size_t end = out.find('\n', at);
            if (end == std::string::npos) end = out.size();
            const std::string line = out.substr(at, end - at);
            const std::string marker = "\"\\" + std::string(kTaskFolder) + "\\";
            const size_t pos = line.find(marker);
            if (pos == 0) {
                // "\agent.cpp\NAME","next run","status"
                const size_t close = line.find('"', marker.size());
                std::printf("%s", line.c_str());
                if (close == std::string::npos || line.back() != '\n') std::printf("\n");
                any = true;
            }
            at = end + 1;
        }
        if (!any) std::printf("no scheduled runs (agent schedule add NAME --every 15m -- run graph.json --yes)\n");
        return 0;
#else
        const auto entries = parse_crontab(read_crontab());
        if (entries.empty()) {
            std::printf("no scheduled runs (agent schedule add NAME --every 15m -- run graph.json --yes)\n");
            return 0;
        }
        for (const CronEntry &e : entries) std::printf("%-20s %-16s %s\n", e.name.c_str(), e.expression.c_str(), e.command.c_str());
        return 0;
#endif
    }

    if (cl.positionals.empty() || !valid_schedule_name(cl.positionals[0]))
        return report_error(make_error(ErrorCode::InvalidConfig,
                                       "give the schedule a name of letters, digits, '-' and '_' (agent schedule " + sub + " NAME)"));
    const std::string name = cl.positionals[0];

    if (sub == "rm") {
#if defined(_WIN32)
        const std::string task = std::string(kTaskFolder) + "\\" + name;
        if (run_status("schtasks /Delete /F /TN " + quote_windows(task) + " >nul 2>&1") != 0)
            return report_error(make_error(ErrorCode::KeyNotFound, "no scheduled run named " + name));
#else
        const std::string current = read_crontab();
        bool found = false;
        for (const CronEntry &e : parse_crontab(current)) found = found || e.name == name;
        if (!found) return report_error(make_error(ErrorCode::KeyNotFound, "no scheduled run named " + name));
        if (!write_crontab(crontab_without(current, name)))
            return report_error(make_error(ErrorCode::FilesystemError, "crontab could not be updated"));
#endif
        std::printf("removed %s\n", name.c_str());
        return 0;
    }

    // add
    const bool has_every = cl.has("every"), has_daily = cl.has("daily"), has_cron = cl.has("cron");
    if (static_cast<int>(has_every) + static_cast<int>(has_daily) + static_cast<int>(has_cron) != 1)
        return report_error(
            make_error(ErrorCode::InvalidConfig, "give exactly one of --every 15m, --daily 09:00 or --cron \"m h dom mon dow\""));
    if (cl.positionals.size() < 2)
        return report_error(make_error(ErrorCode::InvalidConfig, "say what to run after --, for example: agent schedule add " + name +
                                                                     " --every 1h -- run graph.json --yes"));

    // The command: this executable, the same home, then what the user wrote.
    const std::string home = ctx.has_home ? ctx.home.root : std::string();
    std::vector<std::string> args;
    if (!home.empty()) {
        args.push_back("--home");
        args.push_back(home);
    }
    for (size_t i = 1; i < cl.positionals.size(); ++i) args.push_back(cl.positionals[i]);

#if defined(_WIN32)
    if (has_cron)
        return report_error(
            make_error(ErrorCode::UnsupportedProvider, "Task Scheduler has no cron syntax: use --every or --daily on Windows"));
    std::string tr = quote_windows(agent_executable());
    for (const std::string &a : args) tr += " " + quote_windows(a);
    std::string schedule;
    if (has_daily) {
        auto check = cron_from_daily(cl.get("daily"));
        if (!check.ok) return report_error(check.error);
        schedule = "/SC DAILY /ST " + std::string(cl.get("daily"));
    } else {
        const std::string every(cl.get("every"));
        auto check = cron_from_every(every);
        if (!check.ok) return report_error(check.error);
        const std::string n = every.substr(0, every.size() - 1);
        const char unit = every.back();
        schedule = unit == 'm' ? "/SC MINUTE /MO " + n : unit == 'h' ? "/SC HOURLY /MO " + n : "/SC DAILY";
    }
    const std::string task = std::string(kTaskFolder) + "\\" + name;
    const std::string command = "schtasks /Create /F /TN " + quote_windows(task) + " /TR " + quote_windows(tr) + " " + schedule + " >nul";
    if (run_status(command) != 0)
        return report_error(make_error(ErrorCode::SandboxViolation, "schtasks could not create the task (run: " + command + ")"));
    std::printf("scheduled %s: %s\n", name.c_str(), tr.c_str());
#else
    Result<std::string> expr = has_cron    ? check_cron(cl.get("cron"))
                               : has_daily ? cron_from_daily(cl.get("daily"))
                                           : cron_from_every(cl.get("every"));
    if (!expr.ok) return report_error(expr.error);
    std::string command = quote_posix(agent_executable());
    for (const std::string &a : args) command += " " + quote_posix(a);
    const std::string current = crontab_without(read_crontab(), name); // adding an existing name replaces it
    const std::string updated =
        current + (current.empty() || current.back() == '\n' ? "" : "\n") + crontab_block(name, expr.value, command);
    if (!write_crontab(updated))
        return report_error(make_error(ErrorCode::FilesystemError, "crontab is not available or could not be updated"));
    std::printf("scheduled %s: %s  %s\n", name.c_str(), expr.value.c_str(), command.c_str());
#endif
    std::printf(
        "it runs unattended: add --yes to the command if its tools need approval, and use agent run --resume for a waiting graph\n");
    return 0;
}

} // namespace agent::cli
