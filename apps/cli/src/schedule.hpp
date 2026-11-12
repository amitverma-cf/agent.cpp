#pragma once

// `agent schedule`: run an agent command on a schedule with the operating system's own scheduler (cron on Linux, Task Scheduler on
// Windows). agent.cpp keeps no daemon and no timer of its own. This file is the pure half: turning "every 15 minutes" into a cron
// expression, building and editing crontab text, quoting. The part that calls crontab / schtasks is in schedule_cmd.cpp.

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// "15m", "2h", "1d" -> a cron expression ("*/15 * * * *", "0 */2 * * *", "0 0 * * *"). Minutes must divide 60 and hours 24,
// so the interval is regular. Whole days are 1d only.
Result<std::string> cron_from_every(std::string_view every);
// "09:00" or "9:5" -> "0 9 * * *" style: every day at that local time.
Result<std::string> cron_from_daily(std::string_view hh_mm);
// Five fields of a cron expression, checked for shape only.
Result<std::string> check_cron(std::string_view expression);

// POSIX shell quoting: plain words stay as they are, everything else goes in single quotes.
std::string quote_posix(std::string_view word);
// Windows command line quoting (CommandLineToArgvW rules): words with spaces or quotes go in double quotes.
std::string quote_windows(std::string_view word);

struct CronEntry {
    std::string name;       // the tag after "# agent.cpp:"
    std::string expression; // the five cron fields
    std::string command;    // the rest of the line
};

// A crontab line pair: the tag comment, then "<expression> <command>".
std::string crontab_block(std::string_view name, std::string_view expression, std::string_view command);
// The agent.cpp entries of a crontab; other lines are ignored.
std::vector<CronEntry> parse_crontab(std::string_view crontab);
// The crontab without the entry `name` (every other line is kept byte for byte).
std::string crontab_without(std::string_view crontab, std::string_view name);

// Names are used in the tag and in a Task Scheduler path: letters, digits, '-' and '_' only.
bool valid_schedule_name(std::string_view name);

} // namespace agent::cli
