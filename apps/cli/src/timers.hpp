#pragma once

// The agent's own timers: "do X in 20 minutes", "every morning at 9 check Y". They are kept in <home>/timers.json and fire while the chat
// is open: the request is put in front of the agent as a message when it is idle. A timer that came due while the program was closed fires
// once when it is opened again. (For work that must run with the program closed there is `agent schedule`, which uses the system scheduler.)

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

struct Timer {
    std::string id;            // short, unique
    std::string prompt;        // what the agent is asked when it fires
    std::string workspace;     // the project it belongs to
    int64_t next_epoch = 0;    // seconds since the Unix epoch
    int64_t every_seconds = 0; // 0 = once
};

std::vector<Timer> timers_from_json(std::string_view json);
std::string timers_to_json(const std::vector<Timer> &timers);

std::vector<Timer> load_timers(const std::string &home_root);
bool save_timers(const std::string &home_root, const std::vector<Timer> &timers);

// Seconds from `now` (local time) to the next HH:MM. -1 when the text is not a time.
int64_t seconds_until_clock(std::string_view hh_mm, int64_t now_epoch);

// Takes the timers of `workspace` ("*" = of every project) that are due at `now` out of the list (a repeating one is moved to its next time instead, skipping the
// runs that were missed) and returns them.
std::vector<Timer> take_due_timers(std::vector<Timer> &timers, std::string_view workspace, int64_t now_epoch);

// The timer tool: {"action":"add","prompt":"...","in_minutes":20} | {"action":"add","prompt":"...","at":"09:00","repeat":true} |
// {"action":"add","prompt":"...","every_minutes":60} | {"action":"list"} | {"action":"remove","id":"..."}. Returns the answer for the
// model; `failed` is set for a request that could not be done.
std::string timer_tool(const std::string &home_root, const std::string &workspace, std::string_view arguments_json, int64_t now_epoch,
                       bool &failed);

} // namespace agent::cli
