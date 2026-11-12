#pragma once

// Runs paused at a graph "wait" node, remembered in <workspace>/waiting.json so that `agent wake` (usually started by a scheduler) can
// resume the ones that are due. Pure part: the list, its JSON form, and the command line that resumes an entry.

#include <agent-cpp/agent.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

struct WaitingRun {
    std::string thread;               // graph thread id (agent run --resume THREAD)
    std::string graph;                // absolute path of the graph file
    int64_t until = 0;                // wake time, epoch seconds
    std::vector<std::string> flags;   // the original flags of the run, as words: "--model", "qwen", "--yes", ...
};

std::string waiting_to_json(const std::vector<WaitingRun> &runs);
Result<std::vector<WaitingRun>> waiting_from_json(std::string_view json);

// Adds the run, or replaces the entry of the same thread.
void waiting_upsert(std::vector<WaitingRun> &runs, WaitingRun run);
// Removes the entry of the thread; false when there was none.
bool waiting_remove(std::vector<WaitingRun> &runs, std::string_view thread);
// The entries whose wake time has come, soonest first.
std::vector<WaitingRun> waiting_due(const std::vector<WaitingRun> &runs, int64_t now);

// The words after `agent` that resume the entry: run GRAPH <flags> --resume THREAD.
std::vector<std::string> waiting_resume_args(const WaitingRun &run);

} // namespace agent::cli
