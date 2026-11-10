#pragma once

// `agent doctor` and `agent clean`: where the disk space in the global home goes, and what can be removed without
// losing anything that cannot be rebuilt. Models, skills, MCP definitions, memory and secrets are never cleaned.

#include "home.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace agent::cli {

struct UsageEntry {
    std::string label; // "models", "cache/downloads", ...
    std::string path;
    uint64_t bytes = 0;
    uint64_t files = 0;
};

// Bytes and file count under a folder (0 when it does not exist).
UsageEntry measure_folder(const std::string &label, const std::string &path);

// One entry per area of the home, plus the pieces of cache/ that clean can remove.
std::vector<UsageEntry> measure_home(const HomeLayout &home);

enum class CleanTarget { Downloads, Cache, Logs };

struct CleanPlan {
    std::vector<std::string> paths; // files and folders that would be removed
    uint64_t bytes = 0;
};

// What `clean` would remove for the targets:
//   Downloads: partial model downloads (*.part) under models/ and cache/
//   Cache:     everything inside cache/ (kernel and shader caches, saved prefix states); rebuilt on demand
//   Logs:      files in logs/ older than `log_days`
CleanPlan plan_clean(const HomeLayout &home, const std::vector<CleanTarget> &targets, int log_days = 14);

// Removes the planned paths; returns how many bytes were freed.
uint64_t run_clean(const CleanPlan &plan);

// "1.4 GiB", "37 MiB", "512 B".
std::string format_bytes(uint64_t bytes);

} // namespace agent::cli
