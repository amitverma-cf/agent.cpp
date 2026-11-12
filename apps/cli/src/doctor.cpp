#include "doctor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <system_error>

namespace agent::cli {

namespace fs = std::filesystem;

UsageEntry measure_folder(const std::string &label, const std::string &path) {
    UsageEntry e;
    e.label = label;
    e.path = path;
    std::error_code ec;
    if (!fs::is_directory(path, ec)) return e;
    for (auto it = fs::recursive_directory_iterator(path, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code fe;
        if (!it->is_regular_file(fe) || fe) continue;
        const auto size = it->file_size(fe);
        if (!fe) {
            e.bytes += size;
            ++e.files;
        }
    }
    return e;
}

std::vector<UsageEntry> measure_home(const HomeLayout &home) {
    std::vector<UsageEntry> out;
    out.push_back(measure_folder("models", home.models));
    out.push_back(measure_folder("engines", home.engines));
    out.push_back(measure_folder("runtimes", home.runtimes));
    out.push_back(measure_folder("skills", home.skills));
    out.push_back(measure_folder("mcp", home.mcp));
    out.push_back(measure_folder("plugins", home.plugins));
    out.push_back(measure_folder("cache", home.cache));
    out.push_back(measure_folder("logs", home.logs));
    out.push_back(measure_folder("workspaces", home.workspaces));
    return out;
}

namespace {

bool ends_with(const std::string &s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void add_path(CleanPlan &plan, const fs::path &p, uint64_t bytes) {
    plan.paths.push_back(p.string());
    plan.bytes += bytes;
}

} // namespace

CleanPlan plan_clean(const HomeLayout &home, const std::vector<CleanTarget> &targets, int log_days) {
    CleanPlan plan;
    auto wants = [&](CleanTarget t) { return std::find(targets.begin(), targets.end(), t) != targets.end(); };
    std::error_code ec;

    if (wants(CleanTarget::Cache)) {
        // The whole cache folder's contents; the folder itself stays.
        if (fs::is_directory(home.cache, ec))
            for (const auto &entry : fs::directory_iterator(home.cache, ec)) {
                const UsageEntry u = measure_folder("", entry.path().string());
                std::error_code fe;
                const uint64_t size =
                    entry.is_directory(fe) ? u.bytes : (entry.is_regular_file(fe) ? static_cast<uint64_t>(entry.file_size(fe)) : 0);
                add_path(plan, entry.path(), size);
            }
    } else if (wants(CleanTarget::Downloads)) {
        for (const std::string &root : {home.models, home.cache}) {
            if (!fs::is_directory(root, ec)) continue;
            for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
                 !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
                std::error_code fe;
                if (it->is_regular_file(fe) && ends_with(it->path().string(), ".part")) add_path(plan, it->path(), it->file_size(fe));
            }
        }
    }
    if (wants(CleanTarget::Cache) && wants(CleanTarget::Downloads)) {
        // Partial downloads inside models/ are not under cache/: add them as well.
        if (fs::is_directory(home.models, ec))
            for (auto it = fs::recursive_directory_iterator(home.models, fs::directory_options::skip_permission_denied, ec);
                 !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
                std::error_code fe;
                if (it->is_regular_file(fe) && ends_with(it->path().string(), ".part")) add_path(plan, it->path(), it->file_size(fe));
            }
    }

    if (wants(CleanTarget::Logs) && fs::is_directory(home.logs, ec)) {
        const auto cutoff = fs::file_time_type::clock::now() - std::chrono::hours(24 * std::max(0, log_days));
        for (const auto &entry : fs::directory_iterator(home.logs, ec)) {
            std::error_code fe;
            if (!entry.is_regular_file(fe)) continue;
            const auto when = entry.last_write_time(fe);
            if (!fe && when < cutoff) add_path(plan, entry.path(), entry.file_size(fe));
        }
    }
    return plan;
}

uint64_t run_clean(const CleanPlan &plan) {
    uint64_t freed = 0;
    for (const std::string &p : plan.paths) {
        std::error_code ec;
        const UsageEntry u = measure_folder("", p);
        const uint64_t size =
            fs::is_directory(p, ec) ? u.bytes : (fs::is_regular_file(p, ec) ? static_cast<uint64_t>(fs::file_size(p, ec)) : 0);
        const auto removed = fs::remove_all(p, ec);
        if (!ec && removed > 0) freed += size;
    }
    return freed;
}

std::string format_bytes(uint64_t bytes) {
    char buf[32];
    if (bytes >= (1ULL << 30)) std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    else if (bytes >= (1ULL << 20)) std::snprintf(buf, sizeof(buf), "%.0f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    else if (bytes >= (1ULL << 10)) std::snprintf(buf, sizeof(buf), "%.0f KiB", static_cast<double>(bytes) / 1024.0);
    else std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
    return buf;
}

} // namespace agent::cli
