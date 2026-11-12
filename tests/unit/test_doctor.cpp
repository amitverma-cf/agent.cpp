#include "doctor.hpp"

#include <catch_amalgamated.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace agent::cli;

namespace {

void write_bytes(const fs::path &p, size_t n) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << std::string(n, 'x');
}

HomeLayout fresh_home(const char *name) {
    const fs::path root = fs::temp_directory_path() / (std::string("agent-doctor-test-") + name);
    std::error_code ec;
    fs::remove_all(root, ec);
    HomeLayout h = home_layout(root.string());
    REQUIRE(create_home_layout(h).ok);
    return h;
}

bool contains(const CleanPlan &plan, const fs::path &p) {
    for (const std::string &s : plan.paths)
        if (fs::path(s) == p) return true;
    return false;
}

} // namespace

TEST_CASE("doctor: sizes add up per area", "[doctor]") {
    const HomeLayout h = fresh_home("sizes");
    write_bytes(fs::path(h.models) / "a" / "m.gguf", 3000);
    write_bytes(fs::path(h.models) / "b.gguf", 500);
    write_bytes(fs::path(h.cache) / "gpu" / "k.bin", 40);
    const UsageEntry models = measure_folder("models", h.models);
    CHECK(models.bytes == 3500);
    CHECK(models.files == 2);
    CHECK(measure_folder("cache", h.cache).bytes == 40);
    CHECK(measure_folder("nope", (fs::path(h.root) / "missing").string()).bytes == 0);
    CHECK(format_bytes(1536) == "2 KiB");
    CHECK(format_bytes(3 * 1024ULL * 1024 * 1024) == "3.0 GiB");
    std::error_code ec;
    fs::remove_all(h.root, ec);
}

TEST_CASE("clean: downloads removes only partial files, and never a model", "[doctor]") {
    const HomeLayout h = fresh_home("downloads");
    write_bytes(fs::path(h.models) / "x" / "full.gguf", 1000);
    write_bytes(fs::path(h.models) / "x" / "big.gguf.part", 700);
    write_bytes(fs::path(h.cache) / "dl" / "y.bin.part", 300);
    write_bytes(fs::path(h.cache) / "gpu" / "k.bin", 40);

    const CleanPlan plan = plan_clean(h, {CleanTarget::Downloads});
    CHECK(plan.paths.size() == 2);
    CHECK(plan.bytes == 1000);
    CHECK_FALSE(contains(plan, fs::path(h.models) / "x" / "full.gguf"));
    CHECK(run_clean(plan) == 1000);
    CHECK(fs::exists(fs::path(h.models) / "x" / "full.gguf"));
    CHECK_FALSE(fs::exists(fs::path(h.models) / "x" / "big.gguf.part"));
    CHECK(fs::exists(fs::path(h.cache) / "gpu" / "k.bin")); // not asked for
    std::error_code ec;
    fs::remove_all(h.root, ec);
}

TEST_CASE("clean: cache empties cache/ but keeps the folder, models and memory", "[doctor]") {
    const HomeLayout h = fresh_home("cache");
    write_bytes(fs::path(h.cache) / "gpu" / "k.bin", 40);
    write_bytes(fs::path(h.cache) / "kv" / "prefix.bin", 60);
    write_bytes(fs::path(h.models) / "m.gguf", 900);
    write_bytes(fs::path(h.workspaces) / "default" / "memory" / "db.sqlite3", 80);
    write_bytes(fs::path(h.skills) / "s" / "SKILL.md", 10);

    const CleanPlan plan = plan_clean(h, {CleanTarget::Cache});
    CHECK(plan.bytes == 100);
    CHECK(run_clean(plan) == 100);
    CHECK(fs::is_directory(h.cache));
    CHECK(measure_folder("cache", h.cache).bytes == 0);
    CHECK(fs::exists(fs::path(h.models) / "m.gguf"));
    CHECK(fs::exists(fs::path(h.workspaces) / "default" / "memory" / "db.sqlite3"));
    CHECK(fs::exists(fs::path(h.skills) / "s" / "SKILL.md"));
    std::error_code ec;
    fs::remove_all(h.root, ec);
}

TEST_CASE("clean: logs removes only files older than the limit", "[doctor]") {
    const HomeLayout h = fresh_home("logs");
    const fs::path old_log = fs::path(h.logs) / "old.log";
    const fs::path new_log = fs::path(h.logs) / "new.log";
    write_bytes(old_log, 100);
    write_bytes(new_log, 50);
    fs::last_write_time(old_log, fs::file_time_type::clock::now() - std::chrono::hours(24 * 30));

    const CleanPlan plan = plan_clean(h, {CleanTarget::Logs}, 14);
    CHECK(plan.paths.size() == 1);
    CHECK(contains(plan, old_log));
    run_clean(plan);
    CHECK_FALSE(fs::exists(old_log));
    CHECK(fs::exists(new_log));
    std::error_code ec;
    fs::remove_all(h.root, ec);
}
