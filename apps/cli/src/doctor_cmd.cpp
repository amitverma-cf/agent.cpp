// agent doctor / agent clean.

#include "commands.hpp"
#include "doctor.hpp"

#include <agent-cpp/hub.hpp>
#include <cstdio>
#include <filesystem>

namespace agent::cli {

namespace fs = std::filesystem;

int cmd_doctor(const CommandLine &, Context &ctx) {
    std::printf("agent.cpp %.*s\n\n", static_cast<int>(kCliVersion.size()), kCliVersion.data());
    if (!ctx.has_home) {
        std::printf("home:      not set (the older platform folders are in use). Run `agent setup` to choose a home.\n");
        std::printf("config:    %s\n", ctx.config_dir.c_str());
    } else {
        std::error_code ec;
        const auto space = fs::space(ctx.home.root, ec);
        std::printf("home:      %s%s%s\n", ctx.home.root.c_str(),
                    ec ? "" : "   free on this volume: ", ec ? "" : format_bytes(space.available).c_str());
    }
    const uint64_t ram = total_ram_bytes();
    std::printf("RAM:       %s installed, %d CPU cores\n", ram ? format_bytes(ram).c_str() : "unknown", detect_hardware().cpu_cores);
    if (!ctx.project_root.empty()) std::printf("project:   %s (.agents/ found)\n", ctx.project_root.c_str());

    std::printf("\nengines:\n");
    auto engines = load_engines(ctx.engines_path);
    if (!engines.ok || engines.value.empty()) std::printf("  none registered: run `agent setup`\n");
    else
        for (const EngineEntry &e : engines.value)
            std::printf("  %-10s %s  %s\n", e.name.c_str(), file_exists(e.path) ? "ok     " : "MISSING", e.path.c_str());

    auto models = agent::hub::list_models(catalog_path(ctx));
    uint64_t model_bytes = 0;
    size_t missing = 0;
    if (models.ok)
        for (const auto &m : models.value) {
            model_bytes += m.size;
            if (!path_exists(m.path)) ++missing;
        }
    std::printf("\nmodels:    %zu registered, %s%s\n", models.ok ? models.value.size() : 0, format_bytes(model_bytes).c_str(),
                missing ? ("  (" + std::to_string(missing) + " listed but missing on disk: `agent model rm NAME`)").c_str() : "");

    if (ctx.has_home) {
        std::printf("\ndisk use in the home:\n");
        for (const UsageEntry &u : measure_home(ctx.home))
            std::printf("  %-12s %10s  %6llu files\n", u.label.c_str(), format_bytes(u.bytes).c_str(),
                        static_cast<unsigned long long>(u.files));

        const CleanPlan downloads = plan_clean(ctx.home, {CleanTarget::Downloads});
        const CleanPlan cache = plan_clean(ctx.home, {CleanTarget::Cache});
        const CleanPlan logs = plan_clean(ctx.home, {CleanTarget::Logs});
        std::printf("\nsafe to clean (nothing here is lost for good; it is rebuilt or re-downloaded when needed):\n");
        std::printf("  partial downloads  %10s   agent clean --downloads\n", format_bytes(downloads.bytes).c_str());
        std::printf("  cache              %10s   agent clean --cache\n", format_bytes(cache.bytes).c_str());
        std::printf("  logs over 14 days  %10s   agent clean --logs\n", format_bytes(logs.bytes).c_str());
        std::printf("models, skills, MCP definitions, memory and secrets are never removed by `clean`.\n");
    }
    return 0;
}

int cmd_clean(const CommandLine &cl, Context &ctx) {
    if (!ctx.has_home) return report_error(Error{ErrorCode::InvalidConfig, "no home is configured: run `agent setup` first"});
    std::vector<CleanTarget> targets;
    if (cl.has("downloads")) targets.push_back(CleanTarget::Downloads);
    if (cl.has("cache")) targets.push_back(CleanTarget::Cache);
    if (cl.has("logs")) targets.push_back(CleanTarget::Logs);
    if (targets.empty())
        return report_error(Error{ErrorCode::InvalidConfig, "say what to clean: --downloads, --cache and/or --logs (see `agent doctor`)"});

    const CleanPlan plan = plan_clean(ctx.home, targets);
    if (plan.paths.empty()) {
        std::printf("nothing to clean\n");
        return 0;
    }
    std::printf("would remove %zu item%s, %s:\n", plan.paths.size(), plan.paths.size() == 1 ? "" : "s", format_bytes(plan.bytes).c_str());
    for (size_t i = 0; i < plan.paths.size() && i < 12; ++i) std::printf("  %s\n", plan.paths[i].c_str());
    if (plan.paths.size() > 12) std::printf("  ... and %zu more\n", plan.paths.size() - 12);
    if (!cl.has("yes") && !ask_yes_no("Remove them?")) {
        std::printf("nothing removed\n");
        return 0;
    }
    std::printf("freed %s\n", format_bytes(run_clean(plan)).c_str());
    return 0;
}

} // namespace agent::cli
