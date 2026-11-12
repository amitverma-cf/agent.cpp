// agent skill: search skills.sh, install a skill's files from its GitHub repository, list and remove them.

#include "commands.hpp"
#include "doctor.hpp"
#include "registries.hpp"

#include <agent-cpp/hub.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string skills_dir_for(const CommandLine &cl, const Context &ctx) {
    if (cl.has("project")) return ctx.project_root.empty() ? std::string() : (fs::path(ctx.project_root) / ".agents" / "skills").string();
    return ctx.has_home ? ctx.home.skills : std::string();
}

void list_dir(const char *label, const std::string &dir) {
    std::error_code ec;
    if (dir.empty() || !fs::is_directory(dir, ec)) return;
    std::vector<std::string> names;
    for (const auto &e : fs::directory_iterator(dir, ec))
        if (e.is_directory(ec) && !e.path().filename().string().starts_with(".") && fs::exists(e.path() / "SKILL.md", ec)) names.push_back(e.path().filename().string());
    std::sort(names.begin(), names.end());
    if (names.empty()) return;
    std::printf("%s (%s)\n", label, dir.c_str());
    for (const std::string &n : names) std::printf("  %s%s\n", n.c_str(), fs::exists(fs::path(dir) / n / ".source.json", ec) ? "   (from skills.sh)" : "");
}

} // namespace

int cmd_skill(const CommandLine &cl, Context &ctx) {
    const std::string &sub = cl.subcommand;
    if (sub == "ls" || sub.empty()) {
        list_dir("global", ctx.has_home ? ctx.home.skills : std::string());
        list_dir("this project", ctx.project_root.empty() ? std::string() : (fs::path(ctx.project_root) / ".agents" / "skills").string());
        return 0;
    }
    const auto http = agent::hub::http_options_from_env();
    if (sub == "search") {
        auto body = agent::hub::http_get_text(skill_search_url(cl.positionals[0]), http);
        if (!body.ok) return report_error(body.error);
        auto hits = parse_skill_search(body.value);
        if (!hits.ok) return report_error(hits.error);
        if (hits.value.empty()) {
            std::printf("nothing found for '%s'\n", cl.positionals[0].c_str());
            return 0;
        }
        std::printf("%-52s %10s\n", "skill id", "installs");
        for (const SkillHit &h : hits.value) std::printf("%-52s %10llu\n", h.id.c_str(), static_cast<unsigned long long>(h.installs));
        std::printf("\ninstall one with: agent skill add <skill id>\n");
        return 0;
    }
    if (sub == "add") {
        const std::string dir = skills_dir_for(cl, ctx);
        if (dir.empty()) {
            std::fprintf(stderr, "agent skill add: %s\n", cl.has("project") ? "this folder has no .agents project (run `agent init`)" : "no home is set (run `agent setup`)");
            return 1;
        }
        auto hit = parse_skill_id(cl.positionals[0]);
        if (!hit.ok) return report_error(hit.error);
        auto plan = plan_skill(hit.value, http);
        if (!plan.ok) return report_error(plan.error);

        std::vector<std::string> scripts;
        for (const std::string &f : plan.value.files) {
            const std::string ext = fs::path(f).extension().string();
            if (ext != ".md" && ext != ".txt" && ext != ".json" && ext != ".yaml" && ext != ".yml" && ext != ".png" && ext != ".jpg" && ext != ".svg" && !ext.empty()) scripts.push_back(f);
        }
        std::printf("%s\n  from  https://github.com/%s%s%s\n  into  %s\n  files %zu, %s\n", hit.value.id.c_str(), hit.value.source.c_str(),
                    plan.value.dir.empty() ? "" : "/tree/HEAD/", plan.value.dir.c_str(), (fs::path(dir) / hit.value.skill_id).string().c_str(), plan.value.files.size(),
                    format_bytes(plan.value.bytes).c_str());
        if (!scripts.empty()) {
            std::printf("  it includes scripts. They can only run through run_command, in the sandbox, after you approve each command:\n");
            for (const std::string &s : scripts) std::printf("    %s\n", s.c_str());
        }
        std::printf("  A skill is text the model follows. Read it before you rely on it: anyone can publish one.\n");
        if (!cl.has("yes") && !ask_yes_no("Install it?")) {
            std::printf("cancelled\n");
            return 1;
        }
        std::error_code ec;
        fs::create_directories(dir, ec);
        auto done = install_skill(hit.value, plan.value, dir, http);
        if (!done.ok) return report_error(done.error);
        std::printf("installed %s (%zu files) in %s\n", hit.value.skill_id.c_str(), done.value.files.size(), done.value.dir.c_str());
        return 0;
    }
    if (sub == "rm") {
        const std::string name = cl.positionals[0];
        if (name.empty() || name.find_first_of("/\\") != std::string::npos || name == "." || name == "..") {
            std::fprintf(stderr, "agent skill rm: give the skill's folder name\n");
            return 2;
        }
        const std::string dir = skills_dir_for(cl, ctx);
        std::error_code ec;
        const fs::path target = fs::path(dir) / name;
        if (dir.empty() || !fs::exists(target / "SKILL.md", ec)) {
            std::fprintf(stderr, "agent skill rm: no skill '%s' in %s\n", name.c_str(), dir.c_str());
            return 1;
        }
        fs::remove_all(target, ec);
        std::printf("removed %s\n", name.c_str());
        return ec ? 1 : 0;
    }
    std::fprintf(stderr, "agent skill: unknown subcommand '%s'\n", sub.c_str());
    return 2;
}

} // namespace agent::cli
