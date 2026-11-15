// agent workflow: the graphs saved by name (<home>/graphs/NAME.json).

#include "commands.hpp"
#include "workflows.hpp"

#include <cstdio>
#include <fstream>

namespace agent::cli {

int cmd_workflow(const CommandLine &cl, Context &ctx) {
    const std::string &sub = cl.subcommand;
    if (sub == "save") {
        std::ifstream in(cl.positionals[1], std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "agent workflow save: cannot read %s\n", cl.positionals[1].c_str());
            return 1;
        }
        const std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto done = workflow_save(ctx, cl.positionals[0], json);
        if (!done.ok) return report_error(done.error);
        std::printf("%s\n", done.value.c_str());
        return 0;
    }
    if (sub == "rm" || sub == "show") {
        auto done = sub == "rm" ? workflow_remove(ctx, cl.positionals[0]) : workflow_show(ctx, cl.positionals[0]);
        if (!done.ok) return report_error(done.error);
        std::printf("%s\n", done.value.c_str());
        return 0;
    }
    const auto list = workflow_list(ctx);
    if (list.empty()) {
        std::printf("no saved workflows in %s\n", workflows_dir(ctx).c_str());
        return 0;
    }
    for (const WorkflowInfo &w : list) {
        if (!w.problem.empty()) std::printf("%-24s (does not load: %s)\n", w.name.c_str(), w.problem.c_str());
        else std::printf("%-24s %zu steps  %s\n", w.name.c_str(), w.nodes, w.description.c_str());
    }
    std::printf("run one with: agent run NAME\n");
    return 0;
}

} // namespace agent::cli
