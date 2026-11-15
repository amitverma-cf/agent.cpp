#include "workflows.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace agent::cli {

namespace fs = std::filesystem;

std::string workflows_dir(const Context &ctx) {
    return ((ctx.has_home ? fs::path(ctx.home.root) : fs::path(ctx.config_dir)) / "graphs").string();
}

bool valid_workflow_name(std::string_view name) {
    if (name.empty() || name.size() > 64) return false;
    return std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isalnum(c) || c == '-' || c == '_'; });
}

Result<std::string> workflow_save(const Context &ctx, std::string_view name, std::string_view json) {
    if (!valid_workflow_name(name))
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "a workflow name uses letters, digits, '-' and '_' (at most 64), for example daily-report");
    const size_t first = json.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos || json[first] != '{')
        return fail<std::string>(ErrorCode::InvalidConfig, "the workflow must be a JSON object with \"nodes\" and \"edges\"");
    auto parsed = parse_graph_json(json);
    if (!parsed.ok) return fail<std::string>(parsed.error.code, "the workflow is not valid: " + parsed.error.message);
    const fs::path dir = workflows_dir(ctx);
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path file = dir / (std::string(name) + ".json");
    const fs::path tmp = dir / (std::string(name) + ".json.tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << json << (json.back() == '\n' ? "" : "\n");
        if (!out.flush()) return fail<std::string>(ErrorCode::FilesystemError, "cannot write " + tmp.string());
    }
    fs::rename(tmp, file, ec);
    if (ec) return fail<std::string>(ErrorCode::FilesystemError, "cannot save the workflow: " + ec.message());
    std::string text = "Saved the workflow '" + std::string(name) + "' (" + std::to_string(parsed.value.graph.nodes.size()) +
                       " steps). Run it later by name: agent_run [\"run\",\"" + std::string(name) + "\"].";
    for (const std::string &w : parsed.value.warnings) text += " Warning: " + w + ".";
    return ok(std::move(text));
}

std::vector<WorkflowInfo> workflow_list(const Context &ctx) {
    std::vector<WorkflowInfo> out;
    std::error_code ec;
    for (const auto &e : fs::directory_iterator(workflows_dir(ctx), ec)) {
        if (!e.is_regular_file(ec) || e.path().extension() != ".json") continue;
        WorkflowInfo w;
        w.name = e.path().stem().string();
        w.path = e.path().string();
        auto parsed = parse_graph_json(w.path);
        if (parsed.ok) {
            w.description = parsed.value.description;
            w.nodes = parsed.value.graph.nodes.size();
        } else {
            w.problem = parsed.error.message;
        }
        out.push_back(std::move(w));
    }
    std::sort(out.begin(), out.end(), [](const WorkflowInfo &a, const WorkflowInfo &b) { return a.name < b.name; });
    return out;
}

Result<std::string> workflow_remove(const Context &ctx, std::string_view name) {
    if (!valid_workflow_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "that is not a workflow name");
    std::error_code ec;
    if (!fs::remove(fs::path(workflows_dir(ctx)) / (std::string(name) + ".json"), ec))
        return fail<std::string>(ErrorCode::KeyNotFound, "there is no workflow named '" + std::string(name) + "'");
    return ok("Removed the workflow '" + std::string(name) + "'.");
}

Result<std::string> workflow_show(const Context &ctx, std::string_view name) {
    if (!valid_workflow_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "that is not a workflow name");
    std::ifstream in(fs::path(workflows_dir(ctx)) / (std::string(name) + ".json"), std::ios::binary);
    if (!in) return fail<std::string>(ErrorCode::KeyNotFound, "there is no workflow named '" + std::string(name) + "'");
    return ok(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
}

std::string resolve_graph_arg(const Context &ctx, const std::string &arg) {
    std::error_code ec;
    if (fs::exists(arg, ec) || arg.find_first_of("/\\") != std::string::npos) return arg;
    std::string name = arg;
    if (name.size() > 5 && name.ends_with(".json")) name.resize(name.size() - 5);
    if (!valid_workflow_name(name)) return arg;
    const fs::path file = fs::path(workflows_dir(ctx)) / (name + ".json");
    return fs::exists(file, ec) ? file.string() : arg;
}

std::string workflows_prompt(const Context &ctx) {
    const auto list = workflow_list(ctx);
    if (list.empty()) return {};
    std::string text = " Saved workflows (run one with agent_run [\"run\",\"NAME\"]): ";
    for (size_t i = 0; i < list.size(); ++i) {
        text += (i ? "; " : "") + list[i].name;
        if (!list[i].description.empty()) text += " (" + list[i].description.substr(0, 100) + ")";
    }
    return text + ".";
}

} // namespace agent::cli
