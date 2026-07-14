#include "tools_ops.hpp"

#include "../executor/ambient_turn.hpp"

#include <agent-cpp/agent.hpp>

namespace agent::tools {

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
Tool get_terminal_tool(const std::string *workspace_dir);
Tool get_context_compressor_tool(Session **session_cell);

namespace {

void add_if_missing(std::vector<Tool> &tools, Tool t) {
    for (const auto &e : tools)
        if (e.name == t.name) return;
    tools.push_back(std::move(t));
}

Result<void> default_init(Session &session) {
    if (!session.session_ptr_cell) session.session_ptr_cell = std::make_shared<Session *>(&session);

    if (session.config.auto_default_tools) {
        auto fs_ctx = std::make_shared<WorkspaceContext>();
        fs_ctx->workspace_dir = session.workspace_path.get();
        fs_ctx->sandbox = session.config.sandbox_filesystem;
        session.tool_context = fs_ctx;

        auto fs_tools = get_filesystem_tools(fs_ctx);
        for (auto &t : fs_tools) add_if_missing(session.config.tools, std::move(t));

        add_if_missing(session.config.tools, get_terminal_tool(session.workspace_path.get()));
        add_if_missing(session.config.tools, get_context_compressor_tool(session.session_ptr_cell.get()));
    }

    rebuild_tool_index(session);

    if (!session.config.skills_dir.empty()) {
        auto skills_res = load_skills_dir(session, session.config.skills_dir);
        if (!skills_res.ok) return skills_res;
    }

    return ok();
}

Result<std::string> default_execute(Session &session, std::string_view name, std::string_view arguments) {
    if (session.session_ptr_cell) *session.session_ptr_cell = &session;

    auto it = session.tool_by_name.find(std::string(name));
    if (it == session.tool_by_name.end()) return fail<std::string>(ErrorCode::KeyNotFound, "Tool not found: " + std::string(name));

    if (!ambient::allowed_tools.empty()) {
        bool allowed = false;
        for (auto a : ambient::allowed_tools) {
            if (a == name) {
                allowed = true;
                break;
            }
        }
        if (!allowed) return fail<std::string>(ErrorCode::KeyNotFound, "Tool not allowed in current Flow state: " + std::string(name));
    }

    if (session.config.permission_check) {
        auto decision = session.config.permission_check(name, arguments, session.config.permission_user_data);
        if (decision == PermissionDecision::Deny)
            return fail<std::string>(ErrorCode::PermissionDenied, "Tool denied by permission policy: " + std::string(name));
        if (decision == PermissionDecision::Ask) {
            bool approved =
                session.config.permission_prompt && session.config.permission_prompt(name, arguments, session.config.permission_user_data);
            if (!approved) return fail<std::string>(ErrorCode::PermissionDenied, "Tool declined: " + std::string(name));
        }
    }

    const Tool &tool = session.config.tools[it->second];
    return tool.callback(arguments, tool.user_data);
}

constexpr ToolsOps kToolsOps[] = {ToolsOps{.init = default_init, .execute = default_execute}};

} // namespace

const ToolsOps *find_tools_ops() { return &kToolsOps[0]; }

PermissionDecision default_permission_policy(std::string_view tool_name, std::string_view, void *) {
    if (tool_name == "run_command" || tool_name == "delete_path" || tool_name == "move_path") return PermissionDecision::Ask;
    return PermissionDecision::Allow;
}

} // namespace agent::tools
