#include "tools_ops.hpp"

#include <agent-cpp/agent.hpp>

namespace agent::tools {

std::vector<Tool> get_filesystem_tools(const std::string *workspace_dir);
Tool get_terminal_tool(const std::string *workspace_dir);
Tool get_calculator_tool();
Tool get_context_compressor_tool(Session **session_cell);

namespace {

void add_if_missing(std::vector<Tool> &tools, Tool t) {
    for (const auto &e : tools)
        if (e.name == t.name)
            return;
    tools.push_back(std::move(t));
}

Result<void> default_init(Session &session) {
    if (!session.self_ref)
        session.self_ref = std::make_shared<Session *>(&session);

    if (session.config.auto_default_tools) {
        const std::string *ws = session.workspace_dir_ref.get();
        auto fs_tools = get_filesystem_tools(ws);
        for (auto &t : fs_tools)
            add_if_missing(session.config.tools, std::move(t));
        add_if_missing(session.config.tools, get_terminal_tool(ws));
        add_if_missing(session.config.tools, get_calculator_tool());
        add_if_missing(session.config.tools, get_context_compressor_tool(session.self_ref.get()));
    }
    return ok();
}

Result<std::string> default_execute(Session &session, std::string_view name, std::string_view arguments) {
    if (session.self_ref)
        *session.self_ref = &session;

    for (const auto &tool : session.config.tools) {
        if (tool.name == name) {
            return tool.callback(arguments, tool.user_data);
        }
    }
    return fail<std::string>(ErrorCode::KeyNotFound, "Tool not found.");
}

constexpr ToolsOps kToolsOps[] = {ToolsOps{.init = default_init, .execute = default_execute}};

} // namespace

const ToolsOps *find_tools_ops() {
    return &kToolsOps[0];
}

} // namespace agent::tools
