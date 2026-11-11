#include "tools_ops.hpp"

#include "../executor/ambient_turn.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <filesystem>
#include <system_error>

namespace agent::tools {

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
Tool get_terminal_tool(const std::string *workspace_dir);
Tool get_terminal_tool(std::shared_ptr<WorkspaceContext> ctx);
Tool get_context_compressor_tool(Session **session_cell);
Tool get_tool_search_tool(Session **session_cell);
Tool get_load_tool_tool(Session **session_cell);

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
        fs_ctx->command_sandbox = session.config.tool_sandbox;
        fs_ctx->cancel = session.cancel_flag.get(); // shared_ptr: the address survives Session moves
        fs_ctx->grants = std::make_shared<std::vector<PathGrant>>(session.config.path_grants);
        fs_ctx->gate = session.config.path_gate;
        fs_ctx->gate_user_data = session.config.path_gate_user_data;
        session.tool_context = fs_ctx;

        auto fs_tools = get_filesystem_tools(fs_ctx);
        for (auto &t : fs_tools) add_if_missing(session.config.tools, std::move(t));

        add_if_missing(session.config.tools, get_terminal_tool(fs_ctx));
        add_if_missing(session.config.tools, get_context_compressor_tool(session.session_ptr_cell.get()));
    }

    if (session.config.lazy_tools) {
        add_if_missing(session.config.tools, get_tool_search_tool(session.session_ptr_cell.get()));
        add_if_missing(session.config.tools, get_load_tool_tool(session.session_ptr_cell.get()));
    }

    rebuild_tool_index(session);

    if (!session.config.skills_dir.empty()) {
        auto skills_res = load_skills_dir(session, session.config.skills_dir);
        if (!skills_res.ok) return skills_res;
    }

    return ok();
}

// Media paths are handed to engines (possibly in another process with another working directory)
// and persisted with the history: make them absolute.
void absolutize_media(std::vector<Media> &media) {
    for (Media &m : media) {
        if (m.path.empty()) continue;
        std::filesystem::path p(m.path);
        if (p.is_absolute()) continue;
        std::error_code ec;
        auto abs = std::filesystem::absolute(p, ec);
        if (!ec) m.path = abs.lexically_normal().string();
    }
}

Result<ToolResult> default_execute(Session &session, std::string_view name, std::string_view arguments) {
    // Only written when the Session moved: concurrent graph branches on one Session read it.
    if (session.session_ptr_cell && *session.session_ptr_cell != &session) *session.session_ptr_cell = &session;

    auto it = session.tool_by_name.find(std::string(name));
    if (it == session.tool_by_name.end()) return fail<ToolResult>(ErrorCode::KeyNotFound, "Tool not found: " + std::string(name));

    if (!tool_allowed(ambient::allowed_tools, name))
        return fail<ToolResult>(ErrorCode::KeyNotFound, "Tool not allowed for the current agent node: " + std::string(name));
    if (!tool_visible(session, ambient::allowed_tools, name))
        return fail<ToolResult>(ErrorCode::KeyNotFound, "Tool '" + std::string(name) + "' is not loaded yet: call tool_search to find tools, then load_tool with the name.");

    if (session.config.permission_check) {
        auto decision = session.config.permission_check(name, arguments, session.config.permission_user_data);
        if (decision == PermissionDecision::Deny)
            return fail<ToolResult>(ErrorCode::PermissionDenied, "Tool denied by permission policy: " + std::string(name));
        if (decision == PermissionDecision::Ask) {
            bool approved =
                ambient::tool_call_preapproved || (session.config.permission_prompt &&
                                                   session.config.permission_prompt(name, arguments, session.config.permission_user_data));
            if (!approved) return fail<ToolResult>(ErrorCode::PermissionDenied, "Tool declined: " + std::string(name));
        }
    }

    const Tool &tool = session.config.tools[it->second];
    if (tool.callback && tool.parts_callback)
        return fail<ToolResult>(ErrorCode::InvalidConfig, "Tool '" + std::string(name) + "' sets both callback and parts_callback.");
    if (tool.parts_callback) {
        auto res = tool.parts_callback(arguments, tool.user_data);
        if (res.ok) absolutize_media(res.value.media);
        return res;
    }
    if (!tool.callback) return fail<ToolResult>(ErrorCode::InvalidConfig, "Tool '" + std::string(name) + "' has no callback.");
    auto res = tool.callback(arguments, tool.user_data);
    if (!res.ok) return fail<ToolResult>(res.error.code, res.error.message);
    ToolResult out;
    out.text = std::move(res.value);
    return ok(std::move(out));
}

constexpr ToolsOps kToolsOps[] = {ToolsOps{.init = default_init, .execute = default_execute}};

} // namespace

const ToolsOps *find_tools_ops() { return &kToolsOps[0]; }

bool tool_allowed(std::span<const std::string_view> allowed, std::string_view name) {
    if (allowed.empty()) return true;
    for (auto a : allowed)
        if (a == name) return true;
    return false;
}

bool lazy_tools_active(const Session &session, std::span<const std::string_view> allowed) {
    return session.config.lazy_tools && tool_allowed(allowed, "tool_search");
}

bool tool_visible(const Session &session, std::span<const std::string_view> allowed, std::string_view name) {
    if (!tool_allowed(allowed, name)) return false;
    if (!lazy_tools_active(session, allowed)) return true;
    if (name == "tool_search" || name == "load_tool" || name == "use_skill") return true;
    const FlowMemory *memory = ambient::active_memory;
    if (!memory) return false;
    return std::find(memory->active_tools.begin(), memory->active_tools.end(), name) != memory->active_tools.end();
}

std::string render_tool_text(const ToolResult &result) {
    std::string text = result.text;
    for (const Media &m : result.media) {
        if (!text.empty()) text += '\n';
        text += "[attached: ";
        text += m.path;
        text += " (";
        text += m.mime.empty() ? std::string_view("application/octet-stream") : std::string_view(m.mime);
        text += ")]";
    }
    for (const TensorData &t : result.tensors) {
        if (!text.empty()) text += '\n';
        text += "[tensor: ";
        text += t.name;
        text += " [";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i) text += ',';
            text += std::to_string(t.shape[i]);
        }
        text += "]]";
    }
    return text;
}

ToolResult execute_for_message(Session &session, std::string_view name, std::string_view arguments) {
    auto exec = execute_tool_parts(session, name, arguments);
    if (!exec.ok) {
        ToolResult failed;
        failed.text = "Error: " + exec.error.message;
        failed.is_error = true;
        return failed;
    }
    ToolResult out = std::move(exec.value);
    std::string text = render_tool_text(out);
    if (out.is_error) text = "Error: " + (text.empty() ? std::string("tool reported an error") : text);
    out.text = std::move(text);
    return out;
}

void append_tool_list(const Session &session, std::span<const std::string_view> allowed, std::string &out) {
    bool any = false;
    for (const auto &tool : session.config.tools) {
        if (!tool_visible(session, allowed, tool.name)) continue;
        if (!any) out += "You have access to the following tools:\n";
        any = true;
        out += "- ";
        out += tool.name;
        out += ": ";
        out += tool.description;
        out += "\n  Schema: ";
        out += tool.parameter_schema;
        out += "\n";
    }
    if (any)
        out += "When you need a tool, respond with ONLY a single JSON object of "
               "the exact form {\"name\": \"tool_name\", \"arguments\": {...}} and "
               "stop. Wait for the tool result before continuing.";
}

PermissionDecision default_permission_policy(std::string_view tool_name, std::string_view, void *) {
    if (tool_name == "run_command" || tool_name == "delete_path" || tool_name == "move_path") return PermissionDecision::Ask;
    return PermissionDecision::Allow;
}

} // namespace agent::tools
