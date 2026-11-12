#pragma once

#include <agent-cpp/agent.hpp>
#include <span>
#include <string>
#include <string_view>

namespace agent::tools {

using ToolsInitFn = Result<void> (*)(Session &session);

using ToolsExecuteFn = Result<ToolResult> (*)(Session &session, std::string_view name, std::string_view arguments);

struct ToolsOps {
    ToolsInitFn init = nullptr;
    ToolsExecuteFn execute = nullptr;
};

const ToolsOps *find_tools_ops();

// result.text followed by one "[attached: <path> (<mime>)]" line per media item and one
// "[tensor: <name> [d0,d1,...]]" line per tensor.
std::string render_tool_text(const ToolResult &result);

// Runs execute_tool_parts and shapes the outcome as a "tool" message: text is the message content
// (render_tool_text, "Error: ..." for failures and is_error results), media / tensors the attachments.
ToolResult execute_for_message(Session &session, std::string_view name, std::string_view arguments);

// The plain-text tool list ("You have access to the following tools: ..." + call instructions) for
// the tools in `allowed` (empty = every tool). Used by rebuild_tool_index and, filtered by a graph
// Agent node's allowed_tools, by the engine provider.
void append_tool_list(const Session &session, std::span<const std::string_view> allowed, std::string &out);
bool tool_allowed(std::span<const std::string_view> allowed, std::string_view name);

// Lazy tool loading (Config::lazy_tools). True when it applies here: enabled, and tool_search is allowed.
bool lazy_tools_active(const Session &session, std::span<const std::string_view> allowed);
// Whether the model may see / call `name` right now: allowed by the node's list and, in lazy mode, one of the
// always-visible tools (tool_search, load_tool, use_skill) or already loaded in the current conversation.
bool tool_visible(const Session &session, std::span<const std::string_view> allowed, std::string_view name);

// Start of the system message compress_context leaves in place of the history it summarized.
inline constexpr std::string_view kCompressedHistoryPrefix = "[Compressed history]: ";

} // namespace agent::tools
