#pragma once

#include <agent-cpp/agent.hpp>
#include <simdjson.h>

namespace agent::ambient {

extern thread_local FlowMemory *active_memory;
extern thread_local std::span<const FlowHook> active_hooks;

extern thread_local std::span<const std::string_view> allowed_tools;

extern thread_local size_t worker_arena_index;

extern thread_local simdjson::ondemand::parser json_parser;

// Graph runtime hooks (see src/core/graph). When tool_pause_fn is set and returns true for an
// assistant message's tool calls, run_turn returns before executing them and sets tools_paused.
using ToolPauseFn = bool (*)(Session &session, std::span<const ToolCallView> calls);
extern thread_local ToolPauseFn tool_pause_fn;
extern thread_local bool tools_paused;
// Set while executing a tool call a human already approved: a permission_check Ask is treated as Allow.
extern thread_local bool tool_call_preapproved;

Arena &current_arena(Session &session);

} // namespace agent::ambient
