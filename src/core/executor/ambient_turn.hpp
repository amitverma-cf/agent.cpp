#pragma once

#include <agent-cpp/agent.hpp>
#include <simdjson.h>

namespace agent::ambient {

extern thread_local FlowMemory *active_memory;
extern thread_local std::span<const FlowHook> active_hooks;

extern thread_local std::span<const std::string_view> allowed_tools;

extern thread_local size_t worker_arena_index;

extern thread_local simdjson::ondemand::parser json_parser;

Arena &current_arena(Session &session);

} // namespace agent::ambient
