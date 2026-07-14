#include "ambient_turn.hpp"

namespace agent::ambient {

thread_local FlowMemory *active_memory = nullptr;
thread_local std::span<const FlowHook> active_hooks;
thread_local std::span<const std::string_view> allowed_tools;
thread_local size_t worker_arena_index = 0;
thread_local simdjson::ondemand::parser json_parser;

Arena &current_arena(Session &session) {
    if (!session.worker_arenas.empty() && worker_arena_index < session.worker_arenas.size()) {
        return session.worker_arenas[worker_arena_index];
    }
    return session.arena;
}

} // namespace agent::ambient
