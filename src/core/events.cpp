#include <agent-cpp/agent.hpp>

#include <mutex>

namespace agent {

void register_event_hook(Session &session, EventType type, EventHookFn callback,
                         void *user_data) {
    std::lock_guard<std::mutex> lock(*session.event_mutex);
    session.event_hooks.push_back({type, callback, user_data});
}

void trigger_event(Session &session, EventType type, std::span<const uint8_t> payload) {
    std::vector<EventHook> hooks;
    {
        std::lock_guard<std::mutex> lock(*session.event_mutex);
        hooks = session.event_hooks;
    }
    Event event{type, payload};
    for (const auto &hook : hooks) {
        if (hook.type == type && hook.callback)
            hook.callback(event, hook.user_data);
    }
}

} // namespace agent
