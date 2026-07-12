#include <agent-cpp/agent.hpp>

namespace agent {

void register_event_hook(Session &session, EventType type, EventHookFn callback, void *user_data) {
    session.event_hooks.push_back({type, callback, user_data});
}

void trigger_event(Session &session, EventType type, std::span<const uint8_t> payload) {
    Event event{type, payload};
    for (const auto &hook : session.event_hooks) {
        if (hook.type == type) {
            hook.callback(event, hook.user_data);
        }
    }
}

} // namespace agent
