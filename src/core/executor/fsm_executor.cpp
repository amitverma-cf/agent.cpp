#include <agent-cpp/agent.hpp>

namespace agent {

Result<void> run_fsm(Session &session, void *context, FSMExecutor &executor) {
    if (executor.registered_states.empty()) {
        return fail(ErrorCode::InvalidConfig, "FSM has no registered states.");
    }

    FSMState current = executor.registered_states[0];
    bool running = true;

    while (running) {
        trigger_event(session, EventType::OnStateTransition,
                      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(current.state_id.data()),
                                               current.state_id.size()));

        if (!current.on_transition) {
            return fail(ErrorCode::InvalidConfig,
                        "FSM state has no on_transition: " + std::string(current.state_id));
        }

        auto next_res = current.on_transition(session, context);
        if (!next_res.ok) {
            return fail(next_res.error.code, next_res.error.message);
        }

        std::string_view next_state_id = next_res.value;
        if (next_state_id == "exit" || next_state_id.empty()) {
            running = false;
            break;
        }

        bool state_found = false;
        for (const auto &s : executor.registered_states) {
            if (s.state_id == next_state_id) {
                current = s;
                state_found = true;
                break;
            }
        }
        if (!state_found) {
            return fail(ErrorCode::InvalidConfig,
                        "FSM transitioned to unregistered state: " + std::string(next_state_id));
        }
    }

    return ok();
}

} // namespace agent
