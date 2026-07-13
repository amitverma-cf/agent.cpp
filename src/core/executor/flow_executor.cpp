#include "../utils/utils.hpp"
#include "ambient_turn.hpp"

#include <agent-cpp/agent.hpp>
#include <optional>

namespace agent {

static FlowMemory &active_memory_for(Flow &flow, const FlowState &state) {
    if (!state.isolated_memory)
        return flow.memory;
    auto &branch = flow.isolated_memories[std::string(state.state_id)];
    if (branch.id.empty())
        branch.id = flow.id + ":" + std::string(state.state_id);
    return branch;
}

Result<bool> step_flow(Session &session, void *context, Flow &flow) {
    if (flow.has_finished)
        return ok(false);

    if (flow.states.empty())
        return fail<bool>(ErrorCode::InvalidConfig, "Flow has no registered states.");

    if (!flow.has_started) {
        flow.has_started = true;
        flow.current_state_index = 0;
    }

    const FlowState &current = flow.states[flow.current_state_index];
    FlowMemory &memory = active_memory_for(flow, current);

    trigger_event(session, EventType::OnStateTransition,
                  std::span<const uint8_t>(
                      reinterpret_cast<const uint8_t *>(current.state_id.data()),
                      current.state_id.size()));

    if (!current.on_transition)
        return fail<bool>(ErrorCode::InvalidConfig,
                          "Flow state missing on_transition: " + std::string(current.state_id));

    bool injected_system = false;
    std::optional<Message> saved_system;

    std::string_view effective_system_prompt = current.system_prompt;
    std::string context_prefixed_prompt;
    if (current.context_provider) {
        std::string_view extra = current.context_provider(session, memory, context);
        if (!extra.empty()) {
            context_prefixed_prompt.reserve(extra.size() + 1 + current.system_prompt.size());
            context_prefixed_prompt += extra;
            if (!current.system_prompt.empty()) {
                context_prefixed_prompt += '\n';
                context_prefixed_prompt += current.system_prompt;
            }
            effective_system_prompt = context_prefixed_prompt;
        }
    }

    if (!effective_system_prompt.empty()) {
        if (!memory.history.empty() && memory.history[0].role == "system") {
            saved_system = memory.history[0];
            memory.history[0].content = std::string(effective_system_prompt);
            memory.history[0].token_count = -1;
        } else {
            Message sys;
            sys.role = "system";
            sys.content = std::string(effective_system_prompt);
            memory.history.insert(memory.history.begin(), std::move(sys));
            injected_system = true;
        }
    }

    auto saved_allowed_tools = ambient::allowed_tools;
    ambient::allowed_tools = current.allowed_tools;

    auto saved_hooks = ambient::active_hooks;
    ambient::active_hooks = current.hooks;

    auto next_res = current.on_transition(session, memory, context);

    ambient::active_hooks = saved_hooks;
    ambient::allowed_tools = saved_allowed_tools;

    if (!effective_system_prompt.empty()) {
        if (injected_system) {
            if (!memory.history.empty() && memory.history[0].role == "system")
                memory.history.erase(memory.history.begin());
        } else if (saved_system) {
            memory.history[0] = std::move(*saved_system);
            memory.history[0].token_count = -1;
        }
    }

    if (!next_res.ok)
        return fail<bool>(next_res.error.code, next_res.error.message);

    std::string_view next_id = next_res.value;

    if (next_id.empty() || next_id == "exit") {
        flow.has_finished = true;
        return ok(false);
    }

    for (size_t i = 0; i < flow.states.size(); ++i) {
        if (flow.states[i].state_id == next_id) {
            flow.current_state_index = i;
            return ok(true);
        }
    }

    return fail<bool>(ErrorCode::InvalidConfig,
                      "Flow transitioned to unregistered state: " + std::string(next_id));
}

Result<void> run_flow(Session &session, void *context, Flow &flow) {
    while (true) {
        auto res = step_flow(session, context, flow);
        if (!res.ok)
            return fail(res.error.code, res.error.message);
        if (!res.value)
            return ok();
    }
}

} // namespace agent
