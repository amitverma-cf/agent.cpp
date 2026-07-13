#include "../providers/provider_ops.hpp"
#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>

namespace agent {

static Result<void> sync_conversation(Session &session, const ConversationState &state) {
    std::string json;
    size_t estimated_size = 2;
    for (const auto &msg : state.history) {
        estimated_size += msg.role.length() + msg.content.length() + msg.tool_call_id.length() + 128;
        for (const auto &tc : msg.tool_calls) {
            estimated_size += tc.id.length() + tc.name.length() + tc.arguments.length() + 64;
        }
    }
    json.reserve(estimated_size);

    json.append("[");
    for (size_t i = 0; i < state.history.size(); ++i) {
        if (i > 0)
            json.append(",");
        const auto &msg = state.history[i];
        json.append("{\"role\":\"");
        utils::escape_json_string(msg.role, json);
        json.append("\",\"content\":\"");
        utils::escape_json_string(msg.content, json);
        json.append("\"");

        json.append(",\"tool_call_id\":\"");
        utils::escape_json_string(msg.tool_call_id, json);
        json.append("\",\"tool_calls\":[");
        for (size_t j = 0; j < msg.tool_calls.size(); ++j) {
            if (j > 0)
                json.append(",");
            const auto &tc = msg.tool_calls[j];
            json.append("{\"id\":\"");
            utils::escape_json_string(tc.id, json);
            json.append("\",\"name\":\"");
            utils::escape_json_string(tc.name, json);
            json.append("\",\"arguments\":\"");
            utils::escape_json_string(tc.arguments, json);
            json.append("\"}");
        }
        json.append("]}");
    }
    json.append("]");
    return store_memory(session, "history:" + state.id, json);
}

static int count_message_tokens(Session &session, const Message &msg) {
    if (msg.token_count != -1) {
        return msg.token_count;
    }

    int tokens = 0;
    const auto *ops = providers::find_provider_ops(session.config.provider);
    if (ops && ops->count_tokens) {
        std::string formatted;
        formatted.reserve(msg.role.length() + msg.content.length() + 64);
        formatted.append(msg.role);
        formatted.append(": ");
        formatted.append(msg.content);
        formatted.append("\n");
        for (const auto &tc : msg.tool_calls) {
            formatted.append(tc.name);
            formatted.append(tc.arguments);
        }
        auto res = ops->count_tokens(session, formatted);
        if (res.ok) {
            tokens = res.value;
        }
    }

    if (tokens == 0) {
        size_t length = msg.role.length() + 2 + msg.content.length() + 1;
        for (const auto &tc : msg.tool_calls) {
            length += tc.name.length() + tc.arguments.length();
        }
        tokens = static_cast<int>((length + 2) / 3);
    }

    msg.token_count = tokens;
    return tokens;
}

static int estimate_history_tokens(Session &session, const std::vector<Message> &history) {
    int total = 0;
    for (const auto &msg : history) {
        total += count_message_tokens(session, msg);
    }
    const auto *ops = providers::find_provider_ops(session.config.provider);
    if (ops && ops->count_tokens) {
        auto res = ops->count_tokens(session, "assistant: ");
        if (res.ok) {
            total += res.value;
            return total;
        }
    }
    total += static_cast<int>((std::string_view("assistant: ").length() + 2) / 3);
    return total;
}

static Result<void> prune_history_state(Session &session, ConversationState &state) {
    int max_allowed_tokens = session.config.context_window - session.config.max_tokens - 100;
    if (max_allowed_tokens < 200)
        max_allowed_tokens = 200;

    if (state.history.empty())
        return ok();

    bool has_system = (state.history[0].role == "system");
    int current_tokens = estimate_history_tokens(session, state.history);
    size_t drop_count = 0;
    size_t min_required_size = has_system ? 2 : 1;

    while (current_tokens > max_allowed_tokens && (state.history.size() - drop_count) > min_required_size) {
        size_t idx = has_system ? (drop_count + 1) : drop_count;

        if (idx >= state.history.size())
            break;

        if (state.history[idx].role == "tool")
            break;

        size_t group_size = 1;
        if (state.history[idx].role == "assistant" && !state.history[idx].tool_calls.empty()) {
            size_t next = idx + 1;
            while (next < state.history.size() && state.history[next].role == "tool") {
                ++group_size;
                ++next;
            }
        }

        if ((state.history.size() - drop_count - group_size) < min_required_size)
            break;

        for (size_t i = idx; i < idx + group_size; ++i) {
            current_tokens -= count_message_tokens(session, state.history[i]);
        }
        drop_count += group_size;
    }

    if (drop_count > 0) {
        auto start = has_system ? (state.history.begin() + 1) : state.history.begin();
        state.history.erase(start, start + static_cast<ptrdiff_t>(drop_count));
    }

    return ok();
}

Result<std::string_view> run_conversation_turn(Session &session, ConversationState &state, std::string_view user_prompt,
                                               bool stream, TokenCallback on_token, void *token_user_data) {
    session.active_conversation = &state;

    bool loop = true;
    std::string_view final_text;

    if (!user_prompt.empty()) {
        Message user_msg;
        user_msg.role = "user";
        user_msg.content = std::string(user_prompt);
        state.history.push_back(std::move(user_msg));
    }

    {
        int total = estimate_history_tokens(session, state.history);
        if (total > session.config.context_window * 7 / 10 && state.history.size() > 6)
            compress_context(session, state);
    }

    while (loop) {
        auto prune_res = prune_history_state(session, state);
        if (!prune_res.ok) {
            return fail<std::string_view>(prune_res.error.code, prune_res.error.message);
        }

        session.arena.reset();
        std::span<MessageView> message_views = session.arena.allocate_span<MessageView>(state.history.size());
        if (!state.history.empty() && message_views.size() != state.history.size()) {
            return fail<std::string_view>(ErrorCode::DecodeFailed,
                                          "Arena capacity exhausted: insufficient space for message views. "
                                          "Increase Config::arena_capacity or reduce Config::context_window.");
        }
        for (size_t i = 0; i < state.history.size(); ++i) {
            MessageView &mv = message_views[i];

            mv.role = state.history[i].role;
            mv.content = state.history[i].content;

            if (!state.history[i].tool_calls.empty()) {
                std::span<ToolCallView> tcvs =
                    session.arena.allocate_span<ToolCallView>(state.history[i].tool_calls.size());
                for (size_t j = 0; j < state.history[i].tool_calls.size(); ++j) {
                    tcvs[j].id = state.history[i].tool_calls[j].id;
                    tcvs[j].name = state.history[i].tool_calls[j].name;
                    tcvs[j].arguments = state.history[i].tool_calls[j].arguments;
                }
                mv.tool_calls = tcvs;
            } else {
                mv.tool_calls = {};
            }
            mv.tool_call_id = state.history[i].tool_call_id;

            if (!state.history[i].tensors.empty()) {
                std::span<TensorView> tvs = session.arena.allocate_span<TensorView>(state.history[i].tensors.size());
                for (size_t j = 0; j < state.history[i].tensors.size(); ++j) {
                    const TensorData &td = state.history[i].tensors[j];
                    tvs[j].name = td.name;
                    tvs[j].dtype = td.dtype;
                    tvs[j].shape = std::span<const int64_t>(td.shape.data(), td.shape.size());
                    tvs[j].data = std::span<const uint8_t>(td.data.data(), td.data.size());
                }
                mv.tensors = tvs;
            } else {
                mv.tensors = {};
            }
        }

        trigger_event(session, EventType::OnInferenceSubmit, std::span<const uint8_t>());

        ChatRequest request{
            .messages = message_views, .stream = stream, .on_token = on_token, .token_user_data = token_user_data};

        auto response_res = execute_turn(session, request);
        if (!response_res.ok) {
            return fail<std::string_view>(response_res.error.code, response_res.error.message);
        }

        trigger_event(session, EventType::OnInferenceComplete, std::span<const uint8_t>());

        auto &response = response_res.value;
        final_text = response.message.content;

        session.total_prompt_tokens += response.usage.prompt_tokens;
        session.total_completion_tokens += response.usage.completion_tokens;
        session.total_tokens += response.usage.total_tokens;

        Message assistant_msg;
        assistant_msg.role = std::string(response.message.role);
        assistant_msg.content = std::string(response.message.content);

        std::vector<ToolCall> tool_calls;
        if (!response.message.tool_calls.empty()) {
            for (const auto &tcv : response.message.tool_calls) {
                tool_calls.push_back(ToolCall{std::string(tcv.id), std::string(tcv.name), std::string(tcv.arguments)});
            }
        }
        assistant_msg.tool_calls = tool_calls;

        state.history.push_back(std::move(assistant_msg));

        if (!response.message.tool_calls.empty()) {
            for (const auto &tcv : response.message.tool_calls) {
                trigger_event(
                    session, EventType::OnToolCall,
                    std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(tcv.name.data()), tcv.name.size()));

                std::string result_text;
                auto exec_res = execute_tool(session, tcv.name, tcv.arguments);
                if (exec_res.ok) {
                    result_text = std::move(exec_res.value);
                } else {
                    result_text = "Error: " + exec_res.error.message;
                }

                Message tool_msg;
                tool_msg.role = "tool";
                tool_msg.content = result_text;
                tool_msg.tool_call_id = std::string(tcv.id);
                state.history.push_back(std::move(tool_msg));
            }
            continue;
        }

        loop = false;
    }

    auto sync_res = sync_conversation(session, state);
    if (!sync_res.ok) {
        return fail<std::string_view>(sync_res.error.code, sync_res.error.message);
    }

    return ok(final_text);
}

} // namespace agent
