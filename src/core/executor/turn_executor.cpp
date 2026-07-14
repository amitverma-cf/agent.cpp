#include "../providers/provider_ops.hpp"
#include "../utils/utils.hpp"
#include "ambient_turn.hpp"

#include <agent-cpp/agent.hpp>
#include <mutex>
#include <simdjson.h>

namespace agent {

namespace {

struct ActiveMemoryGuard {
    explicit ActiveMemoryGuard(FlowMemory *memory) { ambient::active_memory = memory; }
    ~ActiveMemoryGuard() { ambient::active_memory = nullptr; }
};

static void encode_history_json(const FlowMemory &state, std::string &json) {
    size_t estimated = 2;
    for (const auto &msg : state.history) {
        estimated += msg.role.size() + msg.content.size() + msg.tool_call_id.size() + 128;
        for (const auto &tc : msg.tool_calls) estimated += tc.id.size() + tc.name.size() + tc.arguments.size() + 64;
    }
    json.reserve(estimated);

    json += '[';
    for (size_t i = 0; i < state.history.size(); ++i) {
        if (i > 0) json += ',';
        const auto &msg = state.history[i];
        json += "{\"role\":\"";
        utils::escape_json_string(msg.role, json);
        json += "\",\"content\":\"";
        utils::escape_json_string(msg.content, json);
        json += "\",\"tool_call_id\":\"";
        utils::escape_json_string(msg.tool_call_id, json);
        json += "\",\"tool_calls\":[";
        for (size_t j = 0; j < msg.tool_calls.size(); ++j) {
            if (j > 0) json += ',';
            const auto &tc = msg.tool_calls[j];
            json += "{\"id\":\"";
            utils::escape_json_string(tc.id, json);
            json += "\",\"name\":\"";
            utils::escape_json_string(tc.name, json);
            json += "\",\"arguments\":\"";
            utils::escape_json_string(tc.arguments, json);
            json += "\"}";
        }
        json += "]}";
    }
    json += ']';
}

static int count_message_tokens(Session &session, const Message &msg) {
    if (msg.token_count >= 0) return msg.token_count;

    int tokens = 0;
    const auto *ops = providers::find_provider_ops(session.config.provider);
    if (ops && ops->count_tokens) {
        auto add = [&](std::string_view part) {
            if (part.empty()) return;
            auto res = ops->count_tokens(session, part);
            if (res.ok) tokens += res.value;
        };
        add(msg.role);
        add(msg.content);
        for (const auto &tc : msg.tool_calls) {
            add(tc.name);
            add(tc.arguments);
        }
    }

    if (tokens == 0) {
        size_t len = msg.role.size() + 2 + msg.content.size() + 1;
        for (const auto &tc : msg.tool_calls) len += tc.name.size() + tc.arguments.size();
        tokens = static_cast<int>((len + 3) / 4);
    }

    msg.token_count = tokens;
    return tokens;
}

static int count_text_tokens(Session &session, std::string_view text) {
    if (text.empty()) return 0;
    const auto *ops = providers::find_provider_ops(session.config.provider);
    if (ops && ops->count_tokens) {
        auto res = ops->count_tokens(session, text);
        if (res.ok) return res.value;
    }
    return static_cast<int>((text.size() + 3) / 4);
}

static Result<void> prune_history_state(Session &session, FlowMemory &state) {
    // The llama.cpp path injects session.tools_system_prompt as an extra prompt segment (see
    // run_turn's need_tools_prefix/combined_system logic) that isn't part of state.history, so
    // reserve budget for it here or pruning can leave the real prompt over context_window.
    int tools_prefix_tokens = 0;
    if (session.config.provider == AiProvider::LlamaCpp) tools_prefix_tokens = count_text_tokens(session, session.tools_system_prompt);

    const int max_allowed = std::max(200, session.config.context_window - session.config.max_tokens - 100 - tools_prefix_tokens);

    if (state.history.empty()) return ok();

    const bool has_system = (state.history[0].role == "system");
    const size_t fixed_start = has_system ? 1 : 0;
    const size_t min_keep = fixed_start + 1;

    int total = 0;
    for (const auto &msg : state.history) total += count_message_tokens(session, msg);

    size_t drop_from = fixed_start;
    size_t drop_to = fixed_start;

    while (total > max_allowed) {
        if (drop_to >= state.history.size()) break;
        if ((state.history.size() - (drop_to - drop_from)) <= min_keep) break;

        if (state.history[drop_to].role == "tool") {
            size_t group = 1;
            while (drop_to + group < state.history.size() && state.history[drop_to + group].role == "tool") ++group;
            if ((state.history.size() - (drop_to - drop_from) - group) < min_keep) break;
            for (size_t i = drop_to; i < drop_to + group; ++i) total -= count_message_tokens(session, state.history[i]);
            drop_to += group;
            continue;
        }

        size_t group = 1;
        if (state.history[drop_to].role == "assistant" && !state.history[drop_to].tool_calls.empty()) {
            while (drop_to + group < state.history.size() && state.history[drop_to + group].role == "tool") ++group;
        }

        if ((state.history.size() - (drop_to - drop_from) - group) < min_keep) break;

        for (size_t i = drop_to; i < drop_to + group; ++i) total -= count_message_tokens(session, state.history[i]);

        drop_to += group;
    }

    if (drop_to > drop_from) {
        auto beg = state.history.begin() + static_cast<ptrdiff_t>(drop_from);
        state.history.erase(beg, beg + static_cast<ptrdiff_t>(drop_to - drop_from));
        state.stats.prune_cycles++;

        trigger_event(session, EventType::OnPrune, std::span<const uint8_t>());
        utils::log(session, LogLevel::Debug, "Pruned " + std::to_string(drop_to - drop_from) + " messages from history.");
    }

    return ok();
}

static void fire_tool_hooks(Session &session, FlowMemory &state, FlowHook::When when, std::string_view tool_name, std::string_view args,
                            std::string_view result) {
    for (const auto &hook : ambient::active_hooks) {
        if (hook.when != when) continue;
        if (!hook.tool_name.empty() && hook.tool_name != tool_name) continue;
        if (hook.callback) hook.callback(session, state, tool_name, args, result, hook.user_data);
    }
}

} // namespace

Result<void> save_flow_memory(Session &session, const FlowMemory &memory) {
    if (memory.id.empty()) return fail(ErrorCode::InvalidConfig, "FlowMemory.id is required to save history.");
    std::string json;
    encode_history_json(memory, json);
    auto store = store_memory(session, "history:" + memory.id, json);
    if (!store.ok) return store;
    std::string stats =
        "{\"prompt_tokens\":" + std::to_string(memory.stats.prompt_tokens) +
        ",\"completion_tokens\":" + std::to_string(memory.stats.completion_tokens) +
        ",\"tool_calls\":" + std::to_string(memory.stats.tool_calls) + ",\"compressions\":" + std::to_string(memory.stats.compressions) +
        ",\"prune_cycles\":" + std::to_string(memory.stats.prune_cycles) + ",\"turns\":" + std::to_string(memory.stats.turns) + "}";
    return store_memory(session, "history_stats:" + memory.id, stats);
}

Result<void> load_flow_memory(Session &session, FlowMemory &memory) {
    if (memory.id.empty()) return fail(ErrorCode::InvalidConfig, "FlowMemory.id is required to load history.");

    auto hist = retrieve_memory(session, "history:" + memory.id);
    if (!hist.ok) return fail(hist.error.code, hist.error.message);

    simdjson::padded_string padded(hist.value.data(), hist.value.size());
    simdjson::ondemand::document doc;
    if (ambient::json_parser.iterate(padded).get(doc)) return fail(ErrorCode::ParseError, "Failed to parse stored history JSON.");

    simdjson::ondemand::array arr;
    if (doc.get_array().get(arr)) return fail(ErrorCode::ParseError, "Stored history is not a JSON array.");

    std::vector<Message> loaded;
    for (auto item : arr) {
        simdjson::ondemand::object obj;
        if (item.get_object().get(obj)) continue;
        Message msg;
        std::string_view role;
        std::string_view content;
        std::string_view tool_call_id;
        if (!obj["role"].get_string().get(role)) msg.role = std::string(role);
        if (!obj["content"].get_string().get(content)) msg.content = std::string(content);
        if (!obj["tool_call_id"].get_string().get(tool_call_id)) msg.tool_call_id = std::string(tool_call_id);

        simdjson::ondemand::array tcs;
        if (!obj["tool_calls"].get_array().get(tcs)) {
            for (auto tc_item : tcs) {
                simdjson::ondemand::object tc;
                if (tc_item.get_object().get(tc)) continue;
                ToolCall call;
                std::string_view id, name, arguments;
                if (!tc["id"].get_string().get(id)) call.id = std::string(id);
                if (!tc["name"].get_string().get(name)) call.name = std::string(name);
                if (!tc["arguments"].get_string().get(arguments)) call.arguments = std::string(arguments);
                msg.tool_calls.push_back(std::move(call));
            }
        }
        loaded.push_back(std::move(msg));
    }
    memory.history = std::move(loaded);

    auto stats_res = retrieve_memory(session, "history_stats:" + memory.id);
    if (stats_res.ok) {
        simdjson::padded_string sp(stats_res.value.data(), stats_res.value.size());
        simdjson::ondemand::document sdoc;
        if (!ambient::json_parser.iterate(sp).get(sdoc)) {
            simdjson::ondemand::object sobj;
            if (!sdoc.get_object().get(sobj)) {
                int64_t v = 0;
                if (!sobj["prompt_tokens"].get_int64().get(v)) memory.stats.prompt_tokens = static_cast<int>(v);
                if (!sobj["completion_tokens"].get_int64().get(v)) memory.stats.completion_tokens = static_cast<int>(v);
                if (!sobj["tool_calls"].get_int64().get(v)) memory.stats.tool_calls = static_cast<int>(v);
                if (!sobj["compressions"].get_int64().get(v)) memory.stats.compressions = static_cast<int>(v);
                if (!sobj["prune_cycles"].get_int64().get(v)) memory.stats.prune_cycles = static_cast<int>(v);
                if (!sobj["turns"].get_int64().get(v)) memory.stats.turns = static_cast<int>(v);
            }
        }
    }

    return ok();
}

Result<std::string> run_turn(Session &session, FlowMemory &state, std::string_view user_prompt, bool stream, TokenStreamFn on_token,
                             void *token_user_data) {
    ActiveMemoryGuard guard(&state);

    trigger_event(session, EventType::OnTurnStart, std::span<const uint8_t>());

    if (!user_prompt.empty()) {
        Message user_msg;
        user_msg.role = "user";
        user_msg.content = std::string(user_prompt);
        state.history.push_back(std::move(user_msg));
    }

    {
        int total = 0;
        for (const auto &msg : state.history) total += count_message_tokens(session, msg);
        if (total > session.config.context_window * 7 / 10 && state.history.size() > 6) compress_context(session, state);
    }

    std::string final_text;
    int tool_rounds = 0;
    bool loop = true;

    while (loop) {
        auto prune_res = prune_history_state(session, state);
        if (!prune_res.ok) return fail<std::string>(prune_res.error.code, prune_res.error.message);

        Arena &arena = ambient::current_arena(session);
        arena.reset();

        const bool provider_needs_text_tools = session.config.provider == AiProvider::LlamaCpp;
        const bool has_system = !state.history.empty() && state.history[0].role == "system";
        const bool need_tools_prefix = provider_needs_text_tools && !session.tools_system_prompt.empty() && !has_system;
        const size_t view_count = state.history.size() + (need_tools_prefix ? 1 : 0);

        auto message_views = arena.allocate_span<MessageView>(view_count);
        if (view_count != 0 && message_views.size() != view_count) {
            return fail<std::string>(ErrorCode::DecodeFailed, "Arena capacity exhausted building message views. "
                                                              "Increase Config::arena_capacity or reduce Config::context_window.");
        }

        size_t view_offset = 0;
        if (need_tools_prefix) {
            message_views[0].role = "system";
            message_views[0].content = session.tools_system_prompt;
            message_views[0].tool_calls = {};
            message_views[0].tool_call_id = "";
            message_views[0].tensors = {};
            view_offset = 1;
        }

        std::string_view combined_system;
        if (has_system && provider_needs_text_tools && !session.tools_system_prompt.empty()) {
            std::string combined;
            combined.reserve(state.history[0].content.size() + 1 + session.tools_system_prompt.size());
            combined += state.history[0].content;
            combined += '\n';
            combined += session.tools_system_prompt;
            combined_system = arena.allocate_string(combined);
        }

        for (size_t i = 0; i < state.history.size(); ++i) {
            MessageView &mv = message_views[i + view_offset];
            const Message &m = state.history[i];
            mv.role = m.role;
            if (i == 0 && !combined_system.empty()) mv.content = combined_system;
            else mv.content = m.content;
            mv.tool_call_id = m.tool_call_id;

            if (!m.tool_calls.empty()) {
                auto tcvs = arena.allocate_span<ToolCallView>(m.tool_calls.size());
                for (size_t j = 0; j < m.tool_calls.size(); ++j)
                    tcvs[j] = {m.tool_calls[j].id, m.tool_calls[j].name, m.tool_calls[j].arguments};
                mv.tool_calls = tcvs;
            } else {
                mv.tool_calls = {};
            }

            if (!m.tensors.empty()) {
                auto tvs = arena.allocate_span<TensorView>(m.tensors.size());
                for (size_t j = 0; j < m.tensors.size(); ++j) {
                    const TensorData &td = m.tensors[j];
                    tvs[j] = {td.name, td.dtype, std::span<const int64_t>(td.shape.data(), td.shape.size()),
                              std::span<const uint8_t>(td.data.data(), td.data.size())};
                }
                mv.tensors = tvs;
            } else {
                mv.tensors = {};
            }
        }

        trigger_event(session, EventType::OnInferenceSubmit, std::span<const uint8_t>());

        InferRequest req{.messages = message_views, .stream = stream, .on_token = on_token, .token_user_data = token_user_data};

        auto resp_res = infer(session, req);
        if (!resp_res.ok) return fail<std::string>(resp_res.error.code, resp_res.error.message);

        trigger_event(session, EventType::OnInferenceComplete, std::span<const uint8_t>());

        const auto &resp = resp_res.value;

        state.stats.prompt_tokens += resp.usage.prompt_tokens;
        state.stats.completion_tokens += resp.usage.completion_tokens;
        state.stats.turns++;
        {
            std::lock_guard<std::mutex> lock(*session.stats_mutex);
            session.stats.total_prompt_tokens += resp.usage.prompt_tokens;
            session.stats.total_completion_tokens += resp.usage.completion_tokens;
            session.stats.total_turns++;
        }

        Message assistant_msg;
        assistant_msg.role = std::string(resp.message.role);
        assistant_msg.content = std::string(resp.message.content);
        for (const auto &tcv : resp.message.tool_calls) {
            assistant_msg.tool_calls.push_back(ToolCall{std::string(tcv.id), std::string(tcv.name), std::string(tcv.arguments)});
        }

        final_text = assistant_msg.content;
        state.history.push_back(std::move(assistant_msg));

        if (!resp.message.tool_calls.empty()) {
            if (tool_rounds >= session.config.max_tool_call_rounds) {
                utils::log(session, LogLevel::Warning,
                           "Tool call round limit (" + std::to_string(session.config.max_tool_call_rounds) +
                               ") reached; stopping tool loop.");
                return fail<std::string>(ErrorCode::ToolCallLimitExceeded,
                                         "Exceeded max_tool_call_rounds (" + std::to_string(session.config.max_tool_call_rounds) + ")");
            }
            ++tool_rounds;

            for (const auto &tcv : resp.message.tool_calls) {
                trigger_event(session, EventType::OnToolCall,
                              std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(tcv.name.data()), tcv.name.size()));

                fire_tool_hooks(session, state, FlowHook::When::Pre, tcv.name, tcv.arguments, {});

                std::string result_text;
                auto exec_res = execute_tool(session, tcv.name, tcv.arguments);
                if (exec_res.ok) result_text = std::move(exec_res.value);
                else result_text = "Error: " + exec_res.error.message;

                fire_tool_hooks(session, state, FlowHook::When::Post, tcv.name, tcv.arguments, result_text);

                trigger_event(session, EventType::OnToolResult,
                              std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(result_text.data()), result_text.size()));

                state.stats.tool_calls++;
                {
                    std::lock_guard<std::mutex> lock(*session.stats_mutex);
                    session.stats.total_tool_calls++;
                }

                Message tool_msg;
                tool_msg.role = "tool";
                tool_msg.content = std::move(result_text);
                tool_msg.tool_call_id = std::string(tcv.id);
                state.history.push_back(std::move(tool_msg));
            }
            continue;
        }

        loop = false;
    }

    auto save_res = save_flow_memory(session, state);
    if (!save_res.ok) utils::log(session, LogLevel::Warning, "save_flow_memory failed: " + save_res.error.message);

    return ok(std::move(final_text));
}

} // namespace agent
