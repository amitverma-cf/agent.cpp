#include <agent-cpp/agent.hpp>
#include <simdjson.h>

namespace agent {

Result<void> compress_context(Session &session, ConversationState &state, size_t keep_recent) {
    bool has_system = !state.history.empty() && state.history[0].role == "system";
    size_t start = has_system ? 1 : 0;

    if (state.history.size() <= start + keep_recent + 2)
        return ok();

    size_t compress_end = state.history.size() - keep_recent;

    std::string prompt =
        "Summarize the following conversation. Preserve all key facts, decisions, code, and outputs:\n\n";
    for (size_t i = start; i < compress_end; ++i) {
        const auto &msg = state.history[i];
        prompt += msg.role + ": ";
        if (!msg.tool_calls.empty()) {
            for (const auto &tc : msg.tool_calls)
                prompt += "[tool:" + tc.name + " " + tc.arguments + "] ";
        } else {
            prompt += msg.content;
        }
        prompt += '\n';
    }

    session.arena.reset();
    MessageView sys_mv, user_mv;
    sys_mv.role = "system";
    sys_mv.content = "You are a concise summarizer. Output only the summary, no preamble.";
    user_mv.role = "user";
    user_mv.content = session.arena.allocate_string(prompt);

    MessageView compress_views[2] = {sys_mv, user_mv};
    ChatRequest req{
        .messages = std::span<const MessageView>(compress_views, 2),
        .max_tokens = 512,
    };

    auto res = execute_turn(session, req);
    if (!res.ok)
        return ok();

    Message summary_msg;
    summary_msg.role = "user";
    summary_msg.content = "[Compressed history]: " + std::string(res.value.message.content);

    auto it_start = state.history.begin() + static_cast<ptrdiff_t>(start);
    auto it_end = state.history.begin() + static_cast<ptrdiff_t>(compress_end);
    state.history.erase(it_start, it_end);
    state.history.insert(state.history.begin() + static_cast<ptrdiff_t>(start), std::move(summary_msg));

    return ok();
}

namespace tools {
namespace {

Result<std::string> context_compressor_callback(std::string_view arguments, void *user_data) {
    auto *session_cell = static_cast<Session **>(user_data);
    Session *session = session_cell ? *session_cell : nullptr;
    if (!session)
        return fail<std::string>(ErrorCode::InvalidConfig, "context_compressor: missing session");
    if (!session->active_conversation)
        return fail<std::string>(ErrorCode::InvalidConfig, "context_compressor: no active conversation");

    size_t keep_recent = 6;
    if (!arguments.empty()) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(arguments.data(), arguments.size());
        simdjson::ondemand::document doc;
        if (!parser.iterate(json).get(doc)) {
            simdjson::ondemand::object obj;
            if (!doc.get_object().get(obj)) {
                for (auto field : obj) {
                    auto k = field.unescaped_key();
                    if (k.error())
                        continue;
                    if (k.value() == "keep_recent") {
                        int64_t v;
                        if (!field.value().get_int64().get(v) && v >= 0)
                            keep_recent = static_cast<size_t>(v);
                    }
                }
            }
        }
    }

    auto res = compress_context(*session, *session->active_conversation, keep_recent);
    if (!res.ok)
        return fail<std::string>(res.error.code, res.error.message);
    return ok(std::string("Context compressed."));
}

} // namespace

Tool get_context_compressor_tool(Session **session_cell) {
    return Tool{
        .name = "compress_context",
        .description = "Summarize and compress the active conversation's older history to free up context window "
                       "space. Optional: keep_recent (integer, number of most recent messages to keep uncompressed, "
                       "default 6).",
        .parameter_schema =
            R"({"type":"object","properties":{"keep_recent":{"type":"integer","description":"messages to keep uncompressed"}}})",
        .callback = context_compressor_callback,
        .user_data = session_cell,
    };
}

} // namespace tools

} // namespace agent
