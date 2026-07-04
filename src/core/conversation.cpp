#include <agent-cpp/agent.hpp>
#include <json.hpp>
#include <sstream>

namespace agent {

Result<void> add_message(Conversation &conv, std::string role, std::string content) {
    auto history_res = get_conversation_history(conv);
    std::vector<Message> history;
    if (history_res.ok) {
        history = std::move(history_res.value);
    } else if (history_res.error.code != ErrorCode::KeyNotFound) {
        return fail(history_res.error.code, history_res.error.message);
    }

    history.push_back({std::move(role), std::move(content)});

    nlohmann::json array = nlohmann::json::array();
    for (const auto &msg : history) {
        array.push_back({{"role", msg.role}, {"content", msg.content}});
    }

    auto store_res = store_memory(conv.session, "history", array.dump());
    if (!store_res.ok) {
        return store_res;
    }
    return ok();
}

static Result<void> prune_conversation(Conversation &conv) {
    int max_allowed_tokens = conv.session.config.context_window - conv.session.config.max_tokens - 100;
    if (max_allowed_tokens < 200) max_allowed_tokens = 200;

    auto history_res = get_conversation_history(conv);
    if (!history_res.ok) {
        if (history_res.error.code == ErrorCode::KeyNotFound) {
            return ok();
        }
        return fail(history_res.error.code, history_res.error.message);
    }
    if (history_res.value.empty()) return ok();

    auto history = std::move(history_res.value);

    auto estimate_tokens = [](const std::vector<Message> &hist) {
        size_t total_chars = 0;
        for (const auto &msg : hist) {
            total_chars += msg.role.length() + msg.content.length() + 4;
        }
        return static_cast<int>(total_chars / 4);
    };

    bool has_system = (history[0].role == "system");

    while (estimate_tokens(history) > max_allowed_tokens && history.size() > (has_system ? 2 : 1)) {
        if (has_system) {
            history.erase(history.begin() + 1);
        } else {
            history.erase(history.begin());
        }
    }

    nlohmann::json array = nlohmann::json::array();
    for (const auto &msg : history) {
        array.push_back({{"role", msg.role}, {"content", msg.content}});
    }
    auto store_res = store_memory(conv.session, "history", array.dump());
    if (!store_res.ok) {
        return store_res;
    }
    return ok();
}

Result<std::string> complete_conversation(Conversation &conv) {
    auto prune_res = prune_conversation(conv);
    if (!prune_res.ok) {
        return fail<std::string>(prune_res.error.code, prune_res.error.message);
    }

    auto history_res = get_conversation_history(conv);
    if (!history_res.ok) {
        return fail<std::string>(history_res.error.code, history_res.error.message);
    }

    auto estimate_tokens = [](const std::vector<Message> &hist) {
        size_t total_chars = 0;
        for (const auto &msg : hist) {
            total_chars += msg.role.length() + msg.content.length() + 4;
        }
        return static_cast<int>(total_chars / 4);
    };

    int prompt_tokens_estimate = estimate_tokens(history_res.value) + 5;
    if (prompt_tokens_estimate > conv.session.config.context_window - conv.session.config.max_tokens) {
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "Conversation history is too large to fit in context window even after pruning.");
    }

    std::stringstream ss;
    for (const auto &msg : history_res.value) {
        ss << msg.role << ": " << msg.content << "\n";
    }
    ss << "assistant: ";
    std::string prompt = ss.str();
    auto result = generate_text(conv.session, prompt);
    if (!result.ok) {
        return fail<std::string>(result.error.code, result.error.message);
    }
    auto add_res = add_message(conv, "assistant", result.value.text);
    if (!add_res.ok) {
        return fail<std::string>(add_res.error.code, add_res.error.message);
    }
    return ok(result.value.text);
}

Result<std::vector<Message>> get_conversation_history(Conversation &conv) {
    auto result = retrieve_memory(conv.session, "history");
    if (!result.ok) {
        if (result.error.code == ErrorCode::KeyNotFound) {
            return ok(std::vector<Message>{});
        }
        return fail<std::vector<Message>>(result.error.code, result.error.message);
    }

    auto json = nlohmann::json::parse(result.value, nullptr, false);
    if (json.is_discarded() || !json.is_array()) {
        return fail<std::vector<Message>>(ErrorCode::ParseError, "Failed to parse conversation history from memory.");
    }

    std::vector<Message> history;
    for (const auto &item : json) {
        if (item.is_object() && item.contains("role") && item.contains("content")) {
            history.push_back({
                item["role"].get<std::string>(),
                item["content"].get<std::string>()
            });
        }
    }
    return ok(history);
}

} // namespace agent