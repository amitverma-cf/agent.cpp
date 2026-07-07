#include <agent-cpp/agent.hpp>
#include <json.hpp>
#include <sstream>
#include "providers/provider_ops.hpp"

namespace agent {

static int count_message_tokens(Session &session, const Message &msg) {
    std::string formatted = msg.role + ": " + msg.content + "\n";
    const auto *ops = providers::find_provider_ops(session.config.provider);
    if (ops && ops->count_tokens) {
        auto res = ops->count_tokens(session, formatted);
        if (res.ok) {
            return res.value;
        }
    }
    if (formatted.empty()) return 0;
    return static_cast<int>((formatted.length() + 2) / 3);
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

Conversation::Conversation(Session &sess) : session(sess) {
    auto result = retrieve_memory(session, "history");
    if (result.ok) {
        auto json = nlohmann::json::parse(result.value, nullptr, false);
        if (!json.is_discarded() && json.is_array()) {
            for (const auto &item : json) {
                if (item.is_object() && item.contains("role") && item.contains("content")) {
                    history.push_back({
                        item["role"].get<std::string>(),
                        item["content"].get<std::string>()
                    });
                }
            }
        }
    }
}

Result<void> add_message(Conversation &conv, std::string role, std::string content) {
    conv.history.push_back({std::move(role), std::move(content)});
    return ok();
}

static Result<void> prune_conversation(Conversation &conv) {
    int max_allowed_tokens = conv.session.config.context_window - conv.session.config.max_tokens - 100;
    if (max_allowed_tokens < 200) max_allowed_tokens = 200;

    if (conv.history.empty()) return ok();

    bool has_system = (conv.history[0].role == "system");

    int current_tokens = estimate_history_tokens(conv.session, conv.history);
    size_t drop_count = 0;
    size_t min_required_size = has_system ? 2 : 1;

    while (current_tokens > max_allowed_tokens && (conv.history.size() - drop_count) > min_required_size) {
        size_t idx_to_drop = has_system ? (drop_count + 1) : drop_count;
        int msg_tokens = count_message_tokens(conv.session, conv.history[idx_to_drop]);
        current_tokens -= msg_tokens;
        drop_count++;
    }

    if (drop_count > 0) {
        auto start = has_system ? (conv.history.begin() + 1) : conv.history.begin();
        conv.history.erase(start, start + drop_count);
    }

    return ok();
}

Result<std::string> complete_conversation(Conversation &conv) {
    auto prune_res = prune_conversation(conv);
    if (!prune_res.ok) {
        return fail<std::string>(prune_res.error.code, prune_res.error.message);
    }

    int prompt_tokens_estimate = estimate_history_tokens(conv.session, conv.history);
    if (prompt_tokens_estimate > conv.session.config.context_window - conv.session.config.max_tokens) {
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "Conversation history is too large to fit in context window even after pruning.");
    }

    std::stringstream ss;
    for (const auto &msg : conv.history) {
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

    auto sync_res = sync_conversation(conv);
    if (!sync_res.ok) {
        return fail<std::string>(sync_res.error.code, sync_res.error.message);
    }

    return ok(result.value.text);
}

Result<std::vector<Message>> get_conversation_history(Conversation &conv) {
    return ok(conv.history);
}

Result<void> sync_conversation(Conversation &conv) {
    nlohmann::json array = nlohmann::json::array();
    for (const auto &msg : conv.history) {
        array.push_back({{"role", msg.role}, {"content", msg.content}});
    }
    return store_memory(conv.session, "history", array.dump());
}

} // namespace agent