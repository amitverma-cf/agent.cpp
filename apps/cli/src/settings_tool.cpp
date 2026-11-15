#include "settings_tool.hpp"

#include "speed.hpp"

#include <algorithm>
#include <simdjson.h>
#include <utility>
#include <vector>

namespace agent::cli {

namespace {

struct Allowed {
    const char *key;
    const char *meaning;
    bool reloads; // the model is loaded again
};

// What the agent may change. Not here on purpose: local_only, the web search address, the backend, the memory budget, approvals.
constexpr Allowed kAllowed[] = {
    {"temperature", "0 is exact and steady, 1 is creative, up to 2 (0 to 2)", true},
    {"max_tokens", "the longest answer in tokens, a whole number of at least 16", true},
    {"context_window", "how much of the conversation the model sees, in tokens (at least 2048); more uses more memory", true},
    {"kv_cache", "memory type of the conversation cache: auto, f16, q8_0 or q4_0 (smaller types use less memory)", true},
    {"speculative", "off, prompt lookup (reuses text already in the conversation) or draft model (a small model guesses ahead)", true},
    {"draft_model", "the small installed model that drafts when speculative is draft model", true},
    {"mtp", "on or off: the model's own multi-token drafting, for model files that have it", true},
    {"mode", "ask, plan or auto: when the agent asks before changing things", false},
};

const Allowed *find_allowed(std::string_view key) {
    for (const Allowed &a : kAllowed)
        if (key == a.key) return &a;
    return nullptr;
}

std::vector<std::pair<std::string, std::string>> g_pending; // key, value
bool g_defer = false, g_reload_wanted = false;

constexpr std::string_view kSchema =
    R"({"type":"object","properties":{"action":{"type":"string","enum":["get","set"],"description":"get reads the settings, set changes one"},"key":{"type":"string","enum":["temperature","max_tokens","context_window","kv_cache","speculative","draft_model","mtp","mode"],"description":"the setting; leave out with get to read all of them"},"value":{"type":"string","description":"the new value, with set"}},"required":["action"]})";

std::string describe_all(const SettingsBridge &b) {
    std::string out = "Current settings:\n";
    for (const Allowed &a : kAllowed) out += std::string("- ") + a.key + " = " + b.get(b.user, a.key) + "  (" + a.meaning + ")\n";
    const std::string choices = b.get(b.user, "draft_model_choices");
    out += "Models that can draft: " + (choices.empty() ? std::string("none installed") : choices) + "\n";
    out += "Changing a setting reloads the model (a few seconds) after the answer. local_only and the other privacy switches are not "
           "yours to change: tell the user to open Settings.";
    return out;
}

Result<std::string> settings_cb(std::string_view args, void *user) {
    auto *b = static_cast<SettingsBridge *>(user);
    simdjson::dom::parser parser;
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) || !doc.is_object())
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "settings needs {\"action\":\"get\"} or {\"action\":\"set\",\"key\":...,\"value\":...}");
    std::string_view action, key;
    simdjson::dom::element v;
    const bool has_value = !doc["value"].get(v);
    if (doc["action"].get(action)) action = has_value ? "set" : "get";
    if (doc["key"].get(key)) key = {};
    if (action == "get") {
        if (key.empty()) return ok(describe_all(*b));
        const Allowed *a = find_allowed(key);
        if (!a) return fail<std::string>(ErrorCode::InvalidConfig, "no setting named " + std::string(key) + " that you may read");
        return ok(std::string(a->key) + " = " + b->get(b->user, a->key) + "  (" + a->meaning + ")");
    }
    if (action != "set") return fail<std::string>(ErrorCode::InvalidConfig, "action is get or set");
    if (key == "local_only" || key == "search_url" || key == "backend" || key == "auto_approve" || key == "use_tools")
        return fail<std::string>(ErrorCode::InvalidConfig, std::string(key) + " belongs to the user; ask them to change it in Settings");
    const Allowed *a = find_allowed(key);
    if (!a) return fail<std::string>(ErrorCode::InvalidConfig, "no setting named " + std::string(key) + "; use action get to see them");
    if (!has_value) return fail<std::string>(ErrorCode::InvalidConfig, "set needs a value");
    // The value may come as a JSON number or a boolean.
    std::string value;
    if (v.is_string()) value = std::string(v.get_string().value_unsafe());
    else if (v.is_bool()) value = v.get_bool().value_unsafe() ? "on" : "off";
    else value = simdjson::minify(v);
    if (key == "mtp") {
        if (value == "true") value = "on";
        if (value == "false") value = "off";
    }
    if (auto bad = check_setting_value(key, value); !bad.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, std::string(key) + " must be " + bad);
    const std::string choices = b->get(b->user, "draft_model_choices");
    const std::string none_installed = "none installed (the user can pull a small model with agent model pull)";
    if (key == "draft_model" && (", " + choices + ", ").find(", " + value + ", ") == std::string::npos)
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "draft_model must be one of the installed small models: " + (choices.empty() ? none_installed : choices));
    if (key == "speculative" && value.find("draft") != std::string::npos) {
        const bool queued = std::any_of(g_pending.begin(), g_pending.end(), [](const auto &p) { return p.first == "draft_model"; });
        if (b->get(b->user, "draft_model") == "(none)" && !queued)
            return fail<std::string>(ErrorCode::InvalidConfig,
                                     "choose a draft_model first. Installed small models: " + (choices.empty() ? none_installed : choices));
    }
    const std::string old = b->get(b->user, std::string(key));
    std::erase_if(g_pending, [&](const auto &p) { return p.first == key; });
    g_pending.emplace_back(std::string(key), value);
    return ok("Queued: " + std::string(key) + " " + old + " -> " + value + ". It is applied when this answer is finished" +
              (a->reloads ? ", and the model reloads (a few seconds)" : "") +
              ". Tell the user what you changed and that it takes effect for the next message.");
}

} // namespace

bool settings_defer_reload(bool want_reload) {
    if (!g_defer) return false;
    if (want_reload) g_reload_wanted = true;
    return true;
}

bool settings_call_changes(std::string_view arguments) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(arguments);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) || !doc.is_object()) return false;
    std::string_view action;
    if (!doc["action"].get(action)) return action == "set";
    simdjson::dom::element ignore;
    return !doc["value"].get(ignore);
}

Tool settings_tool(SettingsBridge *bridge) {
    return Tool{.name = "settings",
                .description = "Read or change this chat's settings when the user asks to make answers faster, slower, more creative, "
                               "steadier or longer, or to use a draft model or the memory-saving cache. Use action get to see the values "
                               "and what they mean, then set one at a time. Faster: lower max_tokens, a smaller context_window, kv_cache "
                               "q8_0 or q4_0, speculative prompt lookup or a draft model. More creative: higher temperature. Steadier: "
                               "temperature 0. The user is asked first.",
                .parameter_schema = kSchema,
                .callback = settings_cb,
                .user_data = bridge};
}

std::string apply_pending_settings(SettingsBridge &b) {
    if (g_pending.empty()) return {};
    std::vector<std::pair<std::string, std::string>> work;
    work.swap(g_pending);
    std::string report;
    g_defer = true;
    g_reload_wanted = false;
    for (const auto &[key, value] : work) {
        const std::string before = b.get(b.user, key);
        const std::string err = b.apply(b.user, key, value);
        if (!err.empty()) report += "[settings] " + key + " was not changed: " + err + "\n";
        else report += "[settings] " + key + ": " + before + " -> " + b.get(b.user, key) + "\n";
    }
    g_defer = false;
    if (g_reload_wanted) {
        g_reload_wanted = false;
        report += b.reload(b.user) ? "[settings] the model was reloaded with the new settings\n"
                                   : "[settings] the model did not reload; try /model NAME\n";
    }
    return report;
}

} // namespace agent::cli
