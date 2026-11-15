#pragma once

// The `settings` tool: the agent reads and changes the chat's safe settings when the user asks in plain words ("make it faster",
// "more creative", "longer answers"). It goes through the same checks and the same function as the Settings screen. A change is
// queued and applied when the answer is finished, because most of them reload the model, and a model cannot be reloaded in the
// middle of its own answer. Local only, the web address and the other switches that belong to the user are not offered.

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>

namespace agent::cli {

// What the tool needs from the chat. The functions get `user`.
struct SettingsBridge {
    // The value as the Settings screen shows it ("" for a key it does not know). The key "draft_model_choices" lists the installed
    // models that can draft, separated by ", ".
    std::string (*get)(void *user, const std::string &key) = nullptr;
    // Changes a setting like the Settings screen does; "" when done, else the reason it was refused.
    std::string (*apply)(void *user, const std::string &key, const std::string &value) = nullptr;
    // Loads the model again with the current settings (true when it loaded).
    bool (*reload)(void *user) = nullptr;
    void *user = nullptr;
};

// While apply_pending_settings runs, the Settings code asks for one reload at the end instead of one after every change: it calls this
// with true when it would have reloaded, and gets true back when the reload is left for later.
bool settings_defer_reload(bool want_reload);

// The tool definition. `bridge` must outlive the chat.
Tool settings_tool(SettingsBridge *bridge);

// True when the call (its JSON arguments) would change a setting; reading is not a change.
bool settings_call_changes(std::string_view arguments);

// Applies what the agent asked for during the turn. Returns text for the user ("" when nothing was queued).
std::string apply_pending_settings(SettingsBridge &bridge);

} // namespace agent::cli
