#pragma once

// Small pieces of the chat that need no model: @file mentions, the text of a web page, tool arguments for a permission question,
// hiding <think> blocks, the chat modes, and the per-project list of tools allowed for good.

#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// "Explain @src/main.cpp" -> the text followed by the content of src/main.cpp in a fenced block. A mention is "@" at the start of a
// word followed by a path (no spaces) of a text file that exists, relative to `workspace` or absolute; anything else stays as typed.
// At most 32 KiB per file and 96 KiB in all; a file with a NUL byte is skipped. `attached` receives the paths that were added.
std::string expand_mentions(std::string_view text, const std::string &workspace, std::vector<std::string> *attached = nullptr);

// The readable text of an HTML page: no tags, scripts or styles, a few entities decoded, blank runs collapsed.
std::string html_to_text(std::string_view html);

// A tool call's JSON arguments as lines for a person, "key: value" each, long values cut: at most max_lines lines and max_chars characters.
std::string pretty_arguments(std::string_view json, int max_lines = 8, size_t max_chars = 700);

// Hides <think>...</think> in streamed text (reasoning models), unless `show`. Tags split across tokens are handled.
struct ThinkFilter {
    std::string pending;
    bool inside = false;
    bool show = false;
};
std::string filter_think(ThinkFilter &filter, std::string_view token);
std::string flush_think(ThinkFilter &filter);

// ask: tools that change things ask first (the default); plan: only reading is allowed; auto: nothing asks.
enum class ChatMode { Ask, Plan, Auto };
bool parse_chat_mode(std::string_view text, ChatMode &mode);
const char *chat_mode_name(ChatMode mode);
// Built-in tools that change files or run commands.
bool tool_changes_things(std::string_view tool);

// The tools the user allowed "always" in a project: <workspace>/.agents/permissions.json {"always": ["write_file", ...]}.
std::vector<std::string> load_always_allowed(const std::string &workspace);
bool save_always_allowed(const std::string &workspace, const std::vector<std::string> &tools);

// A conversation index (one id per line) without `id`.
std::string remove_conversation_id(std::string_view index, std::string_view id);

} // namespace agent::cli
