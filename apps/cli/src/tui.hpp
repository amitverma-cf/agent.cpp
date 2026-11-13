#pragma once

// The full-screen chat interface (FTXUI): a welcome screen, a scrolling transcript with markdown-lite rendering and compact tool cards,
// a slash-command palette, an input line with history, a status bar with a context gauge, and a modal for tool permissions. It knows
// nothing about models or sessions: chat.cpp hands it a TuiHost, a table of function pointers (the project avoids virtual calls), and
// the host does the work on a worker thread.

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

enum class TuiKind {
    User,      // what the user typed
    Assistant, // streamed answer text: consecutive pieces extend one block
    Tool,      // "[tool] name args" and "[result] ..." blocks
    Note,      // everything else the host prints (help, stats, switching models)
    Error,
};

// A slash command offered in the palette that opens when the input starts with "/".
struct TuiCommand {
    const char *name; // "/model"
    const char *help; // "show or switch the model"
};

struct TuiHost {
    void *user = nullptr;
    // The title bar after the brand: model, engine, session id.
    std::string (*header)(void *user) = nullptr;
    // The welcome screen's second line: the folder the agent works in, tools, anything useful before the first message.
    std::string (*subtitle)(void *user) = nullptr;
    // The status footer (speed, tokens, KV cache); may be empty.
    std::string (*footer)(void *user) = nullptr;
    // Handles one input line (a message or a slash command) on a worker thread; it reports through tui_emit() and may call
    // tui_ask(). Returns false when the user asked to exit.
    bool (*submit)(void *user, const std::string &line) = nullptr;
    // Cancels the running answer (Esc, or Ctrl-C while it is running). Safe to call from the UI thread.
    void (*cancel)(void *user) = nullptr;
    // The commands listed in the palette.
    std::span<const TuiCommand> commands;
};

// Runs the interface until the user quits; returns the process exit code. Needs a terminal on stdin and stdout.
int run_tui(const TuiHost &host);

// Thread safe, valid only while run_tui() is running. Assistant text extends the last assistant block; other kinds start a new block per
// call (a trailing newline is dropped).
void tui_emit(TuiKind kind, std::string_view text);
// Blocks the calling (worker) thread until the user answers the question: 'y' (true), 'n' (false) or 'a' (true, and *always = true).
bool tui_ask(std::string_view question, bool *always);

// A free-text question from the worker thread: shows `question` (with the default in brackets) above the input line and blocks until the
// user presses Enter; an empty answer means `default_value`. Returns false when the interface is closing.
bool tui_prompt(std::string_view question, std::string_view default_value, std::string &answer);

// The markdown-lite renderer on its own, as plain text laid out for `width` columns (for tests).
std::string render_markdown_text(std::string_view markdown, int width);

// Helpers with no screen behind them (for tests).
// The first `max_lines` lines of a tool result, each cut to `max_width` columns, and "... N more lines" when there are more.
std::string summarize_tool_result(std::string_view result, int max_lines, int max_width);
// Percent of the context window in use, read from a footer such as "... | KV 1191/4096 q8_0 ..."; -1 when the footer has none.
int context_percent(std::string_view footer);
// The commands whose name starts with what the user typed ("/mo" -> "/model"); everything for "/".
std::vector<size_t> filter_commands(std::span<const TuiCommand> commands, std::string_view typed);

} // namespace agent::cli
