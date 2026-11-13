#pragma once

// The full-screen chat interface (FTXUI): a scrolling transcript with markdown-lite rendering, an input line, a status footer,
// slash commands, and a modal for tool permissions. It knows nothing about models or sessions: chat.cpp hands it a TuiHost, a table
// of function pointers (the project avoids virtual calls), and the host does the work on a worker thread.

#include <string>
#include <string_view>

namespace agent::cli {

enum class TuiKind {
    User,      // what the user typed
    Assistant, // streamed answer text: consecutive pieces extend one block
    Tool,      // "[tool] ..." / "[result] ..." lines
    Note,      // everything else the host prints (help, stats, switching models)
    Error,
};

struct TuiHost {
    void *user = nullptr;
    // One or two lines for the title bar (model, engine, workspace).
    std::string (*header)(void *user) = nullptr;
    // The status footer (speed, tokens, KV cache); may be empty.
    std::string (*footer)(void *user) = nullptr;
    // Handles one input line (a message or a slash command) on a worker thread; it reports through tui_emit() and may call
    // tui_ask(). Returns false when the user asked to exit.
    bool (*submit)(void *user, const std::string &line) = nullptr;
    // Cancels the running answer (Esc, or Ctrl-C while it is running). Safe to call from the UI thread.
    void (*cancel)(void *user) = nullptr;
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

} // namespace agent::cli
