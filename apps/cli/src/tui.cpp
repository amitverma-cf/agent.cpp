// The full-screen chat interface. See tui.hpp.

#include "tui.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace agent::cli {

namespace {

using namespace ftxui;

// ------------------------------------------------------------------ markdown-lite

// Inline spans of one line: **bold**, `code`, plain. Words keep their trailing space so hflow wraps them.
Elements inline_words(std::string_view line) {
    Elements words;
    bool bold = false, code = false;
    std::string word;
    auto flush = [&] {
        if (word.empty()) return;
        Element e = text(word);
        if (code) e = e | color(Color::Cyan);
        if (bold) e = e | ftxui::bold;
        words.push_back(e);
        word.clear();
    };
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '`') {
            flush();
            code = !code;
        } else if (!code && c == '*' && i + 1 < line.size() && line[i + 1] == '*') {
            flush();
            bold = !bold;
            ++i;
        } else if (c == ' ') {
            word += c;
            flush();
        } else {
            word += c;
        }
    }
    flush();
    return words;
}

Element render_markdown(std::string_view markdown) {
    Elements blocks;
    size_t at = 0;
    bool in_code = false;
    Elements code_lines;
    auto next_line = [&](std::string_view &line) {
        if (at > markdown.size()) return false;
        size_t end = markdown.find('\n', at);
        if (end == std::string_view::npos) end = markdown.size();
        line = markdown.substr(at, end - at);
        at = end + 1;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        return true;
    };
    std::string_view line;
    while (next_line(line)) {
        if (line.substr(0, 3) == "```") {
            if (in_code) {
                blocks.push_back(vbox(code_lines) | bgcolor(Color::GrayDark) | color(Color::White));
                code_lines.clear();
            }
            in_code = !in_code;
            continue;
        }
        if (in_code) {
            code_lines.push_back(text(" " + std::string(line)));
            continue;
        }
        if (line.empty()) {
            blocks.push_back(text(""));
            continue;
        }
        size_t level = 0;
        while (level < line.size() && line[level] == '#') ++level;
        if (level > 0 && level <= 6 && level < line.size() && line[level] == ' ') {
            blocks.push_back(hflow(inline_words(line.substr(level + 1))) | ftxui::bold | color(Color::Yellow));
            continue;
        }
        size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') ++indent;
        const std::string_view body = line.substr(indent);
        if (body.size() > 2 && (body[0] == '-' || body[0] == '*') && body[1] == ' ') {
            Elements row = inline_words(body.substr(2));
            blocks.push_back(hbox(text(std::string(indent, ' ') + " \xE2\x80\xA2 "), hflow(row) | flex));
            continue;
        }
        blocks.push_back(hflow(inline_words(line)));
    }
    if (in_code && !code_lines.empty()) blocks.push_back(vbox(code_lines) | bgcolor(Color::GrayDark) | color(Color::White));
    return vbox(blocks);
}

// ------------------------------------------------------------------ transcript

struct Block {
    TuiKind kind;
    std::string text;
};

struct Ui {
    std::mutex mu;
    std::vector<Block> blocks;
    bool generating = false;
    // A tool permission question waiting for an answer.
    bool asking = false;
    std::string question;
    int answer = 0; // 0 = none yet, 1 = yes, 2 = no, 3 = always
    std::condition_variable answered;
    ScreenInteractive *screen = nullptr;
    int frame = 0;
};

Ui *g_ui = nullptr;

Element render_block(const Block &b) {
    switch (b.kind) {
    case TuiKind::User: return hbox(text("> ") | ftxui::bold | color(Color::Green), paragraph(b.text) | flex);
    case TuiKind::Assistant: return render_markdown(b.text);
    case TuiKind::Tool: return paragraph(b.text) | dim | color(Color::Cyan);
    case TuiKind::Error: return paragraph(b.text) | color(Color::Red);
    case TuiKind::Note: return paragraph(b.text) | dim;
    }
    return text("");
}

} // namespace

void tui_emit(TuiKind kind, std::string_view text) {
    Ui *ui = g_ui;
    if (!ui) return;
    {
        std::lock_guard<std::mutex> lock(ui->mu);
        if (kind == TuiKind::Assistant) {
            if (!ui->blocks.empty() && ui->blocks.back().kind == TuiKind::Assistant) ui->blocks.back().text.append(text);
            else ui->blocks.push_back({kind, std::string(text)});
        } else {
            std::string t(text);
            while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
            if (!t.empty()) ui->blocks.push_back({kind, std::move(t)});
        }
    }
    if (ui->screen) ui->screen->PostEvent(Event::Custom);
}

bool tui_ask(std::string_view question, bool *always) {
    Ui *ui = g_ui;
    if (!ui) return false;
    std::unique_lock<std::mutex> lock(ui->mu);
    ui->asking = true;
    ui->question = std::string(question);
    ui->answer = 0;
    if (ui->screen) ui->screen->PostEvent(Event::Custom);
    ui->answered.wait(lock, [&] { return ui->answer != 0; });
    ui->asking = false;
    const int a = ui->answer;
    if (ui->screen) ui->screen->PostEvent(Event::Custom);
    if (a == 3 && always) *always = true;
    return a == 1 || a == 3;
}

std::string render_markdown_text(std::string_view markdown, int width) {
    Element e = render_markdown(markdown);
    // Tall enough for any text a test passes: a wrapping paragraph is as tall as its width needs, which Fit cannot know.
    Screen screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(60));
    Render(screen, e);
    return screen.ToString();
}

int run_tui(const TuiHost &host) {
    Ui ui;
    g_ui = &ui;
    auto screen = ScreenInteractive::Fullscreen();
    ui.screen = &screen;

    std::string input;
    float scroll = 1.0f; // 1 = follow the bottom
    std::thread worker;
    std::atomic<bool> quit{false};

    auto join_worker = [&] {
        if (worker.joinable()) worker.join();
    };

    InputOption input_option = InputOption::Default();
    input_option.multiline = false;
    input_option.cursor_position = 0;
    Component input_box = Input(&input, "Message, /help for commands", input_option);

    auto submit_line = [&](const std::string &line) {
        join_worker();
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            ui.generating = true;
        }
        scroll = 1.0f;
        worker = std::thread([&, line] {
            const bool again = host.submit(host.user, line);
            {
                std::lock_guard<std::mutex> lock(ui.mu);
                ui.generating = false;
            }
            if (!again) {
                quit = true;
                screen.PostEvent(Event::Special("agent-exit"));
            } else {
                screen.PostEvent(Event::Custom);
            }
        });
    };

    // A ticker so the spinner and the footer move while the model is busy.
    std::atomic<bool> ticking{true};
    std::thread ticker([&] {
        while (ticking) {
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            bool busy;
            {
                std::lock_guard<std::mutex> lock(ui.mu);
                busy = ui.generating;
                if (busy) ++ui.frame;
            }
            if (busy) screen.PostEvent(Event::Custom);
        }
    });

    Component root = Renderer(input_box, [&] {
        Elements rows;
        bool busy, asking;
        std::string question;
        int frame;
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            for (size_t i = 0; i < ui.blocks.size(); ++i) {
                const Block &b = ui.blocks[i];
                rows.push_back(render_block(b));
                // A blank line between blocks, except inside a run of notes or tool lines.
                const bool quiet =
                    i + 1 < ui.blocks.size() && ui.blocks[i + 1].kind == b.kind && (b.kind == TuiKind::Note || b.kind == TuiKind::Tool);
                if (!quiet) rows.push_back(text(""));
            }
            busy = ui.generating;
            asking = ui.asking;
            question = ui.question;
            frame = ui.frame;
        }
        const std::string header = host.header ? host.header(host.user) : std::string();
        const std::string footer = host.footer ? host.footer(host.user) : std::string();
        static const char *spin[] = {"|", "/", "-", "\\"};

        Element transcript = vbox(rows) | vscroll_indicator | yframe | focusPositionRelative(0, scroll) | flex;
        Elements bottom;
        if (asking) {
            bottom.push_back(hbox(text(" allow ") | ftxui::bold | color(Color::Yellow), paragraph(question) | flex,
                                  text("  [y]es / [n]o / [a]lways ") | ftxui::bold) |
                             borderStyled(ROUNDED, Color::Yellow));
        } else {
            bottom.push_back(hbox(text(busy ? std::string(spin[frame % 4]) + " " : std::string("> ")) | ftxui::bold | color(Color::Green),
                                  input_box->Render() | flex) |
                             borderStyled(ROUNDED));
        }
        Element status = hbox(text(busy ? " Esc: stop   " : " Enter: send   /help   Ctrl-D: quit   "), filler(), text(footer + " ")) | dim;
        return vbox({text(" " + header) | ftxui::bold | inverted, transcript, vbox(bottom), status});
    });

    root = CatchEvent(root, [&](Event e) {
        if (e == Event::Special("agent-exit")) {
            join_worker();
            screen.Exit();
            return true;
        }
        bool asking;
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            asking = ui.asking;
        }
        if (asking && e.is_character()) {
            const std::string c = e.character();
            int a = 0;
            if (c == "y" || c == "Y") a = 1;
            else if (c == "n" || c == "N") a = 2;
            else if (c == "a" || c == "A") a = 3;
            if (a) {
                std::lock_guard<std::mutex> lock(ui.mu);
                ui.answer = a;
                ui.answered.notify_all();
            }
            return true;
        }
        if (asking && (e == Event::Escape || e == Event::CtrlC)) {
            std::lock_guard<std::mutex> lock(ui.mu);
            ui.answer = 2;
            ui.answered.notify_all();
            return true;
        }
        bool busy;
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            busy = ui.generating;
        }
        if (e == Event::CtrlC || e == Event::Escape) {
            if (busy) {
                if (host.cancel) host.cancel(host.user);
            } else if (e == Event::CtrlC) {
                screen.Exit();
            }
            return true;
        }
        if (e == Event::CtrlD && !busy) {
            screen.Exit();
            return true;
        }
        if (e == Event::PageUp || (e.is_mouse() && e.mouse().button == Mouse::WheelUp)) {
            scroll = std::max(0.0f, std::min(1.0f, scroll) - 0.08f);
            return true;
        }
        if (e == Event::PageDown || (e.is_mouse() && e.mouse().button == Mouse::WheelDown)) {
            scroll = std::min(1.0f, scroll + 0.08f);
            return true;
        }
        if (e == Event::Return && !busy) {
            std::string line = input;
            input.clear();
            const size_t first = line.find_first_not_of(" \t");
            if (first == std::string::npos) return true;
            tui_emit(TuiKind::User, line);
            submit_line(line);
            return true;
        }
        if (e == Event::Return) return true; // an answer is running: no second message yet
        return false;
    });

    screen.Loop(root);

    ticking = false;
    if (ticker.joinable()) ticker.join();
    if (worker.joinable()) {
        if (host.cancel) host.cancel(host.user);
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            ui.answer = 2; // a pending permission question must not block the exit
            ui.answered.notify_all();
        }
        worker.join();
    }
    ui.screen = nullptr;
    g_ui = nullptr;
    return 0;
}

} // namespace agent::cli
