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

// ------------------------------------------------------------------ colors

// A cyberpunk palette: neon text on a deep violet background. Everything has an explicit color, so a terminal's own (often white)
// default foreground never shows.
const Color kBackground = Color::RGB(12, 6, 28);
const Color kText = Color::RGB(126, 246, 255);   // neon cyan: assistant text
const Color kAccent = Color::RGB(255, 46, 151);  // hot pink: brand, user bar, headings, input border
const Color kPurple = Color::RGB(191, 125, 255); // neon purple: tool names
const Color kMuted = Color::RGB(150, 118, 196);  // dusty violet: hints, tool arguments, notes
const Color kUserBackground = Color::RGB(40, 14, 62);
const Color kUserText = Color::RGB(255, 178, 224); // soft pink
const Color kOk = Color::RGB(57, 255, 20);         // neon green
const Color kWarn = Color::RGB(255, 230, 0);       // neon yellow
const Color kBad = Color::RGB(255, 64, 96);        // neon red
const Color kCodeBackground = Color::RGB(28, 12, 52);
const Color kCodeText = Color::RGB(57, 255, 20);

// ------------------------------------------------------------------ markdown-lite

// Inline spans of one line: **bold**, `code`, plain. Words keep their trailing space so hflow wraps them.
Elements inline_words(std::string_view line) {
    Elements words;
    bool bold = false, code = false;
    std::string word;
    auto flush = [&] {
        if (word.empty()) return;
        Element e = text(word);
        if (code) e = e | color(kOk);
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

Element code_block(const Elements &lines) { return vbox(lines) | bgcolor(kCodeBackground) | color(kCodeText); }

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
                blocks.push_back(code_block(code_lines));
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
            blocks.push_back(hflow(inline_words(line.substr(level + 1))) | ftxui::bold | color(kAccent));
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
    if (in_code && !code_lines.empty()) blocks.push_back(code_block(code_lines));
    return vbox(blocks);
}

// ------------------------------------------------------------------ small text helpers

size_t glyph_count(std::string_view s) {
    size_t n = 0;
    for (const char c : s)
        if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++n;
    return n;
}

// Cuts to at most `columns` glyphs, adding an ellipsis when something was cut.
std::string clip(std::string_view s, int columns) {
    if (columns <= 0) return {};
    if (glyph_count(s) <= static_cast<size_t>(columns)) return std::string(s);
    size_t glyphs = 0, i = 0;
    while (i < s.size()) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
            if (glyphs == static_cast<size_t>(columns) - 1) break;
            ++glyphs;
        }
        ++i;
    }
    return std::string(s.substr(0, i)) + "\xE2\x80\xA6";
}

std::string trim_both(std::string_view s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string_view::npos) return {};
    const size_t b = s.find_last_not_of(" \t\r\n");
    return std::string(s.substr(a, b - a + 1));
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
    std::chrono::steady_clock::time_point busy_since{};
    // A tool permission question waiting for an answer.
    bool asking = false;
    std::string question;
    // A free-text question (tui_prompt) waiting for the next line the user enters.
    bool prompting = false;
    bool prompt_done = false;
    std::string prompt_text, prompt_default, prompt_answer;
    std::condition_variable prompt_cv;
    int answer = 0; // 0 = none yet, 1 = yes, 2 = no, 3 = always
    std::condition_variable answered;
    ScreenInteractive *screen = nullptr;
    int frame = 0;
};

Ui *g_ui = nullptr;

Element render_tool(const std::string &t) {
    constexpr std::string_view call = "[tool]";
    constexpr std::string_view result = "[result]";
    if (t.compare(0, call.size(), call) == 0) {
        const std::string rest = trim_both(std::string_view(t).substr(call.size()));
        const size_t space = rest.find(' ');
        const std::string name = rest.substr(0, space);
        const std::string args = space == std::string::npos ? std::string() : trim_both(std::string_view(rest).substr(space + 1));
        return hbox(text("  \xE2\x9A\x99 ") | color(kAccent), text(name) | ftxui::bold | color(kPurple), text("  "),
                    text(args) | color(kMuted));
    }
    if (t.compare(0, result.size(), result) == 0) {
        const std::string body = summarize_tool_result(trim_both(std::string_view(t).substr(result.size())), 4, 200);
        const bool failed = body.compare(0, 5, "Error") == 0 ||
                            (body.find("\"exit_code\":0") == std::string::npos && body.find("\"exit_code\":") != std::string::npos);
        Elements lines;
        size_t at = 0;
        bool first = true;
        while (at <= body.size()) {
            size_t end = body.find('\n', at);
            if (end == std::string::npos) end = body.size();
            const std::string line = body.substr(at, end - at);
            lines.push_back(hbox(text(first ? "    \xE2\x94\x94 " : "      ") | color(kMuted), text(line) | color(failed ? kBad : kMuted)));
            first = false;
            at = end + 1;
        }
        return vbox(lines);
    }
    return paragraph(t) | color(kMuted);
}

Element render_block(const Block &b) {
    switch (b.kind) {
    case TuiKind::User:
        return hbox(text("\xE2\x96\x8E") | color(kAccent), text(" "), paragraph(b.text) | color(kUserText) | flex) |
               bgcolor(kUserBackground);
    case TuiKind::Assistant: return hbox(text("  "), render_markdown(b.text) | color(kText) | flex);
    case TuiKind::Tool: return render_tool(b.text);
    case TuiKind::Error: return hbox(text("\xE2\x96\x8E") | color(kBad), text(" "), paragraph(b.text) | color(kBad) | flex);
    case TuiKind::Note: return hbox(text("  "), paragraph(b.text) | flex) | color(kMuted);
    }
    return text("");
}

const char *const kSpinner[] = {"\xE2\xA0\x8B", "\xE2\xA0\x99", "\xE2\xA0\xB9", "\xE2\xA0\xB8", "\xE2\xA0\xBC",
                                "\xE2\xA0\xB4", "\xE2\xA0\xA6", "\xE2\xA0\xA7", "\xE2\xA0\x87", "\xE2\xA0\x8F"};

Element key(const std::string &k) { return text(" " + k + " ") | ftxui::bold | color(kAccent) | bgcolor(kUserBackground); }

Element context_gauge(int percent) {
    constexpr int cells = 10;
    const int filled = std::clamp((percent * cells + 50) / 100, 0, cells);
    std::string bar;
    for (int i = 0; i < cells; ++i) bar += i < filled ? "\xE2\x96\xB0" : "\xE2\x96\xB1";
    const Color c = percent < 60 ? kOk : percent < 85 ? kWarn : kBad;
    return hbox(text("ctx ") | color(kMuted), text(bar) | color(c), text(" " + std::to_string(percent) + "% ") | color(kMuted));
}

} // namespace

// ------------------------------------------------------------------ helpers with no screen (tests)

std::string summarize_tool_result(std::string_view result, int max_lines, int max_width) {
    std::vector<std::string> lines;
    size_t at = 0;
    while (at <= result.size()) {
        size_t end = result.find('\n', at);
        if (end == std::string_view::npos) end = result.size();
        std::string_view line = result.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        lines.emplace_back(line);
        at = end + 1;
    }
    while (!lines.empty() && lines.back().empty()) lines.pop_back();
    std::string out;
    const int shown = std::min<int>(max_lines, static_cast<int>(lines.size()));
    for (int i = 0; i < shown; ++i) {
        if (i) out += '\n';
        out += clip(lines[static_cast<size_t>(i)], max_width);
    }
    if (static_cast<int>(lines.size()) > shown)
        out += "\n\xE2\x80\xA6 " + std::to_string(lines.size() - static_cast<size_t>(shown)) + " more lines";
    return out;
}

int context_percent(std::string_view footer) {
    const size_t at = footer.find("KV ");
    if (at == std::string_view::npos) return -1;
    size_t i = at + 3;
    long used = 0, total = 0;
    bool any = false;
    while (i < footer.size() && footer[i] >= '0' && footer[i] <= '9') used = used * 10 + (footer[i++] - '0'), any = true;
    if (!any || i >= footer.size() || footer[i] != '/') return -1;
    ++i;
    any = false;
    while (i < footer.size() && footer[i] >= '0' && footer[i] <= '9') total = total * 10 + (footer[i++] - '0'), any = true;
    if (!any || total <= 0) return -1;
    return static_cast<int>(std::min<long>(100, used * 100 / total));
}

std::vector<size_t> filter_commands(std::span<const TuiCommand> commands, std::string_view typed) {
    std::vector<size_t> matches;
    for (size_t i = 0; i < commands.size(); ++i)
        if (std::string_view(commands[i].name).substr(0, typed.size()) == typed) matches.push_back(i);
    return matches;
}

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

bool tui_prompt(std::string_view question, std::string_view default_value, std::string &answer) {
    Ui *ui = g_ui;
    if (!ui) return false;
    std::unique_lock<std::mutex> lock(ui->mu);
    ui->prompting = true;
    ui->prompt_done = false;
    ui->prompt_text = std::string(question);
    ui->prompt_default = std::string(default_value);
    if (ui->screen) ui->screen->PostEvent(Event::Custom);
    ui->prompt_cv.wait(lock, [&] { return ui->prompt_done; });
    ui->prompting = false;
    answer = ui->prompt_answer;
    if (ui->screen) ui->screen->PostEvent(Event::Custom);
    return true;
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
    int cursor = 0;
    float scroll = 1.0f; // 1 = follow the bottom
    size_t palette_selected = 0;
    std::vector<std::string> history;
    size_t history_at = 0; // == history.size() while typing a new line
    std::string draft;
    std::thread worker;
    std::atomic<bool> quit{false};

    auto join_worker = [&] {
        if (worker.joinable()) worker.join();
    };
    auto set_input = [&](const std::string &s) {
        input = s;
        cursor = static_cast<int>(glyph_count(s));
    };

    InputOption input_option = InputOption::Default();
    input_option.multiline = false;
    input_option.cursor_position = &cursor;
    Component input_box = Input(&input, "Message, or / for commands", input_option);

    auto submit_line = [&](const std::string &line) {
        join_worker();
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            ui.generating = true;
            ui.busy_since = std::chrono::steady_clock::now();
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

    // The palette is open while the user is typing a command name: the input starts with "/" and has no space yet.
    auto palette_matches = [&]() -> std::vector<size_t> {
        if (host.commands.empty() || input.empty() || input[0] != '/' || input.find(' ') != std::string::npos) return {};
        return filter_commands(host.commands, input);
    };

    // A ticker so the spinner and the footer move while the model is busy.
    std::atomic<bool> ticking{true};
    std::thread ticker([&] {
        while (ticking) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
        bool busy, asking, prompting, empty;
        std::string question, prompt_line, last_text;
        TuiKind last_kind = TuiKind::Note;
        int frame, seconds = 0;
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            for (size_t i = 0; i < ui.blocks.size(); ++i) {
                const Block &b = ui.blocks[i];
                rows.push_back(render_block(b));
                // A blank line between blocks, except inside a run of notes or tool lines.
                const bool quiet = i + 1 < ui.blocks.size() && ((ui.blocks[i + 1].kind == b.kind && b.kind == TuiKind::Note) ||
                                                                (b.kind == TuiKind::Tool && ui.blocks[i + 1].kind == TuiKind::Tool));
                if (!quiet) rows.push_back(text(""));
            }
            empty = ui.blocks.empty();
            if (!ui.blocks.empty()) {
                last_kind = ui.blocks.back().kind;
                last_text = ui.blocks.back().text.substr(0, 8);
            }
            busy = ui.generating;
            asking = ui.asking;
            question = ui.question;
            prompting = ui.prompting;
            prompt_line = ui.prompt_text + (ui.prompt_default.empty() ? "" : " [" + ui.prompt_default + "]");
            frame = ui.frame;
            if (busy)
                seconds = static_cast<int>(
                    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - ui.busy_since).count());
        }
        const std::string header = host.header ? host.header(host.user) : std::string();
        const std::string subtitle = host.subtitle ? host.subtitle(host.user) : std::string();
        const std::string footer = host.footer ? host.footer(host.user) : std::string();

        // What the model is doing, shown at the end of the transcript.
        if (busy && !asking) {
            std::string label = "thinking";
            if (last_kind == TuiKind::Assistant) label = "writing";
            else if (last_kind == TuiKind::Tool && last_text.compare(0, 6, "[tool]") == 0) label = "running a tool";
            rows.push_back(text("  " + std::string(kSpinner[frame % 10]) + " " + label + "  " + std::to_string(seconds) + "s") |
                           color(kAccent));
        }

        Element transcript;
        if (empty && !busy) {
            transcript =
                vbox({filler(), hcenter(text("agent.cpp") | ftxui::bold | color(kAccent)),
                      hcenter(text("local agents, on your machine") | color(kPurple)), text(""), hcenter(text(header) | color(kMuted)),
                      subtitle.empty() ? text("") : hcenter(text(subtitle) | color(kMuted)), text(""),
                      hcenter(hbox(key("/"), text(" commands   "), key("Esc"), text(" stop an answer   "), key("PgUp/PgDn"),
                                   text(" scroll   "), key("Ctrl-D"), text(" quit")) |
                              color(kMuted)),
                      filler()}) |
                flex;
        } else {
            // yframe scrolls so that the focused element is visible, so the focus belongs inside the frame: on a marker row at the end
            // while following the bottom, on a relative position while the user has scrolled up.
            if (scroll >= 1.0f) {
                rows.push_back(text("") | focus);
                transcript = vbox(rows) | vscroll_indicator | yframe | flex;
            } else {
                transcript = vbox(rows) | focusPositionRelative(0, scroll) | vscroll_indicator | yframe | flex;
            }
        }

        Elements bottom;
        if (asking) {
            bottom.push_back(
                vbox({hbox(text(" Permission needed") | ftxui::bold | color(kWarn)), hbox(text(" "), paragraph(question) | flex),
                      hbox(text(" "), key("y"), text(" allow once   "), key("a"), text(" always allow   "), key("n"), text(" deny"))}) |
                borderStyled(ROUNDED, kWarn));
        } else {
            // The command palette: names that match what has been typed, the highlighted one completes with Tab.
            const std::vector<size_t> matches = palette_matches();
            if (!matches.empty() && !busy && !prompting) {
                Elements items;
                const size_t shown = std::min<size_t>(matches.size(), 8);
                for (size_t i = 0; i < shown; ++i) {
                    const TuiCommand &c = host.commands[matches[i]];
                    std::string name = c.name;
                    name.resize(std::max<size_t>(name.size() + 2, 12), ' ');
                    Element row = hbox(text(" " + name) | ftxui::bold | color(kAccent), text(c.help) | color(kMuted), text(" "));
                    if (i == palette_selected % matches.size()) row = row | bgcolor(kUserBackground);
                    items.push_back(row);
                }
                bottom.push_back(vbox(items) | borderStyled(ROUNDED, kPurple));
            }
            if (prompting) bottom.push_back(paragraph(" ? " + prompt_line) | ftxui::bold | color(kWarn));
            bottom.push_back(hbox(text(busy ? " \xE2\x80\xA6 " : " \xE2\x80\xBA ") | ftxui::bold | color(busy ? kWarn : kAccent),
                                  input_box->Render() | color(kUserText) | flex) |
                             borderStyled(ROUNDED, busy ? kMuted : kAccent));
        }

        // Left: key hints (or the speeds of the last answer); right: how full the context window is.
        const int percent = context_percent(footer);
        Elements status;
        if (scroll < 1.0f) status.push_back(text(" scrolled up \xC2\xB7 End or PgDn to follow the newest output") | color(kWarn));
        else if (busy) status.push_back(text(" Esc to stop") | color(kMuted));
        else if (footer.empty() || prompting) status.push_back(text(" / commands   PgUp/PgDn scroll   Ctrl-D quit") | color(kMuted));
        else {
            // The gauge already shows the KV cache, so the footer keeps only the speeds.
            const size_t kv = percent >= 0 ? footer.find(" | KV ") : std::string::npos;
            status.push_back(text(" " + (kv == std::string::npos ? footer : footer.substr(0, kv))) | color(kMuted));
        }
        status.push_back(filler());
        if (percent >= 0) status.push_back(context_gauge(percent));

        const Element title =
            hbox(text(" agent.cpp ") | ftxui::bold | bgcolor(kAccent) | color(kBackground), text("  " + header) | color(kMuted), filler());
        return vbox({title, transcript, vbox(bottom), hbox(status)}) | color(kText) | bgcolor(kBackground);
    });

    root = CatchEvent(root, [&](Event e) {
        if (e == Event::Special("agent-exit")) {
            join_worker();
            screen.Exit();
            return true;
        }
        bool asking, busy;
        {
            std::lock_guard<std::mutex> lock(ui.mu);
            asking = ui.asking;
            busy = ui.generating;
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
        if (e == Event::CtrlC || e == Event::Escape) {
            if (busy) {
                if (host.cancel) host.cancel(host.user);
            } else if (e == Event::CtrlC) {
                screen.Exit();
            } else if (!input.empty()) {
                set_input("");
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
        // Without a mouse: Home and End jump to the top and the bottom while the input line is empty, and the arrow keys scroll the
        // transcript while an answer is running or before there is any history to recall (some terminals turn the wheel into arrows).
        if (input.empty() && e == Event::Home) {
            scroll = 0.0f;
            return true;
        }
        if (input.empty() && e == Event::End) {
            scroll = 1.0f;
            return true;
        }
        if ((busy || (input.empty() && history.empty())) && e == Event::ArrowUp) {
            scroll = std::max(0.0f, std::min(1.0f, scroll) - 0.04f);
            return true;
        }
        if ((busy || (input.empty() && history.empty())) && e == Event::ArrowDown) {
            scroll = std::min(1.0f, scroll + 0.04f);
            return true;
        }

        // The command palette: arrows move, Tab completes.
        const std::vector<size_t> matches = busy ? std::vector<size_t>() : palette_matches();
        if (!matches.empty()) {
            if (e == Event::ArrowDown) {
                palette_selected = (palette_selected + 1) % matches.size();
                return true;
            }
            if (e == Event::ArrowUp) {
                palette_selected = (palette_selected + matches.size() - 1) % matches.size();
                return true;
            }
            const std::string chosen = host.commands[matches[palette_selected % matches.size()]].name;
            if (e == Event::Tab || (e == Event::Return && input != chosen)) {
                set_input(chosen + " ");
                palette_selected = 0;
                return true;
            }
        } else if (!busy && (e == Event::ArrowUp || e == Event::ArrowDown) && !history.empty()) {
            // Input history.
            if (e == Event::ArrowUp && history_at > 0) {
                if (history_at == history.size()) draft = input;
                set_input(history[--history_at]);
            } else if (e == Event::ArrowDown && history_at < history.size()) {
                ++history_at;
                set_input(history_at == history.size() ? draft : history[history_at]);
            }
            return true;
        }
        if (e.is_character() || e == Event::Backspace) palette_selected = 0;

        if (e == Event::Return) {
            std::lock_guard<std::mutex> lock(ui.mu);
            if (ui.prompting && !ui.prompt_done) {
                // The line typed is the answer to the question the worker is waiting on; an empty line takes the default.
                ui.prompt_answer = input.empty() ? ui.prompt_default : input;
                ui.blocks.push_back({TuiKind::User, ui.prompt_answer.empty() ? std::string("(default)") : ui.prompt_answer});
                set_input("");
                ui.prompt_done = true;
                ui.prompt_cv.notify_all();
                return true;
            }
        }
        if (e == Event::Return && !busy) {
            std::string line = input;
            set_input("");
            const size_t first = line.find_first_not_of(" \t");
            if (first == std::string::npos) return true;
            if (history.empty() || history.back() != line) history.push_back(line);
            history_at = history.size();
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
            ui.prompt_answer.clear(); // nor a free-text question
            ui.prompt_done = true;
            ui.prompt_cv.notify_all();
        }
        worker.join();
    }
    ui.screen = nullptr;
    g_ui = nullptr;
    return 0;
}

} // namespace agent::cli
