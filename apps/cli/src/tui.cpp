// The full-screen chat interface, built on RTXUI. See tui.hpp.
//
// The screen is an HTML template with CSS (below); the state it shows lives in plain members of ChatApp. Everything runs on the UI
// thread: the worker that talks to the model hands its output over with PostTask().

#include "tui.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <ctime>
#include <memory>
#include <mutex>
#include <rtxui/headless.hpp>
#include <rtxui/rtxui.hpp>
#include <string>
#include <thread>
#include <vector>

namespace agent::cli {

// ------------------------------------------------------------------ helpers with no screen

namespace {

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

} // namespace

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

// ------------------------------------------------------------------ the screen

namespace {

using rtxui::Event;

// Four colors: black (the background), white (text), neon cyan (tool calls, hints, code, rules) and hot pink (the rarest: the brand,
// prompts, the spinner, errors).
constexpr std::string_view kMarkdownCss = R"css(
  h1, h2, h3, h4, h5, h6, p, ul, ol, pre, blockquote, table, hr { margin-top: 0; }
  h1 { color: #ff147d; font-weight: bold; border-bottom: double; border-color: #00e5ff; margin-bottom: 1; }
  h2 { color: #ffffff; font-weight: bold; border-bottom: solid; border-color: #00e5ff; margin-bottom: 1; }
  h3, h4, h5, h6 { color: #00e5ff; font-weight: bold; }
  p { color: #ffffff; margin-bottom: 1; }
  strong { color: #ffffff; font-weight: bold; }
  em { color: #ffffff; }
  a { color: #00e5ff; text-decoration: underline; }
  code { color: #00e5ff; }
  pre { color: #00e5ff; scrollbar-width: none; border: round; border-color: #00e5ff; padding-left: 1; padding-right: 1; margin-bottom: 1; }
  blockquote { color: #ffffff; border-left: solid; border-color: #00e5ff; padding-left: 1; margin-bottom: 1; }
  blockquote p { margin-bottom: 0; }
  ul, ol { padding-left: 2; margin-bottom: 1; }
  li { color: #ffffff; }
  li p { margin-bottom: 0; }
  table { border: round; border-color: #00e5ff; margin-bottom: 1; }
  th { color: #ff147d; font-weight: bold; border-bottom: solid; border-color: #00e5ff; padding-left: 1; padding-right: 1; }
  td { color: #ffffff; padding-left: 1; padding-right: 1; }
  hr { color: #00e5ff; margin-bottom: 1; }
)css";

// One block of the transcript, as the template reads it.
struct Row {
    TuiKind kind = TuiKind::Note;
    std::string mark; // the symbol in the left gutter
    std::string text;
    std::string cls;  // CSS classes
    int markdown = 0; // 1: render `text` as markdown
    bool operator==(const Row &) const = default;
};

void Fields(const Row &row, rtxui::FieldSink &out) {
    out("mark", row.mark);
    out("text", row.text);
    out("cls", row.cls);
    out("md", row.markdown);
    out("plain", row.markdown ? 0 : 1);
}

struct Line {
    std::string text, cls;
    bool operator==(const Line &) const = default;
};

void Fields(const Line &line, rtxui::FieldSink &out) {
    out("text", line.text);
    out("cls", line.cls);
}

struct Match {
    std::string name, help, cls;
    bool operator==(const Match &) const = default;
};

void Fields(const Match &match, rtxui::FieldSink &out) {
    out("name", match.name);
    out("help", match.help);
    out("cls", match.cls);
}

// What the worker thread waits on: the answer to a permission question or a free-text question.
struct Sync {
    std::mutex mu;
    std::condition_variable cv;
    int answer = 0; // 0 = none yet, 1 = yes, 2 = no, 3 = always
    bool prompt_done = false;
    std::string prompt_answer;
};

class ChatApp;
ChatApp *g_app = nullptr;
Sync *g_sync = nullptr;
rtxui::Screen *g_screen = nullptr;

const char *const kTips[] = {"Tip: type / or press Ctrl+P for commands",   "Tip: Esc stops an answer while it is running",
                             "Tip: select text with the mouse to copy it", "Tip: /image PATH sends a picture with your message",
                             "Tip: End jumps back to the newest line",     "Tip: Up and Down recall what you typed before"};

const char *const kSpinner[] = {"\xE2\xA0\x8B", "\xE2\xA0\x99", "\xE2\xA0\xB9", "\xE2\xA0\xB8", "\xE2\xA0\xBC",
                                "\xE2\xA0\xB4", "\xE2\xA0\xA6", "\xE2\xA0\xA7", "\xE2\xA0\x87", "\xE2\xA0\x8F"};

// The big "agent.cpp" on the empty screen: block letters with shadows, 29 columns wide (a narrow terminal shows the plain name).
const char *const kBanner[3] = {
    "\xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90\xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90\xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90\xE2\x94\x8C\xE2\x94\x90"
    "\xE2\x94\x8C\xE2\x94\x8C\xE2\x94\xAC\xE2\x94\x90   "
    "\xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90\xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90\xE2\x94\x8C\xE2\x94\x80\xE2\x94\x90",
    "\xE2\x94\x9C\xE2\x94\x80\xE2\x94\xA4\xE2\x94\x82 \xE2\x94\xAC\xE2\x94\x9C\xE2\x94\xA4 \xE2\x94\x82\xE2\x94\x82\xE2\x94\x82 "
    "\xE2\x94\x82    \xE2\x94\x82  \xE2\x94\x9C\xE2\x94\x80\xE2\x94\x98\xE2\x94\x9C\xE2\x94\x80\xE2\x94\x98",
    "\xE2\x94\xB4 \xE2\x94\xB4\xE2\x94\x94\xE2\x94\x80\xE2\x94\x98\xE2\x94\x94\xE2\x94\x80\xE2\x94\x98\xE2\x94\x98\xE2\x94\x94\xE2\x94\x98 "
    "\xE2\x94\xB4  \x2E \xE2\x94\x94\xE2\x94\x80\xE2\x94\x98\xE2\x94\xB4  \xE2\x94\xB4  "};

class ChatApp : public rtxui::Component<ChatApp> {
  public:
    explicit ChatApp(const TuiHost &host_) : host(host_) {
        for (size_t i = 0; i < 3; ++i) banner.push_back({kBanner[i], i < 2 ? "banner pink" : "banner neon"});
        tip = kTips[static_cast<size_t>(std::time(nullptr) / 60) % (sizeof kTips / sizeof *kTips)];
        Bind(rows);
        Bind(banner);
        Bind(matches);
        Bind(input);
        Bind(header);
        Bind(subtitle);
        Bind(question);
        Bind(prompt_line);
        Bind(busy_line);
        Bind(status_left);
        Bind(gauge);
        Bind(gauge_class);
        Bind(caret);
        Bind(md_css);
        Bind(asking);
        Bind(prompting);
        Bind(busy);
        Bind(show_welcome);
        Bind(app_class);
        Bind(tip);
        Bind(show_palette);
        Refresh();
    }

    TuiHost host;
    std::vector<Row> rows;
    std::vector<Line> banner;
    std::vector<Match> matches;
    std::string input, header, subtitle, question, prompt_line, busy_line, status_left, gauge, gauge_class = "gauge",
                                                                                               caret = "\xE2\x80\xBA";
    std::string md_css = std::string(kMarkdownCss);
    bool asking = false, prompting = false, busy = false;
    bool show_welcome = true, show_palette = false;
    std::string app_class = "app home", tip;
    bool follow = true; // the transcript stays at the newest line
    std::string prompt_default;
    size_t selected = 0;
    std::vector<std::string> history;
    size_t history_at = 0; // == history.size() while typing a new line
    std::string draft;
    std::thread worker;
    int frame = 0;
    std::chrono::steady_clock::time_point busy_since{};
    bool ticking = false;

    // ---- the transcript

    void Emit(TuiKind kind, const std::string &text) {
        if (kind == TuiKind::Assistant) {
            if (!rows.empty() && rows.back().kind == TuiKind::Assistant) rows.back().text += text;
            else rows.push_back({kind, "", text, "row assistant", 1});
        } else {
            std::string t = text;
            while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
            if (t.empty()) return;
            if (kind == TuiKind::User) {
                rows.push_back({kind, "\xE2\x80\xBA", t, rows.empty() ? "row user" : "row user turn", 0});
            } else if (kind == TuiKind::Error) {
                rows.push_back({kind, "!", t, "row error", 0});
            } else if (kind == TuiKind::Tool) {
                AddTool(t);
            } else {
                rows.push_back({kind, "", t, "row note", 0});
            }
        }
        show_welcome = false;
        app_class = "app";
        Settle();
    }

    void AddTool(const std::string &t) {
        constexpr std::string_view call = "[tool]";
        constexpr std::string_view result = "[result]";
        if (t.compare(0, call.size(), call) == 0) {
            const std::string rest = trim_both(std::string_view(t).substr(call.size()));
            const size_t space = rest.find(' ');
            const std::string name = rest.substr(0, space);
            const std::string args = space == std::string::npos ? std::string() : trim_both(std::string_view(rest).substr(space + 1));
            rows.push_back({TuiKind::Tool, "\xE2\x86\x92", name + "  " + args, "row tool", 0});
        } else if (t.compare(0, result.size(), result) == 0) {
            const std::string body = summarize_tool_result(trim_both(std::string_view(t).substr(result.size())), 4, 200);
            const bool failed = body.compare(0, 5, "Error") == 0 ||
                                (body.find("\"exit_code\":0") == std::string::npos && body.find("\"exit_code\":") != std::string::npos);
            rows.push_back({TuiKind::Tool, "", body, failed ? "row result failed" : "row result", 0});
        } else {
            rows.push_back({TuiKind::Tool, "", t, "row note", 0});
        }
    }

    // After any change: keep the newest output in view.
    void Settle() {
        if (follow) ScrollToEnd();
    }
    void ScrollToEnd() { QueryElement("#t").SetScrollY(INT_MAX / 4); }

    // ---- the header, the status line and the context gauge

    void Refresh() {
        header = host.header ? host.header(host.user) : std::string();
        subtitle = host.subtitle ? host.subtitle(host.user) : std::string();
        const std::string footer = host.footer ? host.footer(host.user) : std::string();
        const int percent = context_percent(footer);
        gauge = percent >= 0 ? "ctx " + std::to_string(percent) + "%" : std::string();
        gauge_class = percent >= 85 ? "gauge hot" : "gauge";
        if (!follow) status_left = "scrolled up \xC2\xB7 End follows the newest output";
        else if (busy) status_left = "Esc to stop";
        else if (footer.empty() || prompting) status_left = "/ commands   PgUp/PgDn scroll   Ctrl-D quit";
        else {
            // The gauge already shows the KV cache, so the footer keeps only the speeds.
            const size_t kv = percent >= 0 ? footer.find(" | KV ") : std::string::npos;
            status_left = kv == std::string::npos ? footer : footer.substr(0, kv);
        }
        app_class = show_welcome ? "app home" : "app";
        caret = busy ? "\xC2\xB7" : "\xE2\x80\xBA";
        UpdateBusyLine();
    }

    void UpdateBusyLine() {
        if (!busy || asking) {
            busy_line.clear();
            return;
        }
        std::string label = "thinking";
        if (!rows.empty()) {
            const Row &last = rows.back();
            if (last.kind == TuiKind::Assistant) label = "writing";
            else if (last.kind == TuiKind::Tool && last.cls == "row tool") label = "running a tool";
        }
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - busy_since).count();
        busy_line = std::string(kSpinner[frame % 10]) + " " + label + "  " + std::to_string(seconds) + "s";
    }

    // The spinner and the footer move while the model is busy.
    void Tick() {
        if (!busy) {
            ticking = false;
            return;
        }
        ++frame;
        Refresh();
        Settle();
        rtxui::PostDelayedTask([this] { Tick(); }, std::chrono::milliseconds(100));
    }

    // ---- sending a line to the host

    void SetInput(const std::string &s) { input = s; }

    void Submit(const std::string &line) {
        if (worker.joinable()) worker.join();
        busy = true;
        busy_since = std::chrono::steady_clock::now();
        follow = true;
        if (!ticking) {
            ticking = true;
            Tick();
        }
        worker = std::thread([this, line] {
            const bool again = host.submit(host.user, line);
            rtxui::PostTask([this, again] {
                busy = false;
                Refresh();
                Settle();
                if (!again && g_screen) g_screen->Exit();
            });
        });
        Refresh();
    }

    // The palette is open while the user is typing a command name: the input starts with "/" and has no space yet.
    std::vector<size_t> PaletteMatches() const {
        if (host.commands.empty() || input.empty() || input[0] != '/' || input.find(' ') != std::string::npos) return {};
        return filter_commands(host.commands, input);
    }

    bool Digest() override {
        const std::vector<size_t> found = (busy || prompting) ? std::vector<size_t>() : PaletteMatches();
        matches.clear();
        if (!found.empty()) selected %= found.size();
        for (size_t i = 0; i < found.size() && i < 8; ++i) {
            const TuiCommand &c = host.commands[found[i]];
            matches.push_back({c.name, c.help, i == selected ? "pitem current" : "pitem"});
        }
        show_palette = !matches.empty();
        return Component<ChatApp>::Digest();
    }

    // ---- keys

    static int Letter(const Event &e) {
        if (auto key = e.get_if<Event::Keyboard>())
            if (!key->modifier().ctrl() && !key->modifier().alt()) return static_cast<int>(key->codepoint());
        return 0;
    }

    bool OnEvent(Event e) override {
        if (asking) {
            const int c = Letter(e);
            int a = 0;
            if (c == 'y' || c == 'Y') a = 1;
            else if (c == 'n' || c == 'N') a = 2;
            else if (c == 'a' || c == 'A') a = 3;
            if (e == Event::Escape() || e == Event::CtrlC()) a = 2;
            if (a && g_sync) {
                std::lock_guard<std::mutex> lock(g_sync->mu);
                g_sync->answer = a;
                g_sync->cv.notify_all();
            }
            return c != 0 || a != 0;
        }
        if (e == Event::CtrlC() || e == Event::Escape()) {
            if (busy) {
                if (host.cancel) host.cancel(host.user);
            } else if (e == Event::CtrlC()) {
                if (g_screen) g_screen->Exit();
            } else if (!input.empty()) {
                SetInput("");
            }
            return true;
        }
        if (e == Event::CtrlP() && !busy) {
            SetInput("/");
            return true;
        }
        if (e == Event::CtrlD() && !busy) {
            if (g_screen) g_screen->Exit();
            return true;
        }
        if (e == Event::PageUp()) return Scroll(-12);
        if (e == Event::PageDown()) return Scroll(12);
        if (auto mouse = e.get_if<Event::Mouse>()) {
            if (mouse->button() == Event::Mouse::WheelUp) {
                follow = false;
                Refresh();
            }
            return Component<ChatApp>::OnEvent(e); // the transcript scrolls itself
        }
        if (input.empty() && e == Event::Home()) {
            follow = false;
            QueryElement("#t").SetScrollY(0);
            Refresh();
            return true;
        }
        if (input.empty() && e == Event::End()) {
            follow = true;
            ScrollToEnd();
            Refresh();
            return true;
        }
        if ((busy || (input.empty() && history.empty())) && e == Event::ArrowUp()) return Scroll(-2);
        if ((busy || (input.empty() && history.empty())) && e == Event::ArrowDown()) return Scroll(2);

        // The command palette: arrows move, Tab completes.
        const std::vector<size_t> found = busy ? std::vector<size_t>() : PaletteMatches();
        if (!found.empty()) {
            if (e == Event::ArrowDown()) {
                selected = (selected + 1) % found.size();
                return true;
            }
            if (e == Event::ArrowUp()) {
                selected = (selected + found.size() - 1) % found.size();
                return true;
            }
            const std::string chosen = host.commands[found[selected % found.size()]].name;
            if (e == Event::Tab() || (e == Event::Return() && input != chosen)) {
                SetInput(chosen + " ");
                selected = 0;
                return true;
            }
        } else if (!busy && (e == Event::ArrowUp() || e == Event::ArrowDown()) && !history.empty()) {
            if (e == Event::ArrowUp() && history_at > 0) {
                if (history_at == history.size()) draft = input;
                SetInput(history[--history_at]);
            } else if (e == Event::ArrowDown() && history_at < history.size()) {
                ++history_at;
                SetInput(history_at == history.size() ? draft : history[history_at]);
            }
            return true;
        }
        if (Letter(e) != 0 || e == Event::Backspace()) selected = 0;

        if (e == Event::Return()) {
            if (prompting && g_sync) {
                // The line typed is the answer to the question the worker is waiting on; an empty line takes the default.
                const std::string answer = input.empty() ? prompt_default : input;
                rows.push_back({TuiKind::User, "\xE2\x80\xBA", answer.empty() ? std::string("(default)") : answer, "row user turn", 0});
                SetInput("");
                {
                    std::lock_guard<std::mutex> lock(g_sync->mu);
                    g_sync->prompt_answer = answer;
                    g_sync->prompt_done = true;
                    g_sync->cv.notify_all();
                }
                Settle();
                return true;
            }
            if (!busy) {
                const std::string line = input;
                SetInput("");
                if (line.find_first_not_of(" \t") == std::string::npos) return true;
                if (history.empty() || history.back() != line) history.push_back(line);
                history_at = history.size();
                Emit(TuiKind::User, line);
                Submit(line);
            }
            return true; // an answer is running: no second message yet
        }
        return Component<ChatApp>::OnEvent(e); // the rest goes to the input
    }

    bool Scroll(int rows_up) {
        auto t = QueryElement("#t");
        if (rows_up < 0) follow = false;
        t.SetScrollY(t.scroll_y() + rows_up, true);
        Refresh();
        return true;
    }

    std::string_view view = R"html(
    <div class="{app_class}">
      <div class="title"><span class="brand">agent.cpp</span><span class="header">{header}</span></div>
      <div class="transcript" id="t">
        <div if="{show_welcome}" class="welcome">
          <for each="{banner}" as="ln"><div class="{ln.cls}">{ln.text}</div></for>
          <div class="tag">local agents · tools · memory · voice · graphs</div>
        </div>
        <for each="{rows}" as="r">
          <div class="{r.cls}">
            <span class="mark">{r.mark}</span>
            <div class="body">
              <markdown if="{r.md}" content="{r.text}" stylesheet="{md_css}"/>
              <span if="{r.plain}">{r.text}</span>
            </div>
          </div>
        </for>
        <div class="busy">{busy_line}</div>
      </div>
      <div class="composer">
        <div if="{asking}" class="ask"><span class="allow">allow</span><span class="q">{question}</span><span class="keys">y once · a always · n deny</span></div>
        <div if="{show_palette}" class="palette">
          <for each="{matches}" as="m"><div class="{m.cls}"><span class="pname">{m.name}</span><span class="phelp">{m.help}</span></div></for>
        </div>
        <div if="{prompting}" class="promptq">? {prompt_line}</div>
        <div class="box"><span class="caret">{caret}</span><input value="{input}" autofocus=""/></div>
        <div class="meta"><span class="model">{header}</span></div>
        <div class="tip" if="{show_welcome}">{tip}</div>
      </div>
      <div class="status"><span class="left">{status_left}</span><span class="{gauge_class}">{gauge}</span></div>
    </div>

    <style>
      .app { display: flex; flex-direction: column; width: 100%; height: 100%; background-color: #000000; color: #ffffff; }
      .home { justify-content: center; }
      .title { border-bottom: solid; border-color: #00e5ff; padding-left: 1; }
      .home .title { display: none; }
      .brand { color: #ff147d; font-weight: bold; }
      .header { color: #00e5ff; padding-left: 2; }
      .transcript { flex-grow: 1; flex-shrink: 1; flex-basis: 0; min-height: 0; overflow-y: scroll; padding-left: 2; padding-right: 2; scrollbar-color: #00e5ff #000000; }
      .home .transcript { flex-grow: 0; flex-shrink: 0; flex-basis: auto; overflow-y: visible; }
      .welcome { display: flex; flex-direction: column; margin-bottom: 1; }
      .welcome > div { width: 100%; text-align: center; }
      .banner { white-space: pre; }
      .pink { color: #ff147d; font-weight: bold; }
      .neon { color: #00e5ff; }
      .tag { color: #ffffff; margin-top: 1; }
      .row { display: flex; flex-direction: row; margin-bottom: 1; }
      .row .mark { width: 2; color: #00e5ff; }
      .row .body { flex-grow: 1; white-space: pre-wrap; }
      .user { border-left: heavy; border-color: #ff147d; padding-left: 1; }
      .user .mark { display: none; }
      .user .body { color: #ffffff; font-weight: bold; }
      .assistant { margin-bottom: 0; padding-left: 1; }
      .assistant .mark { display: none; }
      .assistant .body { white-space: normal; }
      .tool { margin-bottom: 0; padding-left: 1; }
      .tool .body { color: #ffffff; }
      .result { margin-bottom: 1; padding-left: 1; }
      .result .mark { width: 4; }
      .result .body { color: #00e5ff; }
      .result.failed .body { color: #ff147d; }
      .error { padding-left: 1; }
      .error .mark, .error .body { color: #ff147d; }
      .note { padding-left: 1; }
      .note .body { color: #ffffff; }
      .busy { color: #ff147d; padding-left: 1; }
      .composer { padding-left: 2; padding-right: 2; }
      .home .composer { width: 76%; align-self: center; padding-left: 0; padding-right: 0; }
      .box { display: flex; flex-direction: row; border: round; border-color: #00e5ff; padding-left: 1; padding-right: 1; }
      .caret { color: #ff147d; font-weight: bold; width: 2; }
      .box input { flex-grow: 1; background-color: #000000; color: #ffffff; padding-left: 0; }
      .box input:focus, .box input:hover { background-color: #000000; color: #ffffff; }
      .box input::part(cursor), .box input:focus::part(cursor), .box input::part(cursor-focused),
      .box input:focus::part(cursor-focused) { color: #ffffff; }
      .box input::part(selection), .box input:focus::part(selection) { background-color: #00e5ff; color: #000000; }
      .meta { display: flex; flex-direction: row; padding-left: 1; padding-right: 1; }
      .model { color: #00e5ff; flex-grow: 1; white-space: nowrap; }
      .tip { color: #ffffff; padding-left: 1; margin-top: 1; }
      .ask { border-left: heavy; border-color: #ff147d; padding-left: 1; margin-bottom: 1; }
      .allow { color: #ff147d; font-weight: bold; padding-right: 1; }
      .q { flex-grow: 1; }
      .keys { color: #00e5ff; font-weight: bold; padding-left: 2; }
      .palette { border: round; border-color: #00e5ff; padding-left: 1; padding-right: 1; margin-bottom: 0; }
      .pitem { display: flex; flex-direction: row; color: #00e5ff; }
      .pitem .pname { width: 14; }
      .pitem .phelp { color: #ffffff; }
      .current .pname { color: #ff147d; font-weight: bold; }
      .promptq { color: #ff147d; font-weight: bold; padding-left: 1; }
      .status { display: flex; flex-direction: row; padding-left: 2; padding-right: 2; color: #00e5ff; }
      .status .left { flex-grow: 1; }
      .gauge { color: #00e5ff; }
      .gauge.hot { color: #ff147d; }
    </style>
  )html";
};

// ------------------------------------------------------------------ markdown on its own (tests)

class MarkdownProbe : public rtxui::Component<MarkdownProbe> {
  public:
    explicit MarkdownProbe(std::string_view markdown) : content(markdown) { Bind(content); }
    std::string content;
    std::string_view view = R"html(<div><markdown content="{content}"/></div>)html";
};

} // namespace

std::string render_markdown_text(std::string_view markdown, int width) {
    rtxui::HeadlessScreen screen(rtxui::Ref<MarkdownProbe>::New(markdown), width, 60);
    return screen.Text();
}

void tui_emit(TuiKind kind, std::string_view text) {
    if (!g_app) return;
    rtxui::PostTask([kind, t = std::string(text)] {
        if (g_app) g_app->Emit(kind, t);
    });
}

bool tui_prompt(std::string_view question, std::string_view default_value, std::string &answer) {
    if (!g_app || !g_sync) return false;
    {
        std::lock_guard<std::mutex> lock(g_sync->mu);
        g_sync->prompt_done = false;
    }
    rtxui::PostTask([q = std::string(question), d = std::string(default_value)] {
        if (!g_app) return;
        g_app->prompt_default = d;
        g_app->prompt_line = q + (d.empty() ? "" : " [" + d + "]");
        g_app->prompting = true;
        g_app->Refresh();
    });
    Sync *sync = g_sync;
    std::unique_lock<std::mutex> lock(sync->mu);
    sync->cv.wait(lock, [&] { return sync->prompt_done; });
    answer = sync->prompt_answer;
    lock.unlock();
    rtxui::PostTask([] {
        if (!g_app) return;
        g_app->prompting = false;
        g_app->Refresh();
    });
    return true;
}

bool tui_ask(std::string_view question, bool *always) {
    if (!g_app || !g_sync) return false;
    Sync *sync = g_sync;
    {
        std::lock_guard<std::mutex> lock(sync->mu);
        sync->answer = 0;
    }
    rtxui::PostTask([q = std::string(question)] {
        if (!g_app) return;
        g_app->question = q;
        g_app->asking = true;
        g_app->Refresh();
    });
    std::unique_lock<std::mutex> lock(sync->mu);
    sync->cv.wait(lock, [&] { return sync->answer != 0; });
    const int a = sync->answer;
    lock.unlock();
    rtxui::PostTask([] {
        if (!g_app) return;
        g_app->asking = false;
        g_app->Refresh();
    });
    if (a == 3 && always) *always = true;
    return a == 1 || a == 3;
}

int run_tui(const TuiHost &host) {
    Sync sync;
    g_sync = &sync;
    auto app = rtxui::Ref<ChatApp>::New(host);
    g_app = app.get();
    rtxui::Screen screen(app);
    g_screen = &screen;
    screen.SetInline(false);
    screen.Loop();

    // The worker may be inside the model or waiting for an answer: cancel it, answer "no" to a pending question, and let it finish.
    if (app->worker.joinable()) {
        if (host.cancel) host.cancel(host.user);
        {
            std::lock_guard<std::mutex> lock(sync.mu);
            sync.answer = 2;
            sync.prompt_answer.clear();
            sync.prompt_done = true;
            sync.cv.notify_all();
        }
        app->worker.join();
    }
    g_screen = nullptr;
    g_app = nullptr;
    g_sync = nullptr;
    return 0;
}

} // namespace agent::cli
