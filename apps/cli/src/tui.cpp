// The full-screen chat interface, built on RTXUI. See tui.hpp.
//
// The screen is an HTML template with CSS (below); the state it shows lives in plain members of ChatApp. Everything runs on the UI
// thread: the worker that talks to the model hands its output over with PostTask().

#include "tui.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
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

// ------------------------------------------------------------------ keyboard shortcuts

namespace {

using rtxui::Event;

struct KeyName {
    std::string name;
    const Event *event;
};

// Every key the interface can bind, by name: F1 to F12, and the letters with Ctrl, Alt and both.
const std::vector<KeyName> &key_table() {
    static const std::vector<KeyName> table = [] {
        std::vector<KeyName> v;
#define RTXUI_KEY(L, UC)                                                                                                                   \
    v.push_back({"ctrl+" #L, &Event::Ctrl##UC()});                                                                                         \
    v.push_back({"alt+" #L, &Event::Alt##UC()});                                                                                           \
    v.push_back({"ctrl+alt+" #L, &Event::CtrlAlt##UC()});
        // clang-format off
        RTXUI_KEY(a, A) RTXUI_KEY(b, B) RTXUI_KEY(c, C) RTXUI_KEY(d, D) RTXUI_KEY(e, E) RTXUI_KEY(f, F) RTXUI_KEY(g, G) RTXUI_KEY(h, H) RTXUI_KEY(i, I)
        RTXUI_KEY(j, J) RTXUI_KEY(k, K) RTXUI_KEY(l, L) RTXUI_KEY(m, M) RTXUI_KEY(n, N) RTXUI_KEY(o, O) RTXUI_KEY(p, P) RTXUI_KEY(q, Q) RTXUI_KEY(r, R)
        RTXUI_KEY(s, S) RTXUI_KEY(t, T) RTXUI_KEY(u, U) RTXUI_KEY(v, V) RTXUI_KEY(w, W) RTXUI_KEY(x, X) RTXUI_KEY(y, Y) RTXUI_KEY(z, Z)
        // clang-format on
#undef RTXUI_KEY
                    v.push_back({"f1", &Event::F1()});
        v.push_back({"f2", &Event::F2()});
        v.push_back({"f3", &Event::F3()});
        v.push_back({"f4", &Event::F4()});
        v.push_back({"f5", &Event::F5()});
        v.push_back({"f6", &Event::F6()});
        v.push_back({"f7", &Event::F7()});
        v.push_back({"f8", &Event::F8()});
        v.push_back({"f9", &Event::F9()});
        v.push_back({"f10", &Event::F10()});
        v.push_back({"f11", &Event::F11()});
        v.push_back({"f12", &Event::F12()});
        return v;
    }();
    return table;
}

constexpr TuiShortcut kShortcutActions[] = {
    {"settings", "Open the settings", "ctrl+o"},
    {"sessions", "Switch to another session", "ctrl+k"},
    {"new_chat", "Start a new chat", "ctrl+n"},
    {"sidebar", "Show or hide the sidebar", "ctrl+b"},
    {"commands", "List the commands", "ctrl+p"},
    {"help", "Show the keyboard shortcuts", "f1"},
    {"quit", "Quit", "ctrl+d"},
};

} // namespace

std::span<const TuiShortcut> tui_shortcut_actions() { return kShortcutActions; }

bool tui_key_valid(std::string_view name) {
    // Ctrl+I, Ctrl+J, Ctrl+M and Ctrl+H are Tab, Enter and Backspace to a terminal; Ctrl+C always cancels; Ctrl+[ is Escape.
    for (const char *taken : {"ctrl+i", "ctrl+j", "ctrl+m", "ctrl+h", "ctrl+c"})
        if (name == taken) return false;
    for (const KeyName &k : key_table())
        if (k.name == name) return true;
    return false;
}

std::string tui_key_pretty(std::string_view name) {
    if (!tui_key_valid(name)) return {};
    std::string out;
    size_t at = 0;
    while (at <= name.size()) {
        size_t plus = name.find('+', at);
        std::string part(name.substr(at, plus == std::string_view::npos ? std::string_view::npos : plus - at));
        if (!part.empty()) part[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(part[0])));
        out += (out.empty() ? "" : "+") + part;
        if (plus == std::string_view::npos) break;
        at = plus + 1;
    }
    return out;
}

// ------------------------------------------------------------------ the screen

namespace {

using rtxui::Event;

// Four colors: black (the background), white (text), neon cyan (tool calls, hints, code, rules) and hot pink (the rarest: the brand,
// the open session, the spinner, errors).
constexpr std::string_view kMarkdownCss = R"css(
  h1, h2, h3, h4, h5, h6, p, ul, ol, pre, blockquote, table, hr { margin-top: 0; }
  h1 { color: #ffb3d1; font-weight: bold; border-bottom: double; border-color: #4a4a4a; margin-bottom: 1; }
  h2 { color: #ffffff; font-weight: bold; border-bottom: solid; border-color: #4a4a4a; margin-bottom: 1; }
  h3, h4, h5, h6 { color: #40e0d0; font-weight: bold; }
  p { color: #ffffff; margin-bottom: 1; }
  strong { color: #ffffff; font-weight: bold; }
  em { color: #ffffff; }
  a { color: #40e0d0; text-decoration: underline; }
  code { color: #40e0d0; }
  pre { color: #40e0d0; scrollbar-width: none; border: round; border-color: #4a4a4a; padding-left: 1; padding-right: 1; margin-bottom: 1; }
  blockquote { color: #ffffff; border-left: solid; border-color: #4a4a4a; padding-left: 1; margin-bottom: 1; }
  blockquote p { margin-bottom: 0; }
  ul, ol { padding-left: 2; margin-bottom: 1; }
  li { color: #ffffff; }
  li p { margin-bottom: 0; }
  table { border: round; border-color: #4a4a4a; margin-bottom: 1; }
  th { color: #ffb3d1; font-weight: bold; border-bottom: solid; border-color: #4a4a4a; padding-left: 1; padding-right: 1; }
  td { color: #ffffff; padding-left: 1; padding-right: 1; }
  hr { color: #40e0d0; margin-bottom: 1; }
)css";

// One block of the conversation, as the template reads it.
struct Icons {
    const char *assistant, *error, *tool, *result, *current, *other, *project, *caret, *busy, *settings, *copy, *del, *mic, *back, *menu;
};

// Unicode symbols every terminal font has; a Nerd Font (https://www.nerdfonts.com) gives richer glyphs: AGENT_ICONS=nerd.
const Icons &icons() {
    static const Icons unicode{"\xE2\x9C\xA6", "\xE2\x9C\x97", "\xE2\x9D\xAF", "\xE2\x86\xB3", "\xE2\x97\x8F",
                               "\xE2\x97\x8B", "\xE2\x96\xB8", "\xE2\x9D\xAF", "\xC2\xB7",     "\xE2\x9C\xB1",
                               "\xE2\xA7\x89", "\xE2\x9C\x95", "\xE2\x97\x89", "\xE2\x80\xB9", "\xE2\x98\xB0"};
    static const Icons nerd{"\xEF\x80\x85", "\xEF\x80\x8D", "\xEF\x80\x93", "\xEF\x80\x8C", "\xEF\x84\x91",
                            "\xEF\x84\x8C", "\xEF\x81\xBB", "\xEF\x84\x85", "\xC2\xB7",     "\xEF\x80\x93",
                            "\xEF\x83\x85", "\xEF\x87\xB8", "\xEF\x84\xB0", "\xEF\x81\x93", "\xEF\x83\x89"};
    static const Icons ascii{"*", "x", ">", "-", "*", "o", ">", ">", ".", "*", "c", "x", "o", "<", "="};
    static const Icons &chosen = []() -> const Icons & {
        const char *want = std::getenv("AGENT_ICONS");
        if (want && std::string_view(want) == "nerd") return nerd;
        if (want && std::string_view(want) == "ascii") return ascii;
        return unicode;
    }();
    return chosen;
}

struct Row {
    TuiKind kind = TuiKind::Note;
    std::string mark; // the symbol in the left gutter
    std::string text;
    std::string cls;  // CSS classes
    int markdown = 0; // 1: render `text` as markdown
    int id = 0;       // what the copy and delete buttons name
    int user = 0;     // 1: a message of the user (it has a delete button)
    int acts = 0;     // 1: the row has the copy button (user messages and answers)
    std::string meta; // under an answer: how fast and how long
    bool operator==(const Row &) const = default;
};

void Fields(const Row &row, rtxui::FieldSink &out) {
    out("mark", row.mark);
    out("text", row.text);
    out("cls", row.cls);
    out("md", row.markdown);
    out("plain", row.markdown ? 0 : 1);
    out("id", row.id);
    out("user", row.user);
    out("acts", row.acts);
    out("meta", row.meta);
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

// One line of the sidebar: a project heading or a session. `idx` is the session's place in ChatApp::sessions (-1 for a heading).
struct SideRow {
    std::string text, cls;
    int idx = -1;
    bool operator==(const SideRow &) const = default;
};

void Fields(const SideRow &row, rtxui::FieldSink &out) {
    out("text", row.text);
    out("cls", row.cls);
    out("idx", row.idx);
}

// One line of the settings screen: a heading, or a setting with its value.
struct SetRow {
    std::string label, value, cls;
    bool operator==(const SetRow &) const = default;
};

void Fields(const SetRow &row, rtxui::FieldSink &out) {
    out("label", row.label);
    out("value", row.value);
    out("cls", row.cls);
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

const char *const kTips[] = {
    "Tip: type / or press Ctrl+P for commands",           "Tip: Ctrl+K picks another session; Ctrl+N starts a new chat",
    "Tip: Esc stops an answer while it is running",       "Tip: select text with the mouse to copy it",
    "Tip: /image PATH sends a picture with your message", "Tip: Ctrl+B hides the sidebar"};

const char *const kSpinner[] = {"\xE2\xA0\x8B", "\xE2\xA0\x99", "\xE2\xA0\xB9", "\xE2\xA0\xB8", "\xE2\xA0\xBC",
                                "\xE2\xA0\xB4", "\xE2\xA0\xA6", "\xE2\xA0\xA7", "\xE2\xA0\x87", "\xE2\xA0\x8F"};

// The "agent.cpp" logo: two rows of half-block letters, turquoise over pink (a narrow terminal shows the plain name).
const char *const kBanner[2] = {"\xE2\x96\x84"
                                "\xE2\x96\x80"
                                "\xE2\x96\x88"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x80"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x80"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x84"
                                " "
                                "\xE2\x96\x88"
                                " "
                                "\xE2\x96\x80"
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                " "
                                " "
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x80"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x88"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x88",
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x88"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x84"
                                "\xE2\x96\x88"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x88"
                                "\xE2\x96\x84"
                                " "
                                "\xE2\x96\x88"
                                " "
                                "\xE2\x96\x80"
                                "\xE2\x96\x88"
                                " "
                                " "
                                "\xE2\x96\x88"
                                " "
                                " "
                                "\xE2\x96\x84"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x84"
                                "\xE2\x96\x84"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x80"
                                " "
                                "\xE2\x96\x88"
                                "\xE2\x96\x80"
                                "\xE2\x96\x80"};

// The last folder name of a path, for the project headings.
std::string base_name(std::string_view path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) path.remove_suffix(1);
    const size_t slash = path.find_last_of("/\\");
    return std::string(slash == std::string_view::npos ? path : path.substr(slash + 1));
}

class ChatApp : public rtxui::Component<ChatApp> {
  public:
    explicit ChatApp(const TuiHost &host_) : host(host_) {
        for (size_t i = 0; i < 2; ++i) banner.push_back({kBanner[i], i == 0 ? "banner neon" : "banner pink"});
        tip = kTips[static_cast<size_t>(std::time(nullptr) / 60) % (sizeof kTips / sizeof *kTips)];
        Bind(rows);
        Bind(banner);
        Bind(matches);
        Bind(side);
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
        Bind(box_class);
        Bind(md_css);
        Bind(main_title);
        Bind(project_line);
        Bind(side_class);
        Bind(app_class);
        Bind(tip);
        Bind(notice);
        Bind(has_notice);
        Bind(asking);
        Bind(prompting);
        Bind(busy);
        Bind(show_welcome);
        Bind(show_palette);
        Bind(show_settings);
        Bind(set_rows);
        Bind(cat_rows);
        Bind(cat_title);
        Bind(set_help);
        Bind(PickCat);
        Bind(SettingsBack);
        Bind(CloseSettings);
        Bind(set_view);
        Bind(set_note);
        Bind(settings_label);
        Bind(OpenSettings);
        Bind(ico_copy);
        Bind(ico_del);
        Bind(ico_mic);
        Bind(ico_back);
        Bind(ico_menu);
        Bind(ico_settings);
        Bind(mic_class);
        Bind(hint);
        Bind(has_messages);
        Bind(tb_cpu);
        Bind(tb_ram);
        Bind(tb_files);
        Bind(tb_engine);
        Bind(tb_models);
        Bind(tb_mode);
        Bind(tb_ctx);
        Bind(confirm_open);
        Bind(confirm_title);
        Bind(confirm_text);
        Bind(show_keys);
        Bind(key_rows);
        Bind(toast_open);
        Bind(toast_text);
        Bind(CopyRow);
        Bind(AskDelete);
        Bind(AskDeleteSession);
        Bind(ConfirmYes);
        Bind(ConfirmNo);
        Bind(CloseKeys);
        Bind(ToggleMic);
        Bind(Pick);
        Bind(NewChat);
        Bind(ToggleSide);
        LoadKeymap();
        Refresh();
        RefreshSessions();
    }

    TuiHost host;
    std::vector<Row> rows;
    std::vector<Line> banner;
    std::vector<Match> matches;
    std::vector<SideRow> side;
    std::vector<TuiSession> sessions;
    std::string box_class = "box", input, header, subtitle, question, prompt_line, busy_line, status_left, gauge, gauge_class = "gauge",
                caret = "\xE2\x9D\xAF";
    std::string notice;
    bool has_notice = false;
    std::string main_title = "New chat", project_line, side_class = "side", app_class = "main home", tip;
    std::string md_css = std::string(kMarkdownCss);
    bool asking = false, prompting = false, busy = false;
    bool show_welcome = true, show_palette = false;
    // ---- the shortcuts: action -> key name, the defaults with what the user changed on top
    std::map<std::string, std::string> keymap;
    bool set_capturing = false;

    void LoadKeymap() {
        keymap.clear();
        for (const TuiShortcut &a : tui_shortcut_actions()) keymap[a.action] = a.default_key;
        if (host.shortcuts)
            for (const auto &[action, key] : host.shortcuts(host.user))
                if (keymap.count(action) && tui_key_valid(key)) keymap[action] = key;
        key_rows.clear();
        for (const TuiShortcut &a : tui_shortcut_actions()) key_rows.push_back({tui_key_pretty(keymap[a.action]), a.label, ""});
        key_rows.push_back({"? or F1", "Show this list", ""});
        key_rows.push_back({"@file", "Put a file's text in your message", ""});
        key_rows.push_back({"!command", "Run a shell command in your project", ""});
        key_rows.push_back({"Esc", "Stop the answer that is running", ""});
        key_rows.push_back({"PageUp, PageDown", "Scroll the conversation", ""});
        key_rows.push_back({"Home, End", "Oldest message, newest message", ""});
    }
    // Whether this event is the key bound to the action.
    bool Is(const Event &e, const char *action) const {
        const auto bound = keymap.find(action);
        if (bound == keymap.end()) return false;
        for (const KeyName &k : key_table())
            if (k.name == bound->second) return e == *k.event;
        return false;
    }
    // The name of the key an event is, when it is one the interface can bind.
    static std::string NameOf(const Event &e) {
        for (const KeyName &k : key_table())
            if (e == *k.event) return k.name;
        return {};
    }

    // ---- icons, the bottom bar, the hint line, dialogs
    std::string ico_copy = icons().copy, ico_del = icons().del, ico_mic = "mic", ico_back = icons().back, ico_menu = icons().menu,
                ico_settings = icons().settings, mic_class = "mic", hint;
    bool has_messages = false;
    std::string tb_cpu, tb_ram, tb_files, tb_engine, tb_models, tb_mode, tb_ctx;
    bool confirm_open = false, show_keys = false, toast_open = false, recording = false, transcribing = false;
    std::string confirm_title, confirm_text, confirm_kind, toast_text;
    int confirm_row = 0, next_row_id = 0;
    std::vector<SetRow> key_rows;

    void AddRow(Row r) {
        r.id = ++next_row_id;
        r.acts = r.kind == TuiKind::User || r.kind == TuiKind::Assistant ? 1 : 0;
        r.user = r.kind == TuiKind::User ? 1 : 0;
        rows.push_back(std::move(r));
    }
    void Toast(const std::string &text) {
        toast_text = text;
        toast_open = true;
    }
    Row *FindRow(const std::string &id) {
        const int want = std::atoi(id.c_str());
        for (Row &r : rows)
            if (r.id == want) return &r;
        return nullptr;
    }

    static std::string Base64(std::string_view in) {
        static const char *t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        size_t i = 0;
        for (; i + 2 < in.size(); i += 3) {
            const unsigned v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8) |
                               static_cast<unsigned char>(in[i + 2]);
            out += t[v >> 18];
            out += t[(v >> 12) & 63];
            out += t[(v >> 6) & 63];
            out += t[v & 63];
        }
        if (i + 1 == in.size()) {
            const unsigned v = static_cast<unsigned char>(in[i]) << 16;
            out += t[v >> 18];
            out += t[(v >> 12) & 63];
            out += "==";
        } else if (i + 2 == in.size()) {
            const unsigned v = (static_cast<unsigned char>(in[i]) << 16) | (static_cast<unsigned char>(in[i + 1]) << 8);
            out += t[v >> 18];
            out += t[(v >> 12) & 63];
            out += t[(v >> 6) & 63];
            out += '=';
        }
        return out;
    }

    // The text goes to the clipboard: through the terminal (OSC 52, which Windows Terminal, iTerm, kitty, WezTerm, tmux and most others take,
    // also over SSH) and through the system's own clipboard tool where there is one.
    void CopyText(const std::string &text) {
        const std::string seq = "\x1b]52;c;" + Base64(text) + "\x07";
        std::fwrite(seq.data(), 1, seq.size(), stdout);
        std::fflush(stdout);
        std::thread([text] {
#if defined(_WIN32)
            std::error_code ec;
            const auto file = std::filesystem::temp_directory_path(ec) / "agent-clip.txt";
            {
                std::ofstream f(file, std::ios::binary | std::ios::trunc);
                f << text;
            }
            const std::string cmd =
                "powershell -NoProfile -Command \"Get-Content -Raw -Encoding UTF8 '" + file.string() + "' | Set-Clipboard\" >nul 2>nul";
            (void)std::system(cmd.c_str());
            std::filesystem::remove(file, ec);
#else
            if (std::FILE *p = popen("(wl-copy || xclip -selection clipboard || xsel -b -i || pbcopy) 2>/dev/null", "w")) {
                std::fwrite(text.data(), 1, text.size(), p);
                pclose(p);
            }
#endif
        }).detach();
    }

    void CopyRow(std::string id) {
        if (Row *r = FindRow(id)) {
            CopyText(r->text);
            Toast("Copied");
        }
    }

    void AskDelete(std::string id) {
        if (busy) return;
        if (!FindRow(id)) return;
        confirm_kind = "turn";
        confirm_row = std::atoi(id.c_str());
        confirm_title = "Delete this message?";
        confirm_text = "The message and the answers to it are removed from this conversation, and the model forgets them.";
        confirm_open = true;
    }
    void AskDeleteSession() {
        if (busy) return;
        confirm_kind = "session";
        confirm_title = "Delete this conversation?";
        confirm_text = "It is removed from every place it is kept: the list, the saved history and the memory. This cannot be undone.";
        confirm_open = true;
    }
    void ConfirmNo() { confirm_open = false; }
    void ConfirmYes() {
        confirm_open = false;
        if (confirm_kind == "turn") {
            size_t at = rows.size();
            for (size_t i = 0; i < rows.size(); ++i)
                if (rows[i].id == confirm_row) at = i;
            if (at == rows.size()) return;
            int from_end = 0;
            size_t end = rows.size();
            for (size_t i = at + 1; i < rows.size(); ++i)
                if (rows[i].kind == TuiKind::User) {
                    if (end == rows.size()) end = i;
                    ++from_end;
                }
            const bool in_memory = host.delete_turn && host.delete_turn(host.user, from_end);
            rows.erase(rows.begin() + static_cast<std::ptrdiff_t>(at), rows.begin() + static_cast<std::ptrdiff_t>(end));
            if (rows.empty()) {
                show_welcome = true;
                app_class = "main home";
            }
            Toast(in_memory ? "Deleted" : "Removed from the screen; the model had already compacted it");
            RefreshSessions();
        } else if (confirm_kind == "session") {
            for (const TuiSession &s : sessions)
                if (s.active && host.delete_session) {
                    if (!host.delete_session(host.user, s.id, s.workspace)) {
                        Toast("Could not delete the conversation");
                        return;
                    }
                    break;
                }
            if (host.take_repaint && host.take_repaint(host.user)) Repaint();
            RefreshSessions();
            Toast("Conversation deleted");
        }
        Refresh();
    }
    void CloseKeys() { show_keys = false; }

    void ToggleMic() {
        if (busy || transcribing || !host.mic) return;
        if (!recording) {
            const std::string err = host.mic(host.user, true);
            if (!err.empty()) {
                Toast(err);
                return;
            }
            recording = true;
            mic_class = "mic rec";
            return;
        }
        recording = false;
        transcribing = true;
        mic_class = "mic";
        busy_line = "turning your speech into text ...";
        std::thread([this] {
            const std::string words = host.mic(host.user, false);
            rtxui::PostTask([this, words] {
                transcribing = false;
                busy_line.clear();
                if (!words.empty() && words[0] == '!') Toast(words.substr(1));
                else if (!words.empty()) SetInput(input.empty() ? words : input + " " + words);
                Refresh();
            });
        }).detach();
    }

    void RefreshTaskbar() {
        TuiTaskbar tb;
        if (host.taskbar) tb = host.taskbar(host.user);
        char buf[96];
        tb_cpu = tb.cpu_percent >= 0 ? "CPU " + std::to_string(tb.cpu_percent) + "%" : std::string();
        if (tb.ram_total_gb > 0) {
            std::snprintf(buf, sizeof(buf), "RAM %.1f/%.1f GB", tb.ram_used_gb, tb.ram_total_gb);
            tb_ram = buf;
        }
        if (tb.files_gb >= 0) {
            std::snprintf(buf, sizeof(buf), "Files %.1f GB", tb.files_gb);
            tb_files = buf;
        }
        tb_engine = tb.engine;
        tb_models.clear();
        if (!tb.models.empty()) {
            tb_models = clip(tb.models.front(), 12);
            if (tb.models.size() > 1) tb_models += " +" + std::to_string(tb.models.size() - 1);
        }
        tb_mode = tb.mode;
        if (tb.context_total > 0) {
            tb_ctx = "ctx " + std::to_string(tb.context_used) + "/" + std::to_string(tb.context_total);
            gauge_class = tb.context_used * 100 >= tb.context_total * 85 ? "gauge hot" : "gauge";
        } else {
            tb_ctx.clear();
        }
    }
    // Once a second, whether or not an answer is running.
    void StatsTick() {
        RefreshTaskbar();
        ComputeHint();
        rtxui::PostDelayedTask([this] { StatsTick(); }, std::chrono::milliseconds(1000));
    }

    // The line between the input and the bottom bar: what the next useful key or command is, for what is going on right now.
    void ComputeHint() {
        const std::string idle = tui_key_pretty(keymap["help"]) + " shortcuts";
        if (confirm_open) hint = "Enter or y confirms, Esc cancels";
        else if (asking) hint = "y allows this once, a always allows this tool, n refuses";
        else if (recording) hint = "Listening: click the microphone again to stop";
        else if (busy) hint = "Esc stops the answer; the wheel or PageUp scrolls";
        else if (!follow) hint = "End jumps to the newest message";
        else if (!input.empty() && input[0] == '/') hint = "Tab completes the command, Enter runs it";
        else if (!input.empty() && input[0] == '@') hint = "@path puts that file's text in your message, for example @src/main.cpp";
        else if (!input.empty() && input[0] == '!')
            hint = "Runs in your project folder; the output goes into the chat for the model to see";
        else if (!input.empty()) hint = "Enter sends";
        else hint = idle;
    }

    // ---- the settings screen: categories on the left, the settings of the chosen category on the right
    bool show_settings = false, set_editing = false, set_in_content = false;
    std::vector<TuiSetting> set_items;
    std::vector<size_t> shown; // indexes into set_items of the chosen category
    std::vector<std::string> categories;
    std::vector<SideRow> cat_rows;
    std::vector<SetRow> set_rows;
    int cat_sel = 0, set_sel = 0;
    std::string set_view = "setbody cats", set_note, set_help, cat_title, settings_label = std::string(icons().settings) + " Settings";

    void OpenSettings() {
        if (!host.settings || busy) return;
        show_settings = true;
        set_editing = false;
        set_in_content = false;
        cat_sel = 0;
        set_sel = 0;
        set_note.clear();
        SetInput("");
        LoadSettings();
    }
    void CloseSettings() {
        show_settings = false;
        set_editing = false;
        SetInput("");
        Refresh();
    }
    // The back arrow of the header: from the settings of a category to the list of categories, from the list closes the screen.
    void SettingsBack() {
        if (set_in_content && !set_editing) {
            set_in_content = false;
            LoadSettings();
            Refresh();
        } else {
            CloseSettings();
        }
    }
    void PickCat(const std::string &index) {
        const int at = std::atoi(index.c_str());
        if (at < 0 || at >= static_cast<int>(categories.size())) return;
        cat_sel = at;
        set_sel = 0;
        set_in_content = true;
        set_editing = false;
        LoadSettings();
        Refresh();
    }
    void LoadSettings() {
        if (host.settings) set_items = host.settings(host.user);
        set_view = set_in_content ? "setbody content" : "setbody cats";
        categories.clear();
        for (const TuiSetting &s : set_items)
            if (std::find(categories.begin(), categories.end(), s.group) == categories.end()) categories.push_back(s.group);
        if (cat_sel >= static_cast<int>(categories.size())) cat_sel = std::max(0, static_cast<int>(categories.size()) - 1);
        cat_rows.clear();
        for (size_t i = 0; i < categories.size(); ++i)
            cat_rows.push_back({categories[i],
                                static_cast<int>(i) == cat_sel ? (set_in_content ? "setcat cur" : "setcat cur focus") : "setcat",
                                static_cast<int>(i)});
        shown.clear();
        cat_title = categories.empty() ? std::string() : categories[static_cast<size_t>(cat_sel)];
        for (size_t i = 0; i < set_items.size(); ++i)
            if (set_items[i].group == cat_title) shown.push_back(i);
        if (set_sel >= static_cast<int>(shown.size())) set_sel = std::max(0, static_cast<int>(shown.size()) - 1);
        set_rows.clear();
        std::string section;
        for (size_t k = 0; k < shown.size(); ++k) {
            const TuiSetting &s = set_items[shown[k]];
            if (!s.section.empty() && s.section != section) set_rows.push_back({s.section, "", "setgroup"});
            section = s.section;
            const bool current = set_in_content && static_cast<int>(k) == set_sel;
            std::string value = s.value.empty() ? "-" : s.value;
            if (set_capturing && current) value = "press a key ...";
            else if (set_editing && current) value = input + "_";
            else if (!s.options.empty() && s.editable) value = "\xE2\x97\x82 " + value + " \xE2\x96\xB8";
            set_rows.push_back({s.label, value, std::string("setrow") + (current ? " cur" : "") + (s.editable ? "" : " ro")});
        }
        if (set_capturing) {
            set_note = "Press the new key now: F1 to F12, Ctrl+letter or Alt+letter. Esc cancels.";
            set_help = "Press a key  \xC2\xB7  Esc cancels";
        } else if (set_editing) {
            set_note = "Type the new value. Enter keeps it, Esc cancels.";
            set_help = "Enter saves  \xC2\xB7  Esc cancels";
        } else if (set_in_content && !shown.empty()) {
            set_note = set_items[shown[static_cast<size_t>(set_sel)]].hint;
            set_help = "Up and Down choose  \xC2\xB7  Left, Right or Enter change  \xC2\xB7  Esc or Tab back to the list";
        } else {
            set_help = "Up and Down choose a category  \xC2\xB7  Enter or Tab opens it  \xC2\xB7  Esc closes";
            if (!set_in_content) set_note.clear();
        }
    }
    const TuiSetting *CurrentSetting() const {
        if (shown.empty() || set_sel >= static_cast<int>(shown.size())) return nullptr;
        return &set_items[shown[static_cast<size_t>(set_sel)]];
    }
    void ApplySetting(const std::string &value) {
        const TuiSetting *s = CurrentSetting();
        if (!s || !host.set_setting) return;
        const std::string error = host.set_setting(host.user, s->key, value);
        LoadKeymap();
        LoadSettings();
        if (!error.empty()) set_note = error;
        else if (const TuiSetting *now = CurrentSetting()) set_note = "Saved. " + now->hint;
        Refresh();
    }
    void CycleSetting(int step) {
        const TuiSetting *s = CurrentSetting();
        if (!s || !s->editable || s->options.empty()) return;
        size_t at = 0;
        for (size_t i = 0; i < s->options.size(); ++i)
            if (s->options[i] == s->value) at = i;
        const size_t n = s->options.size();
        const std::string next = s->options[(at + n + static_cast<size_t>(step % static_cast<int>(n) + static_cast<int>(n))) % n];
        ApplySetting(next);
    }
    // Keys while the settings screen is open. Returns true when the event was used.
    bool SettingsKey(const Event &e) {
        if (set_capturing) {
            if (e == Event::Escape()) {
                set_capturing = false;
                LoadSettings();
                Refresh();
                return true;
            }
            if (e.is<Event::Mouse>()) return true;
            const std::string name = NameOf(e);
            set_capturing = false;
            if (name.empty() || !tui_key_valid(name)) {
                LoadSettings();
                set_note = "That key cannot be used. Press F1 to F12, Ctrl+letter or Alt+letter.";
                Refresh();
                return true;
            }
            ApplySetting(name);
            return true;
        }
        if (set_editing) {
            if (e == Event::Return()) {
                set_editing = false;
                const std::string value = input;
                SetInput("");
                ApplySetting(value);
                return true;
            }
            if (e == Event::Escape() || e == Event::CtrlC()) {
                set_editing = false;
                SetInput("");
                LoadSettings();
                Refresh();
                return true;
            }
            const bool used = Component<ChatApp>::OnEvent(e); // typing goes to the input line
            LoadSettings();
            Refresh();
            return used;
        }
        if (e.is<Event::Mouse>()) return Component<ChatApp>::OnEvent(e); // clicks reach the category list
        if (e == Event::Tab()) {
            set_in_content = !set_in_content && !shown.empty();
        } else if (!set_in_content) {
            const int count = static_cast<int>(categories.size());
            if (e == Event::Escape() || e == Event::CtrlC()) {
                CloseSettings();
                return true;
            }
            if (e == Event::ArrowDown() && count) cat_sel = (cat_sel + 1) % count, set_sel = 0;
            else if (e == Event::ArrowUp() && count) cat_sel = (cat_sel + count - 1) % count, set_sel = 0;
            else if ((e == Event::Return() || e == Event::ArrowRight()) && !shown.empty()) set_in_content = true;
        } else {
            const int count = static_cast<int>(shown.size());
            if (e == Event::Escape() || e == Event::CtrlC()) set_in_content = false;
            else if (e == Event::ArrowDown() && count) set_sel = (set_sel + 1) % count;
            else if (e == Event::ArrowUp() && count) set_sel = (set_sel + count - 1) % count;
            else if (e == Event::ArrowRight()) CycleSetting(1);
            else if (e == Event::ArrowLeft()) CycleSetting(-1);
            else if (e == Event::Return()) {
                if (const TuiSetting *s = CurrentSetting()) {
                    if (!s->editable) {
                        set_note = "This one is shown for information only.";
                    } else if (!s->options.empty()) {
                        CycleSetting(1);
                        return true;
                    } else if (s->capture) {
                        set_capturing = true;
                    } else {
                        set_editing = true;
                        SetInput(s->value);
                    }
                }
            }
        }
        LoadSettings();
        Refresh();
        return true;
    }

    bool follow = true;       // the conversation stays at the newest line
    bool side_hide = false;   // Ctrl+B on a wide terminal hides the sidebar
    bool side_drawer = false; // on a narrow terminal the sidebar is a drawer over the conversation: Ctrl+K, Ctrl+B or the ☰ button
    bool side_focus = false;  // Ctrl+K: Up/Down pick a session, Enter opens it
    int side_sel = 0;
    std::string side_filter, mode_text;
    std::string prompt_default;
    size_t selected = 0;
    std::vector<std::string> history;
    size_t history_at = 0; // == history.size() while typing a new line
    std::string draft;
    std::thread worker;
    int frame = 0;
    std::chrono::steady_clock::time_point busy_since{};
    bool ticking = false;

    // ---- the conversation

    void Emit(TuiKind kind, const std::string &text) {
        if (kind == TuiKind::Assistant) {
            if (!rows.empty() && rows.back().kind == TuiKind::Assistant) rows.back().text += text;
            else AddRow({kind, icons().assistant, text, "row assistant", 1});
        } else {
            std::string t = text;
            while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
            if (t.empty()) return;
            if (kind == TuiKind::User) AddRow({kind, "", t, "row user", 0});
            else if (kind == TuiKind::Error) AddRow({kind, icons().error, t, "row error", 0});
            else if (kind == TuiKind::Tool) AddTool(t);
            else if (kind == TuiKind::Metrics) {
                for (size_t i = rows.size(); i-- > 0;)
                    if (rows[i].kind == TuiKind::Assistant) {
                        rows[i].meta = t;
                        break;
                    }
                Settle();
                return;
            } else AddRow({kind, "", t, "row note", 0});
        }
        show_welcome = false;
        app_class = "main";
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
            AddRow({TuiKind::Tool, icons().tool, name + "  " + args, "row tool", 0});
        } else if (t.compare(0, result.size(), result) == 0) {
            const std::string body = summarize_tool_result(trim_both(std::string_view(t).substr(result.size())), 4, 200);
            const bool failed = body.compare(0, 5, "Error") == 0 ||
                                (body.find("\"exit_code\":0") == std::string::npos && body.find("\"exit_code\":") != std::string::npos);
            AddRow({TuiKind::Tool, failed ? icons().error : icons().result, body, failed ? "row result failed" : "row result", 0});
        } else {
            AddRow({TuiKind::Tool, "", t, "row note", 0});
        }
    }

    // After any change: keep the newest output in view.
    void Settle() {
        // The position the layout last settled on is where the bottom is; pinning to the end sets a huge offset that only the next layout
        // clamps, and a scroll up that starts from it would land at the bottom again. So the settled value is remembered first.
        if (const int y = QueryElement("#t").scroll_y(); y < kPinned) known_bottom = std::max(known_bottom, y);
        if (follow) ScrollToEnd();
    }
    static constexpr int kPinned = INT_MAX / 8;
    int known_bottom = 0;
    void ScrollToEnd() { QueryElement("#t").SetScrollY(INT_MAX / 4); }

    // ---- the sidebar: sessions grouped by project

    void RefreshSessions() {
        if (!host.sessions || busy) return;
        sessions = host.sessions(host.user);
        main_title = "New chat";
        for (const TuiSession &s : sessions)
            if (s.active) main_title = s.title.empty() ? "New chat" : s.title;
        RebuildSide();
    }

    // Which sessions the filter lets through (by a part of the title or the project, any case).
    bool Visible(const TuiSession &s) const {
        if (side_filter.empty()) return true;
        auto lower = [](std::string text) {
            for (char &c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return text;
        };
        const std::string needle = lower(side_filter);
        return lower(s.title).find(needle) != std::string::npos || lower(s.workspace).find(needle) != std::string::npos;
    }

    void RebuildSide() {
        side.clear();
        std::vector<std::string> seen;
        int projects = 0;
        {
            std::vector<std::string> all;
            for (const TuiSession &s : sessions)
                if (std::find(all.begin(), all.end(), s.workspace) == all.end()) all.push_back(s.workspace);
            projects = static_cast<int>(all.size());
        }
        for (size_t i = 0; i < sessions.size(); ++i) {
            const std::string &ws = sessions[i].workspace;
            if (std::find(seen.begin(), seen.end(), ws) != seen.end()) continue;
            seen.push_back(ws);
            std::vector<SideRow> group;
            for (size_t j = i; j < sessions.size(); ++j) {
                if (sessions[j].workspace != ws || !Visible(sessions[j])) continue;
                const TuiSession &s = sessions[j];
                std::string cls = s.active ? "sess current" : "sess";
                if (side_focus && static_cast<int>(j) == side_sel) cls += " hi";
                const std::string title = s.title.empty() ? "New chat" : s.title;
                group.push_back(
                    {std::string(s.active ? icons().current : icons().other) + " " + clip(title, 22), cls, static_cast<int>(j)});
            }
            if (group.empty()) continue;
            if (projects > 1) side.push_back({std::string(icons().project) + " " + base_name(ws), "proj", -1});
            side.insert(side.end(), group.begin(), group.end());
        }
        if (!side_filter.empty()) side.insert(side.begin(), {"find: " + side_filter, "proj", -1});
    }

    // A click on a session (the index arrives as text), or Enter in the sidebar.
    void Pick(std::string index) {
        const int i = index.empty() ? -1 : std::atoi(index.c_str());
        if (i < 0 || i >= static_cast<int>(sessions.size())) return;
        const TuiSession &s = sessions[static_cast<size_t>(i)];
        side_drawer = false;
        if (s.active) return;
        SwitchTo(s.id, s.workspace);
    }

    // The ☰ button of a narrow terminal, and Ctrl+B: hides the sidebar when there is room for it, opens it as a drawer when there is not.
    void ToggleSide() {
        side_hide = !side_hide;
        side_drawer = !side_drawer;
        Refresh();
    }

    void NewChat() { SwitchTo("", host.subtitle ? host.subtitle(host.user) : std::string()); }

    void SwitchTo(const std::string &id, const std::string &workspace) {
        if (!host.switch_session) return;
        if (busy) {
            Emit(TuiKind::Note, "Wait for the answer to finish before switching.");
            return;
        }
        if (worker.joinable()) worker.join();
        busy = true;
        busy_since = std::chrono::steady_clock::now();
        busy_line = std::string(kSpinner[0]) + " opening the session";
        worker = std::thread([this, id, workspace] {
            const bool opened = host.switch_session(host.user, id, workspace);
            rtxui::PostTask([this, opened] {
                busy = false;
                if (opened) Repaint();
                side_focus = false;
                side_drawer = false;
                Refresh();
                RefreshSessions();
                Settle();
            });
        });
    }

    // The transcript of the open conversation, from the host.
    void Repaint() {
        known_bottom = 0;
        rows.clear();
        show_welcome = true;
        app_class = "main home";
        follow = true;
        if (host.history)
            for (const TuiMessage &m : host.history(host.user)) Emit(m.kind, m.text);
    }

    // ---- the header, the status line and the context gauge

    void Refresh() {
        header = host.header ? host.header(host.user) : std::string();
        mode_text = host.mode ? host.mode(host.user) : std::string();
        subtitle = host.subtitle ? host.subtitle(host.user) : std::string();
        project_line = subtitle;
        notice = host.notice ? host.notice(host.user) : std::string();
        has_notice = !notice.empty();
        has_messages = !show_welcome;
        ComputeHint();
        app_class = show_welcome ? "main home" : "main";
        side_class = std::string("side") + (side_hide ? " hidden" : "") + (side_drawer ? " drawer" : "");
        box_class = (busy || asking || side_focus) ? "box" : "box active";
        UpdateBusyLine();
    }

    void UpdateBusyLine() {
        if (transcribing) return;
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
                if (host.take_repaint && host.take_repaint(host.user)) Repaint();
                Refresh();
                RefreshSessions();
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

    // Up and Down walk over the sessions the filter shows.
    void MoveSide(int step) {
        std::vector<int> visible;
        for (size_t i = 0; i < sessions.size(); ++i)
            if (Visible(sessions[i])) visible.push_back(static_cast<int>(i));
        if (visible.empty()) return;
        auto at = std::find(visible.begin(), visible.end(), side_sel);
        size_t pos = at == visible.end() ? 0 : static_cast<size_t>(at - visible.begin());
        if (at != visible.end() || step != 0)
            pos = (pos + visible.size() + static_cast<size_t>(step < 0 ? visible.size() - 1 : step)) % visible.size();
        side_sel = visible[pos];
        RebuildSide();
    }

    // Delete in the picker: removes the highlighted conversation.
    void DeleteSelected() {
        if (!host.delete_session || side_sel < 0 || side_sel >= static_cast<int>(sessions.size())) return;
        const TuiSession target = sessions[static_cast<size_t>(side_sel)];
        if (!host.delete_session(host.user, target.id, target.workspace)) {
            Emit(TuiKind::Error, "could not delete the conversation");
            return;
        }
        if (host.take_repaint && host.take_repaint(host.user)) Repaint();
        RefreshSessions();
        MoveSide(0);
    }

    bool OnEvent(Event e) override {
        if (confirm_open) {
            const int c = Letter(e);
            if (e == Event::Return() || c == 'y' || c == 'Y') ConfirmYes();
            else if (e == Event::Escape() || c == 'n' || c == 'N') ConfirmNo();
            else if (e.is<Event::Mouse>()) return Component<ChatApp>::OnEvent(e);
            return true;
        }
        if (show_keys) {
            if (e == Event::Escape() || Is(e, "help") || e == Event::F1() || e == Event::Return() || Letter(e) == '?') CloseKeys();
            else if (e.is<Event::Mouse>()) return Component<ChatApp>::OnEvent(e);
            return true;
        }
        if (!asking && !show_settings &&
            (Is(e, "help") || e == Event::F1() || (!busy && input.empty() && !side_focus && Letter(e) == '?'))) {
            show_keys = true;
            return true;
        }
        if (show_settings && !asking) return SettingsKey(e);
        if (!asking && !busy && host.settings && !side_focus && Is(e, "settings")) {
            OpenSettings();
            return true;
        }
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
        // Choosing a session with the keyboard: type to filter, Up/Down, Enter opens, Delete removes, Esc closes.
        if (side_focus) {
            const int letter = Letter(e);
            if (e == Event::ArrowDown()) MoveSide(1);
            else if (e == Event::ArrowUp()) MoveSide(-1);
            else if (e == Event::Return()) {
                side_focus = false;
                side_filter.clear();
                Pick(std::to_string(side_sel));
                RebuildSide();
            } else if (e == Event::Escape() || Is(e, "sessions")) {
                side_focus = false;
                side_drawer = false;
                side_filter.clear();
                RebuildSide();
            } else if (e == Event::Delete()) {
                DeleteSelected();
            } else if (e == Event::Backspace()) {
                if (!side_filter.empty()) side_filter.pop_back();
                RebuildSide();
            } else if (letter >= 32 && letter < 127) {
                side_filter += static_cast<char>(letter);
                RebuildSide();
                MoveSide(0);
            } else if (e.is<Event::Mouse>()) {
                return Component<ChatApp>::OnEvent(e);
            }
            Refresh();
            return true;
        }
        if (Is(e, "sessions")) {
            if (sessions.empty() || busy) return true;
            side_focus = true;
            side_hide = false;
            side_drawer = true;
            side_sel = 0;
            for (size_t i = 0; i < sessions.size(); ++i)
                if (sessions[i].active) side_sel = static_cast<int>(i);
            RebuildSide();
            Refresh();
            return true;
        }
        if (Is(e, "new_chat")) {
            if (!busy) NewChat();
            return true;
        }
        if (Is(e, "sidebar")) {
            ToggleSide();
            return true;
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
        if (Is(e, "commands") && !busy) {
            SetInput("/");
            return true;
        }
        if (Is(e, "quit") && !busy) {
            if (g_screen) g_screen->Exit();
            return true;
        }
        if (e == Event::PageUp()) return Scroll(-12);
        if (e == Event::PageDown()) return Scroll(12);
        if (auto mouse = e.get_if<Event::Mouse>()) {
            // The wheel scrolls the conversation by hand: while the answer streams the offset is pinned to the end, and the component
            // would scroll from there.
            if (mouse->button() == Event::Mouse::WheelUp) return Scroll(-3);
            if (mouse->button() == Event::Mouse::WheelDown) return Scroll(3);
            return Component<ChatApp>::OnEvent(e);
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
                AddRow({TuiKind::User, "", answer.empty() ? std::string("(default)") : answer, "row user", 0});
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

    bool Scroll(int rows) { // negative: up
        auto t = QueryElement("#t");
        if (rows < 0) follow = false;
        int current = t.scroll_y();
        if (current >= kPinned) current = known_bottom; // pinned to the end: start from where the bottom was
        if (rows > 0 && current + rows >= known_bottom) {
            follow = true; // back at the bottom: follow the newest output again
            ScrollToEnd();
        } else {
            t.SetScrollY(std::max(0, std::min(current + rows, known_bottom)));
        }
        Refresh();
        return true;
    }

    std::string_view view = R"html(
    <div class="shell">
      <div class="{side_class}">
        <div class="sback" onclick="ToggleSide">{ico_back}</div>
        <div class="sbrand"><for each="{banner}" as="ln"><div class="{ln.cls}">{ln.text}</div></for></div>
        <div class="newchat" onclick="NewChat"><span class="nico">+</span><span class="nlabel">New chat</span></div>
        <div class="slist">
          <for each="{side}" as="s"><div class="{s.cls}" onclick="Pick({s.idx})">{s.text}</div></for>
        </div>
        <div class="setbtn" onclick="OpenSettings"><span class="setico">{ico_settings}</span><span class="setlabel">Settings</span></div>
      </div>
      <div class="{app_class}">
        <div class="top">
          <span class="burger" onclick="ToggleSide">{ico_menu}</span>
          <span class="ttitle">{main_title}</span>
          <span class="act tdel" if="{has_messages}" onclick="AskDeleteSession">{ico_del}</span>
        </div>
        <div class="sp1"></div>
        <div class="transcript" id="t">
          <div if="{show_welcome}" class="welcome">
            <div class="small-brand">agent.cpp</div>
            <for each="{banner}" as="ln"><div class="{ln.cls}">{ln.text}</div></for>
            <div class="warn" if="{has_notice}">{notice}</div>
          </div>
          <for each="{rows}" as="r">
            <div class="{r.cls}">
              <span class="mark">{r.mark}</span>
              <div class="col">
                <div class="body">
                  <markdown if="{r.md}" content="{r.text}" stylesheet="{md_css}"/>
                  <span if="{r.plain}">{r.text}</span>
                </div>
                <div class="acts" if="{r.acts}">
                  <span class="act" onclick="CopyRow({r.id})">{ico_copy}</span>
                  <span class="act del" if="{r.user}" onclick="AskDelete({r.id})">{ico_del}</span>
                  <span class="rmeta">{r.meta}</span>
                </div>
              </div>
            </div>
          </for>
          <div class="busy">{busy_line}</div>
        </div>
        <div class="composer">
          <div if="{asking}" class="ask"><span class="allow">allow</span><span class="q">{question}</span></div>
          <div if="{show_palette}" class="palette">
            <for each="{matches}" as="m"><div class="{m.cls}"><span class="pname">{m.name}</span><span class="phelp">{m.help}</span></div></for>
          </div>
          <div if="{prompting}" class="promptq">? {prompt_line}</div>
          <div class="{box_class}"><span class="{mic_class}" onclick="ToggleMic">{ico_mic}</span><input value="{input}" placeholder="Message agent.cpp" autofocus=""/></div>
          <div class="hint">{hint}</div>
        </div>
        <div class="sp2"></div>
        <div class="status">
          <span class="sl"><span class="tb cpu">{tb_cpu}</span><span class="tb ram">{tb_ram}</span><span class="tb files">{tb_files}</span></span>
          <span class="sr"><span class="tb engine">{tb_engine}</span><span class="tb models">{tb_models}</span><span class="tb mode">{tb_mode}</span><span class="{gauge_class}">{tb_ctx}</span></span>
        </div>
      </div>
      <div if="{show_settings}" class="settings">
        <div class="sethead"><span class="setback" onclick="SettingsBack">{ico_back}</span><span class="setname">Settings</span><span class="act" onclick="CloseSettings">{ico_del}</span></div>
        <div class="{set_view}">
          <div class="setside"><for each="{cat_rows}" as="c"><div class="{c.cls}" onclick="PickCat({c.idx})">{c.text}</div></for></div>
          <div class="setmain">
            <div class="setcattitle">{cat_title}</div>
            <div class="setlist"><for each="{set_rows}" as="r"><div class="{r.cls}"><span class="slabel">{r.label}</span><span class="sval">{r.value}</span></div></for></div>
            <div class="setnote">{set_note}</div>
          </div>
        </div>
        <div class="sethelp">{set_help}</div>
      </div>
      <dialog open="{confirm_open}" title="{confirm_title}">
        <p>{confirm_text}</p>
        <div class="dlgactions"><span class="btn danger" onclick="ConfirmYes">y Delete  </span><span class="btn" onclick="ConfirmNo">n Cancel  </span></div>
      </dialog>
      <dialog open="{show_keys}" title="Keyboard shortcuts">
        <for each="{key_rows}" as="k"><div class="keyrow"><span class="kname">{k.label}</span><span class="kdesc">{k.value}</span></div></for>
        <div class="dlgactions"><span class="btn" onclick="CloseKeys">Esc Close  </span></div>
      </dialog>
      <toast open="{toast_open}">{toast_text}</toast>
    </div>

    <style>
      .shell { display: flex; flex-direction: row; width: 100%; height: 100%; background-color: #000000; color: #ffffff; border: round; border-color: #40e0d0; }
      .side { width: 33; flex-shrink: 0; display: flex; flex-direction: column; border-right: solid; border-color: #4a4a4a; padding-left: 1; padding-right: 1; }
      .hidden { display: none; }
      .burger, .small-brand, .sback { display: none; }
      .small-brand { color: #ffb3d1; font-weight: bold; }
      .sbrand { padding-bottom: 1; }
      .newchat { display: flex; flex-direction: row; border: round; border-color: #4a4a4a; padding-left: 1; padding-right: 1; margin-bottom: 1; color: #ffffff; }
      .newchat:hover { border-color: #40e0d0; }
      .nico { color: #ffb3d1; font-weight: bold; padding-right: 1; }
      .slist { flex-grow: 1; flex-basis: 0; min-height: 0; overflow-y: scroll; scrollbar-width: none; }
      .proj { color: #40e0d0; font-weight: bold; margin-top: 1; }
      .sess { color: #8a8a8a; padding-left: 1; white-space: nowrap; }
      .sess:hover { color: #ffffff; }
      .sess.current { color: #ffb3d1; font-weight: bold; }
      .sess.hi { background-color: #40e0d0; color: #000000; font-weight: bold; }
      .setbtn { display: flex; flex-direction: row; color: #ffffff; margin-top: 1; border: round; border-color: #4a4a4a; padding-left: 1; padding-right: 1; }
      .setbtn:hover { border-color: #40e0d0; }
      .setico { color: #40e0d0; padding-right: 1; }
      .settings { position: absolute; left: 0; right: 0; top: 0; bottom: 0; z-index: 8; background-color: #000000; border: round; border-color: #40e0d0;
                  display: flex; flex-direction: column; padding-left: 2; padding-right: 2; }
      .sethead { display: flex; flex-direction: row; color: #40e0d0; font-weight: bold; border-bottom: solid; border-color: #4a4a4a; margin-bottom: 1; background-color: #000000; }
      .setname { flex-grow: 1; background-color: #000000; }
      .setback { display: none; padding-right: 2; }
      .setbody { display: flex; flex-direction: row; flex-grow: 1; flex-basis: 0; min-height: 0; }
      .setside { width: 22; flex-shrink: 0; display: flex; flex-direction: column; border-right: solid; border-color: #4a4a4a; padding-right: 1; }
      .setcat { color: #8a8a8a; padding-left: 1; white-space: nowrap; }
      .setcat.cur { color: #ffffff; font-weight: bold; }
      .setcat.focus { background-color: #40e0d0; color: #000000; }
      .setmain { display: flex; flex-direction: column; flex-grow: 1; flex-basis: 0; min-width: 0; padding-left: 2; }
      .setcattitle { color: #ffb3d1; font-weight: bold; }
      .setlist { flex-grow: 1; flex-basis: 0; min-height: 0; overflow-y: scroll; scrollbar-width: none; }
      .setgroup { color: #8a8a8a; margin-top: 1; }
      .setrow { display: flex; flex-direction: row; color: #ffffff; padding-left: 1; }
      .setrow .slabel { width: 28; flex-shrink: 0; }
      .setrow .sval { flex-grow: 1; color: #40e0d0; white-space: nowrap; }
      .setrow.ro .sval { color: #8a8a8a; }
      .setrow.cur { background-color: #40e0d0; color: #000000; font-weight: bold; }
      .setrow.cur .sval { color: #000000; }
      .setnote { color: #8a8a8a; border-top: solid; border-color: #4a4a4a; }
      .sethelp { color: #8a8a8a; }
      .main { flex-grow: 1; min-width: 0; display: flex; flex-direction: column; }
      .sp1, .sp2 { flex-grow: 0; }
      .home .sp1, .home .sp2 { flex-grow: 1; }
      .top { display: flex; flex-direction: row; border-bottom: solid; border-color: #4a4a4a; padding-left: 2; padding-right: 2; }
      .home .top { display: none; }
      .ttitle { color: #ffffff; font-weight: bold; flex-grow: 1; white-space: nowrap; }
      .act { color: #8a8a8a; padding-left: 1; padding-right: 1; }
      .act:hover { color: #40e0d0; }
      .act.del:hover, .tdel:hover { color: #ffb3d1; }
      .transcript { flex-grow: 1; flex-shrink: 1; flex-basis: 0; min-height: 0; overflow-y: scroll; padding-left: 2; padding-right: 2; padding-top: 1; scrollbar-color: #40e0d0 #000000; }
      .home .transcript { flex-grow: 0; flex-shrink: 0; flex-basis: auto; overflow-y: visible; }
      .welcome { display: flex; flex-direction: column; margin-bottom: 1; }
      .welcome > div { width: 100%; text-align: center; }
      .banner { white-space: pre; }
      .pink { color: #ffb3d1; font-weight: bold; }
      .neon { color: #40e0d0; }
      .warn { color: #ffb3d1; margin-top: 1; }
      .row { display: flex; flex-direction: row; margin-bottom: 1; }
      .row .mark { width: 2; color: #40e0d0; }
      .row .col { display: flex; flex-direction: column; flex-grow: 1; min-width: 0; }
      .row .body { white-space: pre-wrap; }
      .acts { display: flex; flex-direction: row; justify-content: flex-end; }
      .assistant .acts { justify-content: flex-start; }
      .rmeta { color: #8a8a8a; padding-left: 1; }
      .user { padding-left: 12; }
      .user .mark { display: none; }
      .user .body { border: round; border-color: #4a4a4a; padding-left: 1; padding-right: 1; color: #ffffff; }
      .assistant { margin-bottom: 0; }
      .assistant .mark { color: #ffb3d1; font-weight: bold; }
      .assistant .body { white-space: normal; }
      .tool { margin-bottom: 0; }
      .tool .body { color: #8a8a8a; }
      .result { margin-bottom: 1; }
      .result .mark { width: 2; }
      .result .body { color: #40e0d0; }
      .result.failed .body { color: #ffb3d1; }
      .error .mark, .error .body { color: #ffb3d1; }
      .note .body { color: #8a8a8a; }
      .busy { color: #ffb3d1; padding-left: 2; }
      .composer { padding-left: 2; padding-right: 2; padding-bottom: 0; }
      .home .composer { width: 76%; align-self: center; padding-left: 0; padding-right: 0; }
      .box { display: flex; flex-direction: row; border: round; border-color: #4a4a4a; padding-left: 1; padding-right: 1; }
      .box.active { border-color: #40e0d0; }
      .mic { color: #ffffff; background-color: #4a4a4a; width: 5; padding-left: 1; padding-right: 1; margin-right: 1; }
      .mic:hover { background-color: #40e0d0; color: #000000; }
      .mic.rec { background-color: #ffb3d1; color: #000000; font-weight: bold; }
      .box input { flex-grow: 1; background-color: #000000; color: #ffffff; padding-left: 0; }
      .box input:focus, .box input:hover { background-color: #000000; color: #ffffff; }
      .box input::part(cursor), .box input:focus::part(cursor), .box input::part(cursor-focused),
      .box input:focus::part(cursor-focused) { color: #ffffff; }
      .box input::part(selection), .box input:focus::part(selection) { background-color: #40e0d0; color: #000000; }
      .hint { color: #8a8a8a; padding-left: 1; white-space: nowrap; overflow-x: hidden; }
      .ask { border: round; border-color: #40e0d0; padding-left: 1; padding-right: 1; margin-bottom: 0; }
      .allow { color: #ffb3d1; font-weight: bold; padding-right: 1; }
      .q { flex-grow: 1; white-space: pre-wrap; }
      .palette { border: round; border-color: #4a4a4a; padding-left: 1; padding-right: 1; }
      .pitem { display: flex; flex-direction: row; color: #40e0d0; }
      .pitem .pname { width: 14; }
      .pitem .phelp { color: #ffffff; }
      .current .pname { color: #ffb3d1; font-weight: bold; }
      .promptq { color: #ffb3d1; font-weight: bold; padding-left: 1; }
      .status { display: flex; flex-direction: row; padding-left: 2; padding-right: 2; color: #8a8a8a; border-top: solid; border-color: #4a4a4a; }
      .sl { display: flex; flex-direction: row; flex-grow: 1; min-width: 0; white-space: nowrap; overflow-x: hidden; }
      .sr { display: flex; flex-direction: row; flex-shrink: 0; white-space: nowrap; }
      .tb { padding-right: 1; white-space: nowrap; flex-shrink: 0; }
      .sr .tb { padding-left: 1; padding-right: 0; color: #8a8a8a; }
      .gauge { color: #40e0d0; padding-left: 2; white-space: nowrap; }
      .gauge.hot { color: #ffb3d1; }
      dialog::part(dialog-box) { border: round; border-color: #40e0d0; background-color: #000000; color: #ffffff; min-width: 56; max-width: 90; }
      dialog::part(dialog-header) { color: #40e0d0; border-color: #4a4a4a; }
      toast::part(toast) { border-color: #40e0d0; background-color: #000000; color: #ffffff; }
      .box input::part(placeholder) { color: #8a8a8a; }
      .dlgactions { display: flex; flex-direction: row; justify-content: flex-end; margin-top: 1; }
      .btn { border: round; border-color: #4a4a4a; padding-left: 1; padding-right: 0; margin-left: 1; color: #ffffff; white-space: nowrap; flex-shrink: 0; }
      .btn:hover { border-color: #40e0d0; color: #40e0d0; }
      .btn.danger { color: #ffb3d1; }
      .btn.danger:hover { border-color: #ffb3d1; color: #ffb3d1; }
      .keyrow { display: flex; flex-direction: row; }
      .kname { width: 20; color: #40e0d0; }
      .kdesc { color: #ffffff; }

      /* Below 84 columns the sidebar is a screen of its own: the list opens over the conversation with the menu button, and picking a session
         goes back to the conversation, like a phone app. */
      @media (max-width: 83) {
        .shell { position: relative; }
        .side { display: none; }
        .side.drawer { display: flex; position: absolute; left: 0; top: 0; bottom: 0; background-color: #000000; z-index: 5; }
        .burger { display: inline; color: #40e0d0; font-weight: bold; padding-right: 2; }
        .sback { display: block; color: #40e0d0; margin-bottom: 1; }
        .home .top { display: flex; border-bottom: none; }
        .home .ttitle, .home .tdel { display: none; }
        .tb.files { display: none; }
      }
      @media (max-width: 69) {
        .side.drawer { width: 100%; border-right: none; }
        .setback { display: inline; }
        .setside { width: 100%; border-right: none; }
        .setbody.cats .setmain { display: none; }
        .setbody.content .setside { display: none; }
        .setmain { padding-left: 0; }
        .setrow .slabel { width: 25; }
        .transcript { padding-left: 1; padding-right: 1; }
        .composer { padding-left: 1; padding-right: 1; }
        .status { padding-left: 1; padding-right: 1; }
        .user { padding-left: 3; }
        .home .composer { width: 100%; padding-left: 1; padding-right: 1; }
        .tb.models, .tb.engine { display: none; }
      }
      @media (max-width: 49) {
        .row .mark { width: 1; }
        .assistant .mark { display: none; }
        .tb.cpu, .tb.mode { display: none; }
        .kname { width: 10; }
      }
      @media (max-width: 35) {
        .banner { display: none; }
        .small-brand { display: block; }
        .hint { display: none; }
        .tb.ram { display: none; }
      }
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

std::string render_chat_text(const TuiHost &host, int width, int height) {
    auto app = rtxui::Ref<ChatApp>::New(host);
    rtxui::HeadlessScreen screen(app, width, height);
    return screen.Text();
}

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
    rtxui::PostTask([] {
        if (g_app) g_app->StatsTick();
    });
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
