#include "chat_extras.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <simdjson.h>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

constexpr size_t kMaxMentionFile = 32 * 1024;
constexpr size_t kMaxMentionTotal = 96 * 1024;

bool is_path_char(char c) {
    return !std::isspace(static_cast<unsigned char>(c)) && c != ',' && c != ';' && c != ')' && c != '(' && c != '"' && c != '\'';
}

std::string lower_copy(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// How many characters at the end of `text` could be the start of `tag`.
size_t partial_suffix(const std::string &text, std::string_view tag) {
    const size_t max = std::min(text.size(), tag.size() - 1);
    for (size_t n = max; n > 0; --n)
        if (text.compare(text.size() - n, n, tag, 0, n) == 0) return n;
    return 0;
}

} // namespace

std::string expand_mentions(std::string_view text, const std::string &workspace, std::vector<std::string> *attached) {
    std::string appended;
    size_t total = 0;
    std::vector<std::string> seen;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '@' || (i > 0 && !std::isspace(static_cast<unsigned char>(text[i - 1])))) continue;
        size_t end = i + 1;
        while (end < text.size() && is_path_char(text[end])) ++end;
        std::string path(text.substr(i + 1, end - i - 1));
        while (!path.empty() && (path.back() == '.' || path.back() == ':')) path.pop_back(); // "see @a.txt." ends a sentence
        if (path.empty() || std::find(seen.begin(), seen.end(), path) != seen.end()) continue;
        fs::path p(path);
        if (p.is_relative()) p = fs::path(workspace) / p;
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) continue;
        const uintmax_t size = fs::file_size(p, ec);
        if (ec || total >= kMaxMentionTotal) continue;
        std::ifstream in(p, std::ios::binary);
        std::string body(std::min<uintmax_t>(size, kMaxMentionFile), '\0');
        in.read(body.data(), static_cast<std::streamsize>(body.size()));
        body.resize(static_cast<size_t>(in.gcount()));
        if (body.find('\0') != std::string::npos) continue;
        total += body.size();
        seen.push_back(path);
        appended += "\n\nFile " + path + (size > kMaxMentionFile ? " (first 32 KiB)" : "") + ":\n```\n" + body +
                    (body.empty() || body.back() != '\n' ? "\n" : "") + "```";
        if (attached) attached->push_back(path);
    }
    return std::string(text) + appended;
}

std::string html_to_text(std::string_view html) {
    std::string out;
    out.reserve(html.size() / 2);
    size_t i = 0;
    auto skip_block = [&](const char *name) {
        // At "<script ..." or "<style ...": skip to the closing tag.
        const std::string close = std::string("</") + name;
        const size_t at = lower_copy(html.substr(i)).find(close);
        if (at == std::string::npos) {
            i = html.size();
            return;
        }
        const size_t gt = html.find('>', i + at);
        i = gt == std::string_view::npos ? html.size() : gt + 1;
    };
    while (i < html.size()) {
        if (html[i] == '<') {
            const std::string head = lower_copy(html.substr(i, 8));
            if (head.rfind("<script", 0) == 0) {
                skip_block("script");
                continue;
            }
            if (head.rfind("<style", 0) == 0) {
                skip_block("style");
                continue;
            }
            if (html.compare(i, 4, "<!--") == 0) {
                const size_t end = html.find("-->", i + 4);
                i = end == std::string_view::npos ? html.size() : end + 3;
                continue;
            }
            const size_t gt = html.find('>', i);
            const bool block = head.rfind("<p", 0) == 0 || head.rfind("<br", 0) == 0 || head.rfind("<div", 0) == 0 ||
                               head.rfind("<li", 0) == 0 || head.rfind("<h", 0) == 0 || head.rfind("</p", 0) == 0 ||
                               head.rfind("</div", 0) == 0 || head.rfind("<tr", 0) == 0;
            if (block && !out.empty() && out.back() != '\n') out += '\n';
            i = gt == std::string_view::npos ? html.size() : gt + 1;
            continue;
        }
        if (html[i] == '&') {
            static const std::pair<std::string_view, char> entities[] = {{"&amp;", '&'},  {"&lt;", '<'},   {"&gt;", '>'},
                                                                         {"&quot;", '"'}, {"&#39;", '\''}, {"&nbsp;", ' '}};
            bool done = false;
            for (const auto &[name, ch] : entities)
                if (html.compare(i, name.size(), name) == 0) {
                    out += ch;
                    i += name.size();
                    done = true;
                    break;
                }
            if (done) continue;
        }
        out += html[i++];
    }
    // Collapse runs of spaces and blank lines.
    std::string clean;
    clean.reserve(out.size());
    int newlines = 0;
    bool space = false;
    for (const char c : out) {
        if (c == '\n') {
            if (newlines < 2) clean += '\n';
            ++newlines;
            space = false;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!space && newlines == 0 && !clean.empty()) clean += ' ';
            space = true;
        } else {
            clean += c;
            newlines = 0;
            space = false;
        }
    }
    const size_t first = clean.find_first_not_of(" \n");
    const size_t last = clean.find_last_not_of(" \n");
    return first == std::string::npos ? std::string() : clean.substr(first, last - first + 1);
}

std::string pretty_arguments(std::string_view json, int max_lines, size_t max_chars) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    simdjson::dom::element doc;
    simdjson::dom::object obj;
    if (parser.parse(padded).get(doc) || doc.get(obj)) {
        std::string text(json.substr(0, max_chars));
        return text;
    }
    std::string out;
    int lines = 0;
    bool cut = false;
    for (auto field : obj) {
        std::string value;
        if (field.value.is_string()) value = std::string(field.value.get_string().value_unsafe());
        else value = simdjson::minify(field.value);
        // A long or multi-line value shows its first lines only.
        std::string shown;
        int value_lines = 0;
        size_t at = 0;
        while (at <= value.size() && value_lines < 4) {
            size_t nl = value.find('\n', at);
            if (nl == std::string::npos) nl = value.size();
            std::string line = value.substr(at, nl - at);
            if (line.size() > 160) line = line.substr(0, 157) + "...";
            shown += (value_lines ? "\n    " : "") + line;
            ++value_lines;
            at = nl + 1;
        }
        if (at < value.size()) shown += "\n    ...";
        if (lines >= max_lines || out.size() + shown.size() > max_chars) {
            cut = true;
            break;
        }
        out += std::string(field.key) + ": " + shown + "\n";
        lines += 1 + value_lines;
    }
    if (cut) out += "...\n";
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

std::string filter_think(ThinkFilter &f, std::string_view token) {
    constexpr std::string_view kOpen = "<think>";
    constexpr std::string_view kClose = "</think>";
    if (f.show) return std::string(token);
    f.pending += token;
    std::string out;
    for (;;) {
        if (!f.inside) {
            const size_t pos = f.pending.find(kOpen);
            if (pos != std::string::npos) {
                out += f.pending.substr(0, pos);
                f.pending.erase(0, pos + kOpen.size());
                f.inside = true;
                continue;
            }
            const size_t keep = partial_suffix(f.pending, kOpen);
            out += f.pending.substr(0, f.pending.size() - keep);
            f.pending.erase(0, f.pending.size() - keep);
            return out;
        }
        const size_t pos = f.pending.find(kClose);
        if (pos != std::string::npos) {
            f.pending.erase(0, pos + kClose.size());
            while (!f.pending.empty() && (f.pending.front() == '\n' || f.pending.front() == ' ')) f.pending.erase(0, 1);
            f.inside = false;
            continue;
        }
        const size_t keep = partial_suffix(f.pending, kClose);
        f.pending.erase(0, f.pending.size() - keep);
        return out;
    }
}

std::string flush_think(ThinkFilter &f) {
    // What is held back outside a block was only a possible tag start: it is text after all. Inside a block it stays hidden.
    std::string rest = f.inside ? std::string() : f.pending;
    f.pending.clear();
    f.inside = false;
    return rest;
}

bool parse_chat_mode(std::string_view text, ChatMode &mode) {
    const std::string t = lower_copy(text);
    if (t == "ask") mode = ChatMode::Ask;
    else if (t == "plan") mode = ChatMode::Plan;
    else if (t == "auto") mode = ChatMode::Auto;
    else return false;
    return true;
}

const char *chat_mode_name(ChatMode mode) { return mode == ChatMode::Plan ? "plan" : mode == ChatMode::Auto ? "auto" : "ask"; }

bool tool_changes_things(std::string_view tool) {
    return tool == "write_file" || tool == "append_file" || tool == "delete_path" || tool == "move_path" || tool == "create_dir" ||
           tool == "run_command";
}

std::vector<std::string> load_always_allowed(const std::string &workspace) {
    std::vector<std::string> tools;
    std::ifstream in(fs::path(workspace) / ".agents" / "permissions.json", std::ios::binary);
    if (!in) return tools;
    const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    simdjson::dom::array list;
    if (parser.parse(padded).get(doc) || doc["always"].get(list)) return tools;
    for (simdjson::dom::element e : list)
        if (e.is_string()) tools.emplace_back(e.get_string().value_unsafe());
    return tools;
}

bool save_always_allowed(const std::string &workspace, const std::vector<std::string> &tools) {
    std::error_code ec;
    const fs::path dir = fs::path(workspace) / ".agents";
    fs::create_directories(dir, ec);
    std::string json = "{\"always\":[";
    for (size_t i = 0; i < tools.size(); ++i) json += (i ? ",\"" : "\"") + tools[i] + "\"";
    json += "]}\n";
    std::ofstream out(dir / "permissions.json", std::ios::binary | std::ios::trunc);
    out << json;
    return static_cast<bool>(out);
}

std::string remove_conversation_id(std::string_view index, std::string_view id) {
    std::string out;
    size_t at = 0;
    while (at < index.size()) {
        size_t end = index.find('\n', at);
        if (end == std::string_view::npos) end = index.size();
        const std::string_view line = index.substr(at, end - at);
        if (!line.empty() && line != id) (out += line) += '\n';
        at = end + 1;
    }
    return out;
}

} // namespace agent::cli
