#include "chat_memory.hpp"

#include "cli_core.hpp"

#include <algorithm>
#include <cctype>

namespace agent::cli {

namespace {

constexpr std::string_view kNoteIndex = "note_index";
constexpr std::string_view kNotePrefix = "note:";
constexpr std::string_view kChatIndex = "cli:conversations";
constexpr std::string_view kTitlePrefix = "cli:title:";

std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string one_line_cut(std::string s, size_t max) {
    for (char &c : s)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    const size_t a = s.find_first_not_of(' ');
    s = a == std::string::npos ? std::string() : s.substr(a);
    if (s.size() > max) s = s.substr(0, max - 3) + "...";
    return s;
}

int score_text(const std::string &haystack_lower, const std::vector<std::string> &words) {
    int score = 0;
    for (const std::string &w : words)
        if (haystack_lower.find(w) != std::string::npos) ++score;
    return score;
}

std::string snippet_around(const std::string &text, const std::vector<std::string> &words) {
    const std::string low = lower(text);
    size_t at = std::string::npos;
    for (const std::string &w : words) {
        const size_t p = low.find(w);
        if (p != std::string::npos && (at == std::string::npos || p < at)) at = p;
    }
    if (at == std::string::npos) return one_line_cut(text, 120);
    const size_t from = at > 50 ? at - 50 : 0;
    return (from ? "..." : "") + one_line_cut(text.substr(from, 140), 140);
}

std::string chat_title(Session &session, const std::string &id, const Conversation &c) {
    if (auto named = retrieve_memory(session, std::string(kTitlePrefix) + id); named.ok && !named.value.empty()) return named.value;
    for (const Message &m : c.history)
        if (m.role == "user" && !without_notes(m.content).empty()) return one_line_cut(std::string(without_notes(m.content)), 80);
    return id;
}

} // namespace

std::vector<std::string> search_words(std::string_view text) {
    std::vector<std::string> words;
    std::string w;
    for (const char c : std::string(text) + " ") {
        if (std::isalnum(static_cast<unsigned char>(c)) || (static_cast<unsigned char>(c) >= 0x80)) {
            w += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            if (w.size() >= 3 && std::find(words.begin(), words.end(), w) == words.end()) words.push_back(w);
            w.clear();
        }
    }
    return words;
}

std::vector<std::string> note_keys(Session &session) {
    std::vector<std::string> keys;
    auto index = retrieve_memory(session, kNoteIndex);
    if (!index.ok) return keys;
    size_t from = 0;
    while (from < index.value.size()) {
        size_t end = index.value.find('\n', from);
        if (end == std::string::npos) end = index.value.size();
        if (end > from) keys.push_back(index.value.substr(from, end - from));
        from = end + 1;
    }
    return keys;
}

Result<void> note_keep(Session &session, std::string_view key, std::string_view value) {
    if (auto r = store_memory(session, std::string(kNotePrefix) + std::string(key), value); !r.ok) return r;
    auto keys = note_keys(session);
    if (std::find(keys.begin(), keys.end(), std::string(key)) == keys.end()) {
        std::string index;
        for (const std::string &k : keys) index += k + "\n";
        index += std::string(key) + "\n";
        return store_memory(session, kNoteIndex, index);
    }
    return ok();
}

std::string note_forget(Session &session, std::string_view key, std::string_view text, bool &removed) {
    removed = false;
    auto keys = note_keys(session);
    auto drop = [&](const std::string &k) -> std::string {
        std::string old;
        if (auto v = retrieve_memory(session, std::string(kNotePrefix) + k); v.ok) old = v.value;
        if (auto r = store_memory(session, std::string(kNotePrefix) + k, ""); !r.ok)
            return "could not remove the note '" + k + "': " + r.error.message;
        std::string index;
        for (const std::string &other : keys)
            if (other != k) index += other + "\n";
        if (auto r = store_memory(session, kNoteIndex, index); !r.ok) return "could not update the list of notes: " + r.error.message;
        removed = true;
        return "removed the note '" + k + "'" + (old.empty() ? std::string() : " (it said: " + one_line_cut(old, 120) + ")");
    };
    if (keys.empty()) return "there are no notes to remove";
    if (!key.empty()) {
        for (const std::string &k : keys)
            if (k == key) return drop(k);
    }
    const std::string query = text.empty() ? std::string(key) : std::string(text);
    const auto words = search_words(query);
    std::vector<std::pair<int, std::string>> found;
    for (const std::string &k : keys) {
        std::string value;
        if (auto v = retrieve_memory(session, std::string(kNotePrefix) + k); v.ok) value = v.value;
        const int score = score_text(lower(k + " " + value), words);
        if (score > 0) found.emplace_back(score, k);
    }
    if (found.empty()) {
        std::string names;
        for (const std::string &k : keys) names += (names.empty() ? "" : ", ") + k;
        return "no note matches '" + query + "'; nothing was removed. The notes are: " + names;
    }
    std::stable_sort(found.begin(), found.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    if (found.size() > 1 && found[0].first == found[1].first) {
        std::string names;
        for (const auto &f : found) names += (names.empty() ? "" : ", ") + f.second;
        return "several notes match '" + query + "' equally (" + names + "); nothing was removed. Ask which one, then forget it by name";
    }
    return drop(found[0].second);
}

std::vector<ChatHit> search_chats(Session &session, std::string_view query, std::string_view skip_id, size_t &searched, size_t max_hits) {
    searched = 0;
    std::vector<ChatHit> hits;
    auto index = retrieve_memory(session, kChatIndex);
    const auto ids = conversation_ids(index.ok ? index.value : std::string());
    const auto words = search_words(query);
    for (auto it = ids.rbegin(); it != ids.rend(); ++it) { // newest first
        if (*it == skip_id) continue;
        Conversation c;
        c.id = *it;
        if (!load_conversation(session, c).ok) continue;
        ++searched;
        ChatHit hit;
        hit.id = *it;
        hit.title = chat_title(session, *it, c);
        if (words.empty()) {
            hit.snippet = one_line_cut(c.history.empty() ? std::string() : c.history.back().content, 120);
            hits.push_back(std::move(hit));
            if (hits.size() >= max_hits) break;
            continue;
        }
        hit.score = score_text(lower(hit.title), words) * 2;
        int best = 0;
        for (const Message &m : c.history) {
            if ((m.role != "user" && m.role != "assistant") || m.content.empty()) continue;
            const std::string said(without_notes(m.content)); // the notes are the same in every chat: they say nothing about this one
            const int s = score_text(lower(said), words);
            if (s > best) {
                best = s;
                hit.snippet = snippet_around(said, words);
            }
        }
        // The conversation as a whole: every word found anywhere counts once.
        std::string all = lower(hit.title);
        for (const Message &m : c.history)
            if ((m.role == "user" || m.role == "assistant")) all += " " + lower(std::string(without_notes(m.content)));
        hit.score = std::max(hit.score, score_text(all, words));
        if (hit.score > 0) {
            if (hit.snippet.empty()) hit.snippet = hit.title;
            hits.push_back(std::move(hit));
        }
    }
    std::stable_sort(hits.begin(), hits.end(), [](const ChatHit &a, const ChatHit &b) { return a.score > b.score; });
    if (hits.size() > max_hits) hits.resize(max_hits);
    return hits;
}

std::string_view without_notes(std::string_view message) {
    constexpr std::string_view open = "[Notes kept from earlier chats", close = "[End of notes]";
    if (!message.starts_with(open)) return message;
    const size_t end = message.find(close);
    if (end == std::string_view::npos) return message;
    message.remove_prefix(end + close.size());
    while (!message.empty() && (message.front() == 10 || message.front() == 13 || message.front() == 32)) message.remove_prefix(1);
    return message;
}

std::vector<std::pair<std::string, std::string>> memory_passages(Session &session, size_t max_chats) {
    std::vector<std::pair<std::string, std::string>> out;
    for (const std::string &k : note_keys(session)) {
        auto v = retrieve_memory(session, std::string(kNotePrefix) + k);
        if (v.ok && !v.value.empty()) out.emplace_back("note:" + k, k + ": " + v.value);
    }
    auto index = retrieve_memory(session, kChatIndex);
    const auto ids = conversation_ids(index.ok ? index.value : std::string());
    size_t chats = 0;
    for (auto it = ids.rbegin(); it != ids.rend() && chats < max_chats; ++it, ++chats) {
        Conversation c;
        c.id = *it;
        if (!load_conversation(session, c).ok) continue;
        for (size_t i = 0; i < c.history.size(); ++i) {
            const Message &m = c.history[i];
            const std::string_view text = without_notes(m.content);
            if ((m.role != "user" && m.role != "assistant") || text.empty()) continue;
            out.emplace_back("chat:" + c.id + ":" + std::to_string(i), m.role + ": " + std::string(text));
        }
    }
    return out;
}

std::string recall_text(Session &session, std::string_view key, std::string_view query, std::string_view current_id) {
    if (!key.empty()) {
        auto exact = retrieve_memory(session, std::string(kNotePrefix) + std::string(key));
        if (exact.ok && !exact.value.empty()) return exact.value;
    }
    const std::string what = query.empty() ? std::string(key) : std::string(query);
    const auto words = search_words(what);
    std::string out;
    int note_hits = 0;
    const auto keys = note_keys(session);
    if (!words.empty()) {
        for (const std::string &k : keys) {
            auto v = retrieve_memory(session, std::string(kNotePrefix) + k);
            if (!v.ok || v.value.empty()) continue;
            if (score_text(lower(k + " " + v.value), words) == 0) continue;
            if (note_hits++ == 0) out += "Notes that match:\n";
            out += "- " + k + ": " + one_line_cut(v.value, 300) + "\n";
        }
    } else if (!keys.empty()) {
        out += "Notes kept: ";
        for (size_t i = 0; i < keys.size(); ++i) out += (i ? ", " : "") + keys[i];
        out += "\n";
    }
    size_t searched = 0;
    const auto chats = search_chats(session, what, current_id, searched);
    out += "Searched " + std::to_string(searched) + " earlier chat" + (searched == 1 ? "" : "s") + " of this project";
    if (chats.empty()) {
        out += words.empty() ? ": there are none yet.\n" : ": nothing matches.\n";
    } else {
        out += words.empty() ? ", the latest:\n" : ", the best matches:\n";
        for (const ChatHit &h : chats) out += "- \"" + h.title + "\" (" + h.id + "): " + h.snippet + "\n";
    }
    if (note_hits == 0 && !words.empty()) out = "No note matches '" + what + "'.\n" + out;
    return out;
}

} // namespace agent::cli
