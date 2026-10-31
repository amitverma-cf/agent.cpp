// chunk_text: size-bounded chunks that prefer paragraph, then line / sentence, then word boundaries.

#include <agent-cpp/agent.hpp>

namespace agent {

namespace {

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool is_utf8_continuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

size_t skip_space(std::string_view t, size_t i) {
    while (i < t.size() && is_space(t[i])) ++i;
    return i;
}

// A blank line ends right before b: t[b-1] is '\n' and the previous non-blank character is '\n' too.
bool paragraph_break(std::string_view t, size_t start, size_t b) {
    if (t[b - 1] != '\n' || b < start + 2) return false;
    size_t j = b - 2;
    while (j > start && (t[j] == ' ' || t[j] == '\t' || t[j] == '\r')) --j;
    return t[j] == '\n';
}

bool line_break(std::string_view t, size_t, size_t b) { return t[b - 1] == '\n'; }

bool sentence_break(std::string_view t, size_t start, size_t b) {
    return b >= start + 2 && is_space(t[b - 1]) && (t[b - 2] == '.' || t[b - 2] == '!' || t[b - 2] == '?');
}

bool word_break(std::string_view t, size_t, size_t b) { return is_space(t[b - 1]) || is_space(t[b]); }

// End (exclusive) of the chunk starting at start, given that t[start, hi) is the largest allowed
// window and hi < t.size(). Searches the back half of the window for the strongest boundary.
size_t find_break(std::string_view t, size_t start, size_t hi) {
    using BreakFn = bool (*)(std::string_view, size_t, size_t);
    static constexpr BreakFn kBreaks[] = {paragraph_break, line_break, sentence_break, word_break};
    const size_t lo = start + (hi - start) / 2;
    for (BreakFn is_break : kBreaks)
        for (size_t b = hi; b > lo; --b)
            if (is_break(t, start, b)) return b;
    size_t b = hi; // no boundary: cut, but not inside a UTF-8 sequence
    while (b > start + 1 && is_utf8_continuation(t[b])) --b;
    return b;
}

std::string_view trimmed(std::string_view t, size_t begin, size_t end) {
    while (begin < end && is_space(t[begin])) ++begin;
    while (end > begin && is_space(t[end - 1])) --end;
    return t.substr(begin, end - begin);
}

} // namespace

Result<std::vector<std::string_view>> chunk_text(std::string_view text, const ChunkOptions &options) {
    using R = std::vector<std::string_view>;
    if (options.size == 0) return fail<R>(ErrorCode::InvalidConfig, "chunk_text: size must be > 0.");
    if (options.overlap >= options.size) return fail<R>(ErrorCode::InvalidConfig, "chunk_text: overlap must be smaller than size.");

    R chunks;
    size_t pos = skip_space(text, 0);
    while (pos < text.size()) {
        if (text.size() - pos <= options.size) {
            if (auto c = trimmed(text, pos, text.size()); !c.empty()) chunks.push_back(c);
            break;
        }
        const size_t end = find_break(text, pos, pos + options.size);
        if (auto c = trimmed(text, pos, end); !c.empty()) chunks.push_back(c);

        size_t next = end;
        if (options.overlap > 0) {
            // Step back by up to overlap bytes, then forward to the start of a word.
            size_t back = end > options.overlap ? end - options.overlap : 0;
            if (back <= pos) back = pos + 1;
            while (back < end && !is_space(text[back - 1])) ++back;
            next = back;
        }
        pos = skip_space(text, next);
    }
    return ok(std::move(chunks));
}

} // namespace agent
