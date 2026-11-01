#include "utils.hpp"

#include "hal/log_file.hpp"

#include <cstdio>
#include <mutex>

namespace agent::utils {

void log(const Session &session, LogLevel level, std::string_view message) {
    if (session.config.logger) session.config.logger(level, message, session.config.logger_user_data);

    if (!session.log_file) return;

    const char *lvl = (level == LogLevel::Debug)     ? "DEBUG"
                      : (level == LogLevel::Warning) ? "WARN"
                      : (level == LogLevel::Error)   ? "ERROR"
                                                     : "INFO";

    std::lock_guard<std::mutex> lock(*session.log_mutex);
    hal::write_log_line(session.log_file.get(), lvl, message);
}

void escape_json_string(std::string_view src, std::string &dst) {
    for (char c : src) {
        switch (c) {
        case '"': dst.append("\\\""); break;
        case '\\': dst.append("\\\\"); break;
        case '\b': dst.append("\\b"); break;
        case '\f': dst.append("\\f"); break;
        case '\n': dst.append("\\n"); break;
        case '\r': dst.append("\\r"); break;
        case '\t': dst.append("\\t"); break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                dst.append(buf);
            } else {
                dst.push_back(c);
            }
        }
    }
}

namespace {

// Length of the well-formed UTF-8 sequence starting at s[i], or 0 when it is malformed or cut off.
size_t utf8_sequence_length(std::string_view s, size_t i) {
    const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char c = b(i);
    if (c < 0x80) return 1;
    size_t n = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) n = 2;
    else if (c >= 0xE0 && c <= 0xEF) {
        n = 3;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F; // no surrogates
    } else if (c >= 0xF0 && c <= 0xF4) {
        n = 4;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;
    } else {
        return 0;
    }
    if (i + n > s.size()) return 0;
    if (b(i + 1) < lo || b(i + 1) > hi) return 0;
    for (size_t k = 2; k < n; ++k)
        if (b(i + k) < 0x80 || b(i + k) > 0xBF) return 0;
    return n;
}

} // namespace

size_t utf8_prefix_length(std::string_view s, size_t max) {
    if (max >= s.size()) return s.size();
    size_t cut = max;
    // Back up over continuation bytes to the start of the sequence that straddles the cut (at most 3).
    size_t back = 0;
    while (cut > 0 && back < 3 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) {
        --cut;
        ++back;
    }
    if (cut == 0 && (static_cast<unsigned char>(s[0]) & 0xC0) == 0x80) return max; // not UTF-8 anyway
    const size_t n = utf8_sequence_length(s, cut);
    return (n != 0 && cut + n <= max) ? cut + n : cut;
}

void sanitize_utf8(std::string &s) {
    size_t i = 0;
    while (i < s.size()) {
        const size_t n = utf8_sequence_length(s, i);
        if (n == 0) break;
        i += n;
    }
    if (i == s.size()) return; // already valid: no copy
    std::string out;
    out.reserve(s.size() + 16);
    out.append(s, 0, i);
    while (i < s.size()) {
        const size_t n = utf8_sequence_length(s, i);
        if (n == 0) {
            out += "\xEF\xBF\xBD";
            ++i;
        } else {
            out.append(s, i, n);
            i += n;
        }
    }
    s = std::move(out);
}

std::string escape_json_string(std::string_view src) {
    std::string dst;
    dst.reserve(src.size());
    escape_json_string(src, dst);
    return dst;
}

} // namespace agent::utils
