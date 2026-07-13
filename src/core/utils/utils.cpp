#include "utils.hpp"

#include "hal/log_file.hpp"

#include <cstdio>
#include <mutex>

namespace agent::utils {

void log(const Session &session, LogLevel level, std::string_view message) {
    if (session.config.logger)
        session.config.logger(level, message, session.config.logger_user_data);

    if (!session.log_file)
        return;

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
        case '"':
            dst.append("\\\"");
            break;
        case '\\':
            dst.append("\\\\");
            break;
        case '\b':
            dst.append("\\b");
            break;
        case '\f':
            dst.append("\\f");
            break;
        case '\n':
            dst.append("\\n");
            break;
        case '\r':
            dst.append("\\r");
            break;
        case '\t':
            dst.append("\\t");
            break;
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

std::string escape_json_string(std::string_view src) {
    std::string dst;
    dst.reserve(src.size());
    escape_json_string(src, dst);
    return dst;
}

} // namespace agent::utils
