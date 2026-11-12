#include "schedule.hpp"

#include <cctype>
#include <charconv>

namespace agent::cli {

namespace {

bool all_digits(std::string_view s) {
    if (s.empty()) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}

bool to_int(std::string_view s, int &out) {
    if (!all_digits(s) || s.size() > 4) return false;
    out = 0;
    for (char c : s) out = out * 10 + (c - '0');
    return true;
}

} // namespace

Result<std::string> cron_from_every(std::string_view every) {
    if (every.size() < 2) return fail<std::string>(ErrorCode::InvalidConfig, "--every expects a number and a unit, such as 15m, 2h or 1d");
    const char unit = every.back();
    int n = 0;
    if (!to_int(every.substr(0, every.size() - 1), n) || n < 1)
        return fail<std::string>(ErrorCode::InvalidConfig, "--every expects a number and a unit, such as 15m, 2h or 1d");
    if (unit == 'm') {
        if (n >= 60 || 60 % n != 0)
            return fail<std::string>(ErrorCode::InvalidConfig,
                                     "--every Nm needs N to divide 60 (1, 2, 3, 5, 10, 15, 20, 30), or use hours");
        return ok(n == 1 ? std::string("* * * * *") : "*/" + std::to_string(n) + " * * * *");
    }
    if (unit == 'h') {
        if (n >= 24 || 24 % n != 0)
            return fail<std::string>(ErrorCode::InvalidConfig, "--every Nh needs N to divide 24 (1, 2, 3, 4, 6, 8, 12), or use --daily");
        return ok(n == 1 ? std::string("0 * * * *") : "0 */" + std::to_string(n) + " * * *");
    }
    if (unit == 'd') {
        if (n != 1) return fail<std::string>(ErrorCode::InvalidConfig, "--every accepts 1d only; use --cron for other day patterns");
        return ok(std::string("0 0 * * *"));
    }
    return fail<std::string>(ErrorCode::InvalidConfig, "--every unit must be m, h or d");
}

Result<std::string> cron_from_daily(std::string_view hh_mm) {
    const size_t colon = hh_mm.find(':');
    int h = 0, m = 0;
    if (colon == std::string_view::npos || !to_int(hh_mm.substr(0, colon), h) || !to_int(hh_mm.substr(colon + 1), m) || h > 23 || m > 59)
        return fail<std::string>(ErrorCode::InvalidConfig, "--daily expects a local time such as 09:00");
    return ok(std::to_string(m) + " " + std::to_string(h) + " * * *");
}

Result<std::string> check_cron(std::string_view expression) {
    std::vector<std::string_view> fields;
    size_t at = 0;
    while (at < expression.size()) {
        while (at < expression.size() && expression[at] == ' ') ++at;
        if (at >= expression.size()) break;
        size_t end = expression.find(' ', at);
        if (end == std::string_view::npos) end = expression.size();
        fields.push_back(expression.substr(at, end - at));
        at = end;
    }
    if (fields.size() != 5)
        return fail<std::string>(ErrorCode::InvalidConfig, "--cron expects five fields: minute hour day-of-month month day-of-week");
    std::string out;
    for (size_t i = 0; i < fields.size(); ++i) {
        for (char c : fields[i])
            if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '*' || c == '/' || c == ',' || c == '-'))
                return fail<std::string>(ErrorCode::InvalidConfig,
                                         "--cron field '" + std::string(fields[i]) + "' has a character cron does not accept");
        out += (i ? " " : "") + std::string(fields[i]);
    }
    return ok(std::move(out));
}

std::string quote_posix(std::string_view word) {
    bool plain = !word.empty();
    for (char c : word)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.' || c == '/' || c == ':' || c == '=' ||
              c == '+' || c == '@' || c == '%' || c == ','))
            plain = false;
    if (plain) return std::string(word);
    std::string out = "'";
    for (char c : word) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out + "'";
}

std::string quote_windows(std::string_view word) {
    if (!word.empty() && word.find_first_of(" \t\"") == std::string_view::npos) return std::string(word);
    std::string out = "\"";
    size_t backslashes = 0;
    for (char c : word) {
        if (c == '\\') {
            ++backslashes;
            continue;
        }
        if (c == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out += '"';
        } else {
            out.append(backslashes, '\\');
            out += c;
        }
        backslashes = 0;
    }
    out.append(backslashes * 2, '\\');
    return out + "\"";
}

namespace {
constexpr std::string_view kTag = "# agent.cpp:";
}

std::string crontab_block(std::string_view name, std::string_view expression, std::string_view command) {
    return std::string(kTag) + std::string(name) + "\n" + std::string(expression) + " " + std::string(command) + "\n";
}

std::vector<CronEntry> parse_crontab(std::string_view crontab) {
    std::vector<CronEntry> out;
    size_t at = 0;
    std::string pending;
    while (at <= crontab.size()) {
        size_t end = crontab.find('\n', at);
        if (end == std::string_view::npos) end = crontab.size();
        std::string_view line = crontab.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.substr(0, kTag.size()) == kTag) {
            pending = std::string(line.substr(kTag.size()));
        } else if (!pending.empty() && !line.empty() && line.front() != '#') {
            // five fields, then the command
            size_t pos = 0;
            for (int f = 0; f < 5; ++f) {
                while (pos < line.size() && line[pos] == ' ') ++pos;
                while (pos < line.size() && line[pos] != ' ') ++pos;
            }
            while (pos < line.size() && line[pos] == ' ') ++pos;
            std::string_view expr = line.substr(0, pos);
            while (!expr.empty() && expr.back() == ' ') expr.remove_suffix(1);
            out.push_back({pending, std::string(expr), std::string(line.substr(pos))});
            pending.clear();
        }
        if (end == crontab.size()) break;
        at = end + 1;
    }
    return out;
}

std::string crontab_without(std::string_view crontab, std::string_view name) {
    std::string out;
    size_t at = 0;
    bool skip_next = false;
    const std::string tag = std::string(kTag) + std::string(name);
    while (at < crontab.size()) {
        size_t end = crontab.find('\n', at);
        const bool has_nl = end != std::string_view::npos;
        if (!has_nl) end = crontab.size();
        std::string_view line = crontab.substr(at, end - at);
        std::string_view bare = line;
        if (!bare.empty() && bare.back() == '\r') bare.remove_suffix(1);
        if (skip_next) {
            skip_next = false; // the command line of the removed entry
        } else if (bare == tag) {
            skip_next = true;
        } else {
            out.append(line);
            if (has_nl) out += '\n';
        }
        at = end + 1;
    }
    return out;
}

bool valid_schedule_name(std::string_view name) {
    if (name.empty() || name.size() > 64) return false;
    for (char c : name)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) return false;
    return true;
}

} // namespace agent::cli
