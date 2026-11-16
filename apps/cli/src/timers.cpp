#include "timers.hpp"

#include "cli_core.hpp"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <simdjson.h>

namespace agent::cli {

namespace {

std::mutex g_file_mutex;

std::string timers_path(const std::string &home_root) { return (std::filesystem::path(home_root) / "timers.json").string(); }

std::string when_text(int64_t epoch, int64_t now) {
    const int64_t d = epoch - now;
    if (d < 90) return "in about a minute";
    if (d < 3600) return "in " + std::to_string((d + 30) / 60) + " minutes";
    const std::time_t t = static_cast<std::time_t>(epoch);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &t);
#else
    localtime_r(&t, &local);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), d < 20 * 3600 ? "at %H:%M" : "on %Y-%m-%d at %H:%M", &local);
    return buf;
}

} // namespace

std::vector<Timer> timers_from_json(std::string_view json) {
    std::vector<Timer> out;
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    simdjson::dom::element doc;
    simdjson::dom::array list;
    if (parser.parse(padded).get(doc) || doc.get(list)) return out;
    for (simdjson::dom::element e : list) {
        Timer t;
        std::string_view id, prompt, workspace;
        if (e["id"].get(id) || e["prompt"].get(prompt) || e["next"].get(t.next_epoch)) continue;
        if (e["workspace"].get(workspace)) workspace = "";
        if (e["every"].get(t.every_seconds)) t.every_seconds = 0;
        t.id = std::string(id);
        t.prompt = std::string(prompt);
        t.workspace = std::string(workspace);
        out.push_back(std::move(t));
    }
    return out;
}

std::string timers_to_json(const std::vector<Timer> &timers) {
    std::string out = "[\n";
    for (size_t i = 0; i < timers.size(); ++i) {
        const Timer &t = timers[i];
        out += "  {\"id\": \"" + json_escape(t.id) + "\", \"prompt\": \"" + json_escape(t.prompt) + "\", \"workspace\": \"" +
               json_escape(t.workspace) + "\", \"next\": " + std::to_string(t.next_epoch) +
               ", \"every\": " + std::to_string(t.every_seconds) + "}" + (i + 1 < timers.size() ? "," : "") + "\n";
    }
    return out + "]\n";
}

std::vector<Timer> load_timers(const std::string &home_root) {
    std::lock_guard<std::mutex> lock(g_file_mutex);
    std::ifstream in(timers_path(home_root), std::ios::binary);
    const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return timers_from_json(body);
}

bool save_timers(const std::string &home_root, const std::vector<Timer> &timers) {
    std::lock_guard<std::mutex> lock(g_file_mutex);
    const std::string path = timers_path(home_root);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << timers_to_json(timers);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

int64_t seconds_until_clock(std::string_view hh_mm, int64_t now_epoch) {
    const size_t colon = hh_mm.find(':');
    if (colon == std::string_view::npos || colon == 0 || colon > 2 || hh_mm.size() - colon - 1 < 1 || hh_mm.size() - colon - 1 > 2)
        return -1;
    for (const char c : hh_mm)
        if (c != ':' && (c < '0' || c > '9')) return -1;
    const int h = std::atoi(std::string(hh_mm.substr(0, colon)).c_str());
    const int m = std::atoi(std::string(hh_mm.substr(colon + 1)).c_str());
    if (h > 23 || m > 59) return -1;
    const std::time_t now = static_cast<std::time_t>(now_epoch);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    const int64_t today = static_cast<int64_t>(local.tm_hour) * 3600 + local.tm_min * 60 + local.tm_sec;
    int64_t d = static_cast<int64_t>(h) * 3600 + m * 60 - today;
    if (d <= 0) d += 24 * 3600;
    return d;
}

std::vector<Timer> take_due_timers(std::vector<Timer> &timers, std::string_view workspace, int64_t now_epoch) {
    std::vector<Timer> due;
    for (auto it = timers.begin(); it != timers.end();) {
        if (it->next_epoch > now_epoch || (workspace != "*" && !it->workspace.empty() && it->workspace != workspace)) {
            ++it;
            continue;
        }
        due.push_back(*it);
        if (it->every_seconds > 0) {
            const int64_t missed = (now_epoch - it->next_epoch) / it->every_seconds + 1;
            it->next_epoch += missed * it->every_seconds;
            ++it;
        } else {
            it = timers.erase(it);
        }
    }
    return due;
}

std::string timer_tool(const std::string &home_root, const std::string &workspace, std::string_view arguments_json, int64_t now_epoch,
                       bool &failed) {
    failed = true;
    simdjson::dom::parser parser;
    simdjson::padded_string padded(arguments_json);
    simdjson::dom::element doc;
    std::string_view action;
    if (parser.parse(padded).get(doc) || doc["action"].get(action)) return "timer needs an action: add, list or remove";
    std::vector<Timer> timers = load_timers(home_root);
    if (action == "list") {
        failed = false;
        std::string out;
        for (const Timer &t : timers)
            if (t.workspace.empty() || t.workspace == workspace)
                out += "- " + t.id + ": \"" + t.prompt + "\" " + when_text(t.next_epoch, now_epoch) +
                       (t.every_seconds > 0 ? ", then every " + std::to_string(t.every_seconds / 60) + " minutes" : "") + "\n";
        return out.empty() ? "there are no timers" : out;
    }
    if (action == "remove") {
        std::string_view id;
        if (doc["id"].get(id)) return "timer remove needs the id (see timer list)";
        const size_t before = timers.size();
        std::erase_if(timers, [&](const Timer &t) { return t.id == id; });
        if (timers.size() == before) return "there is no timer with the id '" + std::string(id) + "'";
        if (!save_timers(home_root, timers)) return "the timers could not be saved";
        failed = false;
        return "removed the timer " + std::string(id);
    }
    if (action != "add") return "unknown timer action '" + std::string(action) + "' (add, list or remove)";
    std::string_view prompt, at;
    if (doc["prompt"].get(prompt) || prompt.empty()) return "timer add needs a prompt: what to do when it fires";
    double in_minutes = 0, every_minutes = 0;
    bool repeat = false;
    if (doc["in_minutes"].get(in_minutes)) in_minutes = 0;
    if (doc["every_minutes"].get(every_minutes)) every_minutes = 0;
    if (doc["repeat"].get(repeat)) repeat = false;
    if (doc["at"].get(at)) at = "";
    Timer t;
    t.prompt = std::string(prompt);
    t.workspace = workspace;
    if (!at.empty()) {
        const int64_t d = seconds_until_clock(at, now_epoch);
        if (d < 0) return "the time '" + std::string(at) + "' is not understood; use HH:MM, for example 09:00";
        t.next_epoch = now_epoch + d;
        t.every_seconds = repeat ? 24 * 3600 : 0;
    } else if (every_minutes > 0) {
        if (every_minutes < 1) return "a repeating timer needs at least one minute between runs";
        t.every_seconds = static_cast<int64_t>(every_minutes * 60);
        t.next_epoch = now_epoch + (in_minutes > 0 ? static_cast<int64_t>(in_minutes * 60) : t.every_seconds);
    } else if (in_minutes > 0) {
        t.next_epoch = now_epoch + static_cast<int64_t>(in_minutes * 60);
    } else {
        return "timer add needs when: in_minutes, every_minutes or at (HH:MM)";
    }
    int n = static_cast<int>(timers.size()) + 1;
    auto taken = [&](const std::string &id) {
        return std::any_of(timers.begin(), timers.end(), [&](const Timer &o) { return o.id == id; });
    };
    while (taken("t" + std::to_string(n))) ++n;
    t.id = "t" + std::to_string(n);
    timers.push_back(t);
    if (!save_timers(home_root, timers)) return "the timer could not be saved";
    failed = false;
    return "timer " + t.id + " set: \"" + t.prompt + "\" " + when_text(t.next_epoch, now_epoch) +
           (t.every_seconds > 0 ? ", then every " + std::to_string(t.every_seconds / 60) + " minutes" : "") +
           ". It fires while the chat is open (or the next time it is opened).";
}

} // namespace agent::cli
