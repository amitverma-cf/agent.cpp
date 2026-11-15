// Scripted stand-in for the api engine's bun bridge (tests/unit/test_engine_api.cpp). Same line-delimited JSON protocol as
// src/engines/api/bridge.js, no network. Arguments:
//   --state FILE   persists the request counter across restarts of this process
//   --log FILE     appends every request line it receives
//   --fail N       the first N requests answer with a retryable error ("HTTP 503")
//   --bad-first    the first request answers with an empty reply (retryable "empty reply")
//   --partial      the first request streams "Hel" "lo " and then fails; the next streams "Hello " "world" and completes
//   --fatal        every request answers with a non-retryable error ("HTTP 401")
//   --die-first    the first request exits the process without an answer
//   --wait-cancel  requests are not answered until a cancel arrives; then the answer is a "cancelled" error
// Without a mode the bridge answers "echo: <last user text>"; a user text "call:NAME" answers a tool call to NAME instead, and a
// request that already holds a tool message answers "tool done".

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <simdjson.h>
#include <string>

namespace {

void send(const std::string &line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string esc(std::string_view s) {
    std::string out;
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

int bump(const std::string &state, int &memory) {
    if (state.empty()) return ++memory;
    int n = 0;
    {
        std::ifstream in(state);
        in >> n;
    }
    ++n;
    std::ofstream(state, std::ios::trunc) << n;
    return n;
}

std::string msg(const std::string &id, const std::string &type, const std::string &rest) {
    return "{\"t\":\"" + type + "\",\"id\":" + id + (rest.empty() ? "" : "," + rest) + "}";
}

std::string done(const std::string &id, const std::string &content, const std::string &calls, int prompt, int completion, int cached,
                 const char *finish) {
    return msg(id, "done",
               "\"content\":\"" + esc(content) + "\",\"tool_calls\":" + calls + ",\"usage\":{\"prompt_tokens\":" + std::to_string(prompt) +
                   ",\"completion_tokens\":" + std::to_string(completion) + ",\"cached_tokens\":" + std::to_string(cached) +
                   "},\"finish\":\"" + finish + "\"");
}

} // namespace

int main(int argc, char **argv) {
    std::string state, log;
    int fail_n = 0;
    bool bad_first = false, partial = false, fatal = false, die_first = false, wait_cancel = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--state" && i + 1 < argc) state = argv[++i];
        else if (a == "--log" && i + 1 < argc) log = argv[++i];
        else if (a == "--fail" && i + 1 < argc) fail_n = std::atoi(argv[++i]);
        else if (a == "--bad-first") bad_first = true;
        else if (a == "--partial") partial = true;
        else if (a == "--fatal") fatal = true;
        else if (a == "--die-first") die_first = true;
        else if (a == "--wait-cancel") wait_cancel = true;
    }
    send("{\"t\":\"ready\"}");
    int memory = 0;
    std::string line;
    int64_t waiting_id = -1;
    simdjson::dom::parser parser;
    while (std::getline(std::cin, line)) {
        simdjson::dom::element doc;
        if (parser.parse(line).get(doc)) continue;
        std::string_view op;
        if (doc["op"].get(op)) continue;
        if (op == "quit") return 0;
        int64_t id = 0;
        if (doc["id"].get(id)) id = 0;
        const std::string ids = std::to_string(id);
        if (op == "cancel") {
            if (waiting_id == id) send(msg(ids, "error", "\"message\":\"cancelled\",\"retryable\":false,\"cancelled\":true"));
            continue;
        }
        if (op != "chat") continue;
        if (!log.empty()) std::ofstream(log, std::ios::app) << line << "\n";
        const int n = bump(state, memory);
        if (die_first && n == 1) return 3;
        if (wait_cancel) {
            waiting_id = id;
            continue;
        }
        if (fatal) {
            send(msg(ids, "error", "\"message\":\"HTTP 401: bad key\",\"retryable\":false,\"status\":401"));
            continue;
        }
        if (n <= fail_n) {
            send(msg(ids, "error", "\"message\":\"HTTP 503: busy\",\"retryable\":true,\"status\":503"));
            continue;
        }
        if (bad_first && n == 1) {
            send(msg(ids, "error", "\"message\":\"empty reply\",\"retryable\":true"));
            continue;
        }
        bool stream = false;
        if (doc["stream"].get(stream)) stream = false;
        if (partial) {
            if (n == 1) {
                send(msg(ids, "delta", "\"text\":\"Hel\""));
                send(msg(ids, "delta", "\"text\":\"lo \""));
                send(msg(ids, "error", "\"message\":\"the reply ended early\",\"retryable\":true"));
            } else {
                send(msg(ids, "delta", "\"text\":\"Hello \""));
                send(msg(ids, "delta", "\"text\":\"world\""));
                send(done(ids, "Hello world", "[]", 5, 2, 0, "stop"));
            }
            continue;
        }
        std::string last_user;
        bool has_tool = false;
        simdjson::dom::array msgs;
        if (!doc["messages"].get(msgs))
            for (simdjson::dom::element m : msgs) {
                std::string_view role, content;
                if (m["role"].get(role)) continue;
                if (role == "tool") has_tool = true;
                if (role == "user" && !m["content"].get(content)) last_user.assign(content);
            }
        if (has_tool) {
            if (stream) send(msg(ids, "delta", "\"text\":\"tool done\""));
            send(done(ids, "tool done", "[]", 9, 2, 4, "stop"));
        } else if (last_user.rfind("call:", 0) == 0) {
            const size_t space = last_user.find(' ');
            const std::string name = last_user.substr(5, space == std::string::npos ? std::string::npos : space - 5);
            send(done(ids, "", "[{\"id\":\"call_1\",\"name\":\"" + esc(name) + "\",\"arguments\":\"{\\\"x\\\":1}\"}]", 8, 3, 0,
                      "tool_calls"));
        } else {
            const std::string text = "echo: " + last_user;
            if (stream) send(msg(ids, "delta", "\"text\":\"" + esc(text) + "\""));
            send(done(ids, text, "[]", 7, 3, 0, "stop"));
        }
    }
    return 0;
}
