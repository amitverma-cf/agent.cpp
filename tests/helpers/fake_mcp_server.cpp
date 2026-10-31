// Minimal MCP stdio server used by tests/unit/test_mcp.cpp. Speaks newline-delimited JSON-RPC 2.0.
//   tools/list: two pages, "echo" on the first, "fail" on the second ("sleep" too with --sleep-tool)
//   echo  -> "echo: <text>" (+ " env=<FAKE_MCP_VAR>" when set); {"image": true} adds an image item
//   fail  -> isError result "boom"
//   sleep -> waits {"ms": N} before answering
// Before answering tools/call it sends a log notification and a ping request (which a client must
// answer or ignore). Modes (argv[1]):
//   --no-init        answer initialize with "method not found"
//   --exit-at-start  exit(2) before reading anything
//   --exit-on-call   exit(3) on tools/call
//   --bad-json       answer tools/call with a line that is not JSON
//   --banner         print a non-JSON banner on stdout first
//   --sleep-tool     also list the "sleep" tool

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <simdjson.h>
#include <string>
#include <thread>

namespace {

std::string mode;

template <typename T> bool try_get(simdjson::simdjson_result<simdjson::dom::element> field, T &out) {
    return field.get(out) == simdjson::SUCCESS;
}

void send(const std::string &line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string escape(std::string_view s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}

void reply(const std::string &id, const std::string &result) { send("{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":" + result + "}"); }

void reply_error(const std::string &id, int code, const std::string &message) {
    send("{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"error\":{\"code\":" + std::to_string(code) + ",\"message\":\"" + escape(message) +
         "\"}}");
}

void handle_call(const std::string &id, simdjson::dom::element params) {
    if (mode == "--exit-on-call") {
        std::fprintf(stderr, "fake: dying on purpose\n");
        std::fflush(stderr);
        std::exit(3);
    }
    if (mode == "--bad-json") {
        send("{this is not json");
        return;
    }
    send(R"({"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"working"}})");
    send(R"({"jsonrpc":"2.0","id":"srv-1","method":"ping"})");

    std::string_view name;
    try_get(params["name"], name);
    simdjson::dom::element args;
    bool has_args = params["arguments"].get(args) == simdjson::SUCCESS;

    if (name == "echo") {
        std::string_view text;
        if (has_args) try_get(args["text"], text);
        std::string out = "echo: " + std::string(text);
        if (const char *v = std::getenv("FAKE_MCP_VAR")) out += " env=" + std::string(v);
        std::string content = "[{\"type\":\"text\",\"text\":\"" + escape(out) + "\"}";
        bool image = false;
        if (has_args) try_get(args["image"], image);
        if (image) content += R"(,{"type":"image","data":"aGVsbG8=","mimeType":"image/png"})";
        content += "]";
        reply(id, "{\"content\":" + content + "}");
    } else if (name == "fail") {
        reply(id, R"({"content":[{"type":"text","text":"boom"}],"isError":true})");
    } else if (name == "sleep") {
        int64_t ms = 0;
        if (has_args) try_get(args["ms"], ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        reply(id, R"({"content":[{"type":"text","text":"slept"}]})");
    } else {
        reply_error(id, -32602, "Unknown tool: " + std::string(name));
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 1) mode = argv[1];
    if (mode == "--exit-at-start") {
        std::fprintf(stderr, "fake: refusing to start\n");
        return 2;
    }
    std::fprintf(stderr, "fake: started\n");
    std::fflush(stderr);
    if (mode == "--banner") send("Fake MCP server v0 ready (not JSON)");

    simdjson::dom::parser parser;
    std::string line;
    while (std::getline(std::cin, line)) {
        simdjson::dom::element msg;
        if (parser.parse(line).get(msg) != simdjson::SUCCESS) continue;
        std::string_view method;
        if (msg["method"].get(method) != simdjson::SUCCESS) continue; // responses (e.g. to our ping)
        simdjson::dom::element id_el;
        if (msg["id"].get(id_el) != simdjson::SUCCESS) continue; // notifications
        std::string id = simdjson::to_string(id_el);
        simdjson::dom::element params;
        try_get(msg["params"], params);

        if (method == "initialize") {
            if (mode == "--no-init") {
                reply_error(id, -32601, "Method not found");
                continue;
            }
            reply(id, R"({"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"0.1"}})");
        } else if (method == "tools/list") {
            std::string_view cursor;
            bool second = params["cursor"].get(cursor) == simdjson::SUCCESS && cursor == "page2";
            if (!second) {
                reply(
                    id,
                    R"({"tools":[{"name":"echo","description":"Echo text back","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}],"nextCursor":"page2"})");
            } else {
                std::string tools = R"({"name":"fail","description":"Always fails","inputSchema":{"type":"object"}})";
                if (mode == "--sleep-tool") tools += R"(,{"name":"sleep","inputSchema":{"type":"object"}})";
                reply(id, "{\"tools\":[" + tools + "]}");
            }
        } else if (method == "tools/call") {
            handle_call(id, params);
        } else {
            reply_error(id, -32601, "Method not found");
        }
    }
    std::fprintf(stderr, "fake: stdin closed, bye\n");
    return 0;
}
