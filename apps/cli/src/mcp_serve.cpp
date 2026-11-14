// `agent mcp serve`: the agent's built-in tools (files of the workspace, and so on) as an MCP server on stdin and stdout, for any MCP client:
// an editor, another agent. JSON-RPC, one message per line; initialize, ping, tools/list and tools/call. The client decides what to approve;
// --read-only offers only the tools that change nothing. The tools keep their own sandbox: they work inside the workspace.

#include "chat_extras.hpp"
#include "commands.hpp"

#include <algorithm>
#include <cstdio>
#include <simdjson.h>

namespace agent::cli {

namespace {

// Tools that need a model or a conversation: not for an MCP client.
bool for_clients(std::string_view name) {
    return name != "use_skill" && name != "tool_search" && name != "load_tool" && name != "compress_context";
}

void send(const std::string &line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string error_reply(const std::string &id, int code, std::string_view message) {
    return "{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"error\":{\"code\":" + std::to_string(code) + ",\"message\":\"" + json_escape(message) +
           "\"}}";
}

} // namespace

int mcp_serve(const CommandLine &cl, Context &ctx) {
    Config cfg;
    cfg.provider = AiProvider::Mock;
    cfg.workspace_dir = workspace_dir(ctx, cl.get("workspace"));
    cfg.auto_default_tools = true;
    apply_sandbox_choice(cfg);
    auto session = init(cfg);
    if (!session.ok) return report_error(session.error);
    const bool read_only = cl.has("read-only");

    std::string line;
    while (read_line(stdin, line) == ReadStatus::Line) {
        simdjson::dom::parser parser;
        simdjson::padded_string padded(line);
        simdjson::dom::element msg;
        if (parser.parse(padded).get(msg) || !msg.is_object()) {
            send(error_reply("null", -32700, "parse error"));
            continue;
        }
        std::string_view method;
        if (msg["method"].get(method)) continue; // a reply to a request we never sent
        simdjson::dom::element id_el;
        const bool has_id = !msg["id"].get(id_el) && !id_el.is_null();
        const std::string id = has_id ? simdjson::to_string(id_el) : std::string("null");
        if (!has_id) continue; // notifications (initialized, cancelled) need no answer

        if (method == "server/discover") {
            send("{\"jsonrpc\":\"2.0\",\"id\":" + id +
                 ",\"result\":{\"resultType\":\"complete\",\"supportedVersions\":[\"2026-07-28\",\"2025-06-18\"],\"capabilities\":{"
                 "\"tools\":{}},"
                 "\"_meta\":{\"io.modelcontextprotocol/serverInfo\":{\"name\":\"agent.cpp\",\"version\":\"" +
                 std::string(kCliVersion) + "\"}},\"ttlMs\":0,\"cacheScope\":\"private\"}}");
        } else if (method == "initialize") {
            send("{\"jsonrpc\":\"2.0\",\"id\":" + id +
                 ",\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{}},\"serverInfo\":{\"name\":\"agent.cpp\","
                 "\"version\":\"" +
                 std::string(kCliVersion) + "\"}}}");
        } else if (method == "ping") {
            send("{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":{}}");
        } else if (method == "tools/list") {
            std::string tools;
            for (const Tool &t : session.value.config.tools) {
                if (!for_clients(t.name) || (read_only && tool_changes_things(t.name))) continue;
                if (!tools.empty()) tools += ',';
                tools += "{\"name\":\"" + json_escape(t.name) + "\",\"description\":\"" + json_escape(t.description) +
                         "\",\"inputSchema\":" +
                         (t.parameter_schema.empty() ? std::string("{\"type\":\"object\"}") : std::string(t.parameter_schema)) + "}";
            }
            send("{\"jsonrpc\":\"2.0\",\"id\":" + id +
                 ",\"result\":{\"resultType\":\"complete\",\"ttlMs\":0,\"cacheScope\":\"private\",\"tools\":[" + tools + "]}}");
        } else if (method == "tools/call") {
            std::string_view name;
            simdjson::dom::element params, arguments;
            if (msg["params"].get(params) || params["name"].get(name)) {
                send(error_reply(id, -32602, "tools/call needs a name"));
                continue;
            }
            const std::string args = params["arguments"].get(arguments) ? std::string("{}") : simdjson::to_string(arguments);
            if (!for_clients(name) || (read_only && tool_changes_things(name))) {
                send(error_reply(id, -32602, "unknown tool: " + std::string(name)));
                continue;
            }
            auto result = execute_tool(session.value, name, args);
            const bool failed = !result.ok;
            const std::string text = failed ? result.error.message : result.value;
            send("{\"jsonrpc\":\"2.0\",\"id\":" + id +
                 ",\"result\":{\"resultType\":\"complete\",\"content\":[{\"type\":\"text\",\"text\":\"" + json_escape(text) +
                 "\"}],\"isError\":" + (failed ? "true" : "false") + "}}");
        } else {
            send(error_reply(id, -32601, "method not found: " + std::string(method)));
        }
    }
    return 0;
}

} // namespace agent::cli
