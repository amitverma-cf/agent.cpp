// tool_search and load_tool: lazy tool loading. With Config::lazy_tools the model is told about only these two tools
// (and use_skill) at the start of a conversation. tool_search lists what else exists, one line each; load_tool(name)
// adds a tool to the conversation's active set, so it is advertised from the next request on and stays loaded across
// context compression. A conversation that never needs the file or shell tools never pays their prompt tokens.

#include "../executor/ambient_turn.hpp"
#include "tools_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cctype>
#include <simdjson.h>
#include <string>
#include <vector>

namespace agent::tools {

namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// The first sentence of a description, capped, for the one-line listing.
std::string first_line(std::string_view description) {
    const size_t end = std::min({description.find(". "), description.find('\n'), size_t{140}});
    std::string line(description.substr(0, end));
    if (end < description.size() && description[end] == '.') line += '.';
    return line;
}

bool is_meta(std::string_view name) { return name == "tool_search" || name == "load_tool" || name == "use_skill"; }

bool is_loaded(std::string_view name) {
    const Conversation *m = ambient::active_memory;
    return m && std::find(m->active_tools.begin(), m->active_tools.end(), name) != m->active_tools.end();
}

Result<std::string> tool_search_callback(std::string_view arguments, void *user_data) {
    Session *session = *static_cast<Session **>(user_data);
    if (!session) return fail<std::string>(ErrorCode::InvalidConfig, "tool_search: session reference is null");
    std::string query;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    std::string_view q;
    if (!arguments.empty() && !parser.parse(arguments).get(doc) && !doc["query"].get(q)) query = lower(q);

    std::string out;
    size_t shown = 0;
    for (const Tool &t : session->config.tools) {
        if (is_meta(t.name) || !tool_allowed(ambient::allowed_tools, t.name)) continue;
        if (!query.empty() && lower(t.name).find(query) == std::string::npos && lower(t.description).find(query) == std::string::npos)
            continue;
        out += "- " + std::string(t.name) + ": " + first_line(t.description) + (is_loaded(t.name) ? " (loaded)" : "") + "\n";
        ++shown;
    }
    if (shown == 0)
        return ok(std::string("No tools match") +
                  (query.empty() ? "." : " '" + query + "'. Call tool_search with no query to list every tool."));
    out += "Call load_tool with a tool's name to use it.";
    return ok(std::move(out));
}

Result<std::string> load_tool_callback(std::string_view arguments, void *user_data) {
    Session *session = *static_cast<Session **>(user_data);
    if (!session) return fail<std::string>(ErrorCode::InvalidConfig, "load_tool: session reference is null");
    Conversation *memory = ambient::active_memory;
    if (!memory) return fail<std::string>(ErrorCode::InvalidConfig, "load_tool: it can only be called during a conversation turn");

    std::vector<std::string> names;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(arguments).get(doc)) return fail<std::string>(ErrorCode::ParseError, "load_tool: arguments must be a JSON object");
    std::string_view one;
    if (!doc["name"].get(one)) names.emplace_back(one);
    simdjson::dom::array many;
    if (!doc["names"].get(many))
        for (simdjson::dom::element e : many)
            if (!e.get(one)) names.emplace_back(one);
    if (names.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "load_tool: give \"name\" (or \"names\") of the tool to load");

    std::string out;
    bool loaded_any = false;
    for (const std::string &name : names) {
        const Tool *tool = nullptr;
        for (const Tool &t : session->config.tools)
            if (t.name == name && !is_meta(t.name) && tool_allowed(ambient::allowed_tools, t.name)) tool = &t;
        if (!tool) {
            out += "No tool named '" + name + "'. Call tool_search to see what exists.\n";
            continue;
        }
        if (!is_loaded(name)) memory->active_tools.push_back(name);
        loaded_any = true;
        // The definition is also given here, in the tool result, so a model whose chat template shows tools only in the
        // system prompt still knows how to call it before the next request advertises it.
        out += "Loaded " + name + ": " + std::string(tool->description) +
               "\nSchema: " + (tool->parameter_schema.empty() ? std::string("{}") : std::string(tool->parameter_schema)) + "\n";
    }
    if (!loaded_any) return fail<std::string>(ErrorCode::KeyNotFound, out);
    return ok(std::move(out));
}

} // namespace

Tool get_tool_search_tool(Session **session_cell) {
    Tool t;
    t.name = "tool_search";
    t.description =
        "List the tools that exist, one line each, optionally only those matching a word. Tools are not loaded until load_tool is called.";
    t.parameter_schema =
        R"({"type":"object","properties":{"query":{"type":"string","description":"a word to look for in tool names and descriptions; omit to list every tool"}}})";
    t.callback = tool_search_callback;
    t.user_data = session_cell;
    return t;
}

Tool get_load_tool_tool(Session **session_cell) {
    Tool t;
    t.name = "load_tool";
    t.description =
        "Load a tool so you can call it. Use the name shown by tool_search. The tool stays available for the rest of the conversation.";
    t.parameter_schema =
        R"({"type":"object","properties":{"name":{"type":"string","description":"the tool to load"},"names":{"type":"array","items":{"type":"string"},"description":"several tools to load at once"}}})";
    t.callback = load_tool_callback;
    t.user_data = session_cell;
    return t;
}

} // namespace agent::tools
