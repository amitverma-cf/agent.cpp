#include "../executor/history_json.hpp"
#include "../utils/utils.hpp"
#include "graph_internal.hpp"

#include <agent-cpp/agent.hpp>
#include <chrono>
#include <simdjson.h>
#include <utility>
#include <vector>

namespace agent::graph {

int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::string checkpoint_key(std::string_view thread_id, std::string_view suffix) {
    std::string key = "checkpoint:";
    key += thread_id;
    key += ':';
    key += suffix;
    return key;
}

namespace {

std::string_view status_name(GraphRunStatus s) {
    switch (s) {
    case GraphRunStatus::Pending: return "pending";
    case GraphRunStatus::Running: return "running";
    case GraphRunStatus::Completed: return "completed";
    case GraphRunStatus::Interrupted: return "interrupted";
    case GraphRunStatus::Failed: return "failed";
    }
    return "pending";
}

GraphRunStatus parse_status(std::string_view s) {
    if (s == "running") return GraphRunStatus::Running;
    if (s == "completed") return GraphRunStatus::Completed;
    if (s == "interrupted") return GraphRunStatus::Interrupted;
    if (s == "failed") return GraphRunStatus::Failed;
    return GraphRunStatus::Pending;
}

void append_string_field(std::string &json, std::string_view name, std::string_view value) {
    json += '"';
    json += name;
    json += "\":\"";
    utils::escape_json_string(value, json);
    json += '"';
}

} // namespace

std::string encode_checkpoint(const GraphRun &run, std::string_view graph_name) {
    std::string json;
    json += "{\"version\":1,";
    append_string_field(json, "graph", graph_name);
    json += ',';
    append_string_field(json, "thread_id", run.thread_id);
    json += ",\"step\":" + std::to_string(run.step) + ',';
    append_string_field(json, "next", run.next_node);
    json += ',';
    append_string_field(json, "status", status_name(run.status));
    json += ',';
    append_string_field(json, "interrupt_node", run.interrupt_node);
    json += ',';
    append_string_field(json, "interrupt_reason", run.interrupt_reason);

    json += ",\"pending_tool_calls\":[";
    for (size_t i = 0; i < run.pending_tool_calls.size(); ++i) {
        if (i > 0) json += ',';
        const ToolCall &tc = run.pending_tool_calls[i];
        json += '{';
        append_string_field(json, "id", tc.id);
        json += ',';
        append_string_field(json, "name", tc.name);
        json += ',';
        append_string_field(json, "arguments", tc.arguments);
        json += '}';
    }

    json += "],\"state\":{";
    bool first = true;
    for (const auto &[key, value] : run.state.values) {
        if (is_temp_key(key)) continue;
        if (!first) json += ',';
        first = false;
        json += '"';
        utils::escape_json_string(key, json);
        json += "\":\"";
        utils::escape_json_string(value, json);
        json += '"';
    }

    json += "},\"histories\":{";
    first = true;
    for (const auto &[node, memory] : run.state.histories) {
        if (!first) json += ',';
        first = false;
        json += '"';
        utils::escape_json_string(node, json);
        json += "\":";
        encode_history_json(memory, json);
    }
    json += "}}";
    return json;
}

Result<void> decode_checkpoint(std::string_view json, GraphRun &out) {
    // A local parser: decode_history_json below uses the thread's shared one.
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc)) return fail(ErrorCode::ParseError, "graph checkpoint: invalid JSON.");
    simdjson::ondemand::object root;
    if (doc.get_object().get(root)) return fail(ErrorCode::ParseError, "graph checkpoint: not a JSON object.");

    GraphState state;
    std::vector<std::pair<std::string, std::string>> raw_histories;
    for (auto field : root) {
        std::string_view key;
        if (field.unescaped_key().get(key)) return fail(ErrorCode::ParseError, "graph checkpoint: bad key.");
        const std::string name(key);
        simdjson::ondemand::value value;
        if (field.value().get(value)) return fail(ErrorCode::ParseError, "graph checkpoint: bad value for " + name);

        std::string_view text;
        if (name == "step") {
            uint64_t step = 0;
            if (value.get_uint64().get(step)) return fail(ErrorCode::ParseError, "graph checkpoint: bad step.");
            out.step = step;
        } else if (name == "next") {
            if (!value.get_string().get(text)) out.next_node = std::string(text);
        } else if (name == "status") {
            if (!value.get_string().get(text)) out.status = parse_status(text);
        } else if (name == "interrupt_node") {
            if (!value.get_string().get(text)) out.interrupt_node = std::string(text);
        } else if (name == "interrupt_reason") {
            if (!value.get_string().get(text)) out.interrupt_reason = std::string(text);
        } else if (name == "pending_tool_calls") {
            simdjson::ondemand::array calls;
            if (value.get_array().get(calls)) return fail(ErrorCode::ParseError, "graph checkpoint: bad pending_tool_calls.");
            out.pending_tool_calls.clear();
            for (auto item : calls) {
                simdjson::ondemand::object obj;
                if (item.get_object().get(obj)) continue;
                ToolCall call;
                std::string_view id, tool, arguments;
                if (!obj["id"].get_string().get(id)) call.id = std::string(id);
                if (!obj["name"].get_string().get(tool)) call.name = std::string(tool);
                if (!obj["arguments"].get_string().get(arguments)) call.arguments = std::string(arguments);
                out.pending_tool_calls.push_back(std::move(call));
            }
        } else if (name == "state") {
            simdjson::ondemand::object obj;
            if (value.get_object().get(obj)) return fail(ErrorCode::ParseError, "graph checkpoint: bad state.");
            for (auto entry : obj) {
                std::string_view k, v;
                if (entry.unescaped_key().get(k)) return fail(ErrorCode::ParseError, "graph checkpoint: bad state key.");
                std::string state_key(k);
                if (entry.value().get_string().get(v))
                    return fail(ErrorCode::ParseError, "graph checkpoint: bad state value: " + state_key);
                state.values.insert_or_assign(std::move(state_key), std::string(v));
            }
        } else if (name == "histories") {
            simdjson::ondemand::object obj;
            if (value.get_object().get(obj)) return fail(ErrorCode::ParseError, "graph checkpoint: bad histories.");
            for (auto entry : obj) {
                std::string_view k, raw;
                if (entry.unescaped_key().get(k)) return fail(ErrorCode::ParseError, "graph checkpoint: bad history key.");
                std::string node(k);
                if (entry.value().raw_json().get(raw)) return fail(ErrorCode::ParseError, "graph checkpoint: bad history: " + node);
                raw_histories.emplace_back(std::move(node), std::string(raw));
            }
        }
    }

    for (auto &[node, raw] : raw_histories) {
        auto messages = decode_history_json(raw);
        if (!messages.ok) return fail(messages.error.code, "graph checkpoint: history of '" + node + "': " + messages.error.message);
        FlowMemory memory;
        memory.id = out.thread_id + ":" + node;
        memory.history = std::move(messages.value);
        state.histories.insert_or_assign(std::move(node), std::move(memory));
    }

    out.state.values = std::move(state.values);
    out.state.histories = std::move(state.histories);
    return ok();
}

Result<void> load_scoped_state(Session &session, GraphState &state, std::set<std::string, std::less<>> &index) {
    auto listed = retrieve_memory(session, kScopedIndexKey);
    if (!listed.ok) {
        if (listed.error.code == ErrorCode::KeyNotFound) return ok();
        return fail(listed.error.code, listed.error.message);
    }

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(listed.value.data(), listed.value.size());
    simdjson::ondemand::document doc;
    simdjson::ondemand::array keys;
    if (parser.iterate(padded).get(doc) || doc.get_array().get(keys))
        return fail(ErrorCode::ParseError, "graph state: stored scoped key index is not a JSON array.");
    for (auto item : keys) {
        std::string_view key;
        if (item.get_string().get(key)) continue;
        index.emplace(key);
    }

    for (const auto &key : index) {
        auto value = retrieve_memory(session, std::string(kScopedValuePrefix) + key);
        if (value.ok) state.values.insert_or_assign(key, std::move(value.value));
        else if (value.error.code != ErrorCode::KeyNotFound) return fail(value.error.code, value.error.message);
    }
    return ok();
}

Result<void> persist_scoped_value(Session &session, std::string_view key, std::string_view value,
                                  std::set<std::string, std::less<>> &index) {
    auto put = store_memory(session, std::string(kScopedValuePrefix) + std::string(key), value);
    if (!put.ok) return put;
    if (!index.emplace(key).second) return ok();

    std::string json = "[";
    bool first = true;
    for (const auto &k : index) {
        if (!first) json += ',';
        first = false;
        json += '"';
        utils::escape_json_string(k, json);
        json += '"';
    }
    json += ']';
    return store_memory(session, kScopedIndexKey, json);
}

} // namespace agent::graph
