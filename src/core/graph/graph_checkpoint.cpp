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

namespace {

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

void encode_calls(std::string &json, const std::vector<ToolCall> &calls) {
    json += '[';
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i > 0) json += ',';
        const ToolCall &tc = calls[i];
        json += '{';
        append_string_field(json, "id", tc.id);
        json += ',';
        append_string_field(json, "name", tc.name);
        json += ',';
        append_string_field(json, "arguments", tc.arguments);
        json += '}';
    }
    json += ']';
}

void encode_values(std::string &json, const std::map<std::string, std::string, std::less<>> &values) {
    json += '{';
    bool first = true;
    for (const auto &[key, value] : values) {
        if (is_temp_key(key)) continue;
        if (!first) json += ',';
        first = false;
        json += '"';
        utils::escape_json_string(key, json);
        json += "\":\"";
        utils::escape_json_string(value, json);
        json += '"';
    }
    json += '}';
}

void encode_strings(std::string &json, const std::vector<std::string> &items) {
    json += '[';
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) json += ',';
        json += '"';
        utils::escape_json_string(items[i], json);
        json += '"';
    }
    json += ']';
}

void encode_histories(std::string &json, const std::map<std::string, FlowMemory, std::less<>> &histories) {
    json += '{';
    bool first = true;
    for (const auto &[node, memory] : histories) {
        if (!first) json += ',';
        first = false;
        json += '"';
        utils::escape_json_string(node, json);
        json += "\":";
        encode_history_json(memory, json);
    }
    json += '}';
}

std::string_view branch_status_name(BranchStatus s) {
    switch (s) {
    case BranchStatus::Pending: return "pending";
    case BranchStatus::Running: return "running";
    case BranchStatus::Done: return "done";
    case BranchStatus::Interrupted: return "interrupted";
    case BranchStatus::Failed: return "failed";
    }
    return "pending";
}

BranchStatus parse_branch_status(std::string_view s) {
    if (s == "running") return BranchStatus::Running;
    if (s == "done") return BranchStatus::Done;
    if (s == "interrupted") return BranchStatus::Interrupted;
    if (s == "failed") return BranchStatus::Failed;
    return BranchStatus::Pending;
}

void encode_progress(std::string &json, const ParallelProgress &progress) {
    json += '{';
    append_string_field(json, "node", progress.node);
    json += ",\"branches\":[";
    for (size_t i = 0; i < progress.branches.size(); ++i) {
        const BranchProgress &b = progress.branches[i];
        if (i > 0) json += ',';
        json += '{';
        append_string_field(json, "entry", b.entry);
        json += ',';
        append_string_field(json, "status", branch_status_name(b.status));
        json += ',';
        append_string_field(json, "next", b.next);
        json += ",\"values\":";
        encode_values(json, b.values);
        json += ",\"erased\":";
        encode_strings(json, b.erased);
        json += ",\"writes\":";
        encode_strings(json, b.writes);
        json += ",\"histories\":";
        encode_histories(json, b.histories);
        json += ',';
        append_string_field(json, "interrupt_node", b.interrupt_node);
        json += ',';
        append_string_field(json, "interrupt_reason", b.interrupt_reason);
        json += ",\"pending_tool_calls\":";
        encode_calls(json, b.pending_tool_calls);
        json += ',';
        append_string_field(json, "error", b.error);
        json += '}';
    }
    json += "]}";
}

// ---- decoding (simdjson on demand) ----

using RawHistories = std::vector<std::pair<std::string, std::string>>; // name -> raw history JSON

Result<void> bad(std::string what) { return fail(ErrorCode::ParseError, "graph checkpoint: " + std::move(what)); }

Result<void> decode_calls(simdjson::ondemand::value value, std::vector<ToolCall> &out) {
    simdjson::ondemand::array calls;
    if (value.get_array().get(calls)) return bad("bad pending_tool_calls.");
    out.clear();
    for (auto item : calls) {
        simdjson::ondemand::object obj;
        if (item.get_object().get(obj)) continue;
        ToolCall call;
        std::string_view id, tool, arguments;
        if (!obj["id"].get_string().get(id)) call.id = std::string(id);
        if (!obj["name"].get_string().get(tool)) call.name = std::string(tool);
        if (!obj["arguments"].get_string().get(arguments)) call.arguments = std::string(arguments);
        out.push_back(std::move(call));
    }
    return ok();
}

Result<void> decode_values(simdjson::ondemand::value value, std::map<std::string, std::string, std::less<>> &out) {
    simdjson::ondemand::object obj;
    if (value.get_object().get(obj)) return bad("bad state.");
    for (auto entry : obj) {
        std::string_view k, v;
        if (entry.unescaped_key().get(k)) return bad("bad state key.");
        std::string key(k);
        if (entry.value().get_string().get(v)) return bad("bad state value: " + key);
        out.insert_or_assign(std::move(key), std::string(v));
    }
    return ok();
}

Result<void> decode_strings(simdjson::ondemand::value value, std::vector<std::string> &out) {
    simdjson::ondemand::array items;
    if (value.get_array().get(items)) return bad("bad string array.");
    for (auto item : items) {
        std::string_view text;
        if (item.get_string().get(text)) return bad("bad string array item.");
        out.emplace_back(text);
    }
    return ok();
}

Result<void> collect_histories(simdjson::ondemand::value value, RawHistories &out) {
    simdjson::ondemand::object obj;
    if (value.get_object().get(obj)) return bad("bad histories.");
    for (auto entry : obj) {
        std::string_view k, raw;
        if (entry.unescaped_key().get(k)) return bad("bad history key.");
        std::string node(k);
        if (entry.value().raw_json().get(raw)) return bad("bad history: " + node);
        out.emplace_back(std::move(node), std::string(raw));
    }
    return ok();
}

Result<void> build_histories(RawHistories &raw, std::string_view thread_id, std::map<std::string, FlowMemory, std::less<>> &out) {
    for (auto &[node, text] : raw) {
        auto messages = decode_history_json(text);
        if (!messages.ok) return fail(messages.error.code, "graph checkpoint: history of '" + node + "': " + messages.error.message);
        FlowMemory memory;
        memory.id = std::string(thread_id) + ":" + node;
        memory.history = std::move(messages.value);
        out.insert_or_assign(std::move(node), std::move(memory));
    }
    return ok();
}

Result<void> decode_progress(simdjson::ondemand::value value, ParallelProgress &out, std::vector<RawHistories> &raw) {
    simdjson::ondemand::object obj;
    if (value.get_object().get(obj)) return bad("bad parallel progress.");
    for (auto field : obj) {
        std::string_view key, text;
        if (field.unescaped_key().get(key)) return bad("bad parallel key.");
        simdjson::ondemand::value v;
        if (field.value().get(v)) return bad("bad parallel value.");
        if (key == "node") {
            if (!v.get_string().get(text)) out.node = std::string(text);
            continue;
        }
        if (key != "branches") continue;
        simdjson::ondemand::array branches;
        if (v.get_array().get(branches)) return bad("bad parallel branches.");
        for (auto item : branches) {
            simdjson::ondemand::object bo;
            if (item.get_object().get(bo)) return bad("bad parallel branch.");
            BranchProgress b;
            RawHistories histories;
            for (auto bf : bo) {
                std::string_view bk;
                if (bf.unescaped_key().get(bk)) return bad("bad branch key.");
                simdjson::ondemand::value bv;
                if (bf.value().get(bv)) return bad("bad branch value.");
                Result<void> r = ok();
                if (bk == "entry") {
                    if (!bv.get_string().get(text)) b.entry = std::string(text);
                } else if (bk == "status") {
                    if (!bv.get_string().get(text)) b.status = parse_branch_status(text);
                } else if (bk == "next") {
                    if (!bv.get_string().get(text)) b.next = std::string(text);
                } else if (bk == "values") {
                    r = decode_values(bv, b.values);
                } else if (bk == "erased") {
                    r = decode_strings(bv, b.erased);
                } else if (bk == "writes") {
                    r = decode_strings(bv, b.writes);
                } else if (bk == "histories") {
                    r = collect_histories(bv, histories);
                } else if (bk == "interrupt_node") {
                    if (!bv.get_string().get(text)) b.interrupt_node = std::string(text);
                } else if (bk == "interrupt_reason") {
                    if (!bv.get_string().get(text)) b.interrupt_reason = std::string(text);
                } else if (bk == "pending_tool_calls") {
                    r = decode_calls(bv, b.pending_tool_calls);
                } else if (bk == "error") {
                    if (!bv.get_string().get(text)) b.error = std::string(text);
                }
                if (!r.ok) return r;
            }
            out.branches.push_back(std::move(b));
            raw.push_back(std::move(histories));
        }
    }
    return ok();
}

} // namespace

std::string encode_checkpoint(const GraphRun &run, std::string_view graph_name, const ParallelProgress *progress) {
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
    json += ",\"pending_tool_calls\":";
    encode_calls(json, run.pending_tool_calls);
    json += ",\"state\":";
    encode_values(json, run.state.values);
    json += ",\"histories\":";
    encode_histories(json, run.state.histories);
    if (progress) {
        json += ",\"parallel\":";
        encode_progress(json, *progress);
    }
    json += '}';
    return json;
}

Result<void> decode_checkpoint(std::string_view json, GraphRun &out, ParallelProgress *progress, bool *has_progress) {
    // A local parser: decode_history_json below uses the thread's shared one.
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc)) return bad("invalid JSON.");
    simdjson::ondemand::object root;
    if (doc.get_object().get(root)) return bad("not a JSON object.");

    GraphState state;
    RawHistories raw_histories;
    ParallelProgress parallel;
    std::vector<RawHistories> raw_branch_histories;
    bool found_progress = false;
    for (auto field : root) {
        std::string_view key;
        if (field.unescaped_key().get(key)) return bad("bad key.");
        const std::string name(key);
        simdjson::ondemand::value value;
        if (field.value().get(value)) return bad("bad value for " + name);

        std::string_view text;
        Result<void> r = ok();
        if (name == "step") {
            uint64_t step = 0;
            if (value.get_uint64().get(step)) return bad("bad step.");
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
            r = decode_calls(value, out.pending_tool_calls);
        } else if (name == "state") {
            r = decode_values(value, state.values);
        } else if (name == "histories") {
            r = collect_histories(value, raw_histories);
        } else if (name == "parallel" && progress) {
            found_progress = true;
            r = decode_progress(value, parallel, raw_branch_histories);
        }
        if (!r.ok) return r;
    }

    auto built = build_histories(raw_histories, out.thread_id, state.histories);
    if (!built.ok) return built;
    if (found_progress) {
        for (size_t i = 0; i < parallel.branches.size(); ++i) {
            built = build_histories(raw_branch_histories[i], out.thread_id, parallel.branches[i].histories);
            if (!built.ok) return built;
        }
        *progress = std::move(parallel);
    }
    if (has_progress) *has_progress = found_progress;

    out.state.values = std::move(state.values);
    out.state.histories = std::move(state.histories);
    return ok();
}

void encode_events(std::span<const GraphEvent> events, std::string &json) {
    json += '[';
    for (size_t i = 0; i < events.size(); ++i) {
        const GraphEvent &e = events[i];
        if (i > 0) json += ',';
        json += "{\"type\":" + std::to_string(static_cast<int>(e.type)) + ',';
        append_string_field(json, "node", e.node);
        json += ',';
        append_string_field(json, "key", e.key);
        json += ",\"ts\":" + std::to_string(e.timestamp_ms) + ',';
        append_string_field(json, "payload", e.payload);
        json += '}';
    }
    json += ']';
}

Result<void> decode_events(std::string_view json, std::vector<GraphEvent> &out) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::ondemand::document doc;
    simdjson::ondemand::array items;
    if (parser.iterate(padded).get(doc) || doc.get_array().get(items))
        return fail(ErrorCode::ParseError, "graph event log: not a JSON array.");
    for (auto item : items) {
        simdjson::ondemand::object obj;
        if (item.get_object().get(obj)) return fail(ErrorCode::ParseError, "graph event log: bad event.");
        GraphEvent e;
        for (auto field : obj) {
            std::string_view key, text;
            if (field.unescaped_key().get(key)) return fail(ErrorCode::ParseError, "graph event log: bad key.");
            int64_t number = 0;
            if (key == "type") {
                if (!field.value().get_int64().get(number)) e.type = static_cast<GraphEventType>(number);
            } else if (key == "ts") {
                if (!field.value().get_int64().get(number)) e.timestamp_ms = number;
            } else if (key == "node") {
                if (!field.value().get_string().get(text)) e.node = std::string(text);
            } else if (key == "key") {
                if (!field.value().get_string().get(text)) e.key = std::string(text);
            } else if (key == "payload") {
                if (!field.value().get_string().get(text)) e.payload = std::string(text);
            }
        }
        out.push_back(std::move(e));
    }
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
