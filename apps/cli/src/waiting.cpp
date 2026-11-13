#include "waiting.hpp"

#include "cli_core.hpp"

#include <algorithm>
#include <simdjson.h>

namespace agent::cli {

std::string waiting_to_json(const std::vector<WaitingRun> &runs) {
    std::string out = "{\n  \"waiting\": [";
    for (size_t i = 0; i < runs.size(); ++i) {
        const WaitingRun &r = runs[i];
        out += i ? ",\n    {" : "\n    {";
        out += "\"thread\": \"" + json_escape(r.thread) + "\", \"graph\": \"" + json_escape(r.graph) + "\", \"cwd\": \"" +
               json_escape(r.cwd) + "\", \"until\": " + std::to_string(r.until) + ", \"flags\": [";
        for (size_t f = 0; f < r.flags.size(); ++f) out += (f ? ", \"" : "\"") + json_escape(r.flags[f]) + "\"";
        out += "]}";
    }
    out += runs.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

Result<std::vector<WaitingRun>> waiting_from_json(std::string_view json) {
    using R = std::vector<WaitingRun>;
    std::vector<WaitingRun> out;
    if (json.empty()) return ok(std::move(out));
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc)) return fail<R>(ErrorCode::DecodeFailed, "waiting.json is not valid JSON");
    simdjson::dom::array arr;
    if (doc["waiting"].get_array().get(arr)) return fail<R>(ErrorCode::DecodeFailed, "waiting.json has no \"waiting\" array");
    for (simdjson::dom::element e : arr) {
        WaitingRun r;
        std::string_view thread, graph;
        int64_t until = 0;
        if (e["thread"].get_string().get(thread) || e["graph"].get_string().get(graph) || e["until"].get_int64().get(until)) continue;
        r.thread = std::string(thread);
        r.graph = std::string(graph);
        std::string_view cwd;
        if (!e["cwd"].get_string().get(cwd)) r.cwd = std::string(cwd);
        r.until = until;
        simdjson::dom::array flags;
        if (!e["flags"].get_array().get(flags))
            for (simdjson::dom::element f : flags) {
                std::string_view word;
                if (!f.get_string().get(word)) r.flags.emplace_back(word);
            }
        out.push_back(std::move(r));
    }
    return ok(std::move(out));
}

void waiting_upsert(std::vector<WaitingRun> &runs, WaitingRun run) {
    for (WaitingRun &r : runs)
        if (r.thread == run.thread) {
            r = std::move(run);
            return;
        }
    runs.push_back(std::move(run));
}

bool waiting_remove(std::vector<WaitingRun> &runs, std::string_view thread) {
    const auto it = std::remove_if(runs.begin(), runs.end(), [&](const WaitingRun &r) { return r.thread == thread; });
    const bool found = it != runs.end();
    runs.erase(it, runs.end());
    return found;
}

std::vector<WaitingRun> waiting_due(const std::vector<WaitingRun> &runs, int64_t now) {
    std::vector<WaitingRun> due;
    for (const WaitingRun &r : runs)
        if (r.until <= now) due.push_back(r);
    std::sort(due.begin(), due.end(), [](const WaitingRun &a, const WaitingRun &b) { return a.until < b.until; });
    return due;
}

std::vector<std::string> waiting_resume_args(const WaitingRun &run) {
    std::vector<std::string> args = {"run", run.graph};
    args.insert(args.end(), run.flags.begin(), run.flags.end());
    args.push_back("--resume");
    args.push_back(run.thread);
    return args;
}

} // namespace agent::cli
