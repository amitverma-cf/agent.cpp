// agent.json loader: builds a Graph from a declarative JSON description. See docs/api/agent-json.md.
//
// JSON cannot hold function pointers, so the loader supplies built-in ones: a router that picks the
// next node from a state value ("route"), a loop condition on a state value ("until"), a tool
// argument builder for static "args", and a no-op function for the entry/exit points of
// "sequential" nodes. Their user_data lives in LoadedGraph::storage.

#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <mutex>
#include <set>
#include <simdjson.h>
#include <sstream>
#include <utility>

namespace agent {

namespace {

// ---- registry of C++ functions for "kind": "function" ----

struct RegisteredFunction {
    GraphFunctionFn fn = nullptr;
    void *user_data = nullptr;
};

std::mutex &registry_mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, RegisteredFunction, std::less<>> &registry() {
    static std::map<std::string, RegisteredFunction, std::less<>> r;
    return r;
}

// ---- built-in callbacks and the storage their user_data points into ----

struct RouteSpec {
    std::string key;
    std::vector<std::pair<std::string, std::string>> cases; // state value -> node
    std::string fallback;                                   // "default"; kGraphEnd when omitted
};

struct UntilSpec {
    std::string key;
    std::string equals;
};

// A "wait" node: pause the run until a time, then continue. The wake time lives in the thread state, so it survives the process that
// started the run: something later (a scheduler running `agent run --resume`) finds the run due and continues it from its checkpoint.
struct WaitSpec {
    std::string node;
    int64_t seconds = 0;     // wait this long after the node is first reached ...
    int64_t until_epoch = 0; // ... or until this moment (local time in the file)
};

struct Storage {
    std::deque<WaitSpec> waits;
    std::deque<RouteSpec> routes;
    std::deque<UntilSpec> untils;
    std::deque<std::string> tool_args;
};

std::string_view trim_ws(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) s.remove_suffix(1);
    return s;
}

Result<std::string_view> route_on_state(Session &, const GraphState &state, void *user_data) {
    const auto *route = static_cast<const RouteSpec *>(user_data);
    if (has_state(state, route->key)) {
        std::string_view value = trim_ws(get_state(state, route->key));
        for (const auto &[match, target] : route->cases)
            if (match == value) return ok(std::string_view(target));
    }
    return ok(std::string_view(route->fallback));
}

Result<bool> until_state_equals(Session &, const GraphState &state, void *user_data) {
    const auto *until = static_cast<const UntilSpec *>(user_data);
    return ok(has_state(state, until->key) && trim_ws(get_state(state, until->key)) == until->equals);
}

Result<std::string> static_tool_args(Session &, const GraphState &, void *user_data) {
    return ok(*static_cast<const std::string *>(user_data));
}

Result<void> noop_function(Session &, GraphState &, void *) { return ok(); }

int64_t epoch_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string local_time_text(int64_t epoch) {
    const std::time_t t = static_cast<std::time_t>(epoch);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

// "2026-11-14T09:00" or "2026-11-14 09:00[:30]", local time.
bool parse_local_time(std::string_view text, int64_t &epoch) {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0;
    const std::string s(text);
    int n = std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%d", &y, &mo, &d, &h, &mi, &sec);
    if (n < 5) n = std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%d", &y, &mo, &d, &h, &mi, &sec);
    if (n < 5 || mo < 1 || mo > 12 || d < 1 || d > 31 || h < 0 || h > 23 || mi < 0 || mi > 59) return false;
    std::tm tm{};
    tm.tm_year = y - 1900;
    tm.tm_mon = mo - 1;
    tm.tm_mday = d;
    tm.tm_hour = h;
    tm.tm_min = mi;
    tm.tm_sec = n >= 6 ? sec : 0;
    tm.tm_isdst = -1;
    const std::time_t t = std::mktime(&tm);
    if (t == static_cast<std::time_t>(-1)) return false;
    epoch = static_cast<int64_t>(t);
    return true;
}

Result<void> wait_function(Session &, GraphState &state, void *user_data) {
    const auto *w = static_cast<const WaitSpec *>(user_data);
    const std::string key = "wait:" + w->node;
    const std::string_view stored = get_state(state, key);
    int64_t until = stored.empty() ? 0 : std::strtoll(std::string(stored).c_str(), nullptr, 10);
    if (until == 0) {
        until = w->until_epoch ? w->until_epoch : epoch_now() + w->seconds;
        set_state(state, key, std::to_string(until));
    }
    if (epoch_now() >= until) {
        set_state(state, key, ""); // due: the node can wait again if a loop reaches it a second time
        set_state(state, "wait:until", "");
        return ok();
    }
    set_state(state, "wait:until", std::to_string(until)); // what a scheduler reads to know when to resume this thread
    request_interrupt(state, "waiting until " + local_time_text(until));
    return ok();
}

constexpr std::string_view kDoneSuffix = "#done";

// ---- parsing helpers ----

struct Parser {
    std::string source; // file path or "<inline>", prefixed to every message
    std::vector<std::string> warnings;

    Error error(std::string_view where, std::string message) const {
        std::string msg = source + ": ";
        if (!where.empty()) msg += std::string(where) + ": ";
        msg += message;
        return make_error(ErrorCode::InvalidConfig, std::move(msg));
    }

    void warn(std::string_view where, std::string message) {
        std::string msg = source + ": ";
        if (!where.empty()) msg += std::string(where) + ": ";
        warnings.push_back(msg + message);
    }

    void check_fields(simdjson::dom::object obj, std::string_view where, std::initializer_list<std::string_view> known) {
        for (auto field : obj) {
            std::string_view k = field.key;
            if (k == "//" || k == "description" || k == "$schema") continue;
            bool found = false;
            for (auto n : known)
                if (n == k) found = true;
            if (!found) warn(where, "unknown field \"" + std::string(k) + "\" (ignored)");
        }
    }
};

// Optional string field. Missing = ok and out untouched; present with the wrong type = error.
bool read_string(const Parser &p, simdjson::dom::object obj, std::string_view key, std::string &out, std::string_view where, Error &err) {
    simdjson::dom::element el;
    if (obj[key].get(el) != simdjson::SUCCESS) return true;
    std::string_view s;
    if (el.get(s) != simdjson::SUCCESS) {
        err = p.error(where, "\"" + std::string(key) + "\" must be a string");
        return false;
    }
    out = std::string(s);
    return true;
}

bool read_required_string(const Parser &p, simdjson::dom::object obj, std::string_view key, std::string &out, std::string_view where,
                          Error &err) {
    simdjson::dom::element el;
    if (obj[key].get(el) != simdjson::SUCCESS) {
        err = p.error(where, "missing required field \"" + std::string(key) + "\"");
        return false;
    }
    if (!read_string(p, obj, key, out, where, err)) return false;
    if (out.empty()) {
        err = p.error(where, "\"" + std::string(key) + "\" must not be empty");
        return false;
    }
    return true;
}

bool read_string_array(const Parser &p, simdjson::dom::object obj, std::string_view key, std::vector<std::string> &out,
                       std::string_view where, Error &err) {
    simdjson::dom::element el;
    if (obj[key].get(el) != simdjson::SUCCESS) return true;
    simdjson::dom::array arr;
    if (el.get(arr) != simdjson::SUCCESS) {
        err = p.error(where, "\"" + std::string(key) + "\" must be an array of strings");
        return false;
    }
    for (simdjson::dom::element item : arr) {
        std::string_view s;
        if (item.get(s) != simdjson::SUCCESS) {
            err = p.error(where, "\"" + std::string(key) + "\" must be an array of strings");
            return false;
        }
        out.emplace_back(s);
    }
    return true;
}

bool read_int(const Parser &p, simdjson::dom::object obj, std::string_view key, int &out, std::string_view where, Error &err) {
    simdjson::dom::element el;
    if (obj[key].get(el) != simdjson::SUCCESS) return true;
    int64_t v = 0;
    if (el.get(v) != simdjson::SUCCESS || v < INT32_MIN || v > INT32_MAX) {
        err = p.error(where, "\"" + std::string(key) + "\" must be an integer");
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool read_bool(const Parser &p, simdjson::dom::object obj, std::string_view key, bool &out, std::string_view where, Error &err) {
    simdjson::dom::element el;
    if (obj[key].get(el) != simdjson::SUCCESS) return true;
    if (el.get(out) != simdjson::SUCCESS) {
        err = p.error(where, "\"" + std::string(key) + "\" must be true or false");
        return false;
    }
    return true;
}

// "{key, equals}" (until) or "{key, cases, default}" (route): state value comparisons.
bool read_object(const Parser &p, simdjson::dom::object obj, std::string_view key, simdjson::dom::object &out, std::string_view where,
                 Error &err, bool &present) {
    simdjson::dom::element el;
    present = obj[key].get(el) == simdjson::SUCCESS;
    if (!present) return true;
    if (el.get(out) != simdjson::SUCCESS) {
        err = p.error(where, "\"" + std::string(key) + "\" must be an object");
        return false;
    }
    return true;
}

std::string node_where(size_t i, std::string_view name) {
    std::string w = "nodes[" + std::to_string(i) + "]";
    if (!name.empty()) w += " (\"" + std::string(name) + "\")";
    return w;
}

bool valid_node_name(std::string_view name) { return !name.empty() && name != kGraphEnd && name.find('#') == std::string_view::npos; }

Result<std::string> read_source(std::string_view path_or_json, std::string &label) {
    std::string_view t = trim_ws(path_or_json);
    if (!t.empty() && (t.front() == '{' || t.front() == '[')) {
        label = "agent.json (inline)";
        return ok(std::string(path_or_json));
    }
    label = std::string(t);
    std::ifstream f{std::filesystem::path(std::string(t)), std::ios::binary};
    if (!f) return fail<std::string>(ErrorCode::FilesystemError, "cannot open graph file '" + label + "'");
    std::stringstream buf;
    buf << f.rdbuf();
    return ok(buf.str());
}

// "sandbox": true | false | {enabled, read_only_paths, read_write_paths, network, memory_limit_mb}.
// An object without "enabled" turns the sandbox on (writing the object is the opt-in).
bool read_sandbox(Parser &p, simdjson::dom::object obj, SandboxPolicy &out, std::string_view where, Error &err) {
    simdjson::dom::element el;
    if (obj["sandbox"].get(el) != simdjson::SUCCESS) return true;
    if (bool on = false; el.get(on) == simdjson::SUCCESS) {
        out.enabled = on;
        return true;
    }
    simdjson::dom::object sb;
    if (el.get(sb) != simdjson::SUCCESS) {
        err = p.error(where, "\"sandbox\" must be true, false or an object");
        return false;
    }
    const std::string sw = std::string(where) + ": sandbox";
    p.check_fields(sb, sw, {"enabled", "read_only_paths", "read_write_paths", "network", "memory_limit_mb"});
    out.enabled = true;
    std::string network;
    if (!read_bool(p, sb, "enabled", out.enabled, sw, err) || !read_string_array(p, sb, "read_only_paths", out.read_only_paths, sw, err) ||
        !read_string_array(p, sb, "read_write_paths", out.read_write_paths, sw, err) || !read_string(p, sb, "network", network, sw, err))
        return false;
    for (const auto *list : {&out.read_only_paths, &out.read_write_paths})
        for (const auto &path : *list)
            if (path.empty()) {
                err = p.error(sw, "paths must not be empty");
                return false;
            }
    if (network == "deny") out.network = SandboxNetwork::Deny;
    else if (network.empty() || network == "allow") out.network = SandboxNetwork::Allow;
    else {
        err = p.error(sw, "\"network\" must be \"allow\" or \"deny\"");
        return false;
    }
    simdjson::dom::element mem;
    if (sb["memory_limit_mb"].get(mem) == simdjson::SUCCESS) {
        // Up to 16 TiB; 0 = no limit.
        constexpr uint64_t kMaxMb = 16ull << 20;
        uint64_t mb = 0;
        if (mem.get(mb) != simdjson::SUCCESS || mb > kMaxMb) {
            err = p.error(sw, "\"memory_limit_mb\" must be an integer between 0 and " + std::to_string(kMaxMb));
            return false;
        }
        out.memory_limit_bytes = mb << 20;
    }
    return true;
}

Result<McpServerConfig> parse_mcp_server(Parser &p, simdjson::dom::element el, std::string_view where) {
    McpServerConfig cfg;
    simdjson::dom::object obj;
    Error err;
    if (el.get(obj) != simdjson::SUCCESS)
        return Result<McpServerConfig>{.ok = false, .value = {}, .error = p.error(where, "must be an object")};
    p.check_fields(obj, where, {"name", "command", "args", "env", "timeout_ms", "working_directory", "sandbox"});
    bool good =
        read_required_string(p, obj, "name", cfg.name, where, err) && read_required_string(p, obj, "command", cfg.command, where, err) &&
        read_string_array(p, obj, "args", cfg.args, where, err) && read_int(p, obj, "timeout_ms", cfg.timeout_ms, where, err) &&
        read_string(p, obj, "working_directory", cfg.working_directory, where, err) && read_sandbox(p, obj, cfg.sandbox, where, err);
    if (good) {
        simdjson::dom::object env;
        bool present = false;
        good = read_object(p, obj, "env", env, where, err, present);
        if (good && present) {
            for (auto field : env) {
                std::string_view v;
                if (field.value.get(v) != simdjson::SUCCESS) {
                    err = p.error(where, "\"env\" values must be strings");
                    good = false;
                    break;
                }
                cfg.env[std::string(field.key)] = std::string(v);
            }
        }
    }
    if (good && cfg.timeout_ms <= 0) {
        err = p.error(where, "\"timeout_ms\" must be > 0");
        good = false;
    }
    if (!good) return Result<McpServerConfig>{.ok = false, .value = {}, .error = err};
    return ok(std::move(cfg));
}

// ---- graph assembly and structural checks ----

struct Builder {
    Parser &p;
    Graph &g;
    std::map<std::string, size_t, std::less<>> index; // node name -> g.nodes index
    std::set<std::string, std::less<>> sequential;    // names of "sequential" nodes
    std::map<std::string, std::vector<std::string>, std::less<>> steps;
    // Ownership of each node by a loop body / parallel branch / sequential step / the top level, to
    // place exit_to and to catch nodes claimed by two of them.
    std::map<std::string, std::pair<std::string, std::string>, std::less<>> owner; // node -> (exit target, label)

    const GraphEdge *edge_from(std::string_view name) const {
        for (const auto &e : g.edges)
            if (e.from == name) return &e;
        return nullptr;
    }

    GraphNode &node(std::string_view name) { return g.nodes[index.find(name)->second]; }

    // Nodes a transition from name can lead to, treating workflow nodes as opaque.
    std::vector<std::string> successors(std::string_view name) const {
        std::vector<std::string> out;
        if (sequential.count(name)) {
            out.push_back(std::string(name) + std::string(kDoneSuffix));
            return out;
        }
        const GraphEdge *e = edge_from(name);
        if (!e) return out;
        if (!e->router) {
            if (e->to != kGraphEnd) out.push_back(e->to);
            return out;
        }
        const auto *route = static_cast<const RouteSpec *>(e->user_data);
        for (const auto &[v, target] : route->cases)
            if (target != kGraphEnd) out.push_back(target);
        if (route->fallback != kGraphEnd) out.push_back(route->fallback);
        return out;
    }

    std::vector<std::string> chain(std::string_view start, std::string_view stop) const {
        std::vector<std::string> out;
        std::set<std::string, std::less<>> seen;
        std::vector<std::string> todo{std::string(start)};
        while (!todo.empty()) {
            std::string n = std::move(todo.back());
            todo.pop_back();
            if (n == stop || !seen.insert(n).second) continue;
            out.push_back(n);
            for (auto &s : successors(n)) todo.push_back(std::move(s));
        }
        return out;
    }

    Result<void> claim(const std::vector<std::string> &members, const std::string &target, const std::string &label) {
        for (const auto &m : members) {
            auto it = owner.find(m);
            if (it != owner.end()) {
                if (it->second.first != target)
                    return Result<void>{.ok = false,
                                        .error = p.error("", "node '" + m + "' is part of both " + it->second.second + " and " + label +
                                                                 "; a node can belong to only one loop body, parallel branch or "
                                                                 "sequential step (or the top level)")};
                continue;
            }
            owner.emplace(m, std::make_pair(target, label));
            if (!target.empty()) node(m).exit_to = target;
        }
        return ok();
    }

    Result<void> assign_exits() {
        auto top = claim(chain(g.entry, {}), "", "the top level (reachable from entry '" + g.entry + "')");
        if (!top.ok) return top;
        for (auto &n : g.nodes) {
            if (n.name.ends_with(kDoneSuffix)) continue;
            Result<void> r = ok();
            if (n.kind == GraphNodeKind::Loop) {
                r = claim(chain(n.loop_body, n.name), n.name, "loop '" + n.name + "' body");
            } else if (n.kind == GraphNodeKind::Parallel) {
                for (const auto &b : n.branches) {
                    r = claim(chain(b, n.name), n.name, "parallel '" + n.name + "' branch '" + b + "'");
                    if (!r.ok) break;
                }
            } else if (sequential.count(n.name)) {
                const auto &s = steps.find(n.name)->second;
                const std::string done = n.name + std::string(kDoneSuffix);
                for (size_t i = 0; i < s.size(); ++i) {
                    const std::string &next = i + 1 < s.size() ? s[i + 1] : done;
                    r = claim(chain(s[i], next), next, "sequential '" + n.name + "' step '" + s[i] + "'");
                    if (!r.ok) break;
                }
            }
            if (!r.ok) return r;
        }
        return ok();
    }

    void warn_unreachable() {
        std::set<std::string, std::less<>> seen;
        std::vector<std::string> todo{g.entry};
        while (!todo.empty()) {
            std::string n = std::move(todo.back());
            todo.pop_back();
            if (!seen.insert(n).second) continue;
            for (auto &s : successors(n)) todo.push_back(std::move(s));
            const GraphNode &gn = node(n);
            if (!gn.exit_to.empty()) todo.push_back(gn.exit_to);
            if (gn.kind == GraphNodeKind::Loop) todo.push_back(gn.loop_body);
            for (const auto &b : gn.branches) todo.push_back(b);
            if (auto it = steps.find(n); it != steps.end())
                for (const auto &s : it->second) todo.push_back(s);
        }
        for (const auto &n : g.nodes)
            if (!n.name.ends_with(kDoneSuffix) && !seen.count(n.name))
                p.warn("", "node '" + n.name + "' is not reachable from entry '" + g.entry + "'");
    }

    // Where a node goes when its transition does not depend on state; empty when it does (route),
    // when the run ends, or for loop controllers (bounded by max_iterations).
    std::string static_next(const GraphNode &n) const {
        if (n.kind == GraphNodeKind::Loop) return {};
        if (sequential.count(n.name)) return steps.find(n.name)->second.front();
        const GraphEdge *e = edge_from(n.name);
        if (e && e->router) return {};
        if (e && e->to != kGraphEnd) return e->to;
        return n.exit_to;
    }

    void warn_static_cycles() {
        std::map<std::string, int, std::less<>> state; // 0 new, 1 on the current walk, 2 done
        for (const auto &start : g.nodes) {
            if (state[start.name] != 0) continue;
            std::vector<std::string> walk;
            std::string cur = start.name;
            while (!cur.empty() && state[cur] == 0) {
                state[cur] = 1;
                walk.push_back(cur);
                cur = static_next(node(cur));
            }
            if (!cur.empty() && state[cur] == 1) {
                std::string text;
                bool in_cycle = false;
                for (const auto &w : walk) {
                    if (w == cur) in_cycle = true;
                    if (in_cycle) text += w + " -> ";
                }
                text += cur;
                p.warn("", "cycle " + text + " has no conditional edge or loop node; it only stops at max_steps or when a node interrupts");
            }
            for (const auto &w : walk) state[w] = 2;
        }
    }
};

Result<LoadedGraph> parse_document(Parser &p, simdjson::dom::element root, std::string default_name) {
    using R = Result<LoadedGraph>;
    auto failure = [](Error e) { return R{.ok = false, .value = {}, .error = std::move(e)}; };

    LoadedGraph out;
    auto storage = std::make_shared<Storage>();
    out.storage = storage;
    Graph &g = out.graph;
    Error err;

    simdjson::dom::object top;
    if (root.get(top) != simdjson::SUCCESS) return failure(p.error("", "the top level must be a JSON object"));
    p.check_fields(top, "", {"name", "entry", "engine", "model", "checkpointer", "max_steps", "mcp_servers", "nodes", "edges"});

    g.name = std::move(default_name);
    if (!read_string(p, top, "name", g.name, "", err) || !read_string(p, top, "description", out.description, "", err) ||
        !read_string(p, top, "engine", out.engine, "", err) || !read_string(p, top, "model", out.model, "", err) ||
        !read_string(p, top, "checkpointer", out.checkpointer, "", err) || !read_string(p, top, "entry", g.entry, "", err) ||
        !read_int(p, top, "max_steps", g.max_steps, "", err))
        return failure(err);
    if (g.max_steps < 1) return failure(p.error("", "\"max_steps\" must be >= 1"));
    if (!out.checkpointer.empty() && out.checkpointer != "sqlite" && out.checkpointer != "memory")
        return failure(p.error("", "\"checkpointer\" must be \"sqlite\" or \"memory\""));

    // mcp_servers
    simdjson::dom::element servers_el;
    if (top["mcp_servers"].get(servers_el) == simdjson::SUCCESS) {
        simdjson::dom::array servers;
        if (servers_el.get(servers) != simdjson::SUCCESS) return failure(p.error("", "\"mcp_servers\" must be an array"));
        size_t i = 0;
        std::set<std::string, std::less<>> names;
        for (simdjson::dom::element s : servers) {
            std::string where = "mcp_servers[" + std::to_string(i++) + "]";
            auto cfg = parse_mcp_server(p, s, where);
            if (!cfg.ok) return failure(cfg.error);
            if (!names.insert(cfg.value.name).second) return failure(p.error(where, "duplicate MCP server name '" + cfg.value.name + "'"));
            out.mcp_servers.push_back(std::move(cfg.value));
        }
    }

    // nodes
    simdjson::dom::array nodes;
    simdjson::dom::element nodes_el;
    if (top["nodes"].get(nodes_el) != simdjson::SUCCESS) return failure(p.error("", "missing required field \"nodes\""));
    if (nodes_el.get(nodes) != simdjson::SUCCESS) return failure(p.error("", "\"nodes\" must be an array"));
    if (nodes.size() == 0) return failure(p.error("", "\"nodes\" is empty"));

    Builder b{.p = p, .g = g, .index = {}, .sequential = {}, .steps = {}, .owner = {}};
    std::vector<std::pair<std::string, std::string>> sequential_edges; // seq -> first step

    size_t i = 0;
    for (simdjson::dom::element el : nodes) {
        const size_t idx = i++;
        simdjson::dom::object obj;
        if (el.get(obj) != simdjson::SUCCESS) return failure(p.error(node_where(idx, ""), "must be an object"));
        GraphNode n;
        std::string kind;
        if (!read_required_string(p, obj, "name", n.name, node_where(idx, ""), err)) return failure(err);
        const std::string where = node_where(idx, n.name);
        if (!valid_node_name(n.name)) return failure(p.error(where, "invalid node name (must not be empty, \"__end__\" or contain '#')"));
        if (b.index.count(n.name)) return failure(p.error(where, "duplicate node name '" + n.name + "'"));
        if (!read_required_string(p, obj, "kind", kind, where, err)) return failure(err);

        if (kind == "agent") {
            n.kind = GraphNodeKind::Agent;
            p.check_fields(obj, where,
                           {"name", "kind", "system_prompt", "prompt", "input_key", "output_key", "allowed_tools", "interrupt_before_tools",
                            "history_key", "model"});
            if (!read_string(p, obj, "system_prompt", n.system_prompt, where, err) ||
                !read_string(p, obj, "prompt", n.prompt, where, err) || !read_string(p, obj, "input_key", n.input_key, where, err) ||
                !read_string(p, obj, "output_key", n.output_key, where, err) ||
                !read_string(p, obj, "history_key", n.history_key, where, err) || !read_string(p, obj, "model", n.model, where, err) ||
                !read_string_array(p, obj, "allowed_tools", n.allowed_tools, where, err) ||
                !read_bool(p, obj, "interrupt_before_tools", n.interrupt_before_tools, where, err))
                return failure(err);
            if (n.prompt.empty() && n.input_key.empty())
                p.warn(where, "agent node has neither \"prompt\" nor \"input_key\"; it will run with an empty user message");
        } else if (kind == "tool") {
            n.kind = GraphNodeKind::Tool;
            p.check_fields(obj, where, {"name", "kind", "tool", "args", "input_key", "output_key"});
            if (!read_required_string(p, obj, "tool", n.tool_name, where, err) ||
                !read_string(p, obj, "input_key", n.input_key, where, err) || !read_string(p, obj, "output_key", n.output_key, where, err))
                return failure(err);
            simdjson::dom::element args;
            if (obj["args"].get(args) == simdjson::SUCCESS) {
                if (!args.is_object()) return failure(p.error(where, "\"args\" must be a JSON object"));
                if (!n.input_key.empty()) p.warn(where, "both \"args\" and \"input_key\" are set; \"args\" wins");
                storage->tool_args.push_back(simdjson::to_string(args));
                n.tool_args = static_tool_args;
                n.user_data = &storage->tool_args.back();
            }
        } else if (kind == "function") {
            n.kind = GraphNodeKind::Function;
            p.check_fields(obj, where, {"name", "kind", "function"});
            std::string fn_name;
            if (!read_required_string(p, obj, "function", fn_name, where, err)) return failure(err);
            std::lock_guard<std::mutex> lock(registry_mutex());
            auto it = registry().find(fn_name);
            if (it == registry().end())
                return failure(p.error(where, "function '" + fn_name + "' is not registered (call agent::register_graph_function first)"));
            n.function = it->second.fn;
            n.user_data = it->second.user_data;
        } else if (kind == "wait") {
            n.kind = GraphNodeKind::Function;
            p.check_fields(obj, where, {"name", "kind", "seconds", "until"});
            WaitSpec spec;
            spec.node = n.name;
            int seconds = -1;
            std::string until;
            if (!read_int(p, obj, "seconds", seconds, where, err) || !read_string(p, obj, "until", until, where, err)) return failure(err);
            if (seconds < 0 && until.empty()) return failure(p.error(where, "wait node needs \"seconds\" or \"until\""));
            if (seconds >= 0 && !until.empty()) return failure(p.error(where, "wait node takes \"seconds\" or \"until\", not both"));
            if (!until.empty() && !parse_local_time(until, spec.until_epoch))
                return failure(p.error(where, "\"until\" must look like 2026-11-14T09:00 (local time)"));
            spec.seconds = seconds < 0 ? 0 : seconds;
            storage->waits.push_back(std::move(spec));
            n.function = wait_function;
            n.user_data = &storage->waits.back();
        } else if (kind == "parallel") {
            n.kind = GraphNodeKind::Parallel;
            p.check_fields(obj, where, {"name", "kind", "branches"});
            if (!read_string_array(p, obj, "branches", n.branches, where, err)) return failure(err);
            if (n.branches.empty()) return failure(p.error(where, "parallel node needs a non-empty \"branches\" array"));
        } else if (kind == "loop") {
            n.kind = GraphNodeKind::Loop;
            p.check_fields(obj, where, {"name", "kind", "body", "max_iterations", "until"});
            n.max_iterations = 0;
            if (!read_required_string(p, obj, "body", n.loop_body, where, err) ||
                !read_int(p, obj, "max_iterations", n.max_iterations, where, err))
                return failure(err);
            if (n.max_iterations < 1) return failure(p.error(where, "loop node needs \"max_iterations\" >= 1"));
            simdjson::dom::object until;
            bool present = false;
            if (!read_object(p, obj, "until", until, where, err, present)) return failure(err);
            if (present) {
                const std::string uw = where + " until";
                UntilSpec spec;
                p.check_fields(until, uw, {"key", "equals"});
                if (!read_required_string(p, until, "key", spec.key, uw, err)) return failure(err);
                simdjson::dom::element eq;
                if (until["equals"].get(eq) != simdjson::SUCCESS) return failure(p.error(uw, "missing required field \"equals\""));
                if (!read_string(p, until, "equals", spec.equals, uw, err)) return failure(err);
                storage->untils.push_back(std::move(spec));
                n.until = until_state_equals;
                n.user_data = &storage->untils.back();
            }
        } else if (kind == "sequential") {
            // Runs as a no-op Function node with an edge to the first step; the last step exits to a
            // synthetic "<name>#done" node that carries the sequential node's own outgoing edge.
            n.kind = GraphNodeKind::Function;
            n.function = noop_function;
            p.check_fields(obj, where, {"name", "kind", "steps"});
            std::vector<std::string> s;
            if (!read_string_array(p, obj, "steps", s, where, err)) return failure(err);
            if (s.empty()) return failure(p.error(where, "sequential node needs a non-empty \"steps\" array"));
            b.sequential.insert(n.name);
            sequential_edges.emplace_back(n.name, s.front());
            b.steps.emplace(n.name, std::move(s));
        } else {
            return failure(p.error(where, "unknown kind '" + kind + "' (expected agent, tool, function, parallel, loop or sequential)"));
        }

        b.index.emplace(n.name, g.nodes.size());
        g.nodes.push_back(std::move(n));
    }
    for (const auto &name : b.sequential) {
        GraphNode done;
        done.name = name + std::string(kDoneSuffix);
        done.kind = GraphNodeKind::Function;
        done.function = noop_function;
        b.index.emplace(done.name, g.nodes.size());
        g.nodes.push_back(std::move(done));
    }

    // References from workflow nodes.
    auto known = [&b](std::string_view name) { return b.index.count(name) && !name.ends_with(kDoneSuffix); };
    for (size_t k = 0; k < g.nodes.size(); ++k) {
        const GraphNode &n = g.nodes[k];
        if (n.name.ends_with(kDoneSuffix)) continue;
        const std::string where = node_where(k, n.name);
        for (const auto &br : n.branches) {
            if (!known(br)) return failure(p.error(where, "unknown node '" + br + "' in \"branches\""));
            if (br == n.name) return failure(p.error(where, "a parallel node cannot be its own branch"));
        }
        if (n.kind == GraphNodeKind::Loop) {
            if (!known(n.loop_body)) return failure(p.error(where, "unknown node '" + n.loop_body + "' in \"body\""));
            if (n.loop_body == n.name) return failure(p.error(where, "a loop node cannot be its own body"));
        }
        if (auto it = b.steps.find(n.name); it != b.steps.end()) {
            std::set<std::string, std::less<>> seen;
            for (const auto &s : it->second) {
                if (!known(s)) return failure(p.error(where, "unknown node '" + s + "' in \"steps\""));
                if (s == n.name) return failure(p.error(where, "a sequential node cannot be its own step"));
                if (!seen.insert(s).second) return failure(p.error(where, "step '" + s + "' is listed twice"));
            }
        }
    }

    // entry
    if (g.entry.empty()) g.entry = g.nodes.front().name;
    else if (!known(g.entry)) return failure(p.error("", "\"entry\" names unknown node '" + g.entry + "'"));

    // edges
    simdjson::dom::element edges_el;
    if (top["edges"].get(edges_el) == simdjson::SUCCESS) {
        simdjson::dom::array edges;
        if (edges_el.get(edges) != simdjson::SUCCESS) return failure(p.error("", "\"edges\" must be an array"));
        size_t e = 0;
        std::map<std::string, size_t, std::less<>> from_seen;
        for (simdjson::dom::element el : edges) {
            const size_t eidx = e++;
            std::string where = "edges[" + std::to_string(eidx) + "]";
            simdjson::dom::object obj;
            if (el.get(obj) != simdjson::SUCCESS) return failure(p.error(where, "must be an object"));
            p.check_fields(obj, where, {"from", "to", "route"});
            GraphEdge edge;
            if (!read_required_string(p, obj, "from", edge.from, where, err)) return failure(err);
            where += " (from \"" + edge.from + "\")";
            if (!known(edge.from)) return failure(p.error(where, "unknown node '" + edge.from + "' in \"from\""));
            if (auto [it, inserted] = from_seen.emplace(edge.from, eidx); !inserted)
                return failure(p.error(where, "node '" + edge.from + "' already has an outgoing edge (edges[" + std::to_string(it->second) +
                                                  "]); use a route for conditional targets or a parallel node to fan out"));
            if (b.sequential.count(edge.from)) edge.from += std::string(kDoneSuffix);

            auto target_ok = [&known](std::string_view t) { return t == kGraphEnd || known(t); };
            simdjson::dom::object route;
            bool has_route = false;
            if (!read_object(p, obj, "route", route, where, err, has_route)) return failure(err);
            simdjson::dom::element to_el;
            const bool has_to = obj["to"].get(to_el) == simdjson::SUCCESS;
            if (has_to == has_route) return failure(p.error(where, "an edge needs exactly one of \"to\" or \"route\""));
            if (has_to) {
                if (!read_required_string(p, obj, "to", edge.to, where, err)) return failure(err);
                if (!target_ok(edge.to)) return failure(p.error(where, "unknown node '" + edge.to + "' in \"to\""));
            } else {
                RouteSpec spec;
                const std::string rw = where + " route";
                p.check_fields(route, rw, {"key", "cases", "default"});
                if (!read_required_string(p, route, "key", spec.key, rw, err)) return failure(err);
                spec.fallback = std::string(kGraphEnd);
                if (!read_string(p, route, "default", spec.fallback, rw, err)) return failure(err);
                if (!target_ok(spec.fallback)) return failure(p.error(rw, "unknown node '" + spec.fallback + "' in \"default\""));
                simdjson::dom::object cases;
                bool has_cases = false;
                if (!read_object(p, route, "cases", cases, rw, err, has_cases)) return failure(err);
                if (!has_cases) return failure(p.error(rw, "missing required field \"cases\""));
                for (auto c : cases) {
                    std::string_view target;
                    if (c.value.get(target) != simdjson::SUCCESS)
                        return failure(p.error(rw, "case \"" + std::string(c.key) + "\" must map to a node name"));
                    if (!target_ok(target))
                        return failure(p.error(rw, "case \"" + std::string(c.key) + "\": unknown node '" + std::string(target) + "'"));
                    spec.cases.emplace_back(std::string(c.key), std::string(target));
                }
                storage->routes.push_back(std::move(spec));
                edge.router = route_on_state;
                edge.user_data = &storage->routes.back();
            }
            g.edges.push_back(std::move(edge));
        }
    }
    for (auto &[from, first] : sequential_edges)
        g.edges.push_back(GraphEdge{.from = from, .to = first, .router = nullptr, .user_data = nullptr});

    auto exits = b.assign_exits();
    if (!exits.ok) return failure(exits.error);
    b.warn_unreachable();
    b.warn_static_cycles();

    out.warnings = std::move(p.warnings);
    return ok(std::move(out));
}

} // namespace

void register_graph_function(std::string_view name, GraphFunctionFn fn, void *user_data) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().insert_or_assign(std::string(name), RegisteredFunction{.fn = fn, .user_data = user_data});
}

void unregister_graph_function(std::string_view name) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    if (auto it = registry().find(name); it != registry().end()) registry().erase(it);
}

Result<LoadedGraph> parse_graph_json(std::string_view path_or_json) {
    Parser p;
    auto text = read_source(path_or_json, p.source);
    if (!text.ok) return Result<LoadedGraph>{.ok = false, .value = {}, .error = text.error};

    simdjson::dom::parser parser;
    simdjson::dom::element root;
    auto parse_err = parser.parse(text.value).get(root);
    if (parse_err != simdjson::SUCCESS)
        return fail<LoadedGraph>(ErrorCode::ParseError, p.source + ": invalid JSON: " + simdjson::error_message(parse_err));

    std::string default_name = "graph";
    if (!p.source.ends_with("(inline)")) default_name = std::filesystem::path(p.source).stem().string();
    return parse_document(p, root, std::move(default_name));
}

Result<LoadedGraph> load_graph_json(Session &session, std::string_view path_or_json) {
    auto loaded = parse_graph_json(path_or_json);
    if (!loaded.ok) return loaded;
    LoadedGraph &lg = loaded.value;

    std::vector<std::string> connected; // by this call, undone on failure
    auto rollback = [&session, &connected]() {
        for (const auto &name : connected) (void)disconnect_mcp_server(session, name);
    };

    const std::vector<std::string> existing = list_mcp_servers(session);
    for (size_t i = 0; i < lg.mcp_servers.size(); ++i) {
        const McpServerConfig &cfg = lg.mcp_servers[i];
        if (std::find(existing.begin(), existing.end(), cfg.name) != existing.end()) {
            lg.warnings.push_back("mcp_servers[" + std::to_string(i) + "]: server '" + cfg.name + "' is already connected; reusing it");
            continue;
        }
        auto res = connect_mcp_server(session, cfg);
        if (!res.ok) {
            rollback();
            return fail<LoadedGraph>(res.error.code, "mcp_servers[" + std::to_string(i) + "] (\"" + cfg.name + "\"): " + res.error.message);
        }
        connected.push_back(cfg.name);
    }

    for (const auto &n : lg.graph.nodes) {
        if (n.kind == GraphNodeKind::Tool && !session.tool_by_name.count(n.tool_name)) {
            rollback();
            return fail<LoadedGraph>(ErrorCode::InvalidConfig,
                                     "graph '" + lg.graph.name + "': tool node '" + n.name + "' uses unknown tool '" + n.tool_name + "'");
        }
        for (const auto &t : n.allowed_tools)
            if (!session.tool_by_name.count(t))
                lg.warnings.push_back("graph '" + lg.graph.name + "': agent node '" + n.name + "' allows unknown tool '" + t + "'");
    }

    for (const auto &w : lg.warnings) utils::log(session, LogLevel::Warning, w);
    return loaded;
}

} // namespace agent
