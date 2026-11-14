// MCP (Model Context Protocol) stdio client: newline-delimited JSON-RPC 2.0 with a server running
// as a child process. Each remote tool is registered as a Tool named "<server>.<tool>".
//
// Ownership: one McpServer per connection, held by a shared_ptr in a DataSource named
// "mcp:<server>" in Session::retained_data_sources, so the Session owns the process and moving the
// Session moves only the shared_ptr. Tool::user_data points at a RemoteTool owned by the McpServer.

#include "../sandbox/sandbox_policy.hpp"
#include "../tools/tools_ops.hpp"
#include "../utils/utils.hpp"
#include "hal/child_process.hpp"
#include "hal/log_file.hpp"

#include <agent-cpp/agent.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <simdjson.h>
#include <thread>
#include <utility>

namespace agent {

namespace {

constexpr std::string_view kProtocolVersion = "2025-06-18";
constexpr std::string_view kModernVersion = "2026-07-28"; // stateless: no handshake, every request carries its version
constexpr std::string_view kDataSourcePrefix = "mcp:";
constexpr size_t kMaxLineBytes = 64u * 1024 * 1024;
constexpr size_t kStderrTailBytes = 2048;
constexpr int kMaxListPages = 1000;

using Clock = std::chrono::steady_clock;

// Optional field: out is left untouched when the field is missing or has another type.
template <typename T> bool try_get(simdjson::simdjson_result<simdjson::dom::element> field, T &out) {
    return field.get(out) == simdjson::SUCCESS;
}

std::string clip(std::string_view s, size_t max = 200) {
    if (s.size() <= max) return std::string(s);
    return std::string(s.substr(0, max)) + "...";
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) s.remove_suffix(1);
    return s;
}

// Logger state copied out of the Session: the stderr thread outlives any particular Session address.
struct LogSink {
    LogFn fn = nullptr;
    void *user_data = nullptr;
    std::shared_ptr<FILE> file;
    std::shared_ptr<std::mutex> mutex;

    void write(LogLevel level, std::string_view message) const {
        if (fn) fn(level, message, user_data);
        if (!file || !mutex) return;
        const char *tag = level == LogLevel::Debug     ? "DEBUG"
                          : level == LogLevel::Warning ? "WARN"
                          : level == LogLevel::Error   ? "ERROR"
                                                       : "INFO";
        std::lock_guard<std::mutex> lock(*mutex);
        hal::write_log_line(file.get(), tag, message);
    }
};

// ---- base64 (image/audio content) ----

int b64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

bool base64_decode(std::string_view in, std::string &out) {
    out.clear();
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        int v = b64_value(c);
        if (v < 0) return false;
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xFFu));
        }
    }
    return true;
}

std::string_view extension_for(std::string_view mime) {
    static constexpr std::pair<std::string_view, std::string_view> kTable[] = {
        {"image/png", "png"},     {"image/jpeg", "jpg"}, {"image/jpg", "jpg"}, {"image/gif", "gif"},   {"image/webp", "webp"},
        {"image/svg+xml", "svg"}, {"image/bmp", "bmp"},  {"audio/wav", "wav"}, {"audio/x-wav", "wav"}, {"audio/wave", "wav"},
        {"audio/mpeg", "mp3"},    {"audio/mp3", "mp3"},  {"audio/ogg", "ogg"}, {"audio/flac", "flac"}, {"audio/webm", "webm"},
    };
    for (const auto &[m, ext] : kTable)
        if (m == mime) return ext;
    return "bin";
}

std::string sanitize_for_filename(std::string_view s) {
    std::string out;
    for (char c : s) {
        bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
        out.push_back(keep ? c : '_');
    }
    return out;
}

bool valid_server_name(std::string_view name) {
    if (name.empty() || name.size() > 64) return false;
    for (char c : name) {
        bool okc = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
        if (!okc) return false;
    }
    return true;
}

class McpServer;

struct RemoteTool {
    std::string full_name; // "<server>.<tool>"
    std::string remote_name;
    std::string description;
    std::string schema;
    McpServer *server = nullptr;
};

// JSON-RPC error responses are reported with this code so callers can tell them apart from
// transport failures (NetworkError) and malformed output (ParseError).
constexpr ErrorCode kRemoteError = ErrorCode::HttpError;

class McpServer {
  public:
    McpServerConfig config;
    LogSink log;
    std::shared_ptr<std::string> workspace;
    std::vector<std::unique_ptr<RemoteTool>> tools;
    std::string protocol_version;
    bool modern_ = false;       // 2026-07-28 revision: per-request _meta instead of initialize
    std::string call_method_, call_name_; // Mcp-Method / Mcp-Name headers of the request in flight (HTTP)
    std::string server_info;
    std::vector<std::string> secret_values; // inserted into the environment; kept out of logs and tool results
    HttpPostFn http_post = nullptr;         // servers reached by URL
    void *http_user = nullptr;
    bool has_resources = false, has_prompts = false;

    // Replaces every secret value in text with "***".
    void redact(std::string &text) const {
        for (const std::string &v : secret_values) {
            if (v.size() < 4) continue; // shorter values would mask ordinary text
            for (size_t at = text.find(v); at != std::string::npos; at = text.find(v, at + 3)) text.replace(at, v.size(), "***");
        }
    }

    McpServer(McpServerConfig cfg, LogSink sink, std::shared_ptr<std::string> ws)
        : config(std::move(cfg)),
          log(std::move(sink)),
          workspace(std::move(ws)) {}
    ~McpServer() { shutdown(); }
    McpServer(const McpServer &) = delete;
    McpServer &operator=(const McpServer &) = delete;

    Result<void> start();
    Result<ToolResult> call_tool(std::string_view remote_name, std::string_view arguments);
    void shutdown();

  private:
    std::mutex mutex_; // serialises requests (and shutdown) on this server
    std::unique_ptr<hal::ChildProcess> process_;
    std::thread stderr_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stderr_closed_{false}; // set by the stderr thread once it has drained everything
    std::mutex stderr_mutex_;
    std::string stderr_tail_;
    std::string stdout_buf_;
    std::string session_id_; // Mcp-Session-Id of a server reached by URL
    int64_t next_id_ = 1;
    bool dead_ = false;
    std::string dead_reason_;
    simdjson::dom::parser parser_;
    uint64_t artifact_counter_ = 0;

    std::string prefix() const { return "mcp server '" + config.name + "'"; }
    std::string stderr_tail();
    std::string exit_details();
    void drain_stderr();
    Result<void> send_line(const std::string &line);
    Result<void> notify(std::string_view method, std::string_view params_json);
    bool remote() const { return !config.url.empty(); }
    Result<HttpReply> post(std::string_view body);
    Result<simdjson::dom::element> request_http(std::string_view method, std::string_view params_json);
    void add_extra_tools();
    void answer_server_request(std::string_view method, simdjson::dom::element id);
    Result<simdjson::dom::element> request(std::string_view method, std::string_view params_json);
    Result<void> mark_dead(std::string reason);
    Result<void> list_tools(bool advertised);
    Result<ToolResult> map_call_result(simdjson::dom::element result, std::string_view remote_name);
    void save_artifact(ToolResult &result, std::string_view remote_name, std::string_view base64, std::string_view mime,
                       std::string_view kind);
};

std::string McpServer::stderr_tail() {
    std::lock_guard<std::mutex> lock(stderr_mutex_);
    return stderr_tail_;
}

void McpServer::drain_stderr() {
    std::string buf;
    auto flush_lines = [this, &buf](bool all) {
        size_t start = 0;
        for (;;) {
            size_t nl = buf.find('\n', start);
            if (nl == std::string::npos) break;
            std::string_view line = trim(std::string_view(buf).substr(start, nl - start));
            if (!line.empty()) {
                std::string text(line);
                redact(text);
                log.write(LogLevel::Info, "[mcp:" + config.name + "] " + text);
            }
            start = nl + 1;
        }
        buf.erase(0, start);
        if (all && !trim(buf).empty()) {
            std::string text(trim(buf));
            redact(text);
            log.write(LogLevel::Info, "[mcp:" + config.name + "] " + text);
        }
        if (all) buf.clear();
    };
    while (!stop_.load()) {
        const size_t before = buf.size();
        hal::PipeRead r = process_->read(hal::ChildStream::Stderr, buf, 100);
        if (r == hal::PipeRead::Data) {
            {
                std::lock_guard<std::mutex> lock(stderr_mutex_);
                stderr_tail_.append(buf, before, std::string::npos);
                if (stderr_tail_.size() > kStderrTailBytes) stderr_tail_.erase(0, stderr_tail_.size() - kStderrTailBytes);
            }
            flush_lines(false);
            if (buf.size() > 64 * 1024) flush_lines(true); // a very long line without '\n'
        } else if (r != hal::PipeRead::Timeout) {
            break;
        }
    }
    flush_lines(true);
    stderr_closed_.store(true);
}

Result<void> McpServer::mark_dead(std::string reason) {
    dead_ = true;
    dead_reason_ = reason;
    return fail(ErrorCode::NetworkError, std::move(reason));
}

// " (exit code N); stderr: ..." for a server whose pipes broke. Waits briefly for the exit status and
// the last stderr lines (usually the explanation) to arrive.
std::string McpServer::exit_details() {
    for (int i = 0; i < 100 && !(process_->exited() && stderr_closed_.load()); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::string out;
    if (process_->exited()) out += " (exit code " + std::to_string(process_->exit_code()) + ")";
    std::string tail = std::string(trim(stderr_tail()));
    if (!tail.empty()) out += "; stderr: " + clip(tail, 500);
    return out;
}

Result<void> McpServer::send_line(const std::string &line) {
    auto w = process_->write_stdin(line);
    if (!w.ok) return mark_dead(prefix() + " is not accepting input (" + w.error.message + ")" + exit_details());
    return ok();
}

Result<void> McpServer::notify(std::string_view method, std::string_view params_json) {
    std::string line = "{\"jsonrpc\":\"2.0\",\"method\":\"";
    utils::escape_json_string(method, line);
    line += "\"";
    if (!params_json.empty()) {
        line += ",\"params\":";
        line += params_json;
    }
    line += "}\n";
    if (remote()) {
        auto sent = post(line);
        if (!sent.ok) return fail(sent.error.code, prefix() + ": " + sent.error.message);
        return ok();
    }
    return send_line(line);
}

// Requests the server sends to us. We advertise no client capabilities, so only ping is answered.
void McpServer::answer_server_request(std::string_view method, simdjson::dom::element id) {
    std::string line = "{\"jsonrpc\":\"2.0\",\"id\":" + simdjson::to_string(id);
    if (method == "ping") line += ",\"result\":{}}\n";
    else {
        line += ",\"error\":{\"code\":-32601,\"message\":\"Method not supported by client: ";
        utils::escape_json_string(method, line);
        line += "\"}}\n";
    }
    (void)send_line(line);
}

// Modern revision: the version, capabilities and client identity travel in the params' _meta.
static std::string with_meta(std::string_view params_json) {
    const std::string meta = std::string("\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"") + std::string(kModernVersion) +
                             "\",\"io.modelcontextprotocol/clientCapabilities\":{},\"io.modelcontextprotocol/clientInfo\":{\"name\":\"agent.cpp\","
                             "\"version\":\"0.1.0\"}}";
    std::string_view rest = params_json;
    while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1);
    if (rest.size() < 2 || rest.front() != '{') return "{" + meta + "}";
    rest.remove_prefix(1);
    while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1);
    return "{" + meta + (rest.front() == '}' ? std::string() : std::string(",")) + std::string(rest);
}

Result<simdjson::dom::element> McpServer::request(std::string_view method_in, std::string_view params_in) {
    using R = Result<simdjson::dom::element>;
    auto failure = [](ErrorCode code, std::string msg) { return R{.ok = false, .value = {}, .error = make_error(code, std::move(msg))}; };
    if (dead_) return failure(ErrorCode::NetworkError, prefix() + " is not running: " + dead_reason_);
    const std::string_view method = method_in;
    const std::string meta_params = modern_ || method == "server/discover" ? with_meta(params_in) : std::string();
    const std::string_view params_json = meta_params.empty() ? params_in : std::string_view(meta_params);
    if (remote()) {
        call_method_ = std::string(method);
        call_name_.clear();
        simdjson::dom::parser pp;
        simdjson::padded_string padded(params_in);
        simdjson::dom::element pe;
        std::string_view nm;
        if (pp.parse(padded).get(pe) == simdjson::SUCCESS && (pe["name"].get(nm) == simdjson::SUCCESS || pe["uri"].get(nm) == simdjson::SUCCESS))
            call_name_ = std::string(nm);
        return request_http(method, params_json);
    }

    const int64_t id = next_id_++;
    std::string line = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) + ",\"method\":\"";
    utils::escape_json_string(method, line);
    line += "\",\"params\":";
    line += params_json.empty() ? std::string_view("{}") : params_json;
    line += "}\n";
    auto sent = send_line(line);
    if (!sent.ok) return failure(sent.error.code, sent.error.message);

    const auto deadline = Clock::now() + std::chrono::milliseconds(config.timeout_ms);
    for (;;) {
        size_t nl = stdout_buf_.find('\n');
        if (nl == std::string::npos) {
            if (stdout_buf_.size() > kMaxLineBytes) {
                auto d = mark_dead(prefix() + " sent a line longer than " + std::to_string(kMaxLineBytes) + " bytes");
                return failure(ErrorCode::ParseError, d.error.message);
            }
            auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (left <= 0) {
                (void)notify("notifications/cancelled", "{\"requestId\":" + std::to_string(id) + ",\"reason\":\"timeout\"}");
                return failure(ErrorCode::NetworkError,
                               prefix() + ": " + std::string(method) + " timed out after " + std::to_string(config.timeout_ms) + " ms");
            }
            hal::PipeRead r = process_->read(hal::ChildStream::Stdout, stdout_buf_, static_cast<int>(left));
            if (r == hal::PipeRead::Closed || r == hal::PipeRead::Error) {
                std::string reason = prefix() + (r == hal::PipeRead::Error ? " stdout read failed" : " exited") + " while waiting for " +
                                     std::string(method) + exit_details();
                auto d = mark_dead(std::move(reason));
                return failure(ErrorCode::NetworkError, d.error.message);
            }
            continue;
        }

        std::string raw = stdout_buf_.substr(0, nl);
        stdout_buf_.erase(0, nl + 1);
        std::string_view text = trim(raw);
        if (text.empty()) continue;
        if (text.front() != '{') {
            // Not JSON-RPC. The spec forbids it, but some launchers print banners on stdout.
            log.write(LogLevel::Warning, prefix() + ": ignoring non-JSON-RPC output: " + clip(text));
            continue;
        }

        simdjson::dom::element msg;
        if (parser_.parse(text.data(), text.size()).get(msg) != simdjson::SUCCESS || !msg.is_object())
            return failure(ErrorCode::ParseError, prefix() + " sent invalid JSON: " + clip(text));

        simdjson::dom::element id_el;
        const bool has_id = msg["id"].get(id_el) == simdjson::SUCCESS && !id_el.is_null();
        std::string_view inbound_method;
        if (msg["method"].get(inbound_method) == simdjson::SUCCESS) {
            if (has_id) {
                answer_server_request(inbound_method, id_el);
            } else if (inbound_method == "notifications/message") {
                simdjson::dom::element params;
                if (msg["params"].get(params) == simdjson::SUCCESS)
                    log.write(LogLevel::Info, "[mcp:" + config.name + "] " + clip(simdjson::to_string(params), 1000));
            }
            continue; // other notifications (progress, list_changed, ...) are ignored
        }

        int64_t got = -1;
        if (!has_id || id_el.get(got) != simdjson::SUCCESS || got != id) continue; // stale reply to a timed-out request

        simdjson::dom::element err;
        if (msg["error"].get(err) == simdjson::SUCCESS) {
            std::string_view message = "unknown error";
            int64_t code = 0;
            try_get(err["message"], message);
            try_get(err["code"], code);
            return failure(kRemoteError, prefix() + ": " + std::string(method) + " failed: " + std::string(message) + " (code " +
                                             std::to_string(code) + ")");
        }
        simdjson::dom::element result;
        if (msg["result"].get(result) != simdjson::SUCCESS)
            return failure(ErrorCode::ParseError, prefix() + ": response to " + std::string(method) + " has neither result nor error");
        return R{.ok = true, .value = result, .error = {}};
    }
}

Result<void> McpServer::start() {
    if (!remote()) {
        hal::SpawnOptions opts;
        opts.program = config.command;
        opts.args = config.args;
        opts.env = config.env;
        opts.working_directory = config.working_directory;
        opts.sandbox = sandbox::mcp_policy(config.sandbox, config.command, config.working_directory);
        opts.sandbox.windows_low_integrity = true; // third-party code: writes only in its own folders (Windows)
        auto spawned = hal::spawn_child(opts);
        if (!spawned.ok) return fail(ErrorCode::ProviderInitFailed, prefix() + ": " + spawned.error.message);
        process_ = std::move(spawned.value);
        for (const auto &w : process_->sandbox_warnings()) log.write(LogLevel::Warning, prefix() + ": " + w);
        stderr_thread_ = std::thread([this]() { drain_stderr(); });
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Newer servers (2026-07-28) have no handshake: probe server/discover, and fall back to initialize on anything that is not a modern answer.
    bool advertised_tools = true;
    bool discovered = false;
    {
        auto disc = request("server/discover", "{}");
        bool speaks_modern = false;
        if (disc.ok) {
            simdjson::dom::array versions;
            if (disc.value["supportedVersions"].get(versions) == simdjson::SUCCESS)
                for (simdjson::dom::element v : versions) {
                    std::string_view sv;
                    if (v.get(sv) == simdjson::SUCCESS && sv == kModernVersion) speaks_modern = true;
                }
        }
        if (speaks_modern) {
            modern_ = discovered = true;
            protocol_version = std::string(kModernVersion);
            std::string_view name, ver;
            simdjson::dom::element info, caps, extra;
            if (disc.value["_meta"]["io.modelcontextprotocol/serverInfo"].get(info) == simdjson::SUCCESS) {
                try_get(info["name"], name);
                try_get(info["version"], ver);
                server_info = std::string(name) + (ver.empty() ? "" : " " + std::string(ver));
            }
            if (disc.value["capabilities"].get(caps) == simdjson::SUCCESS) {
                advertised_tools = caps["tools"].get(extra) == simdjson::SUCCESS;
                has_resources = caps["resources"].get(extra) == simdjson::SUCCESS;
                has_prompts = caps["prompts"].get(extra) == simdjson::SUCCESS;
            }
        } else if (disc.error.message.find("-32022") != std::string::npos) {
            return fail(ErrorCode::ProviderInitFailed, prefix() + ": the server does not speak protocol " + std::string(kModernVersion) + " (" + disc.error.message + ")");
        }
        // Any other error: a server of the earlier revisions.
    }
    std::string init_params = "{\"protocolVersion\":\"" + std::string(kProtocolVersion) +
                              "\",\"capabilities\":{},\"clientInfo\":{\"name\":\"agent.cpp\",\"version\":\"0.1.0\"}}";
    auto init = discovered ? Result<simdjson::dom::element>{.ok = false, .value = {}, .error = make_error(ErrorCode::ParseError, "")}
                           : request("initialize", init_params);
    if (discovered) {
    } else if (init.ok) {
        std::string_view version, name, ver;
        if (init.value["protocolVersion"].get(version) == simdjson::SUCCESS) protocol_version = std::string(version);
        simdjson::dom::element info;
        if (init.value["serverInfo"].get(info) == simdjson::SUCCESS) {
            try_get(info["name"], name);
            try_get(info["version"], ver);
            server_info = std::string(name) + (ver.empty() ? "" : " " + std::string(ver));
        }
        simdjson::dom::element caps, tools_cap;
        advertised_tools = init.value["capabilities"].get(caps) == simdjson::SUCCESS && caps["tools"].get(tools_cap) == simdjson::SUCCESS;
        simdjson::dom::element extra_cap;
        has_resources = init.value["capabilities"].get(caps) == simdjson::SUCCESS && caps["resources"].get(extra_cap) == simdjson::SUCCESS;
        has_prompts = init.value["capabilities"].get(caps) == simdjson::SUCCESS && caps["prompts"].get(extra_cap) == simdjson::SUCCESS;
        auto n = notify("notifications/initialized", "");
        if (!n.ok) return n;
    } else if (init.error.code == kRemoteError) {
        // Servers that predate (or drop) the handshake answer with an error; carry on without it.
        log.write(LogLevel::Warning, prefix() + ": initialize was rejected, continuing without the handshake (" + init.error.message + ")");
    } else {
        return fail(init.error.code == ErrorCode::NetworkError ? ErrorCode::ProviderInitFailed : init.error.code, init.error.message);
    }

    auto listed = list_tools(advertised_tools);
    if (listed.ok) add_extra_tools();
    return listed;
}

Result<void> McpServer::list_tools(bool advertised) {
    std::string cursor;
    std::set<std::string, std::less<>> seen_cursors;
    std::set<std::string, std::less<>> names;
    for (int page = 0;; ++page) {
        if (page >= kMaxListPages)
            return fail(ErrorCode::ParseError, prefix() + ": tools/list did not finish after " + std::to_string(page) + " pages");
        std::string params = "{}";
        if (!cursor.empty()) params = "{\"cursor\":\"" + utils::escape_json_string(cursor) + "\"}";
        auto res = request("tools/list", params);
        if (!res.ok) {
            if (res.error.code == kRemoteError && !advertised && page == 0) {
                log.write(LogLevel::Info, prefix() + " offers no tools");
                return ok();
            }
            return fail(res.error.code, res.error.message);
        }

        simdjson::dom::array list;
        if (res.value["tools"].get(list) != simdjson::SUCCESS)
            return fail(ErrorCode::ParseError, prefix() + ": tools/list result has no \"tools\" array");
        for (simdjson::dom::element t : list) {
            std::string_view name;
            if (t["name"].get(name) != simdjson::SUCCESS || name.empty()) {
                log.write(LogLevel::Warning, prefix() + ": skipping a tool without a name");
                continue;
            }
            if (!names.emplace(name).second) {
                log.write(LogLevel::Warning, prefix() + ": skipping duplicate tool '" + std::string(name) + "'");
                continue;
            }
            auto rt = std::make_unique<RemoteTool>();
            rt->remote_name = std::string(name);
            rt->full_name = config.name + "." + rt->remote_name;
            std::string_view desc, title;
            if (t["description"].get(desc) == simdjson::SUCCESS) rt->description = std::string(desc);
            else if (t["title"].get(title) == simdjson::SUCCESS) rt->description = std::string(title);
            else rt->description = "Tool '" + rt->remote_name + "' of MCP server '" + config.name + "'.";
            simdjson::dom::element schema;
            if (t["inputSchema"].get(schema) == simdjson::SUCCESS && schema.is_object()) rt->schema = simdjson::to_string(schema);
            else rt->schema = "{\"type\":\"object\"}";
            rt->server = this;
            tools.push_back(std::move(rt));
        }

        std::string_view next;
        if (res.value["nextCursor"].get(next) != simdjson::SUCCESS || next.empty()) break;
        if (!seen_cursors.emplace(next).second)
            return fail(ErrorCode::ParseError, prefix() + ": tools/list returned the cursor '" + clip(next) + "' twice");
        cursor = std::string(next);
    }
    return ok();
}

// Decodes base64 content into <workspace>/artifacts/ and adds it to result.media; a content item that
// can't be saved becomes a text note instead.
void McpServer::save_artifact(ToolResult &result, std::string_view remote_name, std::string_view base64, std::string_view mime,
                              std::string_view kind) {
    auto note = [&result](std::string line) {
        if (!result.text.empty()) result.text += "\n";
        result.text += line;
    };
    std::string bytes;
    if (!base64_decode(base64, bytes)) return note("[" + std::string(kind) + " content with invalid base64 data was dropped]");
    if (!workspace || workspace->empty()) return note("[" + std::string(kind) + " content dropped: no workspace]");

    std::error_code ec;
    std::filesystem::path dir = std::filesystem::absolute(std::filesystem::path(*workspace) / "artifacts", ec);
    if (ec) dir = std::filesystem::path(*workspace) / "artifacts";
    std::filesystem::create_directories(dir, ec);
    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string file = "mcp-" + sanitize_for_filename(config.name) + "-" + sanitize_for_filename(remote_name) + "-" +
                       std::to_string(stamp) + "-" + std::to_string(++artifact_counter_) + "." + std::string(extension_for(mime));
    std::filesystem::path path = dir / file;
    std::ofstream out(path, std::ios::binary);
    if (!out) return note("[" + std::string(kind) + " content dropped: cannot write " + path.string() + "]");
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) return note("[" + std::string(kind) + " content dropped: cannot write " + path.string() + "]");
    result.media.push_back(Media{.path = path.string(), .mime = mime.empty() ? "application/octet-stream" : std::string(mime)});
}

// MCP content items -> ToolResult: text items (and text resources) become the text, image / audio
// items and blob resources become media files, isError becomes is_error.
Result<ToolResult> McpServer::map_call_result(simdjson::dom::element result, std::string_view remote_name) {
    ToolResult out;
    auto append = [&out](std::string_view part) {
        if (!out.text.empty()) out.text += "\n";
        out.text += part;
    };

    simdjson::dom::array content;
    if (result["content"].get(content) == simdjson::SUCCESS) {
        for (simdjson::dom::element item : content) {
            std::string_view type;
            try_get(item["type"], type);
            std::string_view value, mime;
            if (type == "text") {
                if (item["text"].get(value) == simdjson::SUCCESS) append(value);
            } else if (type == "image" || type == "audio") {
                try_get(item["mimeType"], mime);
                if (item["data"].get(value) == simdjson::SUCCESS) save_artifact(out, remote_name, value, mime, type);
            } else if (type == "resource") {
                simdjson::dom::element res;
                if (item["resource"].get(res) == simdjson::SUCCESS) {
                    std::string_view uri;
                    try_get(res["uri"], uri);
                    try_get(res["mimeType"], mime);
                    if (res["text"].get(value) == simdjson::SUCCESS) append(value);
                    else if (res["blob"].get(value) == simdjson::SUCCESS) save_artifact(out, remote_name, value, mime, "resource");
                    else append("[resource " + std::string(uri) + "]");
                }
            } else if (type == "resource_link") {
                std::string_view uri, name;
                try_get(item["uri"], uri);
                try_get(item["name"], name);
                append("[resource link " + std::string(name.empty() ? uri : name) + (name.empty() ? "" : ": " + std::string(uri)) + "]");
            } else {
                append("[unsupported content type '" + std::string(type) + "']");
            }
        }
    }
    simdjson::dom::element structured;
    if (out.text.empty() && out.media.empty() && result["structuredContent"].get(structured) == simdjson::SUCCESS)
        out.text = simdjson::to_string(structured);

    try_get(result["isError"], out.is_error);
    if (out.is_error && out.text.empty() && out.media.empty()) out.text = "tool reported an error";
    return ok(std::move(out));
}

Result<ToolResult> McpServer::call_tool(std::string_view remote_name, std::string_view arguments) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!remote() && !process_) return fail<ToolResult>(ErrorCode::NetworkError, prefix() + " is disconnected");
    if (remote() && dead_) return fail<ToolResult>(ErrorCode::NetworkError, prefix() + " is disconnected");

    std::string args = "{}";
    std::string_view trimmed = trim(arguments);
    if (!trimmed.empty()) {
        simdjson::dom::element parsed;
        if (parser_.parse(trimmed.data(), trimmed.size()).get(parsed) != simdjson::SUCCESS)
            return fail<ToolResult>(ErrorCode::ParseError, "arguments are not valid JSON");
        if (!parsed.is_object()) return fail<ToolResult>(ErrorCode::ParseError, "arguments must be a JSON object");
        args = simdjson::to_string(parsed); // minified: the request must stay on one line
    }

    // Resources and prompts are offered to the model as tools named "@resources/list", "@resources/read", "@prompts/list", "@prompts/get":
    // the arguments are the parameters of that request and the answer is shown as the server gave it.
    if (!remote_name.empty() && remote_name.front() == '@') {
        auto answer = request(remote_name.substr(1), args);
        if (!answer.ok) return fail<ToolResult>(answer.error.code, answer.error.message);
        ToolResult shown;
        shown.text = simdjson::to_string(answer.value);
        if (shown.text.size() > 20000) shown.text = shown.text.substr(0, 20000) + "\n[cut]";
        redact(shown.text);
        return ok(std::move(shown));
    }
    std::string params = "{\"name\":\"";
    utils::escape_json_string(remote_name, params);
    params += "\",\"arguments\":";
    params += args;
    params += "}";
    auto res = request("tools/call", params);
    if (!res.ok) return fail<ToolResult>(res.error.code, res.error.message);
    auto mapped = map_call_result(res.value, remote_name);
    if (mapped.ok) redact(mapped.value.text);
    return mapped;
}

void McpServer::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (remote()) {
        dead_ = true;
        dead_reason_ = "disconnected";
        return;
    }
    if (!process_) return;
    const int grace = config.timeout_ms < 2000 ? config.timeout_ms : 2000;
    int code = process_->terminate(grace); // closes stdin first: the spec's stdio shutdown
    stop_.store(true);
    if (stderr_thread_.joinable()) stderr_thread_.join();
    process_.reset();
    dead_ = true;
    dead_reason_ = "disconnected";
    log.write(LogLevel::Debug, prefix() + " stopped (exit code " + std::to_string(code) + ")");
}

// ---- Streamable HTTP ----

// The data of each event of a text/event-stream body ("data:" lines joined).
std::vector<std::string> sse_messages(std::string_view body) {
    std::vector<std::string> events;
    std::string data;
    size_t at = 0;
    while (at <= body.size()) {
        size_t end = body.find('\n', at);
        if (end == std::string_view::npos) end = body.size();
        std::string_view line = body.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        at = end + 1;
        if (line.empty()) {
            if (!data.empty()) events.push_back(std::move(data));
            data.clear();
            continue;
        }
        if (line.rfind("data:", 0) == 0) {
            std::string_view v = line.substr(5);
            if (!v.empty() && v.front() == ' ') v.remove_prefix(1);
            if (!data.empty()) data += '\n';
            data += v;
        }
    }
    if (!data.empty()) events.push_back(std::move(data));
    return events;
}

Result<HttpReply> McpServer::post(std::string_view body) {
    std::vector<std::pair<std::string, std::string>> headers = {{"Content-Type", "application/json"},
                                                                {"Accept", "application/json, text/event-stream"}};
    if (!protocol_version.empty() && !modern_ && call_method_ != "server/discover") headers.emplace_back("MCP-Protocol-Version", protocol_version);
    if (!session_id_.empty()) headers.emplace_back("Mcp-Session-Id", session_id_);
    if (modern_ || call_method_ == "server/discover") {
        headers.emplace_back("MCP-Protocol-Version", std::string(kModernVersion));
        headers.emplace_back("Mcp-Method", call_method_);
        if (!call_name_.empty()) headers.emplace_back("Mcp-Name", call_name_);
    }
    for (const auto &h : config.headers) headers.emplace_back(h.first, h.second);
    return http_post(config.url, headers, body, config.timeout_ms, http_user);
}

Result<simdjson::dom::element> McpServer::request_http(std::string_view method, std::string_view params_json) {
    using R = Result<simdjson::dom::element>;
    auto failure = [](ErrorCode code, std::string msg) { return R{.ok = false, .value = {}, .error = make_error(code, std::move(msg))}; };
    const int64_t id = next_id_++;
    std::string body = "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(id) + ",\"method\":\"";
    utils::escape_json_string(method, body);
    body += "\",\"params\":";
    body += params_json.empty() ? std::string_view("{}") : params_json;
    body += "}";
    auto reply = post(body);
    if (!reply.ok) return failure(ErrorCode::NetworkError, prefix() + ": " + std::string(method) + " failed: " + reply.error.message);
    if (!reply.value.session_id.empty()) session_id_ = reply.value.session_id;
    const int status = reply.value.status;
    if (status == 404 && !session_id_.empty()) {
        dead_ = true;
        dead_reason_ = "the session ended (HTTP 404)";
        return failure(ErrorCode::NetworkError, prefix() + ": the server ended the session");
    }
    if (status < 200 || status >= 300)
        return failure(ErrorCode::HttpError,
                       prefix() + ": " + std::string(method) + " failed: HTTP " + std::to_string(status) + " " + clip(reply.value.body));

    std::vector<std::string> texts;
    if (reply.value.content_type.find("text/event-stream") != std::string::npos) texts = sse_messages(reply.value.body);
    else texts.push_back(reply.value.body);
    for (const std::string &text : texts) {
        simdjson::dom::element top;
        if (parser_.parse(text.data(), text.size()).get(top) != simdjson::SUCCESS)
            return failure(ErrorCode::ParseError, prefix() + " sent invalid JSON: " + clip(text));
        std::vector<simdjson::dom::element> messages;
        if (top.is_array()) {
            for (simdjson::dom::element e : top.get_array().value_unsafe()) messages.push_back(e);
        } else {
            messages.push_back(top);
        }
        for (simdjson::dom::element msg : messages) {
            if (!msg.is_object()) continue;
            simdjson::dom::element id_el;
            std::string_view inbound;
            if (msg["method"].get(inbound) == simdjson::SUCCESS) {
                if (msg["id"].get(id_el) == simdjson::SUCCESS && !id_el.is_null() && inbound == "ping") {
                    (void)post("{\"jsonrpc\":\"2.0\",\"id\":" + simdjson::to_string(id_el) + ",\"result\":{}}");
                }
                continue; // other server requests and notifications are not supported
            }
            int64_t got = -1;
            if (msg["id"].get(id_el) != simdjson::SUCCESS || id_el.get(got) != simdjson::SUCCESS || got != id) continue;
            simdjson::dom::element err;
            if (msg["error"].get(err) == simdjson::SUCCESS) {
                std::string_view message = "unknown error";
                int64_t code = 0;
                try_get(err["message"], message);
                try_get(err["code"], code);
                return failure(kRemoteError, prefix() + ": " + std::string(method) + " failed: " + std::string(message) + " (code " +
                                                 std::to_string(code) + ")");
            }
            simdjson::dom::element result;
            if (msg["result"].get(result) != simdjson::SUCCESS)
                return failure(ErrorCode::ParseError, prefix() + ": response to " + std::string(method) + " has neither result nor error");
            std::string_view result_type;
            if (modern_ && result["resultType"].get(result_type) == simdjson::SUCCESS && result_type == "input_required")
                return failure(kRemoteError, prefix() + ": " + std::string(method) + " needs more input from the client (multi round-trip requests are not supported)");
            return R{.ok = true, .value = result, .error = {}};
        }
    }
    return failure(ErrorCode::ParseError, prefix() + ": no response to " + std::string(method) + " in the reply");
}

// Resources and prompts, for servers that offer them, as four more tools (see call_tool).
void McpServer::add_extra_tools() {
    const auto add = [&](const char *name, const char *remote, const char *description, const char *schema) {
        auto rt = std::make_unique<RemoteTool>();
        rt->full_name = config.name + "." + name;
        rt->remote_name = remote;
        rt->description = description;
        rt->schema = schema;
        rt->server = this;
        tools.push_back(std::move(rt));
    };
    if (has_resources) {
        add("list_resources", "@resources/list", "List the resources (files, records, documents) this MCP server offers.",
            "{\"type\":\"object\",\"properties\":{}}");
        add("read_resource", "@resources/read", "Read one resource of this MCP server by its uri.",
            "{\"type\":\"object\",\"properties\":{\"uri\":{\"type\":\"string\"}},\"required\":[\"uri\"]}");
    }
    if (has_prompts) {
        add("list_prompts", "@prompts/list", "List the prompt templates this MCP server offers.",
            "{\"type\":\"object\",\"properties\":{}}");
        add("get_prompt", "@prompts/get", "Get a prompt template of this MCP server, filled with the arguments.",
            "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"arguments\":{\"type\":\"object\"}},\"required\":["
            "\"name\"]}");
    }
}

// ---- Session glue ----

Result<ToolResult> remote_tool_callback(std::string_view arguments, void *user_data) {
    auto *rt = static_cast<RemoteTool *>(user_data);
    if (!rt || !rt->server) return fail<ToolResult>(ErrorCode::InvalidConfig, "MCP tool has no server");
    return rt->server->call_tool(rt->remote_name, arguments);
}

// DataSource::query: {"name": "<remote tool>", "arguments": {...}} -> tools/call.
Result<std::string> server_query(void *state, std::string_view query, void *) {
    auto *server = static_cast<McpServer *>(state);
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    std::string_view name;
    if (parser.parse(query.data(), query.size()).get(doc) != simdjson::SUCCESS || doc["name"].get(name) != simdjson::SUCCESS)
        return fail<std::string>(ErrorCode::ParseError, "MCP query must be {\"name\": \"<tool>\", \"arguments\": {...}}");
    simdjson::dom::element args;
    std::string args_json = doc["arguments"].get(args) == simdjson::SUCCESS ? simdjson::to_string(args) : std::string("{}");
    auto res = server->call_tool(std::string(name), args_json);
    if (!res.ok) return fail<std::string>(res.error.code, res.error.message);
    std::string text = tools::render_tool_text(res.value);
    if (res.value.is_error) return fail<std::string>(ErrorCode::ToolError, std::move(text));
    return ok(std::move(text));
}

bool is_mcp_entry(const std::shared_ptr<DataSource> &ds) {
    return ds && ds->query == server_query && ds->state && std::string_view(ds->name).starts_with(kDataSourcePrefix);
}

void unregister_tools(Session &session, const McpServer *server) {
    auto &tools = session.config.tools;
    std::erase_if(tools, [server](const Tool &t) {
        return t.parts_callback == remote_tool_callback && t.user_data && static_cast<const RemoteTool *>(t.user_data)->server == server;
    });
    rebuild_tool_index(session);
}

} // namespace

Result<std::vector<Tool>> connect_mcp_server(Session &session, const McpServerConfig &config) {
    using R = Result<std::vector<Tool>>;
    if (!valid_server_name(config.name))
        return fail<std::vector<Tool>>(ErrorCode::InvalidConfig,
                                       "MCP server name '" + config.name + "' is invalid (use 1-64 letters, digits, '_' or '-')");
    if (config.command.empty() && config.url.empty())
        return fail<std::vector<Tool>>(ErrorCode::InvalidConfig, "MCP server '" + config.name + "' has no command or url");
    if (!config.url.empty() && !session.config.http_post)
        return fail<std::vector<Tool>>(ErrorCode::InvalidConfig,
                                       "MCP server '" + config.name + "' is reached by url, but Config::http_post is not set");
    if (config.timeout_ms <= 0)
        return fail<std::vector<Tool>>(ErrorCode::InvalidConfig, "MCP server '" + config.name + "' needs timeout_ms > 0");
    const std::string entry_name = std::string(kDataSourcePrefix) + config.name;
    for (const auto &ds : session.retained_data_sources)
        if (is_mcp_entry(ds) && ds->name == entry_name)
            return fail<std::vector<Tool>>(ErrorCode::InvalidConfig, "MCP server '" + config.name + "' is already connected");

    // Secrets: only the environment may name one, and only through the host's resolver, which checks that the secret is
    // bound to this server and to what it runs. The model has no way to read them.
    McpServerConfig resolved = config;
    std::vector<std::string> secret_values;
    for (const std::string &a : config.args)
        if (a.find("${secret:") != std::string::npos)
            return fail<std::vector<Tool>>(
                ErrorCode::InvalidConfig,
                "MCP server '" + config.name +
                    "': secrets are only allowed in env, not in args (command lines are visible to other processes)");
    for (auto &entry : resolved.env) {
        if (entry.second.find("${secret:") == std::string::npos) continue;
        if (!session.config.secret_resolver)
            return fail<std::vector<Tool>>(ErrorCode::InvalidConfig, "MCP server '" + config.name + "': env " + entry.first +
                                                                         " uses a secret, but no secret resolver is configured");
        std::vector<std::string> identity{config.command};
        identity.insert(identity.end(), config.args.begin(), config.args.end());
        const std::string consumer = "mcp:" + config.name;
        const SecretRequest who{.consumer = consumer, .identity_parts = identity};
        auto expanded = session.config.secret_resolver(entry.second, who, secret_values, session.config.secret_user_data);
        if (!expanded.ok)
            return fail<std::vector<Tool>>(expanded.error.code, "MCP server '" + config.name + "': " + expanded.error.message);
        entry.second = std::move(expanded.value);
    }

    for (auto &header : resolved.headers) {
        if (header.second.find("${secret:") == std::string::npos) continue;
        if (!session.config.secret_resolver)
            return fail<std::vector<Tool>>(ErrorCode::InvalidConfig, "MCP server '" + config.name + "': header " + header.first +
                                                                         " uses a secret, but no secret resolver is configured");
        const std::string consumer = "mcp:" + config.name;
        const std::vector<std::string> identity{config.url};
        const SecretRequest who{.consumer = consumer, .identity_parts = identity};
        auto expanded = session.config.secret_resolver(header.second, who, secret_values, session.config.secret_user_data);
        if (!expanded.ok)
            return fail<std::vector<Tool>>(expanded.error.code, "MCP server '" + config.name + "': " + expanded.error.message);
        header.second = std::move(expanded.value);
    }

    if (resolved.sandbox.enabled) {
        const std::string consumer = "mcp:" + config.name;
        for (const PathGrant &g : session.config.path_grants)
            if (g.consumer == "*" || g.consumer == consumer)
                (g.write ? resolved.sandbox.read_write_paths : resolved.sandbox.read_only_paths).push_back(g.path);
    }
    if (session.config.mcp_launcher) session.config.mcp_launcher(resolved, session.config.mcp_launcher_user_data);
    LogSink sink{
        .fn = session.config.logger, .user_data = session.config.logger_user_data, .file = session.log_file, .mutex = session.log_mutex};
    auto server = std::make_shared<McpServer>(resolved, std::move(sink), session.workspace_path);
    server->secret_values = std::move(secret_values);
    server->http_post = session.config.http_post;
    server->http_user = session.config.http_post_user_data;
    auto started = server->start();
    if (!started.ok) return fail<std::vector<Tool>>(started.error.code, started.error.message);

    for (const auto &rt : server->tools) {
        if (session.tool_by_name.count(rt->full_name))
            return fail<std::vector<Tool>>(ErrorCode::InvalidConfig,
                                           "MCP server '" + config.name + "': a tool named '" + rt->full_name + "' is already registered");
    }

    std::vector<Tool> registered;
    registered.reserve(server->tools.size());
    for (const auto &rt : server->tools) {
        Tool t{.name = rt->full_name,
               .description = rt->description,
               .parameter_schema = rt->schema,
               .user_data = rt.get(),
               .parts_callback = remote_tool_callback};
        session.config.tools.push_back(t);
        registered.push_back(t);
    }

    auto ds = std::make_shared<DataSource>();
    ds->name = entry_name;
    ds->state = server;
    ds->query = server_query;
    session.retained_data_sources.push_back(std::move(ds));
    rebuild_tool_index(session);

    utils::log(session, LogLevel::Info,
               "MCP server '" + config.name + "' connected" + (server->server_info.empty() ? "" : " (" + server->server_info + ")") +
                   (server->protocol_version.empty() ? "" : ", protocol " + server->protocol_version) + ", " +
                   std::to_string(registered.size()) + " tool(s)");
    return R{.ok = true, .value = std::move(registered), .error = {}};
}

Result<void> disconnect_mcp_server(Session &session, std::string_view name) {
    const std::string entry_name = std::string(kDataSourcePrefix) + std::string(name);
    auto &list = session.retained_data_sources;
    for (auto it = list.begin(); it != list.end(); ++it) {
        if (!is_mcp_entry(*it) || (*it)->name != entry_name) continue;
        auto *server = static_cast<McpServer *>((*it)->state.get());
        unregister_tools(session, server);
        server->shutdown();
        list.erase(it);
        return ok();
    }
    return fail(ErrorCode::KeyNotFound, "MCP server '" + std::string(name) + "' is not connected");
}

void disconnect_mcp_servers(Session &session) {
    for (const auto &name : list_mcp_servers(session)) (void)disconnect_mcp_server(session, name);
}

std::vector<std::string> list_mcp_servers(const Session &session) {
    std::vector<std::string> names;
    for (const auto &ds : session.retained_data_sources)
        if (is_mcp_entry(ds)) names.push_back(ds->name.substr(kDataSourcePrefix.size()));
    return names;
}

} // namespace agent
