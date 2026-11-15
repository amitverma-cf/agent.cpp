// agent-engine-api: remote OpenAI-compatible chat inference through a bun child process.
//
// The C++ side owns everything that must be exact: the conversation, tool calls, usage, cancel, and the retry policy (up to
// max_tries attempts with exponential backoff and jitter, a time limit per attempt, a restart of a hung or dead bridge). The
// child is `bun <bridge.js>` (bridge.js is embedded in this plugin and written to bridge_dir); it makes one HTTP attempt per
// request with plain fetch and speaks line-delimited JSON on stdin / stdout, so no provider SDK or HTTP code is in C++.
//
// Options (engine_options JSON):
//   base_url      "https://host/api/v1" (required): POST {base_url}/chat/completions
//   model         the provider's model id (required)
//   api_key_env   name of the environment variable that holds the key (default NUMES_API_KEY). The key is never read here: the
//                 child reads it from its environment, and it is never put in argv, config, logs or error text.
//   bun           path of the bun executable (default "bun" on PATH); the CLI passes its private install
//   bridge_dir    folder the bridge script is written to (default scratch_dir, else the temp folder)
//   bridge_cmd    program to run instead of bun, and bridge_args (array) its arguments: the bridge is then not written (tests)
//   timeout_ms    per attempt: how long the provider may stay silent (default 90000)
//   max_tries     attempts per request (default 6)
//   retry_base_ms first backoff (default 1000; doubles, capped at retry_max_ms, default 20000), then jittered to 50..100 %
//   stream        false: ask for one non-streamed reply (default true when the caller streams)
//   extra_body    object merged into the request body (provider specific fields)
//   local_only    true: refuse to start (the CLI sets it when "Local only" is on)

#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

extern "C" {
extern const unsigned char agent_api_bridge_js[];
extern const unsigned long agent_api_bridge_js_size;
}

namespace {

using Clock = std::chrono::steady_clock;

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

void json_escape(std::string_view in, std::string &out) {
    for (const char ch : in) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else out += ch;
        }
    }
}

void json_string(std::string_view in, std::string &out) {
    out += '"';
    json_escape(in, out);
    out += '"';
}

// ---------------------------------------------------------------------------------------------------------------------
// The child process: stdin / stdout pipes, stderr discarded.
// ---------------------------------------------------------------------------------------------------------------------

enum class Read { Data, Timeout, Closed };

#if defined(_WIN32)

std::wstring widen(const std::string &s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string quote_arg(const std::string &a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string q = "\"";
    size_t slashes = 0;
    for (const char c : a) {
        if (c == '\\') {
            ++slashes;
        } else if (c == '"') {
            q.append(slashes * 2 + 1, '\\');
            q += '"';
            slashes = 0;
        } else {
            q.append(slashes, '\\');
            slashes = 0;
            q += c;
        }
    }
    q.append(slashes * 2, '\\');
    q += '"';
    return q;
}

class Child {
  public:
    ~Child() { stop(); }

    bool start(const std::string &program, const std::vector<std::string> &args, std::string &err) {
        stop();
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr;
        if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0)) {
            err = "cannot create pipes";
            return false;
        }
        SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
        HANDLE nul =
            CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE | FILE_SHARE_READ, &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        STARTUPINFOW si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = in_r;
        si.hStdOutput = out_w;
        si.hStdError = nul;
        std::string cmd = quote_arg(program);
        for (const std::string &a : args) cmd += " " + quote_arg(a);
        std::wstring wcmd = widen(cmd);
        PROCESS_INFORMATION pi{};
        const BOOL ok = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        const DWORD last = GetLastError();
        CloseHandle(in_r);
        CloseHandle(out_w);
        if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
        if (!ok) {
            CloseHandle(in_w);
            CloseHandle(out_r);
            err = "cannot run '" + program + "' (error " + std::to_string(last) + ")";
            return false;
        }
        CloseHandle(pi.hThread);
        process_ = pi.hProcess;
        stdin_ = in_w;
        stdout_ = out_r;
        return true;
    }

    bool write_all(std::string_view data) {
        while (!data.empty()) {
            DWORD n = 0;
            if (!stdin_ || !WriteFile(stdin_, data.data(), static_cast<DWORD>(std::min<size_t>(data.size(), 1 << 16)), &n, nullptr) ||
                n == 0)
                return false;
            data.remove_prefix(n);
        }
        return true;
    }

    Read read(std::string &out, int timeout_ms) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            DWORD avail = 0;
            if (!stdout_ || !PeekNamedPipe(stdout_, nullptr, 0, nullptr, &avail, nullptr)) return Read::Closed;
            if (avail > 0) {
                char buf[16384];
                DWORD n = 0;
                if (!ReadFile(stdout_, buf, static_cast<DWORD>(std::min<size_t>(avail, sizeof(buf))), &n, nullptr) || n == 0)
                    return Read::Closed;
                out.append(buf, n);
                return Read::Data;
            }
            if (Clock::now() >= deadline) return Read::Timeout;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    bool alive() { return process_ && WaitForSingleObject(process_, 0) == WAIT_TIMEOUT; }

    void stop() {
        if (stdin_) CloseHandle(stdin_), stdin_ = nullptr;
        if (process_) {
            if (WaitForSingleObject(process_, 1500) == WAIT_TIMEOUT) TerminateProcess(process_, 1);
            CloseHandle(process_);
            process_ = nullptr;
        }
        if (stdout_) CloseHandle(stdout_), stdout_ = nullptr;
    }

    void kill_now() {
        if (process_) TerminateProcess(process_, 1);
        stop();
    }

    int exit_code() const { return -1; }

  private:
    HANDLE process_ = nullptr, stdin_ = nullptr, stdout_ = nullptr;
};

#else

class Child {
  public:
    ~Child() { stop(); }

    bool start(const std::string &program, const std::vector<std::string> &args, std::string &err) {
        stop();
        int in_pipe[2], out_pipe[2];
        if (::pipe(in_pipe) != 0) {
            err = std::string("cannot create a pipe: ") + std::strerror(errno);
            return false;
        }
        if (::pipe(out_pipe) != 0) {
            err = std::string("cannot create a pipe: ") + std::strerror(errno);
            ::close(in_pipe[0]);
            ::close(in_pipe[1]);
            return false;
        }
        for (const int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]}) ::fcntl(fd, F_SETFD, FD_CLOEXEC);

        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, in_pipe[0], 0);
        posix_spawn_file_actions_adddup2(&fa, out_pipe[1], 1);
        posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        posix_spawnattr_t attr;
        posix_spawnattr_init(&attr);
        sigset_t none, dflt;
        sigemptyset(&none);
        sigemptyset(&dflt);
        sigaddset(&dflt, SIGPIPE);
        posix_spawnattr_setsigmask(&attr, &none);
        posix_spawnattr_setsigdefault(&attr, &dflt);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);

        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(program.c_str()));
        for (const std::string &a : args) argv.push_back(const_cast<char *>(a.c_str()));
        argv.push_back(nullptr);
        pid_t pid = 0;
        const int rc = ::posix_spawnp(&pid, program.c_str(), &fa, &attr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        posix_spawnattr_destroy(&attr);
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);
        if (rc != 0) {
            ::close(in_pipe[1]);
            ::close(out_pipe[0]);
            err = "cannot run '" + program + "': " + std::strerror(rc);
            return false;
        }
        pid_ = pid;
        stdin_ = in_pipe[1];
        stdout_ = out_pipe[0];
        return true;
    }

    // SIGPIPE is blocked for the write on this thread only, so the host's signal settings are not touched.
    bool write_all(std::string_view data) {
        sigset_t pipe_set, old;
        sigemptyset(&pipe_set);
        sigaddset(&pipe_set, SIGPIPE);
        pthread_sigmask(SIG_BLOCK, &pipe_set, &old);
        bool ok = stdin_ >= 0;
        while (ok && !data.empty()) {
            const ssize_t n = ::write(stdin_, data.data(), data.size());
            if (n < 0) {
                if (errno == EINTR) continue;
                ok = false;
            } else data.remove_prefix(static_cast<size_t>(n));
        }
        if (!ok) {
            const struct timespec zero{0, 0};
            while (::sigtimedwait(&pipe_set, nullptr, &zero) > 0) {}
        }
        pthread_sigmask(SIG_SETMASK, &old, nullptr);
        return ok;
    }

    Read read(std::string &out, int timeout_ms) {
        if (stdout_ < 0) return Read::Closed;
        pollfd p{stdout_, POLLIN, 0};
        const int rc = ::poll(&p, 1, timeout_ms);
        if (rc < 0) return errno == EINTR ? Read::Timeout : Read::Closed;
        if (rc == 0) return Read::Timeout;
        char buf[16384];
        const ssize_t n = ::read(stdout_, buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, static_cast<size_t>(n));
            return Read::Data;
        }
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) return Read::Timeout;
        return Read::Closed;
    }

    bool alive() {
        if (pid_ <= 0) return false;
        int status = 0;
        const pid_t r = ::waitpid(pid_, &status, WNOHANG);
        if (r == 0) return true;
        pid_ = -1;
        return false;
    }

    // Closing stdin ends the bridge; after a grace period it is killed.
    void stop() {
        if (stdin_ >= 0) ::close(stdin_), stdin_ = -1;
        if (pid_ > 0) {
            for (int i = 0; i < 30; ++i) {
                int status = 0;
                if (::waitpid(pid_, &status, WNOHANG) != 0) {
                    pid_ = -1;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (pid_ > 0) {
                ::kill(pid_, SIGKILL);
                int status = 0;
                ::waitpid(pid_, &status, 0);
                pid_ = -1;
            }
        }
        if (stdout_ >= 0) ::close(stdout_), stdout_ = -1;
    }

    void kill_now() {
        if (pid_ > 0) ::kill(pid_, SIGKILL);
        stop();
    }

  private:
    pid_t pid_ = -1;
    int stdin_ = -1, stdout_ = -1;
};

#endif

// ---------------------------------------------------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------------------------------------------------

struct Options {
    std::string base_url, model, key_env = "NUMES_API_KEY", bun = "bun", bridge_dir, bridge_cmd, extra_body;
    std::vector<std::string> bridge_args;
    int timeout_ms = 90000;
    int max_tries = 6;
    int retry_base_ms = 1000;
    int retry_max_ms = 20000;
    bool stream = true;
};

struct Attempt {
    enum Kind { Done, Failed, Cancelled } kind = Failed;
    bool retryable = true;
    std::string message;
    std::string content;
    struct Call {
        std::string id, name, arguments;
    };
    std::vector<Call> calls;
    int prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
};

struct Instance {
    Options opt;
    Child child;
    bool ready = false;
    std::string pending; // bytes read from the bridge that do not yet form a whole line
    uint64_t next_id = 1;
    std::string error;
    std::mt19937 rng{std::random_device{}()};

    // What the last response points at.
    std::string reply;
    std::vector<Attempt::Call> calls;
    std::vector<agent_tool_call> c_calls;
    agent_part part{};
};

std::string stable_name(std::string_view content) {
    uint64_t h = 1469598103934665603ull;
    for (const char c : content) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    char buf[40];
    std::snprintf(buf, sizeof(buf), "api-bridge-%016llx.js", static_cast<unsigned long long>(h));
    return buf;
}

// Writes the embedded bridge script (once per content hash) and returns its path.
bool write_bridge(const Options &opt, std::string &path, std::string &err) {
    const std::string_view js(reinterpret_cast<const char *>(agent_api_bridge_js), static_cast<size_t>(agent_api_bridge_js_size));
    std::error_code ec;
    std::filesystem::path dir = opt.bridge_dir.empty() ? std::filesystem::temp_directory_path(ec) : std::filesystem::path(opt.bridge_dir);
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path file = dir / stable_name(js);
    if (std::filesystem::is_regular_file(file, ec) && std::filesystem::file_size(file, ec) == js.size()) {
        path = file.string();
        return true;
    }
    const std::filesystem::path tmp =
        file.string() + ".tmp" + std::to_string(static_cast<unsigned long long>(Clock::now().time_since_epoch().count()));
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(js.data(), static_cast<std::streamsize>(js.size()));
        if (!out.flush()) {
            err = "cannot write the bridge script to " + dir.string();
            return false;
        }
    }
    std::filesystem::rename(tmp, file, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        if (!std::filesystem::is_regular_file(file)) {
            err = "cannot write the bridge script to " + dir.string();
            return false;
        }
    }
    path = file.string();
    return true;
}

enum class Line { Got, Timeout, Closed };

// The next complete line from the bridge (without the newline).
Line next_line(Instance &in, std::string &line, int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const size_t nl = in.pending.find('\n');
        if (nl != std::string::npos) {
            line.assign(in.pending, 0, nl);
            in.pending.erase(0, nl + 1);
            return Line::Got;
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (left <= 0) return Line::Timeout;
        const Read r = in.child.read(in.pending, static_cast<int>(std::min<long long>(left, 50)));
        if (r == Read::Closed) {
            // Whatever is complete has been handled above; the rest of a partial line is lost with the process.
            return Line::Closed;
        }
    }
}

bool start_bridge(Instance &in, std::string &err) {
    in.ready = false;
    in.pending.clear();
    std::string program = in.opt.bun;
    std::vector<std::string> args;
    if (!in.opt.bridge_cmd.empty()) {
        program = in.opt.bridge_cmd;
        args = in.opt.bridge_args;
    } else {
        std::string script;
        if (!write_bridge(in.opt, script, err)) return false;
        args.push_back(script);
    }
    if (!in.child.start(program, args, err)) {
        if (in.opt.bridge_cmd.empty())
            err += "; the api engine runs on bun: install it with `agent runtime install bun` or set the \"bun\" option";
        return false;
    }
    // The bridge announces itself; a runtime that exits at once (a bad install) shows up here.
    std::string line;
    const auto begin = Clock::now();
    for (;;) {
        const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - begin).count();
        if (waited > 20000) {
            in.child.kill_now();
            err = "the bridge did not start within 20 s";
            return false;
        }
        const Line l = next_line(in, line, 200);
        if (l == Line::Closed) {
            in.child.kill_now();
            err = "the bridge exited before it was ready (is '" + program + "' a working bun?)";
            return false;
        }
        if (l == Line::Got && line.find("\"ready\"") != std::string::npos) {
            in.ready = true;
            return true;
        }
    }
}

bool ensure_bridge(Instance &in, std::string &err) {
    if (in.ready && in.child.alive()) return true;
    in.child.kill_now();
    return start_bridge(in, err);
}

int sleep_cancelable(int ms, const volatile int32_t *cancel) {
    const auto end = Clock::now() + std::chrono::milliseconds(ms);
    while (Clock::now() < end) {
        if (cancel && *cancel) return 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return 0;
}

struct Chunker {
    agent_chunk_fn fn;
    void *user;
    size_t emitted = 0;  // bytes already given to the caller, over every attempt
    size_t this_try = 0; // bytes of the current attempt's text
    // A retry re-sends the text from the start: only the part beyond what the caller already has is forwarded.
    void push(std::string_view text) {
        const size_t begin = this_try;
        this_try += text.size();
        if (!fn || this_try <= emitted) return;
        const size_t skip = emitted > begin ? emitted - begin : 0;
        fn(user, AGENT_CHUNK_TEXT, text.data() + skip, text.size() - skip);
        emitted = this_try;
    }
};

// One request / answer exchange with the bridge. Returns when the bridge reports done or error, when the attempt times out, or
// when the bridge dies.
Attempt exchange(Instance &in, const std::string &request_fields, Chunker &chunker, const volatile int32_t *cancel) {
    Attempt a;
    std::string err;
    if (!ensure_bridge(in, err)) {
        a.message = err;
        a.retryable = true;
        return a;
    }
    chunker.this_try = 0;
    const uint64_t id = in.next_id++;
    std::string line = "{\"op\":\"chat\",\"id\":" + std::to_string(id) + "," + request_fields + "}\n";
    if (!in.child.write_all(line)) {
        in.child.kill_now();
        a.message = "the bridge process is gone";
        return a;
    }
    const int silence_ms = in.opt.timeout_ms + 10000; // the bridge enforces timeout_ms itself; this is the guard if it hangs
    auto last_data = Clock::now();
    bool cancel_sent = false;
    auto cancel_at = Clock::now();
    simdjson::dom::parser parser;
    for (;;) {
        if (cancel && *cancel && !cancel_sent) {
            cancel_sent = true;
            cancel_at = Clock::now();
            in.child.write_all("{\"op\":\"cancel\",\"id\":" + std::to_string(id) + "}\n");
        }
        if (cancel_sent && Clock::now() - cancel_at > std::chrono::seconds(3)) {
            in.child.kill_now();
            a.kind = Attempt::Cancelled;
            a.retryable = false;
            a.message = "cancelled";
            return a;
        }
        std::string text;
        const Line l = next_line(in, text, 100);
        if (l == Line::Closed) {
            in.child.kill_now();
            if (cancel_sent) {
                a.kind = Attempt::Cancelled;
                a.retryable = false;
                a.message = "cancelled";
                return a;
            }
            a.message = "the bridge process exited";
            return a;
        }
        if (l == Line::Timeout) {
            if (Clock::now() - last_data > std::chrono::milliseconds(silence_ms)) {
                in.child.kill_now();
                a.message = "timed out after " + std::to_string(in.opt.timeout_ms / 1000) + " s without data";
                return a;
            }
            continue;
        }
        last_data = Clock::now();
        simdjson::dom::element doc;
        if (parser.parse(text).get(doc)) continue; // not ours (a runtime banner); ignore
        int64_t line_id = 0;
        if (doc["id"].get(line_id) || static_cast<uint64_t>(line_id) != id) continue;
        std::string_view type;
        if (doc["t"].get(type)) continue;
        if (type == "delta") {
            std::string_view t;
            if (!doc["text"].get(t)) chunker.push(t);
        } else if (type == "done") {
            a.kind = Attempt::Done;
            std::string_view s;
            if (!doc["content"].get(s)) a.content.assign(s);
            simdjson::dom::array arr;
            if (!doc["tool_calls"].get(arr))
                for (simdjson::dom::element c : arr) {
                    Attempt::Call call;
                    std::string_view v;
                    if (!c["id"].get(v)) call.id.assign(v);
                    if (!c["name"].get(v)) call.name.assign(v);
                    if (!c["arguments"].get(v)) call.arguments.assign(v);
                    a.calls.push_back(std::move(call));
                }
            int64_t n = 0;
            if (!doc["usage"]["prompt_tokens"].get(n)) a.prompt_tokens = static_cast<int>(n);
            if (!doc["usage"]["completion_tokens"].get(n)) a.completion_tokens = static_cast<int>(n);
            if (!doc["usage"]["cached_tokens"].get(n)) a.cached_tokens = static_cast<int>(n);
            return a;
        } else if (type == "error") {
            std::string_view s;
            if (!doc["message"].get(s)) a.message.assign(s);
            bool flag = false;
            if (!doc["retryable"].get(flag)) a.retryable = flag;
            if (!doc["cancelled"].get(flag) && flag) {
                a.kind = Attempt::Cancelled;
                a.retryable = false;
            }
            return a;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------------------------------------------------

void append_messages(const agent_request *req, std::string &out) {
    out += '[';
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        if (i) out += ',';
        out += "{\"role\":";
        json_string(sv(m.role), out);
        std::string text;
        for (size_t p = 0; p < m.n_parts; ++p) {
            const agent_part &part = m.parts[p];
            if (part.kind != AGENT_PART_TEXT) continue; // the engine declares no media input
            if (!text.empty()) text += "\n";
            text.append(sv(part.text));
        }
        out += ",\"content\":";
        json_string(text, out);
        if (m.tool_call_id.size) {
            out += ",\"tool_call_id\":";
            json_string(sv(m.tool_call_id), out);
        }
        if (m.n_tool_calls) {
            out += ",\"tool_calls\":[";
            for (size_t c = 0; c < m.n_tool_calls; ++c) {
                const agent_tool_call &tc = m.tool_calls[c];
                if (c) out += ',';
                out += "{\"id\":";
                json_string(sv(tc.id), out);
                out += ",\"type\":\"function\",\"function\":{\"name\":";
                json_string(sv(tc.name), out);
                out += ",\"arguments\":";
                json_string(tc.arguments.size ? sv(tc.arguments) : std::string_view("{}"), out);
                out += "}}";
            }
            out += ']';
        }
        out += '}';
    }
    out += ']';
}

// Fields every attempt of one run() shares (everything but the op and the id).
std::string request_fields(const Options &opt, const agent_request *req, bool stream) {
    std::string f = "\"base_url\":";
    json_string(opt.base_url, f);
    f += ",\"model\":";
    json_string(opt.model, f);
    f += ",\"key_env\":";
    json_string(opt.key_env, f);
    f += ",\"timeout_ms\":" + std::to_string(opt.timeout_ms);
    f += stream ? ",\"stream\":true" : ",\"stream\":false";

    simdjson::dom::parser parser;
    simdjson::dom::element params;
    const std::string pj(sv(req->params_json));
    if (!pj.empty() && !parser.parse(pj).get(params)) {
        int64_t mt = 0;
        if (!params["max_tokens"].get(mt) && mt > 0) f += ",\"max_tokens\":" + std::to_string(mt);
        double t = 0;
        if (!params["temperature"].get(t) && t >= 0) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), ",\"temperature\":%.4f", t);
            f += buf;
        }
    }
    const std::string_view tools = sv(req->tools_json);
    if (!tools.empty() && tools.front() == '[') {
        f += ",\"tools\":";
        f.append(tools);
    }
    if (!opt.extra_body.empty()) {
        f += ",\"extra_body\":";
        f += opt.extra_body;
    }
    f += ",\"messages\":";
    append_messages(req, f);
    return f;
}

// ---------------------------------------------------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------------------------------------------------

void set_err(char *err, size_t cap, const std::string &msg) {
    if (err && cap) std::snprintf(err, cap, "%s", msg.c_str());
}

int32_t api_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    const std::string text(sv(config_json));
    if (parser.parse(text).get(doc)) {
        set_err(err, err_cap, "api: the config is not valid JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    auto inst = std::make_unique<Instance>();
    Options &o = inst->opt;
    std::string_view s;
    std::string scratch;
    if (!doc["scratch_dir"].get(s)) scratch.assign(s);
    simdjson::dom::object opts;
    if (doc["options"].get(opts)) {
        set_err(err, err_cap, "api: engine options are missing; use `agent model add-api NAME --base-url URL --model ID`");
        return AGENT_ERR_INVALID_CONFIG;
    }
    const auto str = [&](const char *key, std::string &dst) {
        std::string_view v;
        if (!opts[key].get(v) && !v.empty()) dst.assign(v);
    };
    const auto num = [&](const char *key, int &dst, int lo) {
        int64_t v = 0;
        if (!opts[key].get(v) && v >= lo && v <= 3600000) dst = static_cast<int>(v);
    };
    str("base_url", o.base_url);
    str("model", o.model);
    str("api_key_env", o.key_env);
    str("bun", o.bun);
    str("bridge_dir", o.bridge_dir);
    str("bridge_cmd", o.bridge_cmd);
    num("timeout_ms", o.timeout_ms, 1000);
    num("max_tries", o.max_tries, 1);
    num("retry_base_ms", o.retry_base_ms, 0);
    num("retry_max_ms", o.retry_max_ms, 0);
    bool b = true;
    if (!opts["stream"].get(b)) o.stream = b;
    simdjson::dom::array args;
    if (!opts["bridge_args"].get(args))
        for (simdjson::dom::element a : args)
            if (!a.get(s)) o.bridge_args.emplace_back(s);
    simdjson::dom::element extra;
    if (!opts["extra_body"].get(extra) && extra.is_object()) o.extra_body = simdjson::minify(extra);
    if (o.bridge_dir.empty()) o.bridge_dir = scratch;

    bool local_only = false;
    if (!opts["local_only"].get(local_only) && local_only) {
        set_err(err, err_cap,
                "api: this engine sends the conversation to a remote service, and Local only is on (Settings). Turn it off or pick a local "
                "model");
        return AGENT_ERR_UNSUPPORTED;
    }
    if (o.base_url.rfind("http://", 0) != 0 && o.base_url.rfind("https://", 0) != 0) {
        set_err(err, err_cap, "api: option base_url must start with http:// or https://");
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (o.model.empty()) {
        set_err(err, err_cap, "api: option model (the provider's model id) is missing");
        return AGENT_ERR_INVALID_CONFIG;
    }
    std::string why;
    if (!start_bridge(*inst, why)) {
        set_err(err, err_cap, "api: " + why);
        return AGENT_ERR_INIT_FAILED;
    }
    *out_instance = inst.release();
    return AGENT_OK;
}

void api_destroy(void *instance) { delete static_cast<Instance *>(instance); }

int32_t api_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *in = static_cast<Instance *>(instance);
    agent_response_reset(resp);
    in->error.clear();
    if (req->task != AGENT_CAP_TEXT_GEN) {
        in->error = "api: only chat is supported";
        return AGENT_ERR_UNSUPPORTED;
    }
    const bool stream = in->opt.stream && req->on_chunk != nullptr;
    const std::string fields = request_fields(in->opt, req, stream);
    Chunker chunker{req->on_chunk, req->user_data};
    const auto started = Clock::now();
    int retries = 0;
    Attempt result;
    for (int attempt = 1;; ++attempt) {
        if (req->cancel && *req->cancel) {
            in->error = "cancelled";
            return AGENT_ERR_CANCELLED;
        }
        result = exchange(*in, fields, chunker, req->cancel);
        if (result.kind == Attempt::Done) break;
        if (result.kind == Attempt::Cancelled) {
            in->error = "cancelled";
            return AGENT_ERR_CANCELLED;
        }
        if (!result.retryable || attempt >= in->opt.max_tries) {
            in->error = "api: " + result.message;
            if (retries) in->error += " (after " + std::to_string(retries + 1) + " tries)";
            return AGENT_ERR_DECODE_FAILED;
        }
        ++retries;
        long long delay = static_cast<long long>(in->opt.retry_base_ms) << std::min(retries - 1, 16);
        delay = std::min<long long>(delay, in->opt.retry_max_ms);
        std::uniform_int_distribution<long long> jitter(delay / 2, delay);
        if (delay > 0 && sleep_cancelable(static_cast<int>(jitter(in->rng)), req->cancel)) {
            in->error = "cancelled";
            return AGENT_ERR_CANCELLED;
        }
    }

    in->reply = std::move(result.content);
    in->calls = std::move(result.calls);
    in->c_calls.clear();
    for (const Attempt::Call &c : in->calls)
        in->c_calls.push_back(
            agent_tool_call{{c.id.data(), c.id.size()}, {c.name.data(), c.name.size()}, {c.arguments.data(), c.arguments.size()}});
    resp->message.role = agent_str{"assistant", 9};
    if (!in->reply.empty()) {
        in->part = agent_part{};
        in->part.kind = AGENT_PART_TEXT;
        in->part.text = agent_str{in->reply.data(), in->reply.size()};
        resp->message.parts = &in->part;
        resp->message.n_parts = 1;
    }
    if (!in->c_calls.empty()) {
        resp->message.tool_calls = in->c_calls.data();
        resp->message.n_tool_calls = in->c_calls.size();
    }
    resp->usage.prompt_tokens = result.prompt_tokens;
    resp->usage.completion_tokens = result.completion_tokens;
    if (AGENT_HAS_FIELD(resp, agent_response, stats)) {
        resp->stats.cached_tokens = result.cached_tokens;
        resp->stats.decode_ms = static_cast<float>(std::chrono::duration<double, std::milli>(Clock::now() - started).count());
        resp->stats.retries = retries;
    }
    return AGENT_OK;
}

void api_free_response(void *instance, agent_response *resp) {
    (void)instance;
    agent_response_reset(resp);
}

const char *api_last_error(void *instance) { return static_cast<Instance *>(instance)->error.c_str(); }

const agent_engine_v1 kEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "api",
    "1.0",
    AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL,
    AGENT_ENGINE_FLAG_NATIVE_TOOLS,
    api_create,
    api_destroy,
    api_run,
    api_free_response,
    api_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kEngine; }
