#include "engine_remote.hpp"

#include "engine_ipc.hpp"
#include "hal/child_process.hpp"
#include "hal/path.hpp"
#include "hal/platform.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>

namespace agent::engines {

namespace ipc = engine_ipc;
using ipc::Msg;
using Clock = std::chrono::steady_clock;

namespace {

#if defined(AGENT_HAL_WINDOWS)
constexpr const char *kHostName = "agent-engine-host.exe";
#else
constexpr const char *kHostName = "agent-engine-host";
#endif

// Loading a plugin (not the model) should never take this long.
constexpr int kReadyTimeoutMs = 60000;
// Time a host gets to exit on its own after Shutdown before it is killed.
constexpr int kShutdownGraceMs = 2000;
// While a cancel flag is being watched, how often it is polled.
constexpr int kCancelPollMs = 10;
constexpr size_t kStderrTailBytes = 2048;

std::string signal_name(int sig) {
    switch (sig) {
    case 4: return "SIGILL";
    case 6: return "SIGABRT";
    case 7: return "SIGBUS";
    case 8: return "SIGFPE";
    case 9: return "SIGKILL";
    case 11: return "SIGSEGV";
    case 15: return "SIGTERM";
    default: return "signal " + std::to_string(sig);
    }
}

} // namespace

struct RemoteEngine {
    RemoteOptions options;
    RemoteInfo info;
    std::mutex mutex;

    std::unique_ptr<hal::ChildProcess> child; // null while the host is down
    std::thread stderr_thread;
    std::atomic<bool> stderr_stop{false};
    std::mutex tail_mutex;
    std::string stderr_tail;

    // Reused buffers: steady-state calls don't allocate.
    std::string tx;
    std::string rx;
    size_t rx_off = 0;

    int starts = 0;
    std::vector<Clock::time_point> restarts; // restart times within the last minute
    bool skip_restart_count = false;         // the last death was a deliberate kill after cancel
    std::string last_failure;

    ~RemoteEngine() { shutdown(); }

    // ------------------------------------------------------------ process management

    void start_stderr_forwarding() {
        stderr_stop = false;
        {
            std::lock_guard<std::mutex> lk(tail_mutex);
            stderr_tail.clear();
        }
        hal::ChildProcess *c = child.get();
        stderr_thread = std::thread([this, c] {
            std::string buf;
            for (;;) {
                buf.clear();
                const hal::PipeRead r = c->read(hal::ChildStream::Stderr, buf, 100);
                if (r == hal::PipeRead::Data) {
                    // The engine's logs go where an in-process engine's would: our stderr.
                    std::fwrite(buf.data(), 1, buf.size(), stderr);
                    std::lock_guard<std::mutex> lk(tail_mutex);
                    stderr_tail += buf;
                    if (stderr_tail.size() > kStderrTailBytes) stderr_tail.erase(0, stderr_tail.size() - kStderrTailBytes);
                } else if (r == hal::PipeRead::Timeout) {
                    if (stderr_stop) break;
                } else {
                    break;
                }
            }
        });
    }

    // After the child is gone: drain what is left of its stderr and release it.
    void reap() {
        stderr_stop = true;
        if (stderr_thread.joinable()) stderr_thread.join();
        child.reset();
        rx.clear();
        rx_off = 0;
    }

    std::string exit_description() {
        if (child->term_signal()) {
            const int sig = child->term_signal();
            return "killed by " + signal_name(sig) + " (" + std::to_string(sig) + ")";
        }
#if defined(AGENT_HAL_WINDOWS)
        char hex[16];
        std::snprintf(hex, sizeof(hex), "0x%08X", static_cast<unsigned>(child->exit_code()));
        return std::string("exited with code ") + hex;
#else
        return "exited with status " + std::to_string(child->exit_code());
#endif
    }

    std::string stderr_excerpt() {
        std::lock_guard<std::mutex> lk(tail_mutex);
        std::string_view t = stderr_tail;
        while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.remove_suffix(1);
        if (t.empty()) return {};
        // The last few lines are what explains a crash (GGML_ASSERT, abort messages, ...).
        size_t start = t.size();
        for (int lines = 0; start > 0; --start)
            if (t[start - 1] == '\n' && ++lines == 4) break;
        return std::string(t.substr(start));
    }

    std::string engine_label() const { return info.name.empty() ? options.plugin_path : info.name; }

    // The pipe closed or a write failed: the host is dead or dying. Collect how it died.
    Error host_died(std::string_view during) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(1000);
        while (!child->exited() && Clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (!child->exited()) child->terminate(0);
        std::string msg = "Engine host for '" + engine_label() + "' died during " + std::string(during) + ": " + exit_description() + ".";
        reap();
        if (std::string tail = stderr_excerpt(); !tail.empty()) msg += " Last output: " + tail;
        last_failure = msg;
        log(LogLevel::Error, msg);
        return make_error(ErrorCode::EngineCrashed, msg);
    }

    Error kill_host(ErrorCode code, std::string msg) {
        child->terminate(0); // SIGTERM, then SIGKILL after 200 ms
        reap();
        last_failure = msg;
        log(LogLevel::Error, msg);
        return make_error(code, std::move(msg));
    }

    void shutdown() {
        if (!child) return;
        tx.clear();
        ipc::Writer w(tx);
        w.begin(Msg::Shutdown);
        w.finish();
        (void)child->write_stdin(tx);
        child->terminate(kShutdownGraceMs);
        reap();
    }

    void log(LogLevel level, std::string_view msg) const {
        if (options.logger) options.logger(level, msg, options.logger_user_data);
    }

    // ------------------------------------------------------------ framing

    struct Frame {
        Msg type{};
        const char *data = nullptr;
        size_t size = 0;
    };

    Result<void> send() {
        if (auto w = child->write_stdin(tx); !w.ok) return Result<void>{.ok = false, .error = host_died("a write")};
        return ok();
    }

    // Waits for the next frame that is not Chunk / Progress (those go to request's callbacks).
    // request may be null (no callbacks, no cancel). timeout_ms <= 0: no watchdog.
    Result<Frame> wait_reply(std::string_view during, const agent_request *request, int timeout_ms) {
        using R = Result<Frame>;
        auto failure = [](Error e) { return R{.ok = false, .value = {}, .error = std::move(e)}; };

        rx.erase(0, rx_off);
        rx_off = 0;
        const bool has_deadline = timeout_ms > 0;
        const auto deadline = Clock::now() + std::chrono::milliseconds(has_deadline ? timeout_ms : 0);
        const volatile int32_t *cancel = request ? request->cancel : nullptr;
        bool cancel_sent = false;
        Clock::time_point cancel_deadline{};

        for (;;) {
            while (rx.size() - rx_off >= ipc::kHeaderSize) {
                const ipc::FrameHeader h = ipc::read_header(rx.data() + rx_off);
                if (h.size > ipc::kMaxFrame)
                    return failure(kill_host(ErrorCode::EngineCrashed, "Engine host for '" + engine_label() + "' sent a malformed frame."));
                if (rx.size() - rx_off - ipc::kHeaderSize < h.size) break;
                const Frame f{h.type, rx.data() + rx_off + ipc::kHeaderSize, h.size};
                rx_off += ipc::kHeaderSize + h.size;
                if (f.type == Msg::Chunk) {
                    ipc::Reader r(f.data, f.size);
                    const uint32_t kind = r.u32();
                    const uint64_t n = r.u64();
                    const char *bytes = r.bytes(n);
                    if (r.ok() && request && request->on_chunk) request->on_chunk(request->user_data, kind, bytes, static_cast<size_t>(n));
                    continue;
                }
                if (f.type == Msg::Progress) {
                    ipc::Reader r(f.data, f.size);
                    const uint64_t step = r.u64();
                    const uint64_t total = r.u64();
                    if (r.ok() && request && request->on_progress) request->on_progress(request->user_data, step, total);
                    continue;
                }
                return R{.ok = true, .value = f, .error = {}};
            }
            // Consumed chunk frames: keep the buffer from growing during long streams.
            if (rx_off > 0 && rx_off == rx.size()) {
                rx.clear();
                rx_off = 0;
            } else if (rx_off > (64u << 10)) {
                rx.erase(0, rx_off);
                rx_off = 0;
            }

            const auto now = Clock::now();
            if (cancel && !cancel_sent && *cancel != 0) {
                tx.clear();
                ipc::Writer w(tx);
                w.begin(Msg::Cancel);
                w.finish();
                if (auto s = send(); !s.ok) return failure(s.error);
                cancel_sent = true;
                cancel_deadline = now + std::chrono::milliseconds(options.cancel_grace_ms > 0 ? options.cancel_grace_ms : 0);
            }
            if (cancel_sent && now >= cancel_deadline) {
                skip_restart_count = true;
                return failure(kill_host(ErrorCode::Cancelled, "Engine '" + engine_label() + "' did not stop within " +
                                                                   std::to_string(options.cancel_grace_ms) +
                                                                   " ms of the cancel request; its host was killed."));
            }
            if (has_deadline && now >= deadline)
                return failure(kill_host(ErrorCode::EngineCrashed, "Engine '" + engine_label() + "' did not answer within " +
                                                                       std::to_string(timeout_ms) + " ms (" + std::string(during) +
                                                                       "); its host was killed by the watchdog."));

            auto ms_until = [&](Clock::time_point t) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(t - now).count();
                return left > 0 ? static_cast<int>(left) + 1 : 1;
            };
            int wait = -1;
            if (cancel && !cancel_sent) wait = kCancelPollMs;
            if (cancel_sent) wait = ms_until(cancel_deadline);
            if (has_deadline) wait = wait < 0 ? ms_until(deadline) : std::min(wait, ms_until(deadline));

            switch (child->read(hal::ChildStream::Stdout, rx, wait)) {
            case hal::PipeRead::Data:
            case hal::PipeRead::Timeout: break;
            case hal::PipeRead::Closed:
            case hal::PipeRead::Error: return failure(host_died(during));
            }
        }
    }

    // An Error frame -> (status, message).
    static std::pair<int32_t, std::string> parse_error(const Frame &f) {
        ipc::Reader r(f.data, f.size);
        const int32_t status = r.i32();
        const agent_str msg = r.str();
        if (!r.ok()) return {AGENT_ERR_DECODE_FAILED, "malformed error frame"};
        return {status, std::string(msg.data ? msg.data : "", msg.size)};
    }

    Error unexpected(const Frame &f, std::string_view during) {
        return kill_host(ErrorCode::EngineCrashed, "Engine host for '" + engine_label() + "' sent an unexpected message (" +
                                                       std::to_string(static_cast<uint32_t>(f.type)) + ") during " + std::string(during) +
                                                       ".");
    }

    // ------------------------------------------------------------ lifecycle

    Result<void> start() {
        hal::SpawnOptions spawn;
        spawn.program = options.host_path;
        spawn.args = {options.plugin_path};
        spawn.sandbox = options.sandbox;
        auto c = hal::spawn_child(spawn);
        if (!c.ok)
            return fail(ErrorCode::ProviderInitFailed, "Cannot start the engine host '" + options.host_path + "': " + c.error.message +
                                                           " (set Config::engine_host_path or AGENT_ENGINE_HOST).");
        child = std::move(c.value);
        for (const auto &w : child->sandbox_warnings()) log(LogLevel::Warning, w);
        ++starts;
        start_stderr_forwarding();

        auto ready = wait_reply("startup", nullptr, kReadyTimeoutMs);
        if (!ready.ok) return Result<void>{.ok = false, .error = ready.error};
        if (ready.value.type == Msg::Error) {
            auto [status, msg] = parse_error(ready.value);
            (void)status;
            return Result<void>{.ok = false, .error = kill_host(ErrorCode::ProviderInitFailed, msg)};
        }
        if (ready.value.type != Msg::Ready) return Result<void>{.ok = false, .error = unexpected(ready.value, "startup")};
        ipc::Reader r(ready.value.data, ready.value.size);
        const uint32_t protocol = r.u32();
        const uint32_t abi = r.u32();
        RemoteInfo got;
        got.caps = r.u64();
        got.flags = r.u32();
        const uint32_t optional = r.u32();
        const agent_str name = r.str();
        const agent_str version = r.str();
        if (!r.ok() || protocol != ipc::kProtocolVersion || abi != AGENT_ENGINE_ABI_VERSION)
            return Result<void>{.ok = false,
                                .error =
                                    kill_host(ErrorCode::ProviderInitFailed,
                                              "Engine host '" + options.host_path + "' speaks protocol " + std::to_string(protocol) +
                                                  " / ABI " + std::to_string(abi) + "; expected " + std::to_string(ipc::kProtocolVersion) +
                                                  " / " + std::to_string(AGENT_ENGINE_ABI_VERSION) + ".")};
        got.name.assign(name.data ? name.data : "", name.size);
        got.version.assign(version.data ? version.data : "", version.size);
        got.has_count_tokens = (optional & ipc::kHasCountTokens) != 0;
        got.has_reset_cache = (optional & ipc::kHasResetCache) != 0;
        info = std::move(got);

        tx.clear();
        ipc::Writer w(tx);
        w.begin(Msg::Create);
        w.str(options.create_config);
        w.finish();
        if (auto s = send(); !s.ok) return s;
        // Model loading has no deadline (big models take minutes); cancel does not apply to it.
        auto created = wait_reply("model load", nullptr, 0);
        if (!created.ok) return Result<void>{.ok = false, .error = created.error};
        if (created.value.type == Msg::Error) {
            auto [status, msg] = parse_error(created.value);
            (void)status;
            return Result<void>{.ok = false,
                                .error = kill_host(ErrorCode::ModelLoadFailed,
                                                   "Engine '" + info.name + "' failed to load: " + (msg.empty() ? "unknown error" : msg))};
        }
        if (created.value.type != Msg::Created) return Result<void>{.ok = false, .error = unexpected(created.value, "model load")};
        return ok();
    }

    // Restarts a dead host, within options.max_restarts per rolling minute.
    Result<void> ensure_running() {
        if (child) return ok();
        const auto now = Clock::now();
        std::erase_if(restarts, [&](Clock::time_point t) { return now - t > std::chrono::minutes(1); });
        if (!skip_restart_count) {
            if (static_cast<int>(restarts.size()) >= options.max_restarts)
                return fail(ErrorCode::EngineCrashed,
                            "Engine '" + engine_label() + "' is not restarted: " + std::to_string(options.max_restarts) +
                                " restarts in the last minute already (Config::engine_max_restarts). Last failure: " + last_failure);
            restarts.push_back(now);
        }
        skip_restart_count = false;
        log(LogLevel::Warning, "Restarting the engine host for '" + engine_label() + "'.");
        return start();
    }
};

std::string resolve_engine_host_path(std::string_view configured) {
    if (!configured.empty()) return std::string(configured);
    if (const char *env = std::getenv("AGENT_ENGINE_HOST"); env && *env) return env;
    const std::filesystem::path exe = hal::current_executable_path();
    if (!exe.empty()) {
        std::error_code ec;
        for (const auto &candidate : {exe.parent_path() / kHostName, exe.parent_path().parent_path() / "bin" / kHostName})
            if (std::filesystem::is_regular_file(candidate, ec)) return candidate.string();
    }
    return kHostName;
}

Result<std::shared_ptr<RemoteEngine>> start_remote_engine(RemoteOptions options) {
    using R = Result<std::shared_ptr<RemoteEngine>>;
    auto engine = std::make_shared<RemoteEngine>();
    if (options.host_path.empty()) options.host_path = resolve_engine_host_path("");
    if (options.max_restarts < 0) options.max_restarts = 0;
    engine->options = std::move(options);
    std::lock_guard<std::mutex> lk(engine->mutex);
    if (auto s = engine->start(); !s.ok) return R{.ok = false, .value = {}, .error = s.error};
    return R{.ok = true, .value = std::move(engine), .error = {}};
}

const RemoteInfo &remote_info(const RemoteEngine &engine) { return engine.info; }

std::mutex &remote_mutex(RemoteEngine &engine) { return engine.mutex; }

int remote_start_count(const RemoteEngine &engine) { return engine.starts; }

Result<void> remote_run(RemoteEngine &e, const agent_request &request, Arena &arena, agent_response &response) {
    if (auto s = e.ensure_running(); !s.ok) return s;

    e.tx.clear();
    ipc::Writer w(e.tx);
    w.begin(Msg::Run);
    w.request(request, (request.on_chunk ? ipc::kRunStream : 0u) | (request.on_progress ? ipc::kRunProgress : 0u));
    if (!w.finish()) return fail(ErrorCode::InvalidConfig, "Request exceeds the 1 GiB engine IPC frame limit.");
    if (auto s = e.send(); !s.ok) return s;

    auto reply = e.wait_reply("run", &request, e.options.timeout_ms);
    if (!reply.ok) return Result<void>{.ok = false, .error = reply.error};
    const auto &f = reply.value;
    if (f.type == Msg::Error) {
        auto [status, msg] = RemoteEngine::parse_error(f);
        return fail(status == AGENT_ERR_CANCELLED ? ErrorCode::Cancelled : ErrorCode::DecodeFailed,
                    "Engine '" + e.info.name + "': " + (msg.empty() ? std::string("run failed") : msg));
    }
    if (f.type != Msg::Response) return Result<void>{.ok = false, .error = e.unexpected(f, "run")};
    ipc::Reader r(f.data, f.size);
    if (!r.response(response, arena)) {
        if (r.arena_exhausted()) return fail(ErrorCode::DecodeFailed, "Arena capacity exhausted. Increase Config::arena_capacity.");
        return Result<void>{.ok = false,
                            .error =
                                e.kill_host(ErrorCode::EngineCrashed, "Engine host for '" + e.info.name + "' sent a malformed response.")};
    }
    return ok();
}

Result<int> remote_count_tokens(RemoteEngine &e, std::string_view text) {
    std::lock_guard<std::mutex> lk(e.mutex);
    if (!e.info.has_count_tokens) return fail<int>(ErrorCode::UnsupportedProvider, "Engine does not count tokens.");
    if (auto s = e.ensure_running(); !s.ok) return fail<int>(s.error.code, s.error.message);
    e.tx.clear();
    ipc::Writer w(e.tx);
    w.begin(Msg::CountTokens);
    w.str(text);
    w.finish();
    if (auto s = e.send(); !s.ok) return fail<int>(s.error.code, s.error.message);
    auto reply = e.wait_reply("count_tokens", nullptr, e.options.timeout_ms);
    if (!reply.ok) return fail<int>(reply.error.code, reply.error.message);
    if (reply.value.type == Msg::Error) return fail<int>(ErrorCode::DecodeFailed, "Engine token count failed.");
    if (reply.value.type != Msg::Count) {
        Error err = e.unexpected(reply.value, "count_tokens");
        return fail<int>(err.code, err.message);
    }
    ipc::Reader r(reply.value.data, reply.value.size);
    const int32_t n = r.i32();
    if (!r.ok()) return fail<int>(ErrorCode::DecodeFailed, "Engine token count failed.");
    return ok(static_cast<int>(n));
}

Result<void> remote_reset_cache(RemoteEngine &e) {
    std::lock_guard<std::mutex> lk(e.mutex);
    if (!e.info.has_reset_cache) return ok();
    // A host that is down has no cache to reset; the next call starts it fresh.
    if (!e.child) return ok();
    e.tx.clear();
    ipc::Writer w(e.tx);
    w.begin(Msg::ResetCache);
    w.finish();
    if (auto s = e.send(); !s.ok) return s;
    auto reply = e.wait_reply("reset_cache", nullptr, e.options.timeout_ms);
    if (!reply.ok) return Result<void>{.ok = false, .error = reply.error};
    if (reply.value.type == Msg::Error) return fail(ErrorCode::DecodeFailed, "Engine cache reset failed.");
    if (reply.value.type != Msg::Done) return Result<void>{.ok = false, .error = e.unexpected(reply.value, "reset_cache")};
    return ok();
}

} // namespace agent::engines
