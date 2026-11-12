#include "engine_remote.hpp"

#include "engine_ipc.hpp"
#include "engine_shm.hpp"
#include "hal/child_process.hpp"
#include "hal/path.hpp"
#include "hal/platform.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
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
// How often the reading caller wakes up to poll cancel flags and deadlines of the calls in flight.
constexpr int kPollMs = 10;
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

bool is_final(Msg type) { return type != Msg::Chunk && type != Msg::Progress; }

bool shm_enabled_by_env() {
    const char *v = std::getenv("AGENT_ENGINE_SHM");
    return !(v && v[0] == '0' && v[1] == '\0');
}

} // namespace

// One request in flight (or a free slot). Slots are pooled and reused, so steady-state calls don't
// allocate: every buffer keeps its capacity.
struct RemoteCallSlot {
    uint64_t id = 0;
    uint64_t generation = 0; // host generation the request was sent to
    bool active = false;
    const agent_request *request = nullptr; // callbacks and cancel flag; null for control calls

    bool has_deadline = false;
    Clock::time_point deadline{};
    bool cancel_sent = false;
    Clock::time_point cancel_deadline{};

    // Set by the reading caller under RemoteEngine::mutex.
    std::string inbox;         // this call's frames (header + payload), appended in arrival order
    bool final_queued = false; // inbox holds the final frame
    bool failed = false;
    Error error;
    std::condition_variable cv;

    // Owned by the calling thread.
    std::string tx;   // the encoded request frame
    std::string work; // inbox, swapped out for processing; the final frame stays here
    bool done = false;
    Msg final_type{};
    const char *final_data = nullptr;
    size_t final_size = 0;
    // Large request payloads: a segment of this slot's own, reused (and grown) across its calls. The
    // host maps it once and keeps the mapping; after the first exchange the name is removed.
    engine_shm::Segment request_shm;
    uint64_t request_shm_generation = 0; // the host that knows it
    bool request_shm_named = false;      // its name still exists
    std::string retire;                  // a replaced segment the host should unmap (sent with the next Run)
    // The host's response segment this call's response points into, until remote_release (Release).
    const engine_shm::Segment *response_shm = nullptr;
};

struct RemoteEngine {
    RemoteOptions options;
    RemoteInfo info;

    // Lock order: mutex, then write_mutex. serial_mutex is taken before both.
    std::mutex mutex;        // everything below unless noted
    std::mutex write_mutex;  // pipe writes; child / generation changes also take it
    std::mutex serial_mutex; // one run at a time for engines without AGENT_ENGINE_FLAG_CONCURRENT

    std::unique_ptr<hal::ChildProcess> child; // null while the host is down
    uint64_t generation = 0;                  // bumped whenever the host goes down
    long host_pid = 0;
    std::thread stderr_thread;
    std::atomic<bool> stderr_stop{false};
    std::mutex tail_mutex;
    std::string stderr_tail;

    // The pipe is read by one waiting caller at a time (the reader); it hands frames to their calls.
    bool reader_active = false;
    std::string rx; // reader only (or start(), which runs with no call in flight)
    size_t rx_off = 0;
    std::string ctl; // Cancel / Release frames, written under mutex

    std::vector<std::unique_ptr<RemoteCallSlot>> slots;
    uint64_t next_id = 1;
    int in_flight = 0;
    int max_in_flight = 0;

    int starts = 0;
    std::vector<Clock::time_point> restarts; // restart times within the last minute
    bool skip_restart_count = false;         // the last death was a deliberate kill after cancel
    std::string last_failure;
    std::atomic<uint64_t> shm_requests{0};
    std::atomic<uint64_t> shm_responses{0};
    // The host's response segments, mapped once and reused (the host reuses them after Release).
    std::vector<std::unique_ptr<engine_shm::Segment>> response_segments;

    ~RemoteEngine() {
        std::lock_guard<std::mutex> lk(mutex);
        shutdown();
    }

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

    // After the child is gone: drain what is left of its stderr, release it and remove any shared
    // memory it left behind. Holds mutex.
    void reap() {
        stderr_stop = true;
        if (stderr_thread.joinable()) stderr_thread.join();
        {
            std::lock_guard<std::mutex> wl(write_mutex);
            child.reset();
            ++generation;
        }
        if (host_pid) engine_shm::sweep('h', host_pid);
        host_pid = 0;
        response_segments.clear();
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

    // Fails every call in flight on the current host. trigger (if any) gets trigger_code, the others
    // EngineCrashed. Calls whose final frame already arrived keep it.
    void fail_all(const std::string &msg, RemoteCallSlot *trigger, ErrorCode trigger_code) {
        for (auto &s : slots) {
            if (!s->active || s->generation != generation || s->final_queued || s->failed) continue;
            s->failed = true;
            s->error = make_error(s.get() == trigger ? trigger_code : ErrorCode::EngineCrashed, msg);
            s->cv.notify_all();
        }
    }

    // The pipe closed or a write failed: the host is dead or dying. Collect how it died and fail every
    // call in flight. Holds mutex; no other caller is reading.
    Error host_died(std::string_view during) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(1000);
        while (!child->exited() && Clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (!child->exited()) {
            std::lock_guard<std::mutex> wl(write_mutex);
            child->terminate(0);
        }
        std::string msg = "Engine host for '" + engine_label() + "' died during " + std::string(during) + ": " + exit_description() + ".";
        if (in_flight > 1) msg += " Every request in flight on it failed.";
        stderr_stop = true;
        if (stderr_thread.joinable()) stderr_thread.join();
        if (std::string tail = stderr_excerpt(); !tail.empty()) msg += " Last output: " + tail;
        fail_all(msg, nullptr, ErrorCode::EngineCrashed);
        reap();
        last_failure = msg;
        log(LogLevel::Error, msg);
        return make_error(ErrorCode::EngineCrashed, msg);
    }

    // Kills the host on purpose (watchdog, ignored cancel, protocol violation) and fails every call in
    // flight. Holds mutex; no other caller is reading.
    Error kill_host(ErrorCode code, std::string msg, RemoteCallSlot *trigger = nullptr) {
        {
            std::lock_guard<std::mutex> wl(write_mutex);
            child->terminate(0); // SIGTERM, then SIGKILL after 200 ms
        }
        std::string others = msg;
        if (in_flight > 1) others = "Engine host for '" + engine_label() + "' was killed while this request was in flight: " + msg;
        for (auto &s : slots) {
            if (!s->active || s->generation != generation || s->final_queued || s->failed) continue;
            s->failed = true;
            s->error = make_error(s.get() == trigger ? code : ErrorCode::EngineCrashed, s.get() == trigger ? msg : others);
            s->cv.notify_all();
        }
        reap();
        last_failure = msg;
        log(LogLevel::Error, msg);
        return make_error(code, std::move(msg));
    }

    void shutdown() {
        if (!child) return;
        ctl.clear();
        ipc::Writer w(ctl);
        w.begin(Msg::Shutdown);
        w.finish();
        {
            std::lock_guard<std::mutex> wl(write_mutex);
            (void)child->write_stdin(ctl);
            child->terminate(kShutdownGraceMs);
        }
        reap();
    }

    void log(LogLevel level, std::string_view msg) const {
        if (options.logger) options.logger(level, msg, options.logger_user_data);
    }

    // ------------------------------------------------------------ framing

    // Writes one whole frame to the host started as generation gen. false if that host is gone or the
    // write failed (the reader then sees the pipe close and reports how the host died).
    bool write_frame(const std::string &frame, uint64_t gen) {
        std::lock_guard<std::mutex> wl(write_mutex);
        if (!child || gen != generation) return false;
        return child->write_stdin(frame).ok;
    }

    // A small control frame (Cancel / Release) for request id. Holds mutex.
    void send_control(Msg type, uint64_t id) {
        ctl.clear();
        ipc::Writer w(ctl);
        w.begin(type, id);
        w.finish();
        (void)write_frame(ctl, generation);
    }

    RemoteCallSlot *find_slot(uint64_t id) {
        for (auto &s : slots)
            if (s->active && s->id == id) return s.get();
        return nullptr;
    }

    // A Response nobody waits for any more: hand its segment back to the host.
    void drop_frame(const ipc::FrameHeader &h, const char *payload) {
        if (h.type != Msg::Response) return;
        ipc::Reader r(payload, h.size);
        const agent_str name = r.str();
        (void)r.u64();
        const agent_str retire = r.str();
        if (!r.ok()) return;
        if (retire.size) forget_response_segment(std::string_view(retire.data, retire.size));
        if (name.size) send_control(Msg::Release, h.id);
    }

    // The mapping of host response segment name, mapping it on first use. Holds mutex.
    const engine_shm::Segment *response_segment(std::string_view name, size_t size) {
        for (const auto &s : response_segments)
            if (s->name() == name) return s->size() >= size ? s.get() : nullptr;
        auto seg = std::make_unique<engine_shm::Segment>();
        if (!seg->open(name, size)) return nullptr;
        engine_shm::unlink(name); // both sides keep it mapped; the name is no longer needed
        return response_segments.emplace_back(std::move(seg)).get();
    }

    void forget_response_segment(std::string_view name) {
        std::erase_if(response_segments, [&](const std::unique_ptr<engine_shm::Segment> &s) { return s->name() == name; });
    }

    // Hands every complete frame in rx to its call. Holds mutex. false after killing the host for a
    // malformed frame.
    bool dispatch() {
        while (rx.size() - rx_off >= ipc::kHeaderSize) {
            const ipc::FrameHeader h = ipc::read_header(rx.data() + rx_off);
            if (h.size > ipc::kMaxFrame) {
                kill_host(ErrorCode::EngineCrashed, "Engine host for '" + engine_label() + "' sent a malformed frame.");
                return false;
            }
            if (rx.size() - rx_off - ipc::kHeaderSize < h.size) break;
            const size_t total = ipc::kHeaderSize + h.size;
            if (RemoteCallSlot *s = find_slot(h.id); s && s->generation == generation && !s->failed) {
                s->inbox.append(rx, rx_off, total);
                if (is_final(h.type)) s->final_queued = true;
                s->cv.notify_all();
            } else {
                drop_frame(h, rx.data() + rx_off + ipc::kHeaderSize);
            }
            rx_off += total;
        }
        // Consumed frames: keep the buffer from growing during long streams.
        if (rx_off > 0 && rx_off == rx.size()) {
            rx.clear();
            rx_off = 0;
        } else if (rx_off > (64u << 10)) {
            rx.erase(0, rx_off);
            rx_off = 0;
        }
        return true;
    }

    // As the reader: forwards cancel requests and enforces the cancel grace period and the watchdog of
    // every call in flight. Holds mutex. false after killing the host.
    bool service_timers() {
        const auto now = Clock::now();
        for (auto &sp : slots) {
            RemoteCallSlot &s = *sp;
            if (!s.active || s.generation != generation || s.final_queued || s.failed) continue;
            const volatile int32_t *cancel = s.request ? s.request->cancel : nullptr;
            if (cancel && !s.cancel_sent && *cancel != 0) {
                send_control(Msg::Cancel, s.id);
                s.cancel_sent = true;
                s.cancel_deadline = now + std::chrono::milliseconds(options.cancel_grace_ms > 0 ? options.cancel_grace_ms : 0);
            }
            if (s.cancel_sent && now >= s.cancel_deadline) {
                skip_restart_count = true;
                kill_host(ErrorCode::Cancelled,
                          "Engine '" + engine_label() + "' did not stop within " + std::to_string(options.cancel_grace_ms) +
                              " ms of the cancel request; its host was killed.",
                          &s);
                return false;
            }
            if (s.has_deadline && now >= s.deadline) {
                kill_host(ErrorCode::EngineCrashed,
                          "Engine '" + engine_label() + "' did not answer within " + std::to_string(options.timeout_ms) +
                              " ms; its host was killed by the watchdog.",
                          &s);
                return false;
            }
        }
        return true;
    }

    // How long the reader may block in one read.
    int read_wait() const {
        const auto now = Clock::now();
        int wait = kPollMs;
        auto until = [&](Clock::time_point t) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(t - now).count();
            return left > 0 ? static_cast<int>(std::min<long long>(left, kPollMs)) + 1 : 1;
        };
        for (const auto &s : slots) {
            if (!s->active || s->final_queued || s->failed) continue;
            if (s->cancel_sent) wait = std::min(wait, until(s->cancel_deadline));
            if (s->has_deadline) wait = std::min(wait, until(s->deadline));
        }
        return wait;
    }

    void wake_waiters() {
        for (auto &s : slots)
            if (s->active) s->cv.notify_all();
    }

    // Runs chunk / progress callbacks for the frames in s.work and records the final frame. Called
    // without mutex, on the call's own thread. true once the final frame was seen.
    static bool process(RemoteCallSlot &s) {
        size_t off = 0;
        while (s.work.size() - off >= ipc::kHeaderSize) {
            const ipc::FrameHeader h = ipc::read_header(s.work.data() + off);
            const char *payload = s.work.data() + off + ipc::kHeaderSize;
            off += ipc::kHeaderSize + h.size;
            const agent_request *req = s.request;
            if (h.type == Msg::Chunk) {
                ipc::Reader r(payload, h.size);
                const uint32_t kind = r.u32();
                const uint64_t n = r.u64();
                const char *bytes = r.bytes(n);
                if (r.ok() && req && req->on_chunk) req->on_chunk(req->user_data, kind, bytes, static_cast<size_t>(n));
            } else if (h.type == Msg::Progress) {
                ipc::Reader r(payload, h.size);
                const uint64_t step = r.u64();
                const uint64_t total = r.u64();
                if (r.ok() && req && req->on_progress) req->on_progress(req->user_data, step, total);
            } else {
                s.final_type = h.type;
                s.final_data = payload;
                s.final_size = h.size;
                s.done = true;
                return true;
            }
        }
        return false;
    }

    // Waits for s's final frame, reading the pipe whenever no other caller is. Holds mutex (lk).
    Result<void> wait(RemoteCallSlot &s, std::unique_lock<std::mutex> &lk, std::string_view during) {
        for (;;) {
            if (!s.inbox.empty()) {
                s.work.clear();
                s.work.swap(s.inbox);
                lk.unlock();
                const bool done = process(s);
                lk.lock();
                if (done) return ok();
                continue;
            }
            if (s.failed) return Result<void>{.ok = false, .error = s.error};
            if (reader_active) {
                s.cv.wait(lk);
                continue;
            }
            // Become the reader until this call has frames of its own (or fails).
            reader_active = true;
            while (s.inbox.empty() && !s.failed) {
                if (!service_timers()) break;
                const int wait_ms = read_wait();
                lk.unlock();
                const hal::PipeRead r = child->read(hal::ChildStream::Stdout, rx, wait_ms);
                lk.lock();
                if (r == hal::PipeRead::Closed || r == hal::PipeRead::Error) {
                    host_died(during);
                    break;
                }
                if (!dispatch()) break;
            }
            reader_active = false;
            wake_waiters();
        }
    }

    // A free slot for a new call on the current host, with its id and deadline. Holds mutex.
    RemoteCallSlot &acquire_slot(const agent_request *request) {
        RemoteCallSlot *s = nullptr;
        for (auto &sp : slots)
            if (!sp->active) {
                s = sp.get();
                break;
            }
        if (!s) s = slots.emplace_back(std::make_unique<RemoteCallSlot>()).get();
        s->id = next_id++;
        s->generation = generation;
        s->active = true;
        s->request = request;
        s->has_deadline = options.timeout_ms > 0;
        s->deadline = Clock::now() + std::chrono::milliseconds(options.timeout_ms > 0 ? options.timeout_ms : 0);
        s->cancel_sent = false;
        s->inbox.clear();
        s->final_queued = false;
        s->failed = false;
        s->error = Error{};
        s->done = false;
        s->final_data = nullptr;
        s->final_size = 0;
        max_in_flight = std::max(max_in_flight, ++in_flight);
        return *s;
    }

    void release_slot(RemoteCallSlot &s) {
        std::lock_guard<std::mutex> lk(mutex);
        if (s.response_shm) {
            // The response has been copied out: the host may reuse its segment.
            if (child && s.generation == generation) send_control(Msg::Release, s.id);
            s.response_shm = nullptr;
        }
        // Once the host has seen the request segment, its name is not needed any more. If the host
        // never mapped it (it died, or failed to map it), start over with a new one next time.
        if (s.request_shm.mapped() && s.request_shm_named) {
            engine_shm::unlink(s.request_shm.name());
            s.request_shm_named = false;
            if (!s.done || s.final_type == Msg::Error || s.generation != generation) {
                if (s.generation == generation && s.request_shm_generation == generation) s.retire.assign(s.request_shm.name());
                s.request_shm.unmap();
            }
        }
        s.active = false;
        s.request = nullptr;
        s.inbox.clear();
        s.work.clear();
        --in_flight;
    }

    // One request / reply exchange: encode(slot) fills slot.tx (false: nothing to send, error set);
    // then the final frame is in slot.final_*. On error the slot is released already.
    template <typename Encode> Result<RemoteCallSlot *> exchange(const agent_request *request, std::string_view during, Encode &&encode) {
        using R = Result<RemoteCallSlot *>;
        std::unique_lock<std::mutex> lk(mutex);
        if (auto s = ensure_running(); !s.ok) return R{.ok = false, .value = nullptr, .error = s.error};
        RemoteCallSlot &slot = acquire_slot(request);
        lk.unlock();
        if (Result<void> e = encode(slot); !e.ok) {
            release_slot(slot);
            return R{.ok = false, .value = nullptr, .error = e.error};
        }
        (void)write_frame(slot.tx, slot.generation); // a dead host shows up while waiting
        lk.lock();
        if (auto w = wait(slot, lk, during); !w.ok) {
            lk.unlock();
            release_slot(slot);
            return R{.ok = false, .value = nullptr, .error = w.error};
        }
        return R{.ok = true, .value = &slot, .error = {}};
    }

    // An Error frame -> (status, message).
    static std::pair<int32_t, std::string> parse_error(const char *data, size_t size) {
        ipc::Reader r(data, size);
        const int32_t status = r.i32();
        const agent_str msg = r.str();
        if (!r.ok()) return {AGENT_ERR_DECODE_FAILED, "malformed error frame"};
        return {status, std::string(msg.data ? msg.data : "", msg.size)};
    }

    // ------------------------------------------------------------ lifecycle

    // Spawns agent-engine-host for the plugin. The one place a host process is created.
    Result<void> spawn_host() {
        hal::SpawnOptions spawn;
        spawn.program = options.host_path;
        spawn.args = {options.plugin_path};
        spawn.sandbox = options.sandbox;
        spawn.env = options.env;
        auto c = hal::spawn_child(spawn);
        if (!c.ok)
            return fail(ErrorCode::ProviderInitFailed, "Cannot start the engine host '" + options.host_path + "': " + c.error.message +
                                                           " (set Config::engine_host_path or AGENT_ENGINE_HOST).");
        {
            std::lock_guard<std::mutex> wl(write_mutex);
            child = std::move(c.value);
        }
        for (const auto &w : child->sandbox_warnings()) log(LogLevel::Warning, w);
        return ok();
    }

    struct Frame {
        Msg type{};
        const char *data = nullptr;
        size_t size = 0;
    };

    // Startup only (no call in flight; holds mutex): the next frame, read synchronously.
    Result<Frame> startup_frame(std::string_view during, int timeout_ms) {
        using R = Result<Frame>;
        rx.erase(0, rx_off);
        rx_off = 0;
        const bool has_deadline = timeout_ms > 0;
        const auto deadline = Clock::now() + std::chrono::milliseconds(has_deadline ? timeout_ms : 0);
        for (;;) {
            if (rx.size() - rx_off >= ipc::kHeaderSize) {
                const ipc::FrameHeader h = ipc::read_header(rx.data() + rx_off);
                if (h.size > ipc::kMaxFrame)
                    return R{.ok = false,
                             .value = {},
                             .error =
                                 kill_host(ErrorCode::EngineCrashed, "Engine host for '" + engine_label() + "' sent a malformed frame.")};
                if (rx.size() - rx_off - ipc::kHeaderSize >= h.size) {
                    const Frame f{h.type, rx.data() + rx_off + ipc::kHeaderSize, h.size};
                    rx_off += ipc::kHeaderSize + h.size;
                    return R{.ok = true, .value = f, .error = {}};
                }
            }
            int wait = -1;
            if (has_deadline) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
                if (left <= 0)
                    return R{.ok = false,
                             .value = {},
                             .error = kill_host(ErrorCode::EngineCrashed, "Engine host for '" + engine_label() +
                                                                              "' did not answer within " + std::to_string(timeout_ms) +
                                                                              " ms (" + std::string(during) + "); it was killed.")};
                wait = static_cast<int>(left) + 1;
            }
            switch (child->read(hal::ChildStream::Stdout, rx, wait)) {
            case hal::PipeRead::Data:
            case hal::PipeRead::Timeout: break;
            case hal::PipeRead::Closed:
            case hal::PipeRead::Error: return R{.ok = false, .value = {}, .error = host_died(during)};
            }
        }
    }

    Error unexpected(Msg type, std::string_view during) {
        return kill_host(ErrorCode::EngineCrashed, "Engine host for '" + engine_label() + "' sent an unexpected message (" +
                                                       std::to_string(static_cast<uint32_t>(type)) + ") during " + std::string(during) +
                                                       ".");
    }

    // Starts the host and creates the engine. Holds mutex; no call is in flight on a live host.
    Result<void> start() {
        if (auto s = spawn_host(); !s.ok) return s;
        ++starts;
        start_stderr_forwarding();

        auto ready = startup_frame("startup", kReadyTimeoutMs);
        if (!ready.ok) return Result<void>{.ok = false, .error = ready.error};
        if (ready.value.type == Msg::Error) {
            auto [status, msg] = parse_error(ready.value.data, ready.value.size);
            (void)status;
            return Result<void>{.ok = false, .error = kill_host(ErrorCode::ProviderInitFailed, msg)};
        }
        if (ready.value.type != Msg::Ready) return Result<void>{.ok = false, .error = unexpected(ready.value.type, "startup")};
        ipc::Reader r(ready.value.data, ready.value.size);
        const uint32_t protocol = r.u32();
        const uint32_t abi = r.u32();
        RemoteInfo got;
        got.caps = r.u64();
        got.flags = r.u32();
        const uint32_t optional = r.u32();
        const uint32_t host_flags = r.u32();
        const uint64_t pid = r.u64();
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
        got.shm = options.use_shm && (host_flags & ipc::kShmCapable) != 0 && engine_shm::available();
        host_pid = static_cast<long>(pid);
        info = std::move(got);

        ctl.clear();
        ipc::Writer w(ctl);
        w.begin(Msg::Create);
        w.u32(info.shm ? ipc::kShmCapable : 0u);
        w.str(options.create_config);
        w.finish();
        if (!write_frame(ctl, generation)) return Result<void>{.ok = false, .error = host_died("model load")};
        // Model loading has no deadline (big models take minutes); cancel does not apply to it.
        auto created = startup_frame("model load", 0);
        if (!created.ok) return Result<void>{.ok = false, .error = created.error};
        if (created.value.type == Msg::Error) {
            auto [status, msg] = parse_error(created.value.data, created.value.size);
            (void)status;
            return Result<void>{.ok = false,
                                .error = kill_host(ErrorCode::ModelLoadFailed,
                                                   "Engine '" + info.name + "' failed to load: " + (msg.empty() ? "unknown error" : msg))};
        }
        if (created.value.type != Msg::Created) return Result<void>{.ok = false, .error = unexpected(created.value.type, "model load")};
        rx.erase(0, rx_off);
        rx_off = 0;
        return ok();
    }

    // Restarts a dead host, within options.max_restarts per rolling minute. Holds mutex.
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
    if (!shm_enabled_by_env()) options.use_shm = false;
    engine->options = std::move(options);
    std::lock_guard<std::mutex> lk(engine->mutex);
    if (auto s = engine->start(); !s.ok) return R{.ok = false, .value = {}, .error = s.error};
    return R{.ok = true, .value = std::move(engine), .error = {}};
}

const RemoteInfo &remote_info(const RemoteEngine &engine) { return engine.info; }

RemoteStats remote_stats(RemoteEngine &engine) {
    std::lock_guard<std::mutex> lk(engine.mutex);
    return RemoteStats{.starts = engine.starts,
                       .shm_requests = engine.shm_requests.load(),
                       .shm_responses = engine.shm_responses.load(),
                       .max_in_flight = engine.max_in_flight};
}

Result<void> remote_run(RemoteEngine &e, const agent_request &request, Arena &arena, agent_response &response, RemoteCall &call) {
    call.slot = nullptr;
    std::unique_lock<std::mutex> serial(e.serial_mutex, std::defer_lock);
    if (!(e.info.flags & AGENT_ENGINE_FLAG_CONCURRENT)) serial.lock();

    const uint32_t run_flags = (request.on_chunk ? ipc::kRunStream : 0u) | (request.on_progress ? ipc::kRunProgress : 0u);
    auto ex = e.exchange(&request, "run", [&](RemoteCallSlot &slot) -> Result<void> {
        // Large payloads go through the slot's segment (inline if it can't be created). It is reused
        // while it is big enough and the host that mapped it is still the current one.
        ipc::ShmOut shm;
        shm.threshold = e.options.shm_threshold;
        if (slot.request_shm_generation != slot.generation) slot.retire.clear();
        const size_t need = e.info.shm ? ipc::shm_bytes(request, shm.threshold) : 0;
        if (need) {
            const bool reusable =
                slot.request_shm.mapped() && slot.request_shm.size() >= need && slot.request_shm_generation == slot.generation;
            if (!reusable) {
                if (slot.request_shm.mapped()) {
                    if (slot.request_shm_generation == slot.generation) slot.retire.assign(slot.request_shm.name());
                    if (slot.request_shm_named) engine_shm::unlink(slot.request_shm.name());
                    slot.request_shm.unmap();
                }
                constexpr size_t kGrain = 1u << 20;
                if (slot.request_shm.create((need + kGrain - 1) / kGrain * kGrain, 'c')) {
                    slot.request_shm_generation = slot.generation;
                    slot.request_shm_named = true;
                }
            }
            if (slot.request_shm.mapped()) {
                shm.base = slot.request_shm.data();
                shm.size = slot.request_shm.size();
            }
        }
        slot.tx.clear();
        ipc::Writer w(slot.tx);
        w.begin(Msg::Run, slot.id);
        w.str(shm.base ? slot.request_shm.name() : std::string_view());
        w.u64(shm.base ? slot.request_shm.size() : 0);
        w.str(slot.retire);
        slot.retire.clear();
        w.set_shm(shm.base ? &shm : nullptr);
        w.request(request, run_flags);
        if (!w.finish()) return fail(ErrorCode::InvalidConfig, "Request exceeds the 1 GiB engine IPC frame limit.");
        if (shm.count) e.shm_requests.fetch_add(1, std::memory_order_relaxed);
        return ok();
    });
    if (!ex.ok) return Result<void>{.ok = false, .error = ex.error};
    RemoteCallSlot &slot = *ex.value;
    call.slot = &slot;

    if (slot.final_type == Msg::Error) {
        auto [status, msg] = RemoteEngine::parse_error(slot.final_data, slot.final_size);
        return fail(status == AGENT_ERR_CANCELLED ? ErrorCode::Cancelled : ErrorCode::DecodeFailed,
                    "Engine '" + e.info.name + "': " + (msg.empty() ? std::string("run failed") : msg));
    }
    auto malformed = [&](std::string what) {
        std::lock_guard<std::mutex> lk(e.mutex);
        // Only kill the host that sent it, and only if nobody is reading its pipe right now.
        if (e.child && slot.generation == e.generation && !e.reader_active)
            return Result<void>{.ok = false, .error = e.kill_host(ErrorCode::EngineCrashed, std::move(what))};
        return fail(ErrorCode::EngineCrashed, std::move(what));
    };
    if (slot.final_type != Msg::Response)
        return malformed("Engine host for '" + e.info.name + "' sent an unexpected message (" +
                         std::to_string(static_cast<uint32_t>(slot.final_type)) + ") during run.");

    ipc::Reader r(slot.final_data, slot.final_size);
    const agent_str shm_name = r.str();
    const uint64_t shm_size = r.u64();
    const agent_str retire = r.str();
    if (!r.ok()) return malformed("Engine host for '" + e.info.name + "' sent a malformed response.");
    {
        std::lock_guard<std::mutex> lk(e.mutex);
        if (retire.size) e.forget_response_segment(std::string_view(retire.data, retire.size));
        if (shm_name.size) {
            // Mapped once per segment; released (for the host to reuse) in remote_release.
            slot.response_shm = e.response_segment(std::string_view(shm_name.data, shm_name.size), static_cast<size_t>(shm_size));
            if (!slot.response_shm) {
                if (e.child && slot.generation == e.generation) e.send_control(Msg::Release, slot.id);
                return fail(ErrorCode::DecodeFailed, "Cannot map the shared memory of engine '" + e.info.name + "''s response.");
            }
        }
    }
    if (slot.response_shm) {
        r.set_shm(slot.response_shm->data(), static_cast<size_t>(shm_size));
        e.shm_responses.fetch_add(1, std::memory_order_relaxed);
    }
    if (!r.response(response, arena)) {
        if (r.arena_exhausted()) return fail(ErrorCode::DecodeFailed, "Out of memory (arena growth failed).");
        return malformed("Engine host for '" + e.info.name + "' sent a malformed response.");
    }
    return ok();
}

void remote_release(RemoteEngine &e, RemoteCall &call) {
    if (!call.slot) return;
    e.release_slot(*call.slot);
    call.slot = nullptr;
}

Result<int> remote_count_tokens(RemoteEngine &e, std::string_view text) {
    if (!e.info.has_count_tokens) return fail<int>(ErrorCode::UnsupportedProvider, "Engine does not count tokens.");
    auto ex = e.exchange(nullptr, "count_tokens", [&](RemoteCallSlot &slot) -> Result<void> {
        slot.tx.clear();
        ipc::Writer w(slot.tx);
        w.begin(Msg::CountTokens, slot.id);
        w.str(text);
        w.finish();
        return ok();
    });
    if (!ex.ok) return fail<int>(ex.error.code, ex.error.message);
    RemoteCallSlot &slot = *ex.value;
    Result<int> out = fail<int>(ErrorCode::DecodeFailed, "Engine token count failed.");
    if (slot.final_type == Msg::Count) {
        ipc::Reader r(slot.final_data, slot.final_size);
        const int32_t n = r.i32();
        if (r.ok()) out = ok(static_cast<int>(n));
    }
    e.release_slot(slot);
    return out;
}

Result<void> remote_reset_cache(RemoteEngine &e) {
    if (!e.info.has_reset_cache) return ok();
    {
        // A host that is down has no cache to reset; the next call starts it fresh.
        std::lock_guard<std::mutex> lk(e.mutex);
        if (!e.child) return ok();
    }
    auto ex = e.exchange(nullptr, "reset_cache", [&](RemoteCallSlot &slot) -> Result<void> {
        slot.tx.clear();
        ipc::Writer w(slot.tx);
        w.begin(Msg::ResetCache, slot.id);
        w.finish();
        return ok();
    });
    if (!ex.ok) return Result<void>{.ok = false, .error = ex.error};
    const bool done = ex.value->final_type == Msg::Done;
    e.release_slot(*ex.value);
    if (!done) return fail(ErrorCode::DecodeFailed, "Engine cache reset failed.");
    return ok();
}

} // namespace agent::engines
