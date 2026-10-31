// agent-engine-host: runs one engine plugin out of process.
//
//     agent-engine-host <path/to/agent-engine-NAME.so>
//
// Loads the plugin, then serves the engine ABI to its parent over stdin/stdout using the frame
// protocol in src/core/engines/engine_ipc.hpp. The plugin can segfault, abort() or exit() without
// taking the parent down: the parent sees the pipe close, reports the exit status and restarts us.
//
// Threads: a reader thread pulls frames off the protocol input. Run requests go to a pool of worker
// threads, each executing one run() at a time: up to one worker for ordinary engines, as many as there
// are requests in flight for AGENT_ENGINE_FLAG_CONCURRENT engines (workers are created on demand and
// then reused). Cancel <id> sets the cancel flag of that request, so cancel works while run() is busy.
// Every other command (create, token count, cache reset, destroy, shutdown) runs on the main thread in
// arrival order. Frames are written from several threads (workers, engine callbacks); writes are
// serialized.
//
// Large tensor / BYTES payloads arrive and leave in shared-memory segments (engine_shm.hpp), which both
// sides map once and reuse: each core call slot has a request segment (mapped here on first use, dropped
// when the core retires it), and response segments created here are reused once the core has sent
// Release for the response that used them. The core removes the names once both sides have them mapped.
//
// The engine's own stdout output is redirected to stderr (stdout carries the protocol), and stdin
// is replaced by the null device. The parent forwards our stderr to its own.

#include "core/engines/engine_ipc.hpp"
#include "core/engines/engine_shm.hpp"
#include "hal/dylib.hpp"
#include "hal/platform.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(AGENT_HAL_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

namespace ipc = agent::engine_ipc;
namespace shm = agent::engine_shm;
using ipc::Msg;

// Worker threads for AGENT_ENGINE_FLAG_CONCURRENT engines (requests beyond that queue).
constexpr int kMaxConcurrentRuns = 256;

// ------------------------------------------------------------------ protocol I/O

#if defined(AGENT_HAL_WINDOWS)
HANDLE g_in = nullptr;
HANDLE g_out = nullptr;

bool redirect_stdio() {
    HANDLE self = GetCurrentProcess();
    if (!DuplicateHandle(self, GetStdHandle(STD_INPUT_HANDLE), self, &g_in, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;
    if (!DuplicateHandle(self, GetStdHandle(STD_OUTPUT_HANDLE), self, &g_out, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;
    // CRT fd 1 (printf) -> stderr, fd 0 -> NUL. _dup2 on 0..2 also updates the Win32 std handles.
    _dup2(2, 1);
    const int nul = _open("NUL", _O_RDONLY);
    if (nul >= 0) {
        _dup2(nul, 0);
        _close(nul);
    }
    // Console Ctrl-C belongs to the parent, which cancels through the protocol.
    SetConsoleCtrlHandler(nullptr, TRUE);
    return true;
}

bool read_exact(void *dst, size_t n) {
    auto *p = static_cast<char *>(dst);
    while (n) {
        DWORD got = 0;
        const DWORD want = n > (1u << 20) ? (1u << 20) : static_cast<DWORD>(n);
        if (!ReadFile(g_in, p, want, &got, nullptr) || got == 0) return false;
        p += got;
        n -= got;
    }
    return true;
}

bool write_all(const void *src, size_t n) {
    auto *p = static_cast<const char *>(src);
    while (n) {
        DWORD put = 0;
        const DWORD want = n > (1u << 20) ? (1u << 20) : static_cast<DWORD>(n);
        if (!WriteFile(g_out, p, want, &put, nullptr)) return false;
        p += put;
        n -= put;
    }
    return true;
}

long current_pid() { return static_cast<long>(GetCurrentProcessId()); }
#else
int g_in = -1;
int g_out = -1;

bool redirect_stdio() {
    g_in = ::fcntl(STDIN_FILENO, F_DUPFD_CLOEXEC, 3);
    g_out = ::fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
    if (g_in < 0 || g_out < 0) return false;
    const int null_fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (null_fd >= 0) {
        ::dup2(null_fd, STDIN_FILENO);
        ::close(null_fd);
    }
    ::dup2(STDERR_FILENO, STDOUT_FILENO);
    // Ctrl-C in a terminal reaches the whole foreground process group; the parent decides what it
    // means (it cancels through the protocol). A dead parent shows up as EOF / EPIPE instead.
    std::signal(SIGINT, SIG_IGN);
    std::signal(SIGPIPE, SIG_IGN);
    return true;
}

bool read_exact(void *dst, size_t n) {
    auto *p = static_cast<char *>(dst);
    while (n) {
        const ssize_t got = ::read(g_in, p, n);
        if (got > 0) {
            p += got;
            n -= static_cast<size_t>(got);
        } else if (got == 0 || errno != EINTR) {
            return false;
        }
    }
    return true;
}

bool write_all(const void *src, size_t n) {
    auto *p = static_cast<const char *>(src);
    while (n) {
        const ssize_t put = ::write(g_out, p, n);
        if (put > 0) {
            p += put;
            n -= static_cast<size_t>(put);
        } else if (put < 0 && errno != EINTR) {
            return false;
        }
    }
    return true;
}

long current_pid() { return static_cast<long>(::getpid()); }
#endif

// Shared state lives on the heap and is never destroyed: an engine may call exit() on a worker thread,
// and static destructors must not tear down a mutex or condition variable another thread is blocked on.
std::mutex &g_write_mutex = *new std::mutex;

// The parent is gone if it stops reading; nothing left to do but leave.
void send(const std::string &frame) {
    std::lock_guard<std::mutex> lk(g_write_mutex);
    if (!write_all(frame.data(), frame.size())) std::_Exit(1);
}

void send_error(uint64_t id, int32_t status, std::string_view message) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(Msg::Error, id);
    w.i32(status);
    w.str(message);
    w.finish();
    send(buf);
}

void send_empty(Msg type, uint64_t id) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(type, id);
    w.finish();
    send(buf);
}

// ------------------------------------------------------------------ requests

// One received command. Jobs are pooled: buffers (payload, arena, response encoding) keep their capacity.
struct Job {
    uint64_t id = 0;
    Msg type{};
    std::string payload;
    alignas(4) int32_t cancel = 0; // polled by the engine through agent_request::cancel
    agent::Arena arena;
    size_t arena_capacity = 0;
    std::string out;
    std::string retire; // a response segment the core should unmap (sent with this job's Response)
};

// A response segment, reused across responses: busy from the Response that uses it until the core's
// Release for that request.
struct ResponseSegment {
    shm::Segment segment;
    uint64_t busy_id = 0;
    bool busy = false;
};

const agent_engine_v1 *g_engine = nullptr;
std::atomic<void *> g_instance{nullptr};
std::atomic<bool> g_shm{false}; // both sides can map shared memory
int g_max_workers = 1;

std::mutex &g_mutex = *new std::mutex;                             // everything below
std::condition_variable &g_run_cv = *new std::condition_variable;  // run queue / eof
std::condition_variable &g_ctl_cv = *new std::condition_variable;  // control queue / eof
std::condition_variable &g_idle_cv = *new std::condition_variable; // a run finished
std::deque<Job *> &g_run_queue = *new std::deque<Job *>;
std::deque<Job *> &g_ctl_queue = *new std::deque<Job *>;
std::vector<Job *> &g_running = *new std::vector<Job *>;
std::vector<std::unique_ptr<Job>> &g_jobs = *new std::vector<std::unique_ptr<Job>>; // every job ever created
std::vector<Job *> &g_free_jobs = *new std::vector<Job *>;
// Request segments the core announced: each belongs to one of its call slots and is reused for that
// slot's calls, so it is mapped once and kept until the core retires it.
std::vector<std::unique_ptr<shm::Segment>> &g_request_segments = *new std::vector<std::unique_ptr<shm::Segment>>;
std::vector<std::unique_ptr<ResponseSegment>> &g_response_segments = *new std::vector<std::unique_ptr<ResponseSegment>>;
int g_workers = 0;
int g_idle_workers = 0;
bool g_eof = false;

void set_cancel(Job &job, int32_t v) { std::atomic_ref<int32_t>(job.cancel).store(v, std::memory_order_relaxed); }

Job *take_job() { // holds g_mutex
    if (!g_free_jobs.empty()) {
        Job *j = g_free_jobs.back();
        g_free_jobs.pop_back();
        return j;
    }
    return g_jobs.emplace_back(std::make_unique<Job>()).get();
}

// ------------------------------------------------------------------ engine callbacks

void on_chunk(void *user_data, uint32_t kind, const void *data, size_t size) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(Msg::Chunk, static_cast<Job *>(user_data)->id);
    w.u32(kind);
    w.blob(data, size);
    if (w.finish()) send(buf);
}

void on_progress(void *user_data, uint64_t step, uint64_t total) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(Msg::Progress, static_cast<Job *>(user_data)->id);
    w.u64(step);
    w.u64(total);
    w.finish();
    send(buf);
}

const char *last_error(void *instance, const char *fallback) {
    const char *e = instance ? g_engine->last_error(instance) : nullptr;
    return e && *e ? e : fallback;
}

// Executes one Run on the calling worker thread.
void run_job(Job &job) {
    void *instance = g_instance.load();
    if (!instance) {
        send_error(job.id, AGENT_ERR_INIT_FAILED, "agent-engine-host: no engine instance");
        return;
    }
    ipc::Reader r(job.payload.data(), job.payload.size());
    const agent_str seg_name = r.str();
    const uint64_t seg_size = r.u64();
    const agent_str retire = r.str();
    if (!r.ok()) {
        send_error(job.id, AGENT_ERR_PARSE, "agent-engine-host: malformed Run frame");
        return;
    }
    const shm::Segment *request_shm = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        if (retire.size) {
            const std::string_view old(retire.data, retire.size);
            std::erase_if(g_request_segments, [&](const std::unique_ptr<shm::Segment> &s) { return s->name() == old; });
        }
        if (seg_name.size) {
            const std::string_view name(seg_name.data, seg_name.size);
            for (const auto &s : g_request_segments)
                if (s->name() == name) request_shm = s.get();
            if (!request_shm) {
                auto seg = std::make_unique<shm::Segment>();
                if (seg->open(name, static_cast<size_t>(seg_size))) request_shm = g_request_segments.emplace_back(std::move(seg)).get();
            }
        }
    }
    if (seg_name.size) {
        if (!request_shm || request_shm->size() < seg_size) {
            send_error(job.id, AGENT_ERR_INIT_FAILED, "agent-engine-host: cannot map the request's shared memory");
            return;
        }
        r.set_shm(request_shm->data(), static_cast<size_t>(seg_size));
    }
    const size_t need = ipc::decode_arena_bound(job.payload.size());
    if (need > job.arena_capacity) {
        job.arena_capacity = need * 2;
        job.arena = agent::Arena(job.arena_capacity);
    }
    job.arena.reset();
    agent_request req{};
    uint32_t run_flags = 0;
    if (!r.request(req, run_flags, job.arena)) {
        send_error(job.id, AGENT_ERR_PARSE, "agent-engine-host: malformed Run frame");
        return;
    }
    req.on_chunk = (run_flags & ipc::kRunStream) ? on_chunk : nullptr;
    req.on_progress = (run_flags & ipc::kRunProgress) ? on_progress : nullptr;
    req.user_data = &job;
    req.cancel = &job.cancel;

    agent_response resp{};
    resp.struct_size = sizeof(resp);
    const int32_t rc = g_engine->run(instance, &req, &resp);
    if (rc != AGENT_OK) {
        send_error(job.id, rc, last_error(instance, "run failed"));
        return;
    }

    // Large output payloads go through a free response segment (a new one when none is big enough;
    // inline if it can't be created). Marked busy before sending, so the core's Release can't arrive first.
    ResponseSegment *segment = nullptr;
    ipc::ShmOut out_shm;
    job.retire.clear();
    if (g_shm.load()) {
        const size_t bytes = ipc::shm_bytes(resp.message, out_shm.threshold);
        if (bytes) {
            std::lock_guard<std::mutex> lk(g_mutex);
            for (const auto &s : g_response_segments)
                if (!s->busy && s->segment.size() >= bytes && (!segment || s->segment.size() < segment->segment.size())) segment = s.get();
            if (!segment) {
                // Replace one free segment that is too small (the core unmaps it when told), then grow.
                for (auto it = g_response_segments.begin(); it != g_response_segments.end(); ++it)
                    if (!(*it)->busy) {
                        job.retire.assign((*it)->segment.name());
                        shm::unlink(job.retire); // in case the core never mapped it
                        g_response_segments.erase(it);
                        break;
                    }
                constexpr size_t kGrain = 1u << 20;
                auto fresh = std::make_unique<ResponseSegment>();
                if (fresh->segment.create((bytes + kGrain - 1) / kGrain * kGrain, 'h'))
                    segment = g_response_segments.emplace_back(std::move(fresh)).get();
            }
            if (segment) {
                segment->busy = true;
                segment->busy_id = job.id;
                out_shm.base = segment->segment.data();
                out_shm.size = segment->segment.size();
            }
        }
    }
    job.out.clear();
    ipc::Writer w(job.out);
    w.begin(Msg::Response, job.id);
    w.str(segment ? segment->segment.name() : std::string_view());
    w.u64(segment ? segment->segment.size() : 0);
    w.str(job.retire);
    w.set_shm(segment ? &out_shm : nullptr);
    w.response(resp);
    const bool fits = w.finish();
    g_engine->free_response(instance, &resp);
    if (!fits) {
        if (segment) {
            std::lock_guard<std::mutex> lk(g_mutex);
            segment->busy = false;
        }
        send_error(job.id, AGENT_ERR_DECODE_FAILED, "agent-engine-host: response exceeds the 1 GiB frame limit");
        return;
    }
    send(job.out);
}

void worker_main() {
    for (;;) {
        Job *job = nullptr;
        {
            std::unique_lock<std::mutex> lk(g_mutex);
            ++g_idle_workers;
            g_run_cv.wait(lk, [] { return !g_run_queue.empty() || g_eof; });
            --g_idle_workers;
            if (g_run_queue.empty()) return; // EOF
            job = g_run_queue.front();
            g_run_queue.pop_front();
            g_running.push_back(job);
        }
        run_job(*job);
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            std::erase(g_running, job);
            g_free_jobs.push_back(job);
        }
        g_idle_cv.notify_all();
    }
}

// ------------------------------------------------------------------ reader

void reader_main() {
    std::string body;
    char header[ipc::kHeaderSize];
    for (;;) {
        if (!read_exact(header, sizeof(header))) break;
        const ipc::FrameHeader h = ipc::read_header(header);
        if (h.size > ipc::kMaxFrame) break;
        body.resize(h.size);
        if (h.size && !read_exact(body.data(), h.size)) break;

        std::unique_lock<std::mutex> lk(g_mutex);
        if (h.type == Msg::Cancel) {
            for (Job *j : g_running)
                if (j->id == h.id) set_cancel(*j, 1);
            for (Job *j : g_run_queue)
                if (j->id == h.id) set_cancel(*j, 1);
            continue;
        }
        if (h.type == Msg::Release) {
            for (const auto &s : g_response_segments)
                if (s->busy && s->busy_id == h.id) s->busy = false;
            continue;
        }
        Job *job = take_job();
        job->id = h.id;
        job->type = h.type;
        job->payload.swap(body);
        set_cancel(*job, 0);
        if (h.type == Msg::Run) {
            g_run_queue.push_back(job);
            // Every queued run gets a worker, up to the engine's limit.
            if (static_cast<int>(g_run_queue.size()) > g_idle_workers && g_workers < g_max_workers) {
                ++g_workers;
                std::thread(worker_main).detach();
            }
            g_run_cv.notify_one();
        } else {
            g_ctl_queue.push_back(job);
            g_ctl_cv.notify_one();
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_eof = true;
    }
    g_run_cv.notify_all();
    g_ctl_cv.notify_all();
    // EOF means the parent closed the pipe or died. Give an idle main thread time to destroy the
    // engine cleanly; if it is stuck in run(), leave anyway.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::_Exit(0);
}

// ------------------------------------------------------------------ control commands (main thread)

// Destroys the instance once no run is using it (false: runs still busy after 2 s).
bool destroy_instance() {
    void *instance = g_instance.load();
    if (!instance) return true;
    std::unique_lock<std::mutex> lk(g_mutex);
    if (!g_idle_cv.wait_for(lk, std::chrono::seconds(2), [] { return g_running.empty() && g_run_queue.empty(); })) return false;
    g_instance.store(nullptr);
    lk.unlock();
    g_engine->destroy(instance);
    return true;
}

int serve() {
    std::string out;
    for (;;) {
        Job *job = nullptr;
        {
            std::unique_lock<std::mutex> lk(g_mutex);
            g_ctl_cv.wait(lk, [] { return !g_ctl_queue.empty() || g_eof; });
            if (g_ctl_queue.empty()) break; // EOF
            job = g_ctl_queue.front();
            g_ctl_queue.pop_front();
        }
        ipc::Reader r(job->payload.data(), job->payload.size());
        const uint64_t id = job->id;

        switch (job->type) {
        case Msg::Create: {
            const uint32_t client_flags = r.u32();
            const agent_str config = r.str();
            if (!r.ok()) {
                send_error(id, AGENT_ERR_PARSE, "agent-engine-host: malformed Create frame");
                break;
            }
            g_shm.store((client_flags & ipc::kShmCapable) && shm::available());
            if (!destroy_instance()) {
                send_error(id, AGENT_ERR_INIT_FAILED, "agent-engine-host: runs still busy");
                break;
            }
            void *instance = nullptr;
            char err[1024] = {0};
            const int32_t rc = g_engine->create(config, &instance, err, sizeof(err));
            if (rc != AGENT_OK || !instance) {
                send_error(id, rc != AGENT_OK ? rc : AGENT_ERR_INIT_FAILED, err[0] ? err : "unknown error");
            } else {
                g_instance.store(instance);
                send_empty(Msg::Created, id);
            }
            break;
        }
        case Msg::CountTokens: {
            const agent_str text = r.str();
            void *instance = g_instance.load();
            if (!instance || !g_engine->count_tokens) {
                send_error(id, AGENT_ERR_UNSUPPORTED, "engine does not count tokens");
                break;
            }
            int32_t n = 0;
            const int32_t rc = g_engine->count_tokens(instance, text, &n);
            if (rc != AGENT_OK) {
                send_error(id, rc, last_error(instance, "token count failed"));
                break;
            }
            out.clear();
            ipc::Writer w(out);
            w.begin(Msg::Count, id);
            w.i32(n);
            w.finish();
            send(out);
            break;
        }
        case Msg::ResetCache: {
            void *instance = g_instance.load();
            if (instance && g_engine->reset_cache) {
                const int32_t rc = g_engine->reset_cache(instance);
                if (rc != AGENT_OK) {
                    send_error(id, rc, last_error(instance, "cache reset failed"));
                    break;
                }
            }
            send_empty(Msg::Done, id);
            break;
        }
        case Msg::Destroy:
            if (!destroy_instance()) send_error(id, AGENT_ERR_INIT_FAILED, "agent-engine-host: runs still busy");
            else send_empty(Msg::Done, id);
            break;
        case Msg::Shutdown: (void)destroy_instance(); return 0;
        default:
            send_error(id, AGENT_ERR_PARSE, "agent-engine-host: unknown message " + std::to_string(static_cast<uint32_t>(job->type)));
            break;
        }
        std::lock_guard<std::mutex> lk(g_mutex);
        g_free_jobs.push_back(job);
    }
    (void)destroy_instance();
    return 0;
}

const agent_engine_v1 *load(const char *path, std::string &error) {
    void *lib = agent::hal::open_library(path, error);
    if (!lib) return nullptr;
    auto get = reinterpret_cast<agent_engine_get_v1_fn>(agent::hal::find_symbol(lib, AGENT_ENGINE_ENTRY_POINT));
    if (!get) {
        error = std::string(path) + " does not export " AGENT_ENGINE_ENTRY_POINT;
        return nullptr;
    }
    const agent_engine_v1 *e = get();
    if (!e) error = "agent_engine_get_v1 returned null";
    else if (e->abi_version != AGENT_ENGINE_ABI_VERSION)
        error = "engine ABI version " + std::to_string(e->abi_version) + " does not match host ABI version " +
                std::to_string(AGENT_ENGINE_ABI_VERSION);
    else if (e->struct_size < AGENT_ENGINE_V2_SIZE)
        error = "engine table is " + std::to_string(e->struct_size) + " bytes, smaller than the ABI 2 baseline (" +
                std::to_string(AGENT_ENGINE_V2_SIZE) + ")";
    else if (!e->name || !*e->name) error = "engine has no name";
    else if (!e->create || !e->destroy || !e->run || !e->free_response || !e->last_error) error = "engine is missing a required function";
    else return e;
    return nullptr;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("agent-engine-host protocol %u, engine ABI %d\n", ipc::kProtocolVersion, AGENT_ENGINE_ABI_VERSION);
        return 0;
    }
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: %s <engine-plugin>\n"
                     "Serves one agent.cpp engine plugin over stdin/stdout; started by the agent runtime.\n",
                     argc > 0 ? argv[0] : "agent-engine-host");
        return 2;
    }
    if (!redirect_stdio()) {
        std::fprintf(stderr, "agent-engine-host: cannot set up the protocol pipes\n");
        return 2;
    }

    std::string error;
    const agent_engine_v1 *engine = load(argv[1], error);
    if (!engine) {
        send_error(0, AGENT_ERR_INIT_FAILED, "Failed to load engine plugin '" + std::string(argv[1]) + "': " + error);
        return 3;
    }
    g_engine = engine;
    g_max_workers = (engine->flags & AGENT_ENGINE_FLAG_CONCURRENT) ? kMaxConcurrentRuns : 1;

    std::string ready;
    ipc::Writer w(ready);
    w.begin(Msg::Ready);
    w.u32(ipc::kProtocolVersion);
    w.u32(engine->abi_version);
    w.u64(engine->caps);
    w.u32(engine->flags);
    w.u32((engine->count_tokens ? ipc::kHasCountTokens : 0u) | (engine->reset_cache ? ipc::kHasResetCache : 0u));
    w.u32(shm::available() ? ipc::kShmCapable : 0u);
    w.u64(static_cast<uint64_t>(current_pid()));
    w.str(std::string_view(engine->name));
    w.str(std::string_view(engine->version ? engine->version : ""));
    w.finish();
    send(ready);

    std::thread(reader_main).detach();
    const int rc = serve();
    std::fflush(nullptr);
    std::_Exit(rc);
}
