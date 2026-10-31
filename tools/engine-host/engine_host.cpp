// agent-engine-host: runs one engine plugin out of process.
//
//     agent-engine-host <path/to/agent-engine-NAME.so>
//
// Loads the plugin, then serves the engine ABI to its parent over stdin/stdout using the frame
// protocol in src/core/engines/engine_ipc.hpp. The plugin can segfault, abort() or exit() without
// taking the parent down: the parent sees the pipe close, reports the exit status and restarts us.
//
// Threads: a reader thread pulls frames off the protocol input and turns Cancel into the cancel flag
// the engine polls (so cancel works while run() is busy); the main thread executes one command at a
// time. Chunk / progress frames may be written from engine threads; writes are serialized.
//
// The engine's own stdout output is redirected to stderr (stdout carries the protocol), and stdin
// is replaced by the null device. The parent forwards our stderr to its own.

#include "core/engines/engine_ipc.hpp"
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
#include <mutex>
#include <string>
#include <thread>

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
using ipc::Msg;

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
#endif

std::mutex g_write_mutex;

// The parent is gone if it stops reading; nothing left to do but leave.
void send(const std::string &frame) {
    std::lock_guard<std::mutex> lk(g_write_mutex);
    if (!write_all(frame.data(), frame.size())) std::_Exit(1);
}

void send_error(int32_t status, std::string_view message) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(Msg::Error);
    w.i32(status);
    w.str(message);
    w.finish();
    send(buf);
}

void send_empty(Msg type) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(type);
    w.finish();
    send(buf);
}

// ------------------------------------------------------------------ command mailbox

// The parent sends one command at a time and waits for its reply, so one slot is enough; Cancel
// bypasses it. Buffers are swapped, not copied, so their capacity is reused.
struct Mailbox {
    std::mutex mutex;
    std::condition_variable cv;
    bool full = false;
    bool eof = false;
    Msg type{};
    std::string payload;
};

Mailbox g_mailbox;
alignas(4) int32_t g_cancel = 0; // polled by the engine through agent_request::cancel

void set_cancel(int32_t v) { std::atomic_ref<int32_t>(g_cancel).store(v, std::memory_order_relaxed); }

void reader_main() {
    std::string body;
    char header[ipc::kHeaderSize];
    for (;;) {
        if (!read_exact(header, sizeof(header))) break;
        const ipc::FrameHeader h = ipc::read_header(header);
        if (h.size > ipc::kMaxFrame) break;
        body.resize(h.size);
        if (h.size && !read_exact(body.data(), h.size)) break;
        if (h.type == Msg::Cancel) {
            set_cancel(1);
            continue;
        }
        // Reset in stream order, so a Cancel that follows this Run is never lost.
        if (h.type == Msg::Run) set_cancel(0);
        std::unique_lock<std::mutex> lk(g_mailbox.mutex);
        g_mailbox.cv.wait(lk, [] { return !g_mailbox.full; });
        g_mailbox.type = h.type;
        g_mailbox.payload.swap(body);
        g_mailbox.full = true;
        g_mailbox.cv.notify_all();
    }
    {
        std::lock_guard<std::mutex> lk(g_mailbox.mutex);
        g_mailbox.eof = true;
    }
    g_mailbox.cv.notify_all();
    // EOF means the parent closed the pipe or died. Give an idle main thread time to destroy the
    // engine cleanly; if it is stuck in run(), leave anyway.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::_Exit(0);
}

// ------------------------------------------------------------------ engine callbacks

void on_chunk(void *, uint32_t kind, const void *data, size_t size) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(Msg::Chunk);
    w.u32(kind);
    w.blob(data, size);
    if (w.finish()) send(buf);
}

void on_progress(void *, uint64_t step, uint64_t total) {
    thread_local std::string buf;
    buf.clear();
    ipc::Writer w(buf);
    w.begin(Msg::Progress);
    w.u64(step);
    w.u64(total);
    w.finish();
    send(buf);
}

const char *last_error(const agent_engine_v1 *engine, void *instance, const char *fallback) {
    const char *e = instance ? engine->last_error(instance) : nullptr;
    return e && *e ? e : fallback;
}

int serve(const agent_engine_v1 *engine) {
    void *instance = nullptr;
    agent::Arena arena;
    size_t arena_capacity = 0;
    std::string work;
    std::string out;

    for (;;) {
        Msg type{};
        {
            std::unique_lock<std::mutex> lk(g_mailbox.mutex);
            g_mailbox.cv.wait(lk, [] { return g_mailbox.full || g_mailbox.eof; });
            if (!g_mailbox.full) break; // EOF
            type = g_mailbox.type;
            work.swap(g_mailbox.payload);
            g_mailbox.full = false;
            g_mailbox.cv.notify_all();
        }
        ipc::Reader r(work.data(), work.size());

        switch (type) {
        case Msg::Create: {
            const agent_str config = r.str();
            if (!r.ok()) {
                send_error(AGENT_ERR_PARSE, "agent-engine-host: malformed Create frame");
                break;
            }
            if (instance) engine->destroy(instance);
            instance = nullptr;
            char err[1024] = {0};
            const int32_t rc = engine->create(config, &instance, err, sizeof(err));
            if (rc != AGENT_OK || !instance) {
                instance = nullptr;
                send_error(rc != AGENT_OK ? rc : AGENT_ERR_INIT_FAILED, err[0] ? err : "unknown error");
            } else {
                send_empty(Msg::Created);
            }
            break;
        }
        case Msg::Run: {
            if (!instance) {
                send_error(AGENT_ERR_INIT_FAILED, "agent-engine-host: no engine instance");
                break;
            }
            const size_t need = ipc::decode_arena_bound(work.size());
            if (need > arena_capacity) {
                arena_capacity = need * 2;
                arena = agent::Arena(arena_capacity);
            }
            arena.reset();
            agent_request req{};
            uint32_t run_flags = 0;
            if (!r.request(req, run_flags, arena)) {
                send_error(AGENT_ERR_PARSE, "agent-engine-host: malformed Run frame");
                break;
            }
            req.on_chunk = (run_flags & ipc::kRunStream) ? on_chunk : nullptr;
            req.on_progress = (run_flags & ipc::kRunProgress) ? on_progress : nullptr;
            req.user_data = nullptr;
            req.cancel = &g_cancel;

            agent_response resp{};
            const int32_t rc = engine->run(instance, &req, &resp);
            if (rc != AGENT_OK) {
                send_error(rc, last_error(engine, instance, "run failed"));
                break;
            }
            out.clear();
            ipc::Writer w(out);
            w.begin(Msg::Response);
            w.response(resp);
            const bool fits = w.finish();
            engine->free_response(instance, &resp);
            if (fits) send(out);
            else send_error(AGENT_ERR_DECODE_FAILED, "agent-engine-host: response exceeds the 1 GiB frame limit");
            break;
        }
        case Msg::CountTokens: {
            const agent_str text = r.str();
            if (!instance || !engine->count_tokens) {
                send_error(AGENT_ERR_UNSUPPORTED, "engine does not count tokens");
                break;
            }
            int32_t n = 0;
            const int32_t rc = engine->count_tokens(instance, text, &n);
            if (rc != AGENT_OK) {
                send_error(rc, last_error(engine, instance, "token count failed"));
                break;
            }
            out.clear();
            ipc::Writer w(out);
            w.begin(Msg::Count);
            w.i32(n);
            w.finish();
            send(out);
            break;
        }
        case Msg::ResetCache: {
            if (instance && engine->reset_cache) {
                const int32_t rc = engine->reset_cache(instance);
                if (rc != AGENT_OK) {
                    send_error(rc, last_error(engine, instance, "cache reset failed"));
                    break;
                }
            }
            send_empty(Msg::Done);
            break;
        }
        case Msg::Destroy:
            if (instance) engine->destroy(instance);
            instance = nullptr;
            send_empty(Msg::Done);
            break;
        case Msg::Shutdown:
            if (instance) engine->destroy(instance);
            return 0;
        default: send_error(AGENT_ERR_PARSE, "agent-engine-host: unknown message " + std::to_string(static_cast<uint32_t>(type))); break;
        }
    }
    if (instance) engine->destroy(instance);
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
        send_error(AGENT_ERR_INIT_FAILED, "Failed to load engine plugin '" + std::string(argv[1]) + "': " + error);
        return 3;
    }

    std::string ready;
    ipc::Writer w(ready);
    w.begin(Msg::Ready);
    w.u32(ipc::kProtocolVersion);
    w.u32(engine->abi_version);
    w.u64(engine->caps);
    w.u32(engine->flags);
    w.u32((engine->count_tokens ? ipc::kHasCountTokens : 0u) | (engine->reset_cache ? ipc::kHasResetCache : 0u));
    w.str(std::string_view(engine->name));
    w.str(std::string_view(engine->version ? engine->version : ""));
    w.finish();
    send(ready);

    std::thread(reader_main).detach();
    const int rc = serve(engine);
    std::fflush(nullptr);
    std::_Exit(rc);
}
