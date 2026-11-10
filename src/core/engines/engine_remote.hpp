#pragma once

// Client side of out-of-process engines (EngineIsolation::Process): runs an engine plugin inside an
// agent-engine-host child process and drives it over the engine_ipc protocol. Used by
// src/core/providers/engine_provider.cpp; see docs/api/engines.md, "Isolation".
//
// Several calls may be in flight at once (from several threads): every frame carries its request's
// id, and whichever waiting caller currently holds the "reader" role reads the pipe and hands the
// other callers their frames. Engines without AGENT_ENGINE_FLAG_CONCURRENT get one run() at a time.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace agent::engines {

struct RemoteOptions {
    std::string host_path;     // empty = resolve_engine_host_path("")
    std::string plugin_path;   // passed to agent-engine-host
    std::string create_config; // create() config JSON, replayed on every restart
    int timeout_ms = 0;        // per-request watchdog, 0 = none
    int cancel_grace_ms = 2000;
    int max_restarts = 3; // per rolling minute
    SandboxPolicy sandbox; // applied to the host before exec; see core/sandbox/sandbox_policy.hpp
    std::map<std::string, std::string> env; // added to the host's environment (GPU cache folders, ...)
    // Tensor / BYTES payloads of at least shm_threshold bytes go through shared memory when both sides
    // can map it ($AGENT_ENGINE_SHM=0 turns it off); otherwise they are copied through the pipe.
    bool use_shm = true;
    size_t shm_threshold = 64 * 1024;
    LogFn logger = nullptr;
    void *logger_user_data = nullptr;
};

// What the host reported about the plugin at startup.
struct RemoteInfo {
    std::string name;
    std::string version;
    uint64_t caps = 0;
    uint32_t flags = 0;
    bool has_count_tokens = false;
    bool has_reset_cache = false;
    bool shm = false; // both sides can use shared memory
};

// Diagnostics for tests.
struct RemoteStats {
    int starts = 0;             // how many times the host was (re)started
    uint64_t shm_requests = 0;  // Run frames whose payloads went through shared memory
    uint64_t shm_responses = 0; // Response frames whose payloads came through shared memory
    int max_in_flight = 0;      // most calls in flight at once
};

struct RemoteEngine;
struct RemoteCallSlot;

// A finished remote_run's response buffers. Pass it to remote_release once the response has been
// copied out (also after a failed remote_run).
struct RemoteCall {
    RemoteCallSlot *slot = nullptr;
};

// Config::engine_host_path if set, else $AGENT_ENGINE_HOST, else agent-engine-host next to the current
// executable, else in <exe dir>/../bin, else the bare name (looked up on PATH).
std::string resolve_engine_host_path(std::string_view configured);

// Spawns the host, loads the plugin and creates the engine instance. Errors: ProviderInitFailed (host
// missing, plugin failed to load), ModelLoadFailed (create() failed), EngineCrashed (host died).
Result<std::shared_ptr<RemoteEngine>> start_remote_engine(RemoteOptions options);

const RemoteInfo &remote_info(const RemoteEngine &engine);

// Runs request in the host; thread-safe. on_chunk / on_progress / user_data / cancel of request are
// honoured on the calling thread (chunks and progress arrive as frames; cancel is forwarded as a Cancel
// frame and escalates to killing the host after cancel_grace_ms). On success response views point into
// call's buffers (strings, mapped shared memory) and arena (arrays, shapes) until remote_release.
// Restarts a dead host first, within the restart limit. A host that dies fails every call in flight.
Result<void> remote_run(RemoteEngine &engine, const agent_request &request, Arena &arena, agent_response &response, RemoteCall &call);
void remote_release(RemoteEngine &engine, RemoteCall &call);
Result<int> remote_count_tokens(RemoteEngine &engine, std::string_view text);
Result<void> remote_reset_cache(RemoteEngine &engine);

RemoteStats remote_stats(RemoteEngine &engine);

} // namespace agent::engines
