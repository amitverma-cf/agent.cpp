#pragma once

// Client side of out-of-process engines (EngineIsolation::Process): runs an engine plugin inside an
// agent-engine-host child process and drives it over the engine_ipc protocol. Used by
// src/core/providers/engine_provider.cpp; see docs/api/engines.md, "Isolation".

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <cstdint>
#include <memory>
#include <mutex>
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
};

struct RemoteEngine;

// Config::engine_host_path if set, else $AGENT_ENGINE_HOST, else agent-engine-host next to the current
// executable, else in <exe dir>/../bin, else the bare name (looked up on PATH).
std::string resolve_engine_host_path(std::string_view configured);

// Spawns the host, loads the plugin and creates the engine instance. Errors: ProviderInitFailed (host
// missing, plugin failed to load), ModelLoadFailed (create() failed), EngineCrashed (host died).
Result<std::shared_ptr<RemoteEngine>> start_remote_engine(RemoteOptions options);

const RemoteInfo &remote_info(const RemoteEngine &engine);

// Every remote call holds this mutex: one request is in flight per host (v1 serializes even
// AGENT_ENGINE_FLAG_CONCURRENT engines). remote_run's response views stay valid while it is held.
std::mutex &remote_mutex(RemoteEngine &engine);

// Runs request in the host. on_chunk / on_progress / user_data / cancel of request are honoured
// locally (chunks and progress arrive as frames; cancel is forwarded as a Cancel frame and escalates
// to killing the host after cancel_grace_ms). On success response views point into the engine's
// receive buffer (strings, tensor data) and arena (arrays, shapes) until the next call. Restarts a
// dead host first, within the restart limit. Caller holds remote_mutex().
Result<void> remote_run(RemoteEngine &engine, const agent_request &request, Arena &arena, agent_response &response);
Result<int> remote_count_tokens(RemoteEngine &engine, std::string_view text);
Result<void> remote_reset_cache(RemoteEngine &engine);

// Diagnostics for tests: how many times the host was (re)started, and its pid (0 when not running).
int remote_start_count(const RemoteEngine &engine);

} // namespace agent::engines
