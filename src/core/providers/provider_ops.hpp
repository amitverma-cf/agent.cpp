#pragma once

#include <agent-cpp/agent.hpp>

namespace agent::providers {

using ProviderInitFn = Result<void> (*)(Session &session);
using ProviderInferFn = Result<InferResponse> (*)(Session &session, const InferRequest &request);
using ProviderCountTokensFn = Result<int> (*)(Session &session, std::string_view text);
using ProviderClearKvCacheFn = Result<void> (*)(Session &session);

void init_global_backends();
void free_global_backends();

struct ProviderOps {
    AiProvider provider = AiProvider::Mock;
    ProviderInitFn init = nullptr;
    ProviderInferFn infer = nullptr;
    ProviderCountTokensFn count_tokens = nullptr;
    ProviderClearKvCacheFn clear_kv_cache = nullptr;
};

const ProviderOps *find_provider_ops(AiProvider provider);

// Capabilities of the active provider, valid after the provider's init().
// true when concurrent infer() calls are safe and worthwhile (engine flag AGENT_ENGINE_FLAG_CONCURRENT).
bool supports_parallel(const Session &session);
// Plain-text instructions the core folds into the system message for this provider: the tool list
// and skills for engines without AGENT_ENGINE_FLAG_NATIVE_TOOLS, only the skills for engines that
// format tools natively, nothing otherwise.
std::string_view injected_system_text(const Session &session);
// Process isolation: how many times the session's agent-engine-host was started (1 + restarts);
// -1 when the engine runs in-process. For tests and diagnostics.
int engine_host_starts(const Session &session);
// Process isolation diagnostics (all zero for in-process engines): host starts, Run / Response frames
// whose payloads went through shared memory, the most calls in flight at once, and whether shared
// memory is in use at all.
struct EngineHostStats {
    int starts = 0;
    uint64_t shm_requests = 0;
    uint64_t shm_responses = 0;
    int max_in_flight = 0;
    bool shm = false;
};
EngineHostStats engine_host_stats(const Session &session);

// Runs request on the session's engine with an explicit task (one agent_capability bit, e.g.
// AGENT_CAP_EMBED) instead of the engine's default. The converted request and the response views
// live in arena, which the caller owns (infer() uses the session's per-turn arena). Fails with
// UnsupportedProvider for non-engine sessions and engines that don't declare the capability.
Result<InferResponse> infer_task(Session &session, uint32_t task, const InferRequest &request, Arena &arena);

} // namespace agent::providers
