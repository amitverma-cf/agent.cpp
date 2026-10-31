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
// true when the core must inject its plain-text tool/skill instructions into the prompt.
bool needs_text_tools(const Session &session);

} // namespace agent::providers
