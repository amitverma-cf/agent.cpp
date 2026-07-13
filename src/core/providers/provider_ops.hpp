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

} // namespace agent::providers
