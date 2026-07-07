#pragma once

#include <agent-cpp/agent.hpp>

namespace agent::providers {

using ProviderInitFn = Result<void> (*)(Session &session);
using ProviderGenerateFn = Result<GenerationResult> (*)(Session &session, std::string_view prompt);
using ProviderStreamFn = Result<void> (*)(Session &session, std::string_view prompt, TokenCallback on_token,
                                          void *user_data);
using ProviderCountTokensFn = Result<int> (*)(Session &session, std::string_view text);

void init_global_backends();
void free_global_backends();

struct ProviderOps {
    AiProvider provider = AiProvider::Mock;
    ProviderInitFn init = nullptr;
    ProviderGenerateFn generate = nullptr;
    ProviderStreamFn stream = nullptr;
    ProviderCountTokensFn count_tokens = nullptr;
};

const ProviderOps *find_provider_ops(AiProvider provider);

} // namespace agent::providers
