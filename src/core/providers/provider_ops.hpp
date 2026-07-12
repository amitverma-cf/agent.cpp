#pragma once

#include <agent-cpp/agent.hpp>

namespace agent::providers {

using ProviderInitFn = Result<void> (*)(Session &session);

using ProviderExecuteTurnFn = Result<ChatResponse> (*)(Session &session, const ChatRequest &request);

using ProviderCountTokensFn = Result<int> (*)(Session &session, std::string_view text);

void init_global_backends();

void free_global_backends();

struct ProviderOps {
    AiProvider provider = AiProvider::Mock;
    ProviderInitFn init = nullptr;
    ProviderExecuteTurnFn execute_turn = nullptr;
    ProviderCountTokensFn count_tokens = nullptr;
};

const ProviderOps *find_provider_ops(AiProvider provider);

} // namespace agent::providers
