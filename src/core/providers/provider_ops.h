#pragma once

#include <agent-cpp/agent.h>

namespace agent::providers {

using ProviderInitFn = Result<void> (*)(Session &session);
using ProviderGenerateFn = Result<std::string> (*)(Session &session, std::string_view prompt);
using ProviderStreamFn = Result<void> (*)(Session &session, std::string_view prompt, TokenCallback on_token,
                                          void *user_data);

void init_global_backends();
void free_global_backends();

struct ProviderOps {
    Provider provider = Provider::Mock;
    ProviderInitFn init = nullptr;
    ProviderGenerateFn generate = nullptr;
    ProviderStreamFn stream = nullptr;
};

const ProviderOps *find_provider_ops(Provider provider);

} // namespace agent::providers
