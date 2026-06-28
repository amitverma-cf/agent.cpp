#include "provider_ops.h"

#ifdef AGENT_HAS_LLAMACPP
#include <llama.h>
#endif

namespace agent::providers {

#ifdef AGENT_HAS_LLAMACPP
Result<void> init_llama_cpp(Session &session);
Result<std::string> generate_text_llama_cpp(Session &session, std::string_view prompt);
Result<void> stream_text_llama_cpp(Session &session, std::string_view prompt, TokenCallback on_token, void *user_data);
#endif

#ifdef AGENT_HAS_OPENAICOMPATIBLE
Result<void> init_openai_compatible(Session &session);
Result<std::string> generate_text_openai_compatible(Session &session, std::string_view prompt);
Result<void> stream_text_openai_compatible(Session &session, std::string_view prompt, TokenCallback on_token,
                                           void *user_data);
#endif

namespace {

Result<void> init_mock(Session &) {
    return ok();
}

Result<std::string> generate_mock(Session &, std::string_view) {
    return ok(std::string("Mock Response"));
}

Result<void> stream_mock(Session &, std::string_view, TokenCallback on_token, void *user_data) {
    on_token("Mock Response Streamed", user_data);
    return ok();
}

constexpr ProviderOps kProviderOps[] = {
    ProviderOps{.provider = Provider::Mock, .init = init_mock, .generate = generate_mock, .stream = stream_mock},
#ifdef AGENT_HAS_LLAMACPP
    ProviderOps{.provider = Provider::LlamaCpp,
                .init = init_llama_cpp,
                .generate = generate_text_llama_cpp,
                .stream = stream_text_llama_cpp},
#endif
#ifdef AGENT_HAS_OPENAICOMPATIBLE
    ProviderOps{.provider = Provider::OpenAICompatible,
                .init = init_openai_compatible,
                .generate = generate_text_openai_compatible,
                .stream = stream_text_openai_compatible},
#endif
};

} // namespace

void init_global_backends() {
    #ifdef AGENT_HAS_LLAMACPP
        llama_backend_init();
    #endif
}

void free_global_backends() {
    #ifdef AGENT_HAS_LLAMACPP
        llama_backend_free();
    #endif
}

const ProviderOps *find_provider_ops(Provider provider) {
    for (const ProviderOps &ops : kProviderOps) {
        if (ops.provider == provider) {
            return &ops;
        }
    }
    return nullptr;
}

} // namespace agent::providers
