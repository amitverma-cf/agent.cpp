#include "provider_ops.hpp"

#include <agent-cpp/agent.hpp>

#ifdef AGENT_HAS_LLAMACPP
#include <llama.h>
#endif

namespace agent::providers {

#ifdef AGENT_HAS_LLAMACPP
Result<void> init_llama_cpp(Session &session);
Result<ChatResponse> execute_turn_llama_cpp(Session &session, const ChatRequest &request);
Result<int> count_tokens_llama_cpp(Session &session, std::string_view text);
#endif

#ifdef AGENT_HAS_OPENAICOMPATIBLE
Result<void> init_openai_compatible(Session &session);
Result<ChatResponse> execute_turn_openai_compatible(Session &session, const ChatRequest &request);
#endif

#ifdef AGENT_HAS_ONNX
Result<void> init_onnx(Session &session);
Result<ChatResponse> execute_turn_onnx(Session &session, const ChatRequest &request);
#endif

namespace {

Result<void> init_mock(Session &) {
    return ok();
}

Result<ChatResponse> execute_turn_mock(Session &session, const ChatRequest &request) {
    if (request.stream && request.on_token) {
        request.on_token("Mock Response Streamed", request.token_user_data);
    }
    std::string_view response_text = session.arena.allocate_string("Mock Response");

    MessageView msg;
    msg.role = "assistant";
    msg.content = response_text;
    msg.tool_calls = {};
    msg.tool_call_id = "";

    return ok(
        ChatResponse{.message = msg, .usage = Usage{.prompt_tokens = 2, .completion_tokens = 2, .total_tokens = 4}});
}

constexpr ProviderOps kProviderOps[] = {
    ProviderOps{
        .provider = AiProvider::Mock, .init = init_mock, .execute_turn = execute_turn_mock, .count_tokens = nullptr},
#ifdef AGENT_HAS_LLAMACPP
    ProviderOps{.provider = AiProvider::LlamaCpp,
                .init = init_llama_cpp,
                .execute_turn = execute_turn_llama_cpp,
                .count_tokens = count_tokens_llama_cpp},
#endif
#ifdef AGENT_HAS_OPENAICOMPATIBLE
    ProviderOps{.provider = AiProvider::OpenAICompatible,
                .init = init_openai_compatible,
                .execute_turn = execute_turn_openai_compatible},
#endif
#ifdef AGENT_HAS_ONNX
    ProviderOps{.provider = AiProvider::OnnxRuntime,
                .init = init_onnx,
                .execute_turn = execute_turn_onnx,
                .count_tokens = nullptr},
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

const ProviderOps *find_provider_ops(AiProvider provider) {
    for (const ProviderOps &ops : kProviderOps) {
        if (ops.provider == provider) {
            return &ops;
        }
    }
    return nullptr;
}

} // namespace agent::providers
