#include "provider_ops.hpp"

#include "../executor/ambient_turn.hpp"

#include <agent-cpp/agent.hpp>

namespace agent::providers {

Result<void> init_engine(Session &session);
Result<InferResponse> infer_engine(Session &session, const InferRequest &request);
Result<int> count_tokens_engine(Session &session, std::string_view text);
Result<void> clear_kv_cache_engine(Session &session);

#ifdef AGENT_HAS_ONNX
Result<void> init_onnx(Session &session);
Result<InferResponse> infer_onnx(Session &session, const InferRequest &request);
#endif

namespace {

Result<void> init_mock(Session &) { return ok(); }

Result<InferResponse> infer_mock(Session &session, const InferRequest &request) {
    if (request.stream && request.on_token) request.on_token("Mock Response Streamed", request.token_user_data);

    std::string_view response_text = ambient::current_arena(session).allocate_string("Mock Response");

    MessageView msg;
    msg.role = "assistant";
    msg.content = response_text;
    msg.tool_calls = {};
    msg.tool_call_id = "";

    return ok(InferResponse{.message = msg, .usage = TokenUsage{.prompt_tokens = 2, .completion_tokens = 2, .total_tokens = 4}});
}

constexpr ProviderOps kProviderOps[] = {
    ProviderOps{.provider = AiProvider::Mock, .init = init_mock, .infer = infer_mock, .count_tokens = nullptr, .clear_kv_cache = nullptr},
    ProviderOps{.provider = AiProvider::Engine,
                .init = init_engine,
                .infer = infer_engine,
                .count_tokens = count_tokens_engine,
                .clear_kv_cache = clear_kv_cache_engine},
#ifdef AGENT_HAS_ONNX
    ProviderOps{
        .provider = AiProvider::OnnxRuntime, .init = init_onnx, .infer = infer_onnx, .count_tokens = nullptr, .clear_kv_cache = nullptr},
#endif
};

} // namespace

// Engine plugins initialise their own backends; nothing is compiled into the core today.
void init_global_backends() {}

void free_global_backends() {}

const ProviderOps *find_provider_ops(AiProvider provider) {
    for (const ProviderOps &ops : kProviderOps) {
        if (ops.provider == provider) return &ops;
    }
    return nullptr;
}

} // namespace agent::providers
