#include "../utils/utils.hpp"
#include "provider_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <filesystem>
#include <functional>
#include <limits>
#include <vector>

#ifdef AGENT_HAS_LLAMACPP
#include <llama.h>

namespace agent::providers {

struct LlamaState {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *smpl = nullptr;

    ~LlamaState() {
        if (smpl)
            llama_sampler_free(smpl);
        if (ctx)
            llama_free(ctx);
        if (model)
            llama_model_free(model);
    }
};

static std::filesystem::path resolve_path(const std::string_view rpath, const std::string_view dir) {
    std::filesystem::path p(rpath);
    if (p.is_absolute()) {
        return p;
    }
    return std::filesystem::absolute(std::filesystem::path(dir) / p);
}

Result<void> init_llama_cpp(Session &session) {
    if (session.config.model.empty()) {
        return fail(ErrorCode::InvalidConfig, "LlamaCpp provider requires model.");
    }

    auto state = std::make_shared<LlamaState>();

    llama_model_params model_params = llama_model_default_params();
    std::filesystem::path target_model = resolve_path(session.config.model, session.config.workspace_dir);

    state->model = llama_model_load_from_file(target_model.string().c_str(), model_params);
    if (!state->model) {
        std::string message = "Failed to load Llama model from: " + target_model.string();
        utils::log(session, LogLevel::Error, message);
        return fail(ErrorCode::ModelLoadFailed, std::move(message));
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(session.config.context_window);

    state->ctx = llama_init_from_model(state->model, ctx_params);
    if (!state->ctx) {
        utils::log(session, LogLevel::Error, "Failed to instantiate Llama execution context.");
        return fail(ErrorCode::ProviderInitFailed, "Failed to instantiate Llama execution context.");
    }

    state->smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!state->smpl) {
        utils::log(session, LogLevel::Error, "Failed to create Llama sampler.");
        return fail(ErrorCode::ProviderInitFailed, "Failed to create Llama sampler.");
    }

    llama_sampler *temp_sampler = llama_sampler_init_temp(session.config.temperature);
    if (!temp_sampler) {
        utils::log(session, LogLevel::Error, "Failed to create Llama temperature sampler.");
        return fail(ErrorCode::ProviderInitFailed, "Failed to create Llama temperature sampler.");
    }
    llama_sampler_chain_add(state->smpl, temp_sampler);

    llama_sampler *dist_sampler = llama_sampler_init_dist(LLAMA_DEFAULT_SEED);
    if (!dist_sampler) {
        utils::log(session, LogLevel::Error, "Failed to create Llama distribution sampler.");
        return fail(ErrorCode::ProviderInitFailed, "Failed to create Llama distribution sampler.");
    }
    llama_sampler_chain_add(state->smpl, dist_sampler);

    session.provider_state = std::move(state);
    utils::log(session, LogLevel::Info, "LlamaCpp provider initialized.");
    return ok();
}

static Result<void> run_llama_inference(Session &session, std::string_view prompt,
                                        Usage &usage,
                                        const std::function<void(std::string_view)> &on_piece) {
    auto state = std::static_pointer_cast<LlamaState>(session.provider_state);
    if (!state || !state->ctx) {
        return fail(ErrorCode::ProviderInitFailed, "Llama context not initialized.");
    }

    const llama_vocab *vocab = llama_model_get_vocab(state->model);
    if (!vocab) {
        return fail(ErrorCode::ProviderInitFailed, "Llama vocabulary not initialized.");
    }
    if (prompt.length() > static_cast<size_t>(std::numeric_limits<int>::max() - 2)) {
        return fail(ErrorCode::InvalidConfig, "Prompt is too large for Llama tokenization.");
    }

    llama_memory_clear(llama_get_memory(state->ctx), true);

    std::vector<llama_token> tokens(prompt.length() + 2);
    int n_tokens = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.length()), tokens.data(), static_cast<int>(tokens.size()), true, true);
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.length()), tokens.data(), static_cast<int>(tokens.size()), true, true);
    }
    if (n_tokens < 0) {
        return fail(ErrorCode::ParseError, "Failed to tokenize prompt.");
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    if (tokens.size() > static_cast<size_t>(session.config.context_window)) {
        return fail(ErrorCode::DecodeFailed, "Tokenized prompt length exceeds configured context_window limit.");
    }

    usage.prompt_tokens = static_cast<int>(tokens.size());

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int>(tokens.size()));
    const int max_tokens_to_generate = session.config.max_tokens;

    if (llama_decode(state->ctx, batch)) {
        return fail(ErrorCode::DecodeFailed, "Initial prompt decode failed.");
    }

    llama_token new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);

    for (int i = 0; i < max_tokens_to_generate; i++) {
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n >= 0) {
            on_piece(std::string_view(buf, n)); // Dispatch the text chunk
        } else {
            std::vector<char> dynamic_buf(static_cast<size_t>(-n));
            int n2 = llama_token_to_piece(vocab, new_token_id, dynamic_buf.data(), static_cast<int>(dynamic_buf.size()), 0, true);
            if (n2 >= 0) {
                on_piece(std::string_view(dynamic_buf.data(), static_cast<size_t>(n2)));
            }
        }

        usage.completion_tokens++;

        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(state->ctx, batch)) {
            return fail(ErrorCode::DecodeFailed, "Execution graph decode failed.");
        }
        new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);
    }
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
    return ok();
}

Result<GenerationResult> generate_text_llama_cpp(Session &session, std::string_view prompt) {
    std::string response;
    Usage usage;

    auto res = run_llama_inference(session, prompt, usage, [&](std::string_view piece) { response += piece; });

    if (!res.ok)
        return fail<GenerationResult>(res.error.code, res.error.message);
    return ok(GenerationResult{.text = std::move(response), .usage = usage});
}

Result<void> stream_text_llama_cpp(Session &session, std::string_view prompt, TokenCallback on_token, void *user_data) {
    Usage dummy;
    return run_llama_inference(session, prompt, dummy, [&](std::string_view piece) { on_token(piece, user_data); });
}

Result<int> count_tokens_llama_cpp(Session &session, std::string_view text) {
    auto state = std::static_pointer_cast<LlamaState>(session.provider_state);
    if (!state || !state->ctx || !state->model) {
        return fail<int>(ErrorCode::ProviderInitFailed, "Llama context not initialized.");
    }
    const llama_vocab *vocab = llama_model_get_vocab(state->model);
    if (!vocab) {
        return fail<int>(ErrorCode::ProviderInitFailed, "Llama vocabulary not initialized.");
    }
    if (text.empty()) {
        return ok(0);
    }
    int n_tokens = llama_tokenize(vocab, text.data(), static_cast<int>(text.length()), nullptr, 0, true, true);
    int count = n_tokens < 0 ? -n_tokens : n_tokens;
    return ok(count);
}

} // namespace agent::providers
#endif