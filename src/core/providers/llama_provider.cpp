#include <agent-cpp/agent.h>
#include <filesystem>
#include <iostream>
#include <vector>

#ifdef AGENT_HAS_LLAMACPP
#include <llama.h>

namespace agent {

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
        llama_backend_free();
    }
};

static std::filesystem::path resolve_path(const std::string_view rpath, const std::string_view dir) {
    std::filesystem::path p(rpath);
    if (p.is_absolute()) {
        return p;
    }
    return std::filesystem::absolute(std::filesystem::path(dir) / p);
}

void init_llama_cpp(Session &session) {
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    std::filesystem::path target_model = resolve_path(session.config.model, session.config.workspace_dir);

    llama_model *model = llama_model_load_from_file(target_model.string().c_str(), model_params);
    if (!model) {
        llama_backend_free();
        std::cerr << "Error: Failed to load Llama model from: " << target_model.string() << "\n";
        return;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        llama_backend_free();
        std::cerr << "Error: Failed to instantiate execution context.\n";
        return;
    }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto state = std::make_shared<LlamaState>();
    state->model = model;
    state->ctx = ctx;
    state->smpl = smpl;

    session.state = state;
}

std::string generate_text_llama_cpp(Session &session, std::string_view prompt) {
    auto state = std::static_pointer_cast<LlamaState>(session.state);
    if (!state || !state->ctx)
        return "Error: Llama context not initialized.";

    const llama_vocab *vocab = llama_model_get_vocab(state->model);

    llama_memory_clear(llama_get_memory(state->ctx), true);
    std::vector<llama_token> tokens(prompt.length() + 2);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.data(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    std::string response = "";
    const int max_tokens_to_generate = 512;
    if (llama_decode(state->ctx, batch)) {
        return "[Error: Initial prompt decode failure]";
    }
    llama_token new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);
    for (int i = 0; i < max_tokens_to_generate; i++) {
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n >= 0) {
            response.append(buf, n);
        }
        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(state->ctx, batch)) {
            response += "\n[Error: Execution graph decode failure]";
            break;
        }
        new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);
    }
    return response;
}

void stream_text_llama_cpp(Session &session, std::string_view prompt,
                           const std::function<void(std::string_view)> &on_token) {
    auto state = std::static_pointer_cast<LlamaState>(session.state);
    if (!state || !state->ctx) {
        on_token("Error: Llama context not initialized.");
        return;
    }

    const llama_vocab *vocab = llama_model_get_vocab(state->model);

    llama_memory_clear(llama_get_memory(state->ctx), true);
    std::vector<llama_token> tokens(prompt.length() + 2);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.data(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    const int max_tokens_to_generate = 512;
    if (llama_decode(state->ctx, batch)) {
        on_token("[Error: Initial prompt decode failure]");
        return;
    }
    llama_token new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);
    for (int i = 0; i < max_tokens_to_generate; i++) {
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n >= 0) {
            on_token(std::string_view(buf, n));
        }
        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(state->ctx, batch)) {
            on_token("\n[Error: Execution graph decode failure]");
            break;
        }
        new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);
    }
}

} // namespace agent
#endif