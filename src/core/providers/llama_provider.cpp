#include "provider_ops.hpp"

#include <agent-cpp/agent.hpp>

#ifdef AGENT_HAS_LLAMACPP
#include <cctype>
#include <cmath>
#include <llama.h>
#include <string>
#include <vector>

namespace agent::providers {

struct LlamaState {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *smpl = nullptr;
    float applied_temperature = -1.0f;

    ~LlamaState() {
        if (smpl) {
            llama_sampler_free(smpl);
        }
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }
};

Result<void> init_llama_cpp(Session &session) {
    if (session.config.model.empty()) {
        return fail(ErrorCode::InvalidConfig, "llama.cpp requires a model path.");
    }

    auto params = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(session.config.model.c_str(), params);
    if (!model) {
        return fail(ErrorCode::ModelLoadFailed, "Failed to load llama.cpp model from: " + session.config.model);
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(session.config.context_window);

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        return fail(ErrorCode::ModelLoadFailed, "Failed to create context for llama.cpp model.");
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(session.config.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto state = std::make_shared<LlamaState>();
    state->model = model;
    state->ctx = ctx;
    state->smpl = smpl;
    state->applied_temperature = session.config.temperature;
    session.provider_state = state;

    return ok();
}

// Scans `text` starting at `from` for the first balanced {...} span (honoring quoted
// strings/escapes) and returns its [start, end] byte offsets (inclusive). Models that
// aren't grammar-constrained typically wrap a tool call in surrounding prose rather
// than emitting it as the entire response, so callers should search rather than
// require an exact whole-string match.
bool find_balanced_json_object(std::string_view text, size_t from, size_t &obj_start, size_t &obj_end) {
    for (size_t i = from; i < text.size(); ++i) {
        if (text[i] != '{')
            continue;
        size_t depth = 0;
        bool in_string = false;
        bool escape = false;
        for (size_t j = i; j < text.size(); ++j) {
            char c = text[j];
            if (in_string) {
                if (escape)
                    escape = false;
                else if (c == '\\')
                    escape = true;
                else if (c == '"')
                    in_string = false;
                continue;
            }
            if (c == '"') {
                in_string = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth == 0) {
                    obj_start = i;
                    obj_end = j;
                    return true;
                }
            }
        }
    }
    return false;
}

Result<ChatResponse> execute_turn_llama_cpp(Session &session, const ChatRequest &request) {
    auto state = std::static_pointer_cast<LlamaState>(session.provider_state);
    if (!state || !state->model || !state->ctx || !state->smpl) {
        return fail<ChatResponse>(ErrorCode::ProviderInitFailed, "Llama model not initialized.");
    }

    std::string prompt;
    if (!session.config.tools.empty()) {
        prompt += "system: You have access to the following tools:\n";
        for (const auto &tool : session.config.tools) {
            prompt += "- " + std::string(tool.name) + ": " + std::string(tool.description) + "\n";
            prompt += "  Schema: " + std::string(tool.parameter_schema) + "\n";
        }
        prompt += "When you need a tool, respond with nothing but a single JSON object of the exact form "
                  "{\"name\": \"tool_name\", \"arguments\": {...}} and stop; wait for the tool result before "
                  "continuing.\n\n";
    }

    for (const auto &msg : request.messages) {
        prompt += std::string(msg.role) + ": " + std::string(msg.content) + "\n";
    }
    prompt += "assistant: ";

    const llama_vocab *vocab = llama_model_get_vocab(state->model);

    llama_memory_clear(llama_get_memory(state->ctx), true);

    std::vector<llama_token> tokens(prompt.length() + 4);
    int n_tokens = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.length()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, true);
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.length()), tokens.data(),
                                  static_cast<int>(tokens.size()), true, true);
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int>(tokens.size()));
    if (llama_decode(state->ctx, batch)) {
        return fail<ChatResponse>(ErrorCode::DecodeFailed, "Initial prompt decode failure");
    }

    const float effective_temperature =
        (request.temperature >= 0.0f) ? request.temperature : session.config.temperature;
    if (effective_temperature != state->applied_temperature) {
        llama_sampler_free(state->smpl);
        auto sparams = llama_sampler_chain_default_params();
        state->smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(state->smpl, llama_sampler_init_temp(effective_temperature));
        llama_sampler_chain_add(state->smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        state->applied_temperature = effective_temperature;
    }

    std::string response;
    const int max_tokens_to_generate = (request.max_tokens > 0)
                                           ? request.max_tokens
                                           : (session.config.max_tokens > 0 ? session.config.max_tokens : 512);
    llama_token new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);

    for (int i = 0; i < max_tokens_to_generate; i++) {
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n >= 0) {
            std::string_view token_piece(buf, static_cast<size_t>(n));
            response.append(token_piece);
            if (request.stream && request.on_token) {
                request.on_token(token_piece, request.token_user_data);
            }
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(state->ctx, batch)) {
            response += "\n[Error: Execution graph decode failure]";
            break;
        }
        new_token_id = llama_sampler_sample(state->smpl, state->ctx, -1);
    }

    std::string_view response_text = session.arena.allocate_string(response);

    MessageView msg;
    msg.role = "assistant";
    msg.content = response_text;
    msg.tool_calls = {};
    msg.tool_call_id = "";

    if (!session.config.tools.empty()) {
        size_t search_from = 0;
        size_t obj_start = 0, obj_end = 0;
        while (find_balanced_json_object(response_text, search_from, obj_start, obj_end)) {
            std::string_view candidate = response_text.substr(obj_start, obj_end - obj_start + 1);
            search_from = obj_end + 1;

            simdjson::ondemand::parser parser;
            simdjson::padded_string padded_candidate(candidate.data(), candidate.size());
            simdjson::ondemand::document doc;
            if (parser.iterate(padded_candidate).get(doc))
                continue;

            simdjson::ondemand::object obj;
            if (doc.get_object().get(obj))
                continue;

            std::string_view tool_name;
            if (obj["name"].get_string().get(tool_name) || tool_name.empty())
                continue;

            auto raw_args_res = obj["arguments"];
            std::string args_str;
            std::string_view raw_json;
            if (!raw_args_res.get_string().get(raw_json)) {
                args_str = std::string(raw_json);
            } else if (!raw_args_res.raw_json().get(raw_json)) {
                args_str = std::string(raw_json);
            }

            std::span<ToolCallView> tcs = session.arena.allocate_span<ToolCallView>(1);
            tcs[0].id = session.arena.allocate_string("call_" + std::string(tool_name));
            tcs[0].name = session.arena.allocate_string(tool_name);
            tcs[0].arguments = session.arena.allocate_string(args_str.empty() ? "{}" : args_str);
            msg.tool_calls = tcs;

            std::string_view visible = response_text.substr(0, obj_start);
            while (!visible.empty() && std::isspace(static_cast<unsigned char>(visible.back())))
                visible.remove_suffix(1);
            msg.content = visible;
            break;
        }
    }

    return ok(ChatResponse{.message = msg,
                           .usage = Usage{.prompt_tokens = n_tokens,
                                          .completion_tokens = static_cast<int>(response.length() / 4),
                                          .total_tokens = n_tokens + static_cast<int>(response.length() / 4)}});
}

Result<int> count_tokens_llama_cpp(Session &session, std::string_view text) {
    auto state = std::static_pointer_cast<LlamaState>(session.provider_state);
    if (!state || !state->model || !state->ctx) {
        return fail<int>(ErrorCode::ProviderInitFailed, "Llama model not initialized.");
    }

    const llama_vocab *vocab = llama_model_get_vocab(state->model);
    std::vector<llama_token> tokens(text.size() + 4);
    int n_tokens = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokens.data(),
                                  static_cast<int>(tokens.size()), false, false);
    if (n_tokens < 0) {
        return ok(static_cast<int>((text.length() + 2) / 3));
    }
    return ok(n_tokens);
}

} // namespace agent::providers
#endif
