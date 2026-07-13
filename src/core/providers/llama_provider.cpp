#include "../executor/ambient_turn.hpp"
#include "provider_ops.hpp"

#include <agent-cpp/agent.hpp>

#ifdef AGENT_HAS_LLAMACPP
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <llama.h>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace agent::providers {

class LlamaEngine {
  public:
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

    ~LlamaEngine() {
        if (ctx)
            llama_free(ctx);
        if (model)
            llama_model_free(model);
    }

    Result<llama_seq_id> acquire_seq() {
        std::lock_guard<std::mutex> lock(mu_);
        if (free_seq_ids_.empty())
            return fail<llama_seq_id>(
                ErrorCode::ProviderInitFailed,
                "LlamaEngine: no free sequence slots (raise Config::max_parallel_subagents).");
        llama_seq_id seq = free_seq_ids_.back();
        free_seq_ids_.pop_back();
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        next_pos_[seq] = 0;
        return ok(seq);
    }

    void release_seq(llama_seq_id seq) {
        std::lock_guard<std::mutex> lock(mu_);
        next_pos_.erase(seq);
        free_seq_ids_.push_back(seq);
    }

    void seed_free_list(int n_seq_max) {
        free_seq_ids_.reserve(static_cast<size_t>(n_seq_max));
        for (int i = n_seq_max - 1; i >= 0; --i)
            free_seq_ids_.push_back(i);
    }

    Result<llama_token> prefill(llama_seq_id seq, const std::vector<llama_token> &tokens,
                                llama_sampler *sampler) {
        if (tokens.empty())
            return fail<llama_token>(ErrorCode::InvalidConfig, "LlamaEngine: empty prompt.");

        std::lock_guard<std::mutex> lock(mu_);
        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        llama_pos pos0 = next_pos_[seq];
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = pos0 + static_cast<llama_pos>(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i] = (i + 1 == tokens.size());
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());

        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0)
            return fail<llama_token>(ErrorCode::DecodeFailed, "LlamaEngine: prefill decode failed.");

        next_pos_[seq] = pos0 + static_cast<llama_pos>(tokens.size());
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        return ok(tok);
    }

    Result<llama_token> next_token(llama_seq_id seq, llama_token last_token,
                                   llama_sampler *sampler) {
        std::unique_lock<std::mutex> lock(mu_);
        pending_.push_back(RoundReq{seq, last_token, sampler});

        if (batching_) {
            cv_.wait(lock, [&] { return results_.count(seq) != 0; });
            RoundResult res = results_[seq];
            results_.erase(seq);
            if (!res.ok)
                return fail<llama_token>(ErrorCode::DecodeFailed,
                                         "LlamaEngine: batched decode failed.");
            return ok(res.token);
        }

        batching_ = true;
        lock.unlock();
        std::this_thread::sleep_for(kDebounceWindow);
        lock.lock();

        std::vector<RoundReq> round;
        round.swap(pending_);

        llama_batch batch = llama_batch_init(static_cast<int32_t>(round.size()), 0, 1);
        for (size_t i = 0; i < round.size(); ++i) {
            batch.token[i] = round[i].token;
            batch.pos[i] = next_pos_[round[i].seq];
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = round[i].seq;
            batch.logits[i] = true;
        }
        batch.n_tokens = static_cast<int32_t>(round.size());

        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        for (size_t i = 0; i < round.size(); ++i) {
            llama_seq_id s = round[i].seq;
            next_pos_[s] += 1;
            RoundResult rr;
            rr.ok = (rc == 0);
            if (rr.ok)
                rr.token = llama_sampler_sample(round[i].sampler, ctx, static_cast<int32_t>(i));
            results_[s] = rr;
        }

        batching_ = false;
        cv_.notify_all();

        RoundResult res = results_[seq];
        results_.erase(seq);
        if (!res.ok)
            return fail<llama_token>(ErrorCode::DecodeFailed, "LlamaEngine: batched decode failed.");
        return ok(res.token);
    }

  private:
    static constexpr std::chrono::milliseconds kDebounceWindow{2};

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<llama_seq_id> free_seq_ids_;
    std::unordered_map<llama_seq_id, llama_pos> next_pos_;

    struct RoundReq {
        llama_seq_id seq;
        llama_token token;
        llama_sampler *sampler;
    };
    std::vector<RoundReq> pending_;
    bool batching_ = false;

    struct RoundResult {
        llama_token token = -1;
        bool ok = false;
    };
    std::unordered_map<llama_seq_id, RoundResult> results_;
};

static llama_sampler *build_sampler_chain(float temperature) {
    auto sp = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return smpl;
}

Result<void> init_llama_cpp(Session &session) {
    if (session.config.model.empty())
        return fail(ErrorCode::InvalidConfig, "llama.cpp requires a model path.");

    auto params = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(session.config.model.c_str(), params);
    if (!model)
        return fail(ErrorCode::ModelLoadFailed,
                    "Failed to load llama.cpp model from: " + session.config.model);

    const int n_seq_max = std::max(1, session.config.max_parallel_subagents);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(session.config.context_window) *
                       static_cast<uint32_t>(n_seq_max);
    ctx_params.n_seq_max = static_cast<uint32_t>(n_seq_max);

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        return fail(ErrorCode::ModelLoadFailed, "Failed to create context for llama.cpp model.");
    }

    auto engine = std::make_shared<LlamaEngine>();
    engine->model = model;
    engine->ctx = ctx;
    engine->seed_free_list(n_seq_max);
    session.provider_state = engine;

    return ok();
}

Result<void> clear_kv_cache_llama_cpp(Session &session) {
    auto engine = std::static_pointer_cast<LlamaEngine>(session.provider_state);
    if (!engine || !engine->ctx)
        return fail(ErrorCode::ProviderInitFailed, "Llama model not initialized.");
    llama_memory_clear(llama_get_memory(engine->ctx), true);
    return ok();
}

static bool find_balanced_json_object(std::string_view text, size_t from, size_t &obj_start,
                                      size_t &obj_end) {
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
            if (c == '"')
                in_string = true;
            else if (c == '{')
                ++depth;
            else if (c == '}') {
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

static std::string format_prompt_manual(std::span<const MessageView> messages) {
    std::string prompt;
    for (const auto &msg : messages) {
        prompt += msg.role;
        prompt += ": ";
        prompt += msg.content;
        prompt += "\n";
    }
    prompt += "assistant: ";
    return prompt;
}

static std::string format_prompt_templated(const llama_model *model,
                                            std::span<const MessageView> messages) {
    try {
        const char *tmpl = llama_model_chat_template(model, nullptr);
        if (!tmpl)
            return {};

        std::vector<std::string> owned;
        owned.reserve(messages.size() * 2);
        std::vector<llama_chat_message> chat;
        chat.reserve(messages.size());
        for (const auto &msg : messages) {
            owned.emplace_back(msg.role);
            const char *role_c = owned.back().c_str();
            owned.emplace_back(msg.content);
            const char *content_c = owned.back().c_str();
            chat.push_back(llama_chat_message{role_c, content_c});
        }

        size_t guess = 0;
        for (const auto &msg : messages)
            guess += msg.role.size() + msg.content.size() + 8;
        guess = guess * 2 + 256;

        std::vector<char> buf(guess);
        int32_t written = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true,
                                                     buf.data(), static_cast<int32_t>(buf.size()));
        if (written < 0)
            return {};
        if (static_cast<size_t>(written) > buf.size()) {
            buf.resize(static_cast<size_t>(written));
            written = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(),
                                                static_cast<int32_t>(buf.size()));
            if (written < 0)
                return {};
        }
        return std::string(buf.data(), static_cast<size_t>(written));
    } catch (...) {
        return {};
    }
}

Result<InferResponse> infer_llama_cpp(Session &session, const InferRequest &request) {
    auto engine = std::static_pointer_cast<LlamaEngine>(session.provider_state);
    if (!engine || !engine->model || !engine->ctx)
        return fail<InferResponse>(ErrorCode::ProviderInitFailed, "Llama model not initialized.");

    std::string prompt = format_prompt_templated(engine->model, request.messages);
    if (prompt.empty())
        prompt = format_prompt_manual(request.messages);

    const llama_vocab *vocab = llama_model_get_vocab(engine->model);

    std::vector<llama_token> tokens(prompt.length() + 4);
    int n_tokens =
        llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.length()), tokens.data(),
                       static_cast<int>(tokens.size()), true, true);
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.length()),
                                  tokens.data(), static_cast<int>(tokens.size()), true, true);
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    auto seq_res = engine->acquire_seq();
    if (!seq_res.ok)
        return fail<InferResponse>(seq_res.error.code, seq_res.error.message);
    llama_seq_id seq = seq_res.value;

    const float eff_temp =
        (request.temperature >= 0.0f) ? request.temperature : session.config.temperature;
    llama_sampler *smpl = build_sampler_chain(eff_temp);

    auto cleanup = [&]() {
        if (smpl)
            llama_sampler_free(smpl);
        engine->release_seq(seq);
    };

    auto prefill_res = engine->prefill(seq, tokens, smpl);
    if (!prefill_res.ok) {
        cleanup();
        return fail<InferResponse>(prefill_res.error.code, prefill_res.error.message);
    }

    std::string response;
    const int max_gen = request.max_tokens > 0   ? request.max_tokens
                        : session.config.max_tokens > 0 ? session.config.max_tokens
                                                       : 512;
    llama_token tok = prefill_res.value;

    for (int i = 0; i < max_gen; ++i) {
        if (llama_vocab_is_eog(vocab, tok))
            break;

        std::vector<char> piece_buf(128);
        int n = llama_token_to_piece(vocab, tok, piece_buf.data(), static_cast<int32_t>(piece_buf.size()), 0, true);
        if (n < 0) {
            piece_buf.resize(static_cast<size_t>(-n));
            n = llama_token_to_piece(vocab, tok, piece_buf.data(), static_cast<int32_t>(piece_buf.size()), 0, true);
        }
        if (n >= 0) {
            std::string_view piece(piece_buf.data(), static_cast<size_t>(n));
            response.append(piece);
            if (request.stream && request.on_token)
                request.on_token(piece, request.token_user_data);
        }

        auto next_res = engine->next_token(seq, tok, smpl);
        if (!next_res.ok) {
            response += "\n[Error: decode failure]";
            break;
        }
        tok = next_res.value;
    }

    cleanup();

    std::string_view response_sv = ambient::current_arena(session).allocate_string(response);

    MessageView msg;
    msg.role = "assistant";
    msg.content = response_sv;
    msg.tool_calls = {};
    msg.tool_call_id = "";

    if (!session.config.tools.empty()) {
        size_t search_from = 0;
        size_t obj_start = 0, obj_end = 0;
        while (find_balanced_json_object(response_sv, search_from, obj_start, obj_end)) {
            std::string_view candidate = response_sv.substr(obj_start, obj_end - obj_start + 1);
            search_from = obj_end + 1;

            simdjson::padded_string padded(candidate.data(), candidate.size());
            simdjson::ondemand::document doc;
            if (ambient::json_parser.iterate(padded).get(doc))
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
            if (!raw_args_res.get_string().get(raw_json))
                args_str = std::string(raw_json);
            else if (!raw_args_res.raw_json().get(raw_json))
                args_str = std::string(raw_json);

            auto &arena = ambient::current_arena(session);
            auto tcs = arena.allocate_span<ToolCallView>(1);
            tcs[0].id = arena.allocate_string("call_" + std::string(tool_name));
            tcs[0].name = arena.allocate_string(tool_name);
            tcs[0].arguments = arena.allocate_string(args_str.empty() ? "{}" : args_str);
            msg.tool_calls = tcs;

            std::string_view visible = response_sv.substr(0, obj_start);
            while (!visible.empty() && std::isspace(static_cast<unsigned char>(visible.back())))
                visible.remove_suffix(1);
            msg.content = visible;
            break;
        }
    }

    return ok(InferResponse{
        .message = msg,
        .usage = TokenUsage{.prompt_tokens = n_tokens,
                       .completion_tokens = static_cast<int>(response.size() / 4),
                       .total_tokens = n_tokens + static_cast<int>(response.size() / 4)}});
}

Result<int> count_tokens_llama_cpp(Session &session, std::string_view text) {
    auto engine = std::static_pointer_cast<LlamaEngine>(session.provider_state);
    if (!engine || !engine->model)
        return fail<int>(ErrorCode::ProviderInitFailed, "Llama model not initialized.");

    const llama_vocab *vocab = llama_model_get_vocab(engine->model);
    std::vector<llama_token> tokens(text.size() + 4);
    int n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokens.data(),
                           static_cast<int>(tokens.size()), false, false);
    if (n < 0)
        return ok(static_cast<int>((text.size() + 3) / 4));
    return ok(n);
}

}
#endif
