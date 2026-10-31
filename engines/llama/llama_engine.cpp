// agent-engine-llama: llama.cpp text generation as an agent.cpp engine plugin.
//
// One instance = one model + one llama_context with n_seq_max = "parallel" sequence slots.
// Concurrent run() calls (AgentScheduler subagents) are continuously batched: each generation's
// next-token step joins a shared round, and one llama_decode serves every active sequence.

#include <agent-cpp/engine_abi.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <llama.h>
#include <mutex>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

// ------------------------------------------------------------------ batching engine

class Batcher {
  public:
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    int seq_ctx = 0; // tokens available to one sequence (context_window)

    ~Batcher() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    void seed_free_list(int n_seq_max) {
        for (int i = n_seq_max - 1; i >= 0; --i) free_seq_ids_.push_back(i);
    }

    bool acquire_seq(llama_seq_id &seq) {
        std::lock_guard<std::mutex> lock(mu_);
        if (free_seq_ids_.empty()) return false;
        seq = free_seq_ids_.back();
        free_seq_ids_.pop_back();
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        next_pos_[seq] = 0;
        return true;
    }

    void release_seq(llama_seq_id seq) {
        std::lock_guard<std::mutex> lock(mu_);
        next_pos_.erase(seq);
        free_seq_ids_.push_back(seq);
    }

    // Prompt prefill runs as its own decodes (not batched across sequences), split into chunks of
    // at most n_batch tokens: a single llama_decode larger than n_batch aborts inside llama.cpp.
    bool prefill(llama_seq_id seq, const std::vector<llama_token> &tokens, llama_sampler *sampler, llama_token &out) {
        if (tokens.empty()) return false;
        std::lock_guard<std::mutex> lock(mu_);
        const size_t chunk = std::max<size_t>(1, llama_n_batch(ctx));
        llama_batch batch = llama_batch_init(static_cast<int32_t>(std::min(chunk, tokens.size())), 0, 1);
        llama_pos pos = next_pos_[seq];
        bool ok = true;
        for (size_t begin = 0; ok && begin < tokens.size(); begin += chunk) {
            const size_t n = std::min(chunk, tokens.size() - begin);
            for (size_t i = 0; i < n; ++i) {
                batch.token[i] = tokens[begin + i];
                batch.pos[i] = pos + static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = (begin + i + 1 == tokens.size());
            }
            batch.n_tokens = static_cast<int32_t>(n);
            ok = llama_decode(ctx, batch) == 0;
            pos += static_cast<llama_pos>(n);
        }
        llama_batch_free(batch);
        if (!ok) return false;
        next_pos_[seq] = pos;
        out = llama_sampler_sample(sampler, ctx, -1);
        return true;
    }

    // The first caller in a round leads it: waits a short debounce window so other active
    // generations can join, then decodes every joined sequence's next token in one call.
    bool next_token(llama_seq_id seq, llama_token last, llama_sampler *sampler, llama_token &out) {
        std::unique_lock<std::mutex> lock(mu_);
        pending_.push_back(RoundReq{seq, last, sampler});

        if (batching_) {
            cv_.wait(lock, [&] { return results_.count(seq) != 0; });
            RoundResult res = results_[seq];
            results_.erase(seq);
            out = res.token;
            return res.ok;
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
            if (rr.ok) rr.token = llama_sampler_sample(round[i].sampler, ctx, static_cast<int32_t>(i));
            results_[s] = rr;
        }

        batching_ = false;
        cv_.notify_all();

        RoundResult res = results_[seq];
        results_.erase(seq);
        out = res.token;
        return res.ok;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        llama_memory_clear(llama_get_memory(ctx), true);
    }

  private:
    static constexpr std::chrono::milliseconds kDebounceWindow{2};

    struct RoundReq {
        llama_seq_id seq;
        llama_token token;
        llama_sampler *sampler;
    };
    struct RoundResult {
        llama_token token = -1;
        bool ok = false;
    };

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<llama_seq_id> free_seq_ids_;
    std::unordered_map<llama_seq_id, llama_pos> next_pos_;
    std::vector<RoundReq> pending_;
    bool batching_ = false;
    std::unordered_map<llama_seq_id, RoundResult> results_;
};

// ------------------------------------------------------------------ per-call output

// Response storage for one in-flight run(). Owned by the instance, keyed by thread, so concurrent
// run() calls never share buffers; released by free_response().
struct CallOutput {
    std::string text;
    std::string visible;
    std::string tool_id, tool_name, tool_args;
    agent_part part{};
    agent_tool_call call{};
};

struct Instance {
    Batcher batcher;
    std::mutex out_mu;
    std::unordered_map<std::thread::id, CallOutput> outputs;
    std::unordered_map<std::thread::id, std::string> errors;
    std::string idle_error;

    CallOutput &output() {
        std::lock_guard<std::mutex> lk(out_mu);
        return outputs[std::this_thread::get_id()];
    }
    int32_t fail(int32_t code, std::string msg) {
        std::lock_guard<std::mutex> lk(out_mu);
        errors[std::this_thread::get_id()] = std::move(msg);
        return code;
    }
};

// ------------------------------------------------------------------ helpers

std::once_flag g_backend_once;

llama_sampler *build_sampler(float temperature) {
    llama_sampler *s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(s, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(s, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    return s;
}

std::string message_text(const agent_message &m) {
    std::string out;
    for (size_t p = 0; p < m.n_parts; ++p) {
        const agent_part &part = m.parts[p];
        if (part.kind == AGENT_PART_TEXT) out.append(part.text.data, part.text.size);
    }
    // Earlier assistant tool calls are replayed in the same JSON form the model is asked to emit.
    for (size_t c = 0; c < m.n_tool_calls; ++c) {
        if (!out.empty()) out += '\n';
        out += "{\"name\": \"";
        out.append(m.tool_calls[c].name.data, m.tool_calls[c].name.size);
        out += "\", \"arguments\": ";
        out.append(m.tool_calls[c].arguments.data, m.tool_calls[c].arguments.size);
        out += '}';
    }
    return out;
}

std::string format_manual(const std::vector<std::string> &roles, const std::vector<std::string> &contents) {
    std::string prompt;
    for (size_t i = 0; i < roles.size(); ++i) prompt += roles[i] + ": " + contents[i] + "\n";
    prompt += "assistant: ";
    return prompt;
}

// Returns empty when the model has no built-in template or llama.cpp can't apply it.
std::string format_templated(const llama_model *model, const std::vector<std::string> &roles, const std::vector<std::string> &contents) {
    const char *tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) return {};
    std::vector<llama_chat_message> chat;
    chat.reserve(roles.size());
    size_t guess = 256;
    for (size_t i = 0; i < roles.size(); ++i) {
        chat.push_back(llama_chat_message{roles[i].c_str(), contents[i].c_str()});
        guess += (roles[i].size() + contents[i].size() + 16) * 2;
    }
    std::vector<char> buf(guess);
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), static_cast<int32_t>(buf.size()));
    if (n < 0) return {};
    if (static_cast<size_t>(n) > buf.size()) {
        buf.resize(static_cast<size_t>(n));
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(), static_cast<int32_t>(buf.size()));
        if (n < 0) return {};
    }
    return std::string(buf.data(), static_cast<size_t>(n));
}

bool find_json_object(std::string_view text, size_t from, size_t &start, size_t &end) {
    for (size_t i = from; i < text.size(); ++i) {
        if (text[i] != '{') continue;
        size_t depth = 0;
        bool in_str = false, esc = false;
        for (size_t j = i; j < text.size(); ++j) {
            char c = text[j];
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}' && --depth == 0) {
                start = i;
                end = j;
                return true;
            }
        }
    }
    return false;
}

// Finds the first {"name": ..., "arguments": ...} object in the reply.
bool extract_tool_call(std::string_view text, CallOutput &out) {
    simdjson::ondemand::parser parser;
    size_t from = 0, start = 0, end = 0;
    while (find_json_object(text, from, start, end)) {
        from = end + 1;
        simdjson::padded_string json(text.substr(start, end - start + 1));
        simdjson::ondemand::document doc;
        if (parser.iterate(json).get(doc)) continue;
        simdjson::ondemand::object obj;
        if (doc.get_object().get(obj)) continue;
        std::string_view name;
        if (obj["name"].get_string().get(name) || name.empty()) continue;
        std::string args;
        std::string_view raw;
        auto a = obj["arguments"];
        if (!a.get_string().get(raw)) args = std::string(raw);
        else if (!a.raw_json().get(raw)) args = std::string(raw);
        out.tool_name = std::string(name);
        out.tool_id = "call_" + out.tool_name;
        out.tool_args = args.empty() ? "{}" : args;
        std::string_view visible = text.substr(0, start);
        while (!visible.empty() && std::isspace(static_cast<unsigned char>(visible.back()))) visible.remove_suffix(1);
        out.visible = std::string(visible);
        return true;
    }
    return false;
}

template <typename T> T json_number(simdjson::ondemand::object &obj, const char *key, T fallback) {
    double d = 0;
    if (obj[key].get_double().get(d)) return fallback;
    return static_cast<T>(d);
}

// ------------------------------------------------------------------ ABI functions

int32_t llama_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    std::call_once(g_backend_once, [] { llama_backend_init(); });

    simdjson::ondemand::parser parser;
    simdjson::padded_string json(sv(config_json));
    simdjson::ondemand::document doc;
    simdjson::ondemand::object cfg;
    if (parser.iterate(json).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "llama: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    std::string_view model_path;
    if (cfg["model"].get_string().get(model_path) || model_path.empty()) {
        std::snprintf(err, err_cap, "llama: config.model (GGUF path) is required");
        return AGENT_ERR_INVALID_CONFIG;
    }
    const std::string model_str(model_path);
    const int n_ctx = std::max(1, json_number<int>(cfg, "context_window", 2048));
    const int n_seq = std::max(1, json_number<int>(cfg, "parallel", 1));
    int gpu_layers = -1, threads = 0;
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        gpu_layers = json_number<int>(options, "gpu_layers", -1);
        threads = json_number<int>(options, "threads", 0);
    }

    auto mparams = llama_model_default_params();
    if (gpu_layers >= 0) mparams.n_gpu_layers = gpu_layers;
    llama_model *model = llama_model_load_from_file(model_str.c_str(), mparams);
    if (!model) {
        std::snprintf(err, err_cap, "llama: failed to load model from %s", model_str.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(n_ctx) * static_cast<uint32_t>(n_seq);
    cparams.n_seq_max = static_cast<uint32_t>(n_seq);
    if (threads > 0) {
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
    }
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        std::snprintf(err, err_cap, "llama: failed to create context");
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }

    auto *inst = new Instance();
    inst->batcher.model = model;
    inst->batcher.ctx = ctx;
    inst->batcher.seq_ctx = n_ctx;
    inst->batcher.seed_free_list(n_seq);
    *out_instance = inst;
    return AGENT_OK;
}

void llama_destroy(void *instance) { delete static_cast<Instance *>(instance); }

int32_t llama_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    if (req->task != AGENT_CAP_TEXT_GEN) return inst->fail(AGENT_ERR_UNSUPPORTED, "llama: only TEXT_GEN is supported");

    int max_tokens = 512;
    float temperature = 0.7f;
    {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(sv(req->params_json));
        simdjson::ondemand::document doc;
        simdjson::ondemand::object params;
        if (!req->params_json.size || parser.iterate(json).get(doc) || doc.get_object().get(params)) {
            // keep defaults
        } else {
            max_tokens = json_number<int>(params, "max_tokens", max_tokens);
            temperature = json_number<float>(params, "temperature", temperature);
        }
    }

    std::vector<std::string> roles, contents;
    roles.reserve(req->n_messages);
    contents.reserve(req->n_messages);
    for (size_t i = 0; i < req->n_messages; ++i) {
        roles.emplace_back(sv(req->messages[i].role));
        contents.push_back(message_text(req->messages[i]));
    }

    Batcher &b = inst->batcher;
    std::string prompt = format_templated(b.model, roles, contents);
    if (prompt.empty()) prompt = format_manual(roles, contents);

    const llama_vocab *vocab = llama_model_get_vocab(b.model);
    std::vector<llama_token> tokens(prompt.size() + 4);
    int n_prompt = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                  static_cast<int32_t>(tokens.size()), true, true);
    if (n_prompt < 0) {
        tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                  static_cast<int32_t>(tokens.size()), true, true);
    }
    if (n_prompt <= 0) return inst->fail(AGENT_ERR_DECODE_FAILED, "llama: tokenization failed");
    tokens.resize(static_cast<size_t>(n_prompt));
    if (n_prompt >= b.seq_ctx)
        return inst->fail(AGENT_ERR_INVALID_CONFIG, "llama: prompt is " + std::to_string(n_prompt) + " tokens but context_window is " +
                                                        std::to_string(b.seq_ctx));
    max_tokens = std::min(max_tokens, b.seq_ctx - n_prompt);

    llama_seq_id seq = 0;
    if (!b.acquire_seq(seq)) return inst->fail(AGENT_ERR_DECODE_FAILED, "llama: no free sequence slot (raise max_parallel_subagents)");
    llama_sampler *sampler = build_sampler(temperature);

    CallOutput &out = inst->output();
    out = CallOutput{};
    int completion = 0;
    int32_t status = AGENT_OK;
    llama_token tok = 0;

    if (!b.prefill(seq, tokens, sampler, tok)) {
        status = inst->fail(AGENT_ERR_DECODE_FAILED, "llama: prompt decode failed");
    } else {
        char piece[256];
        for (int i = 0; i < max_tokens; ++i) {
            if (llama_vocab_is_eog(vocab, tok)) break;
            if (req->cancel && *req->cancel) {
                status = inst->fail(AGENT_ERR_CANCELLED, "llama: cancelled");
                break;
            }
            int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
            if (n > 0) {
                out.text.append(piece, static_cast<size_t>(n));
                if (req->on_chunk) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, piece, static_cast<size_t>(n));
            }
            ++completion;
            if (!b.next_token(seq, tok, sampler, tok)) {
                status = inst->fail(AGENT_ERR_DECODE_FAILED, "llama: decode failed");
                break;
            }
        }
    }
    llama_sampler_free(sampler);
    b.release_seq(seq);
    if (status != AGENT_OK) return status;

    *resp = agent_response{};
    resp->message.role = agent_str{"assistant", 9};
    resp->usage = agent_usage{n_prompt, completion};

    std::string_view text = out.text;
    if (req->tools_json.size) {
        if (extract_tool_call(out.text, out)) {
            out.call = agent_tool_call{agent_str{out.tool_id.data(), out.tool_id.size()}, agent_str{out.tool_name.data(), out.tool_name.size()},
                                       agent_str{out.tool_args.data(), out.tool_args.size()}};
            resp->message.tool_calls = &out.call;
            resp->message.n_tool_calls = 1;
            text = out.visible;
        }
    }
    out.part = agent_part{};
    out.part.kind = AGENT_PART_TEXT;
    out.part.text = agent_str{text.data(), text.size()};
    resp->message.parts = &out.part;
    resp->message.n_parts = 1;
    return AGENT_OK;
}

void llama_free_response(void *instance, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    std::lock_guard<std::mutex> lk(inst->out_mu);
    inst->outputs.erase(std::this_thread::get_id());
    *resp = agent_response{};
}

const char *llama_last_error(void *instance) {
    auto *inst = static_cast<Instance *>(instance);
    std::lock_guard<std::mutex> lk(inst->out_mu);
    auto it = inst->errors.find(std::this_thread::get_id());
    return it == inst->errors.end() ? inst->idle_error.c_str() : it->second.c_str();
}

int32_t llama_count_tokens(void *instance, agent_str text, int32_t *out_count) {
    auto *inst = static_cast<Instance *>(instance);
    const llama_vocab *vocab = llama_model_get_vocab(inst->batcher.model);
    int32_t n = llama_tokenize(vocab, text.data, static_cast<int32_t>(text.size), nullptr, 0, false, false);
    *out_count = n < 0 ? -n : n;
    return AGENT_OK;
}

int32_t llama_reset_cache(void *instance) {
    static_cast<Instance *>(instance)->batcher.clear();
    return AGENT_OK;
}

const agent_engine_v1 kLlamaEngine = {
    AGENT_ENGINE_ABI_VERSION,
    "llama",
    "0.1",
    AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL,
    AGENT_ENGINE_FLAG_CONCURRENT,
    llama_create,
    llama_destroy,
    llama_run,
    llama_free_response,
    llama_last_error,
    llama_count_tokens,
    llama_reset_cache,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kLlamaEngine; }
