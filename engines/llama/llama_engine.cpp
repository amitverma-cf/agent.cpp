// agent-engine-llama: llama.cpp text generation as an agent.cpp engine plugin.
//
// One instance = one model + one llama_context with n_seq_max = "parallel" sequence slots.
// Concurrent run() calls (AgentScheduler subagents) are continuously batched: each generation's
// next-token step joins a shared round, and one llama_decode serves every active sequence.
//
// Prompts are rendered with the model's own Jinja chat template and the reply is parsed into
// content + tool calls by llama.cpp's common_chat (each model's native tool-call format).

#include <agent-cpp/engine_abi.h>

#include <algorithm>
#include <cctype>
#include <chat.h>
#include <exception>
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
    std::string content;
    common_chat_msg parsed;
    std::vector<agent_tool_call> calls;
    agent_part part{};
};

struct Instance {
    Batcher batcher;
    common_chat_templates_ptr templates;
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

template <typename T> T json_number(simdjson::ondemand::object &obj, const char *key, T fallback) {
    double d = 0;
    if (obj[key].get_double().get(d)) return fallback;
    return static_cast<T>(d);
}

common_chat_templates_inputs build_inputs(const agent_request *req) {
    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.add_generation_prompt = true;
    in.enable_thinking = false;
    in.messages.reserve(req->n_messages);
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        common_chat_msg msg;
        msg.role = std::string(sv(m.role));
        for (size_t p = 0; p < m.n_parts; ++p)
            if (m.parts[p].kind == AGENT_PART_TEXT) msg.content += sv(m.parts[p].text);
        for (size_t c = 0; c < m.n_tool_calls; ++c)
            msg.tool_calls.push_back(common_chat_tool_call{std::string(sv(m.tool_calls[c].name)), std::string(sv(m.tool_calls[c].arguments)),
                                                           std::string(sv(m.tool_calls[c].id))});
        msg.tool_call_id = std::string(sv(m.tool_call_id));
        in.messages.push_back(std::move(msg));
    }

    // tools_json: [{"name": ..., "description": ..., "parameters": {...}}, ...]
    if (req->tools_json.size) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(sv(req->tools_json));
        simdjson::ondemand::document doc;
        simdjson::ondemand::array arr;
        if (!parser.iterate(json).get(doc) && !doc.get_array().get(arr)) {
            for (auto el : arr) {
                simdjson::ondemand::object obj;
                if (el.get_object().get(obj)) continue;
                common_chat_tool tool;
                std::string_view s;
                for (auto field : obj) {
                    std::string_view key;
                    if (field.unescaped_key().get(key)) continue;
                    if (key == "name" && !field.value().get_string().get(s)) tool.name = std::string(s);
                    else if (key == "description" && !field.value().get_string().get(s)) tool.description = std::string(s);
                    else if (key == "parameters" && !field.value().raw_json().get(s)) tool.parameters = std::string(s);
                }
                if (!tool.name.empty()) {
                    if (tool.parameters.empty()) tool.parameters = "{}";
                    in.tools.push_back(std::move(tool));
                }
            }
        }
    }
    return in;
}

// Exception firewall. llama.cpp's common_chat (Jinja rendering, PEG parsing) reports errors by
// throwing; nothing may unwind across the C plugin ABI into the host, so these two calls are the
// only places in agent.cpp that catch. Everything else in the plugin and the core is exception-free.
bool start_session(const Instance &inst, const llama_vocab *vocab, const common_chat_templates_inputs &in, common_chat_session &out,
                   std::string &error) {
    try {
        out = common_chat_session(inst.templates.get(), vocab, in);
        return true;
    } catch (const std::exception &e) {
        error = std::string("llama: chat template failed: ") + e.what();
    } catch (...) { error = "llama: chat template failed"; }
    return false;
}

bool finish_session(common_chat_session &session, const common_chat_input &generated, common_chat_msg &out, std::string &error) {
    try {
        out = session.finish(generated);
        return true;
    } catch (const std::exception &e) {
        error = std::string("llama: could not parse model output: ") + e.what();
    } catch (...) { error = "llama: could not parse model output"; }
    return false;
}

// Template parts may carry special tokens; user-provided parts are tokenized as plain text so they
// can't inject control tokens.
bool tokenize_prompt(const llama_vocab *vocab, const common_chat_prompt &prompt, std::vector<llama_token> &tokens) {
    bool first = true;
    for (const auto &part : prompt.parts) {
        if (part.text.empty()) continue;
        const int32_t len = static_cast<int32_t>(part.text.size());
        const size_t at = tokens.size();
        tokens.resize(at + part.text.size() + 4);
        int32_t n = llama_tokenize(vocab, part.text.data(), len, tokens.data() + at, static_cast<int32_t>(tokens.size() - at), first,
                                   !part.is_input);
        if (n < 0) {
            tokens.resize(at + static_cast<size_t>(-n));
            n = llama_tokenize(vocab, part.text.data(), len, tokens.data() + at, -n, first, !part.is_input);
        }
        if (n < 0) return false;
        tokens.resize(at + static_cast<size_t>(n));
        first = false;
    }
    return !tokens.empty();
}

bool ends_with_stop(const std::string &text, const std::vector<std::string> &stops, size_t &cut) {
    for (const auto &s : stops) {
        if (!s.empty() && text.size() >= s.size() && text.compare(text.size() - s.size(), s.size(), s) == 0) {
            cut = text.size() - s.size();
            return true;
        }
    }
    return false;
}

// Fallback for models that drift from their own tool-call format (e.g. small Qwen2.5 models copy
// the "{{...}}" example their template ships with): finds the first {"name": ..., "arguments": ...}
// object anywhere in the text, retrying from each '{' so a stray outer brace is skipped.
bool find_loose_tool_call(std::string_view text, common_chat_tool_call &call) {
    simdjson::ondemand::parser parser;
    for (size_t i = text.find('{'); i != std::string_view::npos; i = text.find('{', i + 1)) {
        size_t depth = 0, end = std::string_view::npos;
        bool in_str = false, esc = false;
        for (size_t j = i; j < text.size() && end == std::string_view::npos; ++j) {
            char c = text[j];
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
            } else if (c == '"') in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}' && --depth == 0) end = j;
        }
        if (end == std::string_view::npos) return false;
        simdjson::padded_string json(text.substr(i, end - i + 1));
        simdjson::ondemand::document doc;
        simdjson::ondemand::object obj;
        if (parser.iterate(json).get(doc) || doc.get_object().get(obj)) continue;
        std::string_view name, args;
        if (obj["name"].get_string().get(name) || name.empty()) continue;
        auto a = obj["arguments"];
        if (a.get_string().get(args) && a.raw_json().get(args)) args = "{}";
        call.name = std::string(name);
        call.arguments = std::string(args);
        return true;
    }
    return false;
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
    std::string chat_template;
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        gpu_layers = json_number<int>(options, "gpu_layers", -1);
        threads = json_number<int>(options, "threads", 0);
        std::string_view t;
        if (!options["chat_template"].get_string().get(t)) chat_template = std::string(t);
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
    // Exception firewall (see start_session): template loading parses Jinja and may throw.
    try {
        inst->templates = common_chat_templates_init(model, chat_template);
    } catch (const std::exception &e) {
        std::snprintf(err, err_cap, "llama: invalid chat template: %s", e.what());
    } catch (...) { std::snprintf(err, err_cap, "llama: invalid chat template"); }
    if (!inst->templates) {
        delete inst;
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    *out_instance = inst;
    return AGENT_OK;
}

void llama_destroy(void *instance) { delete static_cast<Instance *>(instance); }

int32_t llama_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    if (req->task != AGENT_CAP_TEXT_GEN) return inst->fail(AGENT_ERR_UNSUPPORTED, "llama: only TEXT_GEN is supported");

    int max_tokens = 512;
    float temperature = 0.7f;
    if (req->params_json.size) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(sv(req->params_json));
        simdjson::ondemand::document doc;
        simdjson::ondemand::object params;
        if (!parser.iterate(json).get(doc) && !doc.get_object().get(params)) {
            max_tokens = json_number<int>(params, "max_tokens", max_tokens);
            temperature = json_number<float>(params, "temperature", temperature);
        }
    }

    Batcher &b = inst->batcher;
    const llama_vocab *vocab = llama_model_get_vocab(b.model);

    common_chat_session session;
    std::string error;
    if (!start_session(*inst, vocab, build_inputs(req), session, error)) return inst->fail(AGENT_ERR_INVALID_CONFIG, error);

    std::vector<llama_token> tokens;
    if (!tokenize_prompt(vocab, session.prompt(), tokens)) return inst->fail(AGENT_ERR_DECODE_FAILED, "llama: tokenization failed");
    const int n_prompt = static_cast<int>(tokens.size());
    if (n_prompt >= b.seq_ctx)
        return inst->fail(AGENT_ERR_INVALID_CONFIG, "llama: prompt is " + std::to_string(n_prompt) + " tokens but context_window is " +
                                                        std::to_string(b.seq_ctx));
    max_tokens = std::min(max_tokens, b.seq_ctx - n_prompt);

    llama_seq_id seq = 0;
    if (!b.acquire_seq(seq)) return inst->fail(AGENT_ERR_DECODE_FAILED, "llama: no free sequence slot (raise max_parallel_subagents)");
    llama_sampler *sampler = build_sampler(temperature);

    common_chat_input generated;
    const std::vector<std::string> &stops = session.additional_stops();
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
            int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
            if (n > 0) {
                generated.append(std::string(piece, static_cast<size_t>(n)), tok);
                if (req->on_chunk) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, piece, static_cast<size_t>(n));
            }
            ++completion;
            size_t cut = 0;
            if (ends_with_stop(generated.text, stops, cut)) {
                generated.truncate(cut);
                break;
            }
            if (!b.next_token(seq, tok, sampler, tok)) {
                status = inst->fail(AGENT_ERR_DECODE_FAILED, "llama: decode failed");
                break;
            }
        }
    }
    llama_sampler_free(sampler);
    b.release_seq(seq);
    if (status != AGENT_OK) return status;

    CallOutput &out = inst->output();
    out = CallOutput{};
    if (finish_session(session, generated, out.parsed, error)) out.content = out.parsed.content;
    else out.content = generated.text; // unparseable output is returned verbatim rather than failing the turn

    common_chat_tool_call loose;
    if (out.parsed.tool_calls.empty() && req->tools_json.size && find_loose_tool_call(generated.text, loose)) {
        out.parsed.tool_calls.push_back(std::move(loose));
        out.content.clear();
    }
    out.calls.reserve(out.parsed.tool_calls.size());
    for (size_t i = 0; i < out.parsed.tool_calls.size(); ++i) {
        common_chat_tool_call &c = out.parsed.tool_calls[i];
        if (c.id.empty()) c.id = "call_" + std::to_string(i);
        if (c.arguments.empty()) c.arguments = "{}";
        out.calls.push_back(agent_tool_call{agent_str{c.id.data(), c.id.size()}, agent_str{c.name.data(), c.name.size()},
                                            agent_str{c.arguments.data(), c.arguments.size()}});
    }

    *resp = agent_response{};
    resp->message.role = agent_str{"assistant", 9};
    resp->usage = agent_usage{n_prompt, completion};
    out.part = agent_part{};
    out.part.kind = AGENT_PART_TEXT;
    out.part.text = agent_str{out.content.data(), out.content.size()};
    resp->message.parts = &out.part;
    resp->message.n_parts = 1;
    resp->message.tool_calls = out.calls.data();
    resp->message.n_tool_calls = out.calls.size();
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
    "0.2",
    AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL,
    AGENT_ENGINE_FLAG_CONCURRENT | AGENT_ENGINE_FLAG_NATIVE_TOOLS,
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
