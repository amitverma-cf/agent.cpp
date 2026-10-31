// agent-engine-llama: llama.cpp text generation, embeddings and reranking as an agent.cpp engine plugin.
//
// One instance = one model + one llama_context with n_seq_max = "parallel" sequence slots.
// Concurrent run() calls (AgentScheduler subagents) are continuously batched: each generation's
// next-token step joins a shared round, and one llama_decode serves every active sequence.
//
// Prompts are rendered with the model's own Jinja chat template and the reply is parsed into
// content + tool calls by llama.cpp's common_chat (each model's native tool-call format).
//
// With a multimodal projector (options.mmproj) image and audio parts are fed through libmtmd: each
// media part becomes mtmd's media marker in the rendered chat, and the prompt is prefilled with
// mtmd_helper_eval_chunks on the request's sequence before generation joins the shared rounds.
//
// With options.mode "embed" / "rerank" the context is created in embedding mode with pooling (rank
// pooling for rerankers) and run() serves AGENT_CAP_EMBED / AGENT_CAP_RERANK instead of chat.

#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <cctype>
#include <chat.h>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <llama.h>
#include <memory>
#include <mtmd-helper.h>
#include <mtmd.h>
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

    // Prefill for prompts with media. mtmd encodes images/audio with the projector and decodes text
    // and embedding chunks in turn; positions come back from the helper because M-RoPE models
    // advance fewer positions than tokens for an image. Runs under the batching mutex like prefill().
    bool prefill_media(llama_seq_id seq, mtmd_context *mctx, const mtmd_input_chunks *chunks, llama_sampler *sampler,
                       llama_token &out) {
        std::lock_guard<std::mutex> lock(mu_);
        llama_pos n_past = next_pos_[seq];
        if (mtmd_helper_eval_chunks(mctx, ctx, chunks, n_past, seq, static_cast<int32_t>(llama_n_batch(ctx)), true, &n_past) != 0)
            return false;
        next_pos_[seq] = n_past;
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
    // AGENT_CAP_EMBED / AGENT_CAP_RERANK: tensor parts pointing into floats.
    std::vector<float> floats;
    int64_t shape = 0;
    std::vector<agent_part> parts;
};

// What the context was created for (create() option "mode"). Embedding and reranking models need a
// context in embedding mode with pooling, which can't generate, so one instance does one of the three.
enum class Mode { Generate, Embed, Rerank };

struct Instance {
    Batcher batcher;
    Mode mode = Mode::Generate;
    std::mutex pool_mu;     // serializes embed / rerank decodes (they use every sequence slot)
    mtmd::context_ptr mtmd; // projector for image/audio input; null without options.mmproj. Freed before the model.
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

bool is_media(const agent_part &p) { return p.kind == AGENT_PART_FILE || p.kind == AGENT_PART_BYTES; }

void erase_all(std::string &text, std::string_view needle) {
    if (needle.empty()) return;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at)) text.erase(at, needle.size());
}

// marker: mtmd's media marker, or nullptr for a text-only model. Each media part becomes one marker at
// its position in the message; markers typed by the user are stripped so they can't misalign media.
common_chat_templates_inputs build_inputs(const agent_request *req, const char *marker) {
    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.add_generation_prompt = true;
    in.enable_thinking = false;
    in.messages.reserve(req->n_messages);
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        common_chat_msg msg;
        msg.role = std::string(sv(m.role));
        for (size_t p = 0; p < m.n_parts; ++p) {
            if (m.parts[p].kind == AGENT_PART_TEXT) {
                if (!marker) {
                    msg.content += sv(m.parts[p].text);
                    continue;
                }
                std::string text(sv(m.parts[p].text));
                erase_all(text, marker);
                msg.content += text;
            } else if (marker && is_media(m.parts[p])) {
                msg.content += marker;
            }
        }
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

// Decodes every media part of the request, in prompt order, into mtmd bitmaps. The helper sniffs
// the format from the bytes (stb_image for images, miniaudio for wav/mp3/flac), so mime is only used
// to reject video, which needs mtmd's ffmpeg subprocess path and isn't built into the plugin.
int32_t load_media(const Instance &inst, const agent_request *req, std::vector<mtmd::bitmap> &out, std::string &error) {
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        for (size_t p = 0; p < m.n_parts; ++p) {
            const agent_part &part = m.parts[p];
            if (!is_media(part)) continue;
            const std::string_view mime = sv(part.mime);
            if (mime.substr(0, 6) == "video/") {
                error = "llama: video input is not supported";
                return AGENT_ERR_UNSUPPORTED;
            }
            mtmd_helper_bitmap_wrapper w{};
            std::string label;
            if (part.kind == AGENT_PART_FILE) {
                label = std::string(sv(part.path));
                w = mtmd_helper_bitmap_init_from_file(inst.mtmd.get(), label.c_str(), false, mtmd_helper_init_opt_default());
            } else {
                label = "<bytes>";
                if (part.data && part.size)
                    w = mtmd_helper_bitmap_init_from_buf(inst.mtmd.get(), static_cast<const unsigned char *>(part.data), part.size, false,
                                                         mtmd_helper_init_opt_default());
            }
            if (w.video_ctx) mtmd_helper_video_free(w.video_ctx);
            mtmd::bitmap bmp(w.bitmap);
            if (!bmp.ptr) {
                error = "llama: could not decode media " + label + " (" + std::string(mime) + ")";
                return part.kind == AGENT_PART_FILE ? AGENT_ERR_FILESYSTEM : AGENT_ERR_PARSE;
            }
            const bool audio = mtmd_bitmap_is_audio(bmp.ptr.get());
            if (audio ? !mtmd_support_audio(inst.mtmd.get()) : !mtmd_support_vision(inst.mtmd.get())) {
                error = std::string("llama: the loaded mmproj does not support ") + (audio ? "audio" : "image") + " input";
                return AGENT_ERR_UNSUPPORTED;
            }
            out.push_back(std::move(bmp));
        }
    }
    return AGENT_OK;
}

// mtmd counterpart of tokenize_prompt(): parts are split at the media markers (build_inputs stripped
// user-typed ones, so every marker is ours) and bitmaps go in at the markers. User-input text is
// tokenized as plain text, template text with special tokens.
bool tokenize_media_prompt(const Instance &inst, const common_chat_prompt &prompt, const std::vector<mtmd::bitmap> &bitmaps,
                           mtmd_input_chunks *chunks, std::string &error) {
    const std::string_view marker = mtmd_default_marker();
    std::vector<mtmd_input_text> texts;
    std::vector<std::pair<bool, size_t>> order; // (is_bitmap, index into texts / bitmaps)
    texts.reserve(prompt.parts.size() + bitmaps.size() + 1);
    size_t n_bitmaps = 0;
    auto add_text = [&](std::string_view t, bool parse_special) {
        if (t.empty()) return;
        texts.push_back(mtmd_input_text{t.data(), t.size(), false, parse_special});
        order.emplace_back(false, texts.size() - 1);
    };
    for (const auto &part : prompt.parts) {
        std::string_view rest = part.text;
        for (size_t at = rest.find(marker); at != std::string_view::npos; at = rest.find(marker)) {
            add_text(rest.substr(0, at), !part.is_input);
            order.emplace_back(true, n_bitmaps++);
            rest.remove_prefix(at + marker.size());
        }
        add_text(rest, !part.is_input);
    }
    if (n_bitmaps != bitmaps.size()) {
        error = "llama: chat template kept " + std::to_string(n_bitmaps) + " of " + std::to_string(bitmaps.size()) + " media parts";
        return false;
    }
    std::vector<mtmd_input_part> parts(order.size());
    std::vector<const mtmd_input_part *> ptrs(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i].first) parts[i] = mtmd_input_part{nullptr, bitmaps[order[i].second].ptr.get()};
        else parts[i] = mtmd_input_part{&texts[order[i].second], nullptr};
        ptrs[i] = &parts[i];
    }
    if (mtmd_tokenize_from_parts(inst.mtmd.get(), chunks, ptrs.data(), ptrs.size(), true) != 0) {
        error = "llama: media tokenization failed";
        return false;
    }
    return true;
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

// ------------------------------------------------------------------ embeddings and reranking

constexpr uint32_t kPoolSeqs = 16; // sequences packed into one embed / rerank decode

std::vector<llama_token> tokenize_text(const llama_vocab *vocab, std::string_view text, bool add_special, bool parse_special) {
    std::vector<llama_token> out(text.size() + 4);
    const int32_t len = static_cast<int32_t>(text.size());
    int32_t n = llama_tokenize(vocab, text.data(), len, out.data(), static_cast<int32_t>(out.size()), add_special, parse_special);
    if (n < 0) {
        out.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.data(), len, out.data(), -n, add_special, parse_special);
    }
    out.resize(n < 0 ? 0 : static_cast<size_t>(n));
    return out;
}

// Inputs longer than one batch are cut; a trailing EOS/SEP added by the tokenizer is kept.
void truncate_tokens(const llama_vocab *vocab, std::vector<llama_token> &tokens, size_t max) {
    if (tokens.size() <= max || max == 0) return;
    const llama_token last = tokens.back();
    tokens.resize(max);
    if (last == llama_vocab_eos(vocab) || last == llama_vocab_sep(vocab)) tokens.back() = last;
}

// Rerank input for one (query, document) pair, as llama-server builds it: the model's "rerank" chat
// template when it has one ({query} / {document} placeholders), otherwise
// [BOS] query [EOS] [SEP] document [EOS] following the vocab's add_* flags. Template text may carry
// special tokens; the query and document are tokenized as plain text so they can't inject any.
std::vector<llama_token> rerank_tokens(const llama_model *model, const llama_vocab *vocab, std::string_view query, std::string_view doc,
                                       size_t max) {
    std::vector<llama_token> q = tokenize_text(vocab, query, false, false);
    std::vector<llama_token> d = tokenize_text(vocab, doc, false, false);
    // Fit the pair into one batch: cut the document first, then the query (leaving room for the
    // template / special tokens).
    const size_t reserve = std::min<size_t>(max / 4, 32);
    const size_t budget = max > reserve ? max - reserve : max;
    if (q.size() + d.size() > budget) {
        const size_t q_max = std::max(budget / 2, budget - std::min(budget, d.size())); // a short document leaves the query more room
        if (q.size() > q_max) q.resize(q_max);
        d.resize(std::min(d.size(), budget - q.size()));
    }

    std::vector<llama_token> out;
    auto append = [&](const std::vector<llama_token> &t) { out.insert(out.end(), t.begin(), t.end()); };
    if (const char *tmpl = llama_model_chat_template(model, "rerank")) {
        const std::string_view t(tmpl);
        size_t lit = 0;
        for (size_t pos = 0; pos < t.size();) {
            const bool is_q = t.compare(pos, 7, "{query}") == 0, is_d = !is_q && t.compare(pos, 10, "{document}") == 0;
            if (!is_q && !is_d) {
                ++pos;
                continue;
            }
            append(tokenize_text(vocab, t.substr(lit, pos - lit), false, true));
            append(is_q ? q : d);
            pos += is_q ? 7 : 10;
            lit = pos;
        }
        append(tokenize_text(vocab, t.substr(lit), false, true));
    } else {
        llama_token eos = llama_vocab_eos(vocab);
        if (eos == LLAMA_TOKEN_NULL) eos = llama_vocab_sep(vocab);
        if (llama_vocab_get_add_bos(vocab)) out.push_back(llama_vocab_bos(vocab));
        append(q);
        if (llama_vocab_get_add_eos(vocab)) out.push_back(eos);
        if (llama_vocab_get_add_sep(vocab)) out.push_back(llama_vocab_sep(vocab));
        append(d);
        if (llama_vocab_get_add_eos(vocab)) out.push_back(eos);
    }
    if (out.size() > max) out.resize(max);
    return out;
}

// Runs every input on its own sequence, packing up to kPoolSeqs inputs (and at most n_batch tokens,
// so a non-causal model sees each whole sequence in one micro-batch) into one decode, and copies
// the first n_out floats of each sequence's pooled output into out[i * n_out].
int32_t pooled_decode(Instance &inst, const std::vector<std::vector<llama_token>> &inputs, size_t n_out, const volatile int32_t *cancel,
                      std::vector<float> &out, std::string &error) {
    std::lock_guard<std::mutex> lock(inst.pool_mu);
    llama_context *ctx = inst.batcher.ctx;
    const size_t n_batch = llama_n_batch(ctx);
    const size_t n_seq_max = std::min<size_t>(kPoolSeqs, llama_n_seq_max(ctx));
    out.assign(inputs.size() * n_out, 0.0f);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch), 0, 1);
    int32_t status = AGENT_OK;
    for (size_t i = 0; i < inputs.size() && status == AGENT_OK;) {
        if (cancel && *cancel) {
            error = "llama: cancelled";
            status = AGENT_ERR_CANCELLED;
            break;
        }
        const size_t first = i;
        batch.n_tokens = 0;
        while (i < inputs.size() && i - first < n_seq_max && static_cast<size_t>(batch.n_tokens) + inputs[i].size() <= n_batch) {
            const auto seq = static_cast<llama_seq_id>(i - first);
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                const int32_t k = batch.n_tokens++;
                batch.token[k] = inputs[i][j];
                batch.pos[k] = static_cast<llama_pos>(j);
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = seq;
                batch.logits[k] = true;
            }
            ++i;
        }
        if (i == first) { // callers truncate every input to n_batch, so this means an empty input
            error = "llama: empty input";
            status = AGENT_ERR_INVALID_CONFIG;
            break;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch) != 0) {
            error = "llama: embedding decode failed";
            status = AGENT_ERR_DECODE_FAILED;
            break;
        }
        for (size_t s = 0; s < i - first; ++s) {
            const float *e = llama_get_embeddings_seq(ctx, static_cast<llama_seq_id>(s));
            if (!e) {
                error = "llama: the context produced no pooled output";
                status = AGENT_ERR_DECODE_FAILED;
                break;
            }
            std::copy(e, e + n_out, out.begin() + static_cast<std::ptrdiff_t>((first + s) * n_out));
        }
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch_free(batch);
    return status;
}

// One f32 tensor part per row of out.floats (rows of out.shape floats each), all with the same name.
void tensor_parts(CallOutput &out, const char *name, size_t n_rows, agent_response *resp) {
    out.parts.assign(n_rows, agent_part{});
    for (size_t r = 0; r < n_rows; ++r) {
        agent_part &p = out.parts[r];
        p.kind = AGENT_PART_TENSOR;
        p.name = agent_str{name, std::char_traits<char>::length(name)};
        p.dtype = AGENT_DTYPE_F32;
        p.shape = &out.shape;
        p.ndim = 1;
        p.data = out.floats.data() + r * static_cast<size_t>(out.shape);
        p.size = static_cast<size_t>(out.shape) * sizeof(float);
    }
    *resp = agent_response{};
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = out.parts.data();
    resp->message.n_parts = out.parts.size();
}

int32_t reject_media(Instance &inst, const agent_request *req) {
    for (size_t i = 0; i < req->n_messages; ++i)
        for (size_t p = 0; p < req->messages[i].n_parts; ++p)
            if (is_media(req->messages[i].parts[p])) return inst.fail(AGENT_ERR_UNSUPPORTED, "llama: embed / rerank take text parts only");
    return AGENT_OK;
}

// AGENT_CAP_EMBED: one pooled, L2-normalized "embedding" tensor (f32 [n_embd]) per TEXT part, in order.
int32_t run_embed(Instance &inst, const agent_request *req, agent_response *resp) {
    if (int32_t rc = reject_media(inst, req); rc != AGENT_OK) return rc;
    const llama_model *model = inst.batcher.model;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const size_t n_batch = llama_n_batch(inst.batcher.ctx);
    std::vector<std::vector<llama_token>> inputs;
    int32_t n_prompt = 0;
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        for (size_t p = 0; p < m.n_parts; ++p) {
            if (m.parts[p].kind != AGENT_PART_TEXT) continue;
            std::vector<llama_token> t = tokenize_text(vocab, sv(m.parts[p].text), true, false);
            if (t.empty()) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: cannot embed an empty text");
            truncate_tokens(vocab, t, n_batch);
            n_prompt += static_cast<int32_t>(t.size());
            inputs.push_back(std::move(t));
        }
    }
    const auto n_embd = static_cast<size_t>(llama_model_n_embd_out(model));
    std::vector<float> floats;
    std::string error;
    if (int32_t rc = pooled_decode(inst, inputs, n_embd, req->cancel, floats, error); rc != AGENT_OK) return inst.fail(rc, error);
    for (size_t r = 0; r < inputs.size(); ++r) {
        float *v = floats.data() + r * n_embd;
        double sum = 0.0;
        for (size_t k = 0; k < n_embd; ++k) sum += static_cast<double>(v[k]) * static_cast<double>(v[k]);
        const float inv = sum > 0.0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0f;
        for (size_t k = 0; k < n_embd; ++k) v[k] *= inv;
    }
    CallOutput &out = inst.output();
    out = CallOutput{};
    out.floats = std::move(floats);
    out.shape = static_cast<int64_t>(n_embd);
    tensor_parts(out, "embedding", inputs.size(), resp);
    resp->usage = agent_usage{n_prompt, 0};
    return AGENT_OK;
}

// AGENT_CAP_RERANK: the first TEXT part is the query, every later one a document. Returns one
// "scores" tensor (f32 [n_docs]) with the classifier's raw relevance score per document (higher =
// more relevant), in document order.
int32_t run_rerank(Instance &inst, const agent_request *req, agent_response *resp) {
    if (int32_t rc = reject_media(inst, req); rc != AGENT_OK) return rc;
    const llama_model *model = inst.batcher.model;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const size_t n_batch = llama_n_batch(inst.batcher.ctx);
    std::string_view query;
    bool have_query = false;
    std::vector<std::vector<llama_token>> inputs;
    int32_t n_prompt = 0;
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        for (size_t p = 0; p < m.n_parts; ++p) {
            if (m.parts[p].kind != AGENT_PART_TEXT) continue;
            if (!have_query) {
                query = sv(m.parts[p].text);
                have_query = true;
                continue;
            }
            std::vector<llama_token> t = rerank_tokens(model, vocab, query, sv(m.parts[p].text), n_batch);
            n_prompt += static_cast<int32_t>(t.size());
            inputs.push_back(std::move(t));
        }
    }
    if (inputs.empty()) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: rerank needs a query and at least one document text part");
    std::vector<float> scores;
    std::string error;
    if (int32_t rc = pooled_decode(inst, inputs, 1, req->cancel, scores, error); rc != AGENT_OK) return inst.fail(rc, error);
    CallOutput &out = inst.output();
    out = CallOutput{};
    out.floats = std::move(scores);
    out.shape = static_cast<int64_t>(inputs.size());
    tensor_parts(out, "scores", 1, resp);
    resp->usage = agent_usage{n_prompt, 0};
    return AGENT_OK;
}

// llama.cpp has no public "has a classification head" query. Reranker GGUFs carry no
// <arch>.pooling_type or declare rank pooling; embedding models declare mean / cls / last. So a
// declared non-rank pooling type rules the model out.
bool maybe_reranker(const llama_model *model) {
    char arch[64], value[16];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) <= 0) return true;
    const std::string key = std::string(arch) + ".pooling_type";
    if (llama_model_meta_val_str(model, key.c_str(), value, sizeof(value)) <= 0) return true;
    return std::strtol(value, nullptr, 10) == LLAMA_POOLING_TYPE_RANK;
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
    std::string chat_template, mmproj;
    Mode mode = Mode::Generate;
    enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        gpu_layers = json_number<int>(options, "gpu_layers", -1);
        threads = json_number<int>(options, "threads", 0);
        std::string_view t;
        if (!options["chat_template"].get_string().get(t)) chat_template = std::string(t);
        if (!options["mmproj"].get_string().get(t)) mmproj = std::string(t);
        if (!options["mode"].get_string().get(t)) {
            if (t == "embed") mode = Mode::Embed;
            else if (t == "rerank") mode = Mode::Rerank;
            else if (t != "generate") {
                std::snprintf(err, err_cap, "llama: options.mode must be \"generate\", \"embed\" or \"rerank\"");
                return AGENT_ERR_INVALID_CONFIG;
            }
        }
        if (!options["pooling"].get_string().get(t)) {
            if (t == "mean") pooling = LLAMA_POOLING_TYPE_MEAN;
            else if (t == "cls") pooling = LLAMA_POOLING_TYPE_CLS;
            else if (t == "last") pooling = LLAMA_POOLING_TYPE_LAST;
            else {
                std::snprintf(err, err_cap, "llama: options.pooling must be \"mean\", \"cls\" or \"last\"");
                return AGENT_ERR_INVALID_CONFIG;
            }
        }
    }
    if (mode != Mode::Generate && !mmproj.empty()) {
        std::snprintf(err, err_cap, "llama: options.mmproj is only used in generate mode");
        return AGENT_ERR_INVALID_CONFIG;
    }

    auto mparams = llama_model_default_params();
    if (gpu_layers >= 0) mparams.n_gpu_layers = gpu_layers;
    llama_model *model = llama_model_load_from_file(model_str.c_str(), mparams);
    if (!model) {
        std::snprintf(err, err_cap, "llama: failed to load model from %s", model_str.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }

    mtmd::context_ptr mctx;
    if (!mmproj.empty()) {
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = gpu_layers != 0;
        mp.print_timings = false;
        if (threads > 0) mp.n_threads = threads;
        mctx.reset(mtmd_init_from_file(mmproj.c_str(), model, mp));
        if (!mctx) {
            llama_model_free(model);
            std::snprintf(err, err_cap, "llama: failed to load mmproj from %s", mmproj.c_str());
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(n_ctx) * static_cast<uint32_t>(n_seq);
    cparams.n_seq_max = static_cast<uint32_t>(n_seq);
    if (threads > 0) {
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
    }
    // Projectors with non-causal attention over image tokens (e.g. Gemma 3) need a whole image in
    // one micro-batch.
    if (mctx && mtmd_decode_use_non_causal(mctx.get(), nullptr)) cparams.n_ubatch = cparams.n_batch;
    if (mode != Mode::Generate) {
        // Embed / rerank: pooled outputs, one decode per group of inputs, cleared after each. An input
        // is at most one batch (and the model's training context: BERT-style position embeddings end
        // there), and the batch is one micro-batch so non-causal models see whole sequences.
        const int n_train = llama_model_n_ctx_train(model);
        const auto seq_tokens = static_cast<uint32_t>(n_train > 0 ? std::min(n_ctx, n_train) : n_ctx);
        cparams.embeddings = true;
        cparams.pooling_type = mode == Mode::Rerank ? LLAMA_POOLING_TYPE_RANK : pooling;
        cparams.n_ctx = seq_tokens;
        cparams.n_batch = seq_tokens;
        cparams.n_ubatch = seq_tokens;
        cparams.n_seq_max = kPoolSeqs;
        cparams.kv_unified = true;
    }
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx && mode == Mode::Embed && llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_NONE) {
        // The model declares no pooling (e.g. a generative model used as an embedder): mean-pool.
        llama_free(ctx);
        cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        ctx = llama_init_from_model(model, cparams);
    }
    if (!ctx) {
        mctx.reset();
        llama_model_free(model);
        std::snprintf(err, err_cap, "llama: failed to create context");
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    if (mode == Mode::Rerank &&
        (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_RANK || llama_model_n_cls_out(model) < 1 || !maybe_reranker(model))) {
        llama_free(ctx);
        llama_model_free(model);
        std::snprintf(err, err_cap, "llama: %s is not a reranker (no classification head); rerank mode needs a reranker model",
                      model_str.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }

    auto *inst = new Instance();
    inst->batcher.model = model;
    inst->batcher.ctx = ctx;
    inst->batcher.seq_ctx = n_ctx;
    inst->mode = mode;
    if (mode != Mode::Generate) { // no chat template or sequence slots needed
        *out_instance = inst;
        return AGENT_OK;
    }
    inst->batcher.seed_free_list(n_seq);
    inst->mtmd = std::move(mctx);
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
    if (inst->mode == Mode::Embed && req->task == AGENT_CAP_EMBED) return run_embed(*inst, req, resp);
    if (inst->mode == Mode::Rerank && req->task == AGENT_CAP_RERANK) return run_rerank(*inst, req, resp);
    if (inst->mode != Mode::Generate || req->task != AGENT_CAP_TEXT_GEN) {
        static constexpr const char *kModeTask[] = {"TEXT_GEN", "EMBED", "RERANK"};
        static constexpr const char *kModeName[] = {"generate", "embed", "rerank"};
        const auto m = static_cast<size_t>(inst->mode);
        return inst->fail(AGENT_ERR_UNSUPPORTED, std::string("llama: this instance was created with mode \"") + kModeName[m] +
                                                     "\" and only runs " + kModeTask[m] + " requests (set options.mode)");
    }

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

    bool has_media = false;
    for (size_t i = 0; i < req->n_messages && !has_media; ++i)
        for (size_t p = 0; p < req->messages[i].n_parts && !has_media; ++p) has_media = is_media(req->messages[i].parts[p]);
    if (has_media && !inst->mtmd)
        return inst->fail(AGENT_ERR_UNSUPPORTED, "llama: image/audio input needs a multimodal projector (engine option \"mmproj\")");

    std::string error;
    std::vector<mtmd::bitmap> bitmaps;
    if (has_media) {
        int32_t rc = load_media(*inst, req, bitmaps, error);
        if (rc != AGENT_OK) return inst->fail(rc, error);
    }

    common_chat_session session;
    if (!start_session(*inst, vocab, build_inputs(req, inst->mtmd ? mtmd_default_marker() : nullptr), session, error))
        return inst->fail(AGENT_ERR_INVALID_CONFIG, error);

    std::vector<llama_token> tokens;
    mtmd::input_chunks chunks(has_media ? mtmd_input_chunks_init() : nullptr);
    int n_prompt = 0;
    if (has_media) {
        if (!tokenize_media_prompt(*inst, session.prompt(), bitmaps, chunks.ptr.get(), error))
            return inst->fail(AGENT_ERR_DECODE_FAILED, error);
        n_prompt = static_cast<int>(mtmd_helper_get_n_tokens(chunks.ptr.get()));
    } else {
        if (!tokenize_prompt(vocab, session.prompt(), tokens)) return inst->fail(AGENT_ERR_DECODE_FAILED, "llama: tokenization failed");
        n_prompt = static_cast<int>(tokens.size());
    }
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

    const bool prefilled =
        has_media ? b.prefill_media(seq, inst->mtmd.get(), chunks.ptr.get(), sampler, tok) : b.prefill(seq, tokens, sampler, tok);
    if (!prefilled) {
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
    "0.4",
    // Static: what the plugin can do. Image/audio input also needs options.mmproj (and audio a
    // projector that supports it); otherwise run() returns AGENT_ERR_UNSUPPORTED for media parts.
    // EMBED / RERANK need an instance created with options.mode "embed" / "rerank"; an instance runs
    // only its own mode's task.
    AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL | AGENT_CAP_VISION_IN | AGENT_CAP_AUDIO_IN | AGENT_CAP_EMBED | AGENT_CAP_RERANK,
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
