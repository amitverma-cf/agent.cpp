// agent-engine-llama: llama.cpp text generation, embeddings and reranking as an agent.cpp engine plugin.
//
// One instance = one model + one llama_context with n_seq_max = "parallel" sequence slots.
// Concurrent run() calls (parallel graph branches) are continuously batched: each generation's
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
//
// In generate mode AGENT_CAP_DECIDE scores a fixed set of answers (choice / score / yes-no) from the
// model's log-probabilities after one prompt prefill, without generating, and AGENT_CAP_OCR
// transcribes image parts with the OCR prompt of the model's family (PaddleOCR-VL, GLM-OCR, ...).

#include "upstream_engine.hpp"

#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <cctype>
#include <chat.h>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <llama.h>
#include <memory>
#include <mtmd-helper.h>
#include <mtmd.h>
#include <mutex>
#include <sampling.h>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

// ------------------------------------------------------------------ sampling

// Samples the next token and feeds it back to the sampler (its grammar state, if any, follows the output).
llama_token sample_token(common_sampler *sampler, llama_context *ctx, int32_t idx) {
    const llama_token token = common_sampler_sample(sampler, ctx, idx);
    common_sampler_accept(sampler, token, true);
    return token;
}

// The tokens that followed the most recent earlier occurrence of the last `n` tokens of `history`, at most `max_draft`.
std::vector<llama_token> ngram_draft(const std::vector<llama_token> &history, size_t n, size_t max_draft) {
    std::vector<llama_token> draft;
    if (history.size() <= n || max_draft == 0) return draft;
    const size_t end = history.size();
    for (size_t start = end - n; start-- > 0;) { // candidate starts, newest first, never the suffix itself
        if (!std::equal(history.begin() + static_cast<std::ptrdiff_t>(start), history.begin() + static_cast<std::ptrdiff_t>(start + n),
                        history.end() - static_cast<std::ptrdiff_t>(n)))
            continue;
        for (size_t j = start + n; j < end && draft.size() < max_draft; ++j) draft.push_back(history[j]);
        break;
    }
    return draft;
}

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

    // Prefix cache. Every sequence slot keeps the tokens whose keys/values are in its KV cache after a request.
    // A new text prompt takes the free slot whose tokens share the longest prefix with it, drops only the
    // differing tail and decodes just the rest, instead of reading the whole conversation again. The KV
    // buffer of every slot is allocated when the context is created, so this costs no extra memory.
    // Models with recurrent state (hybrid conv/attention such as LFM, Mamba) cannot cut a sequence back to an
    // earlier point, so they use checkpoints instead: when a prompt has been decoded, the recurrent part of the
    // state is saved (a few MiB); the next request restores it, drops the attention keys after it and decodes the
    // rest. Off for models that keep only a sliding window of keys (SWA); those recompute the whole prompt.
    bool prefix_cache = true;
    bool use_checkpoints = false;
    bool spec_enabled = false; // prompt-lookup speculation allowed (see speculate) // recurrent or hybrid model: reuse through checkpoints
    uint64_t checkpoint_restores = 0;
    uint64_t reused_tokens = 0, prompt_tokens = 0; // totals over all text requests, for stats and tests
    int kv_type_id = AGENT_KV_F16;                 // agent_kv_type of the K/V cache
    uint64_t kv_bytes = 0;                         // estimated size of the whole KV cache

    // Tokens currently in the slot's KV sequence.
    int position(llama_seq_id seq) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = next_pos_.find(seq);
        return it == next_pos_.end() ? 0 : static_cast<int>(it->second);
    }

    // Takes a free slot and clears it: DECIDE scoring and media prompts, which do not use the prefix cache.
    bool acquire_seq(llama_seq_id &seq) {
        std::lock_guard<std::mutex> lock(mu_);
        if (free_seq_ids_.empty()) return false;
        seq = free_seq_ids_.back();
        free_seq_ids_.pop_back();
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        next_pos_[seq] = 0;
        cached_[seq].clear();
        ckpt_.erase(seq);
        reusable_[seq] = false;
        return true;
    }

    // Takes the free slot that best matches `tokens` and trims it to the shared prefix. *reused tokens are
    // already in the KV cache: the caller decodes tokens[*reused ...]. One token is always left to decode,
    // because its logits give the first reply token.
    bool acquire_seq_for(const std::vector<llama_token> &tokens, llama_seq_id &seq, size_t &reused) {
        std::lock_guard<std::mutex> lock(mu_);
        if (free_seq_ids_.empty()) return false;
        size_t best = free_seq_ids_.size() - 1; // default: the most recently released slot
        size_t best_lcp = 0;
        if (prefix_cache && reuse_ok_ && !tokens.empty()) {
            for (size_t i = free_seq_ids_.size(); i-- > 0;) {
                const llama_seq_id id = free_seq_ids_[i];
                const std::vector<llama_token> &c = cached_[id];
                const size_t n = std::min(c.size(), tokens.size());
                size_t k = 0;
                while (k < n && c[k] == tokens[k]) ++k;
                if (std::getenv("AGENT_LLAMA_TRACE"))
                    std::fprintf(stderr, "llama: select slot %d lcp=%zu cached=%zu prompt=%zu ckpt_pos=%zu\n", static_cast<int>(id), k,
                                 c.size(), tokens.size(), ckpt_.count(id) ? ckpt_[id].pos : static_cast<size_t>(0));
                if (use_checkpoints) { // only the position of the saved checkpoint can be resumed from
                    const auto ck = ckpt_.find(id);
                    k = (ck != ckpt_.end() && ck->second.pos <= k) ? ck->second.pos : 0;
                }
                if (k > best_lcp) {
                    best_lcp = k;
                    best = i;
                }
            }
        }
        seq = free_seq_ids_[best];
        free_seq_ids_.erase(free_seq_ids_.begin() + static_cast<std::ptrdiff_t>(best));

        reused = tokens.empty() ? 0 : std::min(best_lcp, tokens.size() - 1);
        llama_memory_t mem = llama_get_memory(ctx);
        if (use_checkpoints) {
            // Restore the recurrent state saved when the earlier prompt ended (it sits exactly at `reused`), then
            // drop the attention keys of the reply that followed it.
            bool restored = false;
            if (reused > 0) {
                const auto ck = ckpt_.find(seq);
                bool set_ok = false;
                if (ck != ckpt_.end() && ck->second.pos == reused) {
                    set_ok = llama_state_seq_set_data_ext(ctx, ck->second.data.data(), ck->second.data.size(), seq,
                                                          LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 0;
                    restored = set_ok && llama_memory_seq_rm(mem, seq, static_cast<llama_pos>(reused), -1);
                }
                if (std::getenv("AGENT_LLAMA_TRACE"))
                    std::fprintf(stderr, "llama: checkpoint slot %d saved_pos=%zu want=%zu state_set=%d trimmed=%d\n",
                                 static_cast<int>(seq), ck == ckpt_.end() ? size_t{0} : ck->second.pos, reused, set_ok ? 1 : 0,
                                 restored ? 1 : 0);
            } else if (std::getenv("AGENT_LLAMA_TRACE")) {
                const auto ck = ckpt_.find(seq);
                std::fprintf(stderr, "llama: no resumable prefix: slot %d has_checkpoint=%d checkpoint_pos=%zu cached_tokens=%zu\n",
                             static_cast<int>(seq), ck == ckpt_.end() ? 0 : 1, ck == ckpt_.end() ? size_t{0} : ck->second.pos,
                             cached_[seq].size());
            }
            if (restored) ++checkpoint_restores;
            else reused = 0;
        } else if (reused > 0 && !llama_memory_seq_rm(mem, seq, static_cast<llama_pos>(reused), -1)) {
            // This model cannot drop only the end of a sequence: stop trying and recompute the whole prompt.
            reuse_ok_ = false;
            reused = 0;
        }
        if (reused == 0) {
            llama_memory_seq_rm(mem, seq, -1, -1);
            ckpt_.erase(seq);
        }
        cached_[seq].resize(reused);
        next_pos_[seq] = static_cast<llama_pos>(reused);
        reusable_[seq] = true;
        reused_tokens += reused;
        prompt_tokens += tokens.size();
        return true;
    }

    // keep = false when the request failed or was cancelled: the slot's KV cache is then not trusted.
    void release_seq(llama_seq_id seq, bool keep = true) {
        std::lock_guard<std::mutex> lock(mu_);
        next_pos_.erase(seq);
        if (!keep || !reusable_[seq]) {
            cached_[seq].clear();
            ckpt_.erase(seq);
        }
        reusable_[seq] = false;
        free_seq_ids_.push_back(seq);
    }

    // Prompt prefill runs as its own decodes (not batched across sequences), split into chunks of
    // at most n_batch tokens: a single llama_decode larger than n_batch aborts inside llama.cpp.
    bool prefill(llama_seq_id seq, const std::vector<llama_token> &tokens, size_t begin, common_sampler *sampler, llama_token &out) {
        if (begin >= tokens.size()) return false;
        std::lock_guard<std::mutex> lock(mu_);
        if (use_checkpoints && reusable_[seq] && tokens.size() > begin + kCheckpointMargin + 1) {
            // Checkpoint a few tokens before the end of the prompt, not at the very end: the next turn's prompt ends the
            // earlier one differently (the assistant header is followed by the reply, and its last tokens can split
            // differently), so only the part before that is certain to be shared.
            const size_t split = tokens.size() - kCheckpointMargin;
            if (!decode_prompt_locked(seq, tokens, begin, split)) return false;
            save_checkpoint_locked(seq);
            if (!decode_prompt_locked(seq, tokens, split)) return false;
        } else {
            if (!decode_prompt_locked(seq, tokens, begin)) return false;
            if (use_checkpoints && reusable_[seq]) save_checkpoint_locked(seq);
        }
        out = sample_token(sampler, ctx, -1);
        return true;
    }

    // Prefill for prompts with media. mtmd encodes images/audio with the projector and decodes text
    // and embedding chunks in turn; positions come back from the helper because M-RoPE models
    // advance fewer positions than tokens for an image. Runs under the batching mutex like prefill().
    bool prefill_media(llama_seq_id seq, mtmd_context *mctx, const mtmd_input_chunks *chunks, common_sampler *sampler, llama_token &out) {
        std::lock_guard<std::mutex> lock(mu_);
        llama_pos n_past = next_pos_[seq];
        if (mtmd_helper_eval_chunks(mctx, ctx, chunks, n_past, seq, static_cast<int32_t>(llama_n_batch(ctx)), true, &n_past) != 0)
            return false;
        next_pos_[seq] = n_past;
        out = sample_token(sampler, ctx, -1);
        return true;
    }

    // Prompt-lookup speculation. Decodes `last` followed by the drafted tokens in ONE call, with logits at every position, and
    // samples after each. The draft tokens that the model itself would have produced are accepted, the rest are dropped from
    // the KV cache again, and `next` is the token after the last accepted one (the model's correction, or a free extra token
    // when the whole draft was right). The tokens are exactly those greedy decoding would give, up to the usual float
    // differences between batch shapes. Only for attention models that can drop the end of a sequence.
    bool speculate(llama_seq_id seq, llama_token last, const std::vector<llama_token> &draft, common_sampler *sampler,
                   std::vector<llama_token> &accepted, llama_token &next) {
        std::lock_guard<std::mutex> lock(mu_);
        const llama_pos base = next_pos_[seq];
        const int32_t n = static_cast<int32_t>(draft.size()) + 1;
        // Hybrid / recurrent models cannot cut their state back to an earlier position, so the recurrent part is saved before the
        // draft is decoded and restored if part of it is rejected (the accepted prefix is then decoded again).
        std::vector<uint8_t> snapshot;
        if (use_checkpoints) {
            snapshot.resize(llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            if (llama_state_seq_get_data_ext(ctx, snapshot.data(), snapshot.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0)
                return false;
        }
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int32_t i = 0; i < n; ++i) {
            batch.token[i] = i == 0 ? last : draft[static_cast<size_t>(i - 1)];
            batch.pos[i] = base + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq;
            batch.logits[i] = true;
        }
        batch.n_tokens = n;
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            reusable_[seq] = false;
            cached_[seq].clear();
            return false;
        }
        accepted.clear();
        size_t i = 0;
        llama_token t = sample_token(sampler, ctx, 0);
        while (i < draft.size() && t == draft[i]) {
            accepted.push_back(t);
            ++i;
            t = sample_token(sampler, ctx, static_cast<int32_t>(i));
        }
        next = t;
        const llama_pos keep = base + 1 + static_cast<llama_pos>(i); // last + the accepted drafts stay in the cache
        if (i < draft.size() && use_checkpoints) {
            // back to the state before `last`, drop every key from base on, then decode `last` and the accepted drafts again
            bool restored =
                llama_state_seq_set_data_ext(ctx, snapshot.data(), snapshot.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 0 &&
                llama_memory_seq_rm(llama_get_memory(ctx), seq, base, -1);
            if (restored) {
                llama_batch redo = llama_batch_init(static_cast<int32_t>(i) + 1, 0, 1);
                for (int32_t j = 0; j <= static_cast<int32_t>(i); ++j) {
                    redo.token[j] = j == 0 ? last : accepted[static_cast<size_t>(j - 1)];
                    redo.pos[j] = base + j;
                    redo.n_seq_id[j] = 1;
                    redo.seq_id[j][0] = seq;
                    redo.logits[j] = false;
                }
                redo.n_tokens = static_cast<int32_t>(i) + 1;
                restored = llama_decode(ctx, redo) == 0;
                llama_batch_free(redo);
            }
            if (!restored) {
                reusable_[seq] = false;
                cached_[seq].clear();
                return false;
            }
        } else if (i < draft.size() && !llama_memory_seq_rm(llama_get_memory(ctx), seq, keep, -1)) {
            reusable_[seq] = false; // cannot happen for the models this is enabled for; stop trusting this slot
            cached_[seq].clear();
            return false;
        }
        next_pos_[seq] = keep;
        if (reusable_[seq]) {
            cached_[seq].push_back(last);
            cached_[seq].insert(cached_[seq].end(), accepted.begin(), accepted.end());
        }
        return true;
    }

    // The first caller in a round leads it: waits a short debounce window so other active
    // generations can join, then decodes every joined sequence's next token in one call.
    bool next_token(llama_seq_id seq, llama_token last, common_sampler *sampler, llama_token &out) {
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
            if (rc == 0) {
                if (reusable_[s]) cached_[s].push_back(round[i].token); // this token's keys/values are now in the cache
            } else {
                reusable_[s] = false;
                cached_[s].clear();
            }
            RoundResult rr;
            rr.ok = (rc == 0);
            if (rr.ok) rr.token = sample_token(round[i].sampler, ctx, static_cast<int32_t>(i));
            results_[s] = rr;
        }

        batching_ = false;
        cv_.notify_all();

        RoundResult res = results_[seq];
        results_.erase(seq);
        out = res.token;
        return res.ok;
    }

    // AGENT_CAP_DECIDE: prefills the prompt (text tokens, or mtmd chunks when chunks is set) on seq, then
    // adds up the log-probability of each option's tokens following it. Every option is decoded after the
    // prompt and removed again, so one sequence serves all of them. Runs under the batching mutex.
    bool score_options(llama_seq_id seq, const std::vector<llama_token> &tokens, mtmd_context *mctx, const mtmd_input_chunks *chunks,
                       const std::vector<std::vector<llama_token>> &options, std::vector<double> &logprobs) {
        std::lock_guard<std::mutex> lock(mu_);
        if (chunks) {
            llama_pos n_past = next_pos_[seq];
            if (mtmd_helper_eval_chunks(mctx, ctx, chunks, n_past, seq, static_cast<int32_t>(llama_n_batch(ctx)), true, &n_past) != 0)
                return false;
            next_pos_[seq] = n_past;
        } else if (!decode_prompt_locked(seq, tokens)) {
            return false;
        }
        const llama_pos base = next_pos_[seq];
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        // The prompt's logits are overwritten by the first option decode: take every option's first token now.
        const float *last = llama_get_logits_ith(ctx, -1);
        if (!last) return false;
        logprobs.assign(options.size(), 0.0);
        for (size_t o = 0; o < options.size(); ++o) logprobs[o] = token_logprob(last, n_vocab, options[o].front());

        size_t longest = 1;
        for (const auto &opt : options) longest = std::max(longest, opt.size());
        llama_batch batch = llama_batch_init(static_cast<int32_t>(longest), 0, 1);
        bool ok = true;
        // A recurrent / hybrid model cannot drop the tail of a sequence: the state after the prompt is saved and put back instead.
        const bool snapshot = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        std::vector<uint8_t> snap;
        if (snapshot) {
            snap.resize(llama_state_seq_get_size(ctx, seq));
            ok = llama_state_seq_get_data(ctx, snap.data(), snap.size(), seq) == snap.size();
        }
        for (size_t o = 0; ok && o < options.size(); ++o) {
            const std::vector<llama_token> &opt = options[o];
            if (opt.size() < 2) continue;
            for (size_t i = 0; i + 1 < opt.size(); ++i) {
                batch.token[i] = opt[i];
                batch.pos[i] = base + static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = true;
            }
            batch.n_tokens = static_cast<int32_t>(opt.size() - 1);
            ok = llama_decode(ctx, batch) == 0;
            for (size_t i = 0; ok && i + 1 < opt.size(); ++i) {
                const float *logits = llama_get_logits_ith(ctx, static_cast<int32_t>(i));
                ok = logits != nullptr;
                if (ok) logprobs[o] += token_logprob(logits, n_vocab, opt[i + 1]);
            }
            if (snapshot) {
                llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
                ok = ok && llama_state_seq_set_data(ctx, snap.data(), snap.size(), seq) != 0;
            } else {
                llama_memory_seq_rm(llama_get_memory(ctx), seq, base, -1);
            }
        }
        llama_batch_free(batch);
        return ok;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        llama_memory_clear(llama_get_memory(ctx), true);
        for (auto &entry : cached_) entry.second.clear();
        for (auto &entry : reusable_) entry.second = false;
        ckpt_.clear();
    }

    // Turns the prefix cache off for good (model without trimmable memory, option "prefix_cache": false).
    void disable_prefix_reuse() {
        std::lock_guard<std::mutex> lock(mu_);
        reuse_ok_ = false;
    }

  private:
    // Decodes tokens on seq after its current position in chunks of at most n_batch, with logits for
    // the last token only. mu_ must be held.
    static constexpr size_t kCheckpointMargin = 8; // tokens decoded after the checkpoint (see prefill)

    bool decode_prompt_locked(llama_seq_id seq, const std::vector<llama_token> &tokens, size_t first = 0, size_t last = SIZE_MAX) {
        const size_t end = std::min(last, tokens.size());
        if (first >= end) return false;
        const size_t chunk = std::max<size_t>(1, llama_n_batch(ctx));
        llama_batch batch = llama_batch_init(static_cast<int32_t>(std::min(chunk, end - first)), 0, 1);
        llama_pos pos = next_pos_[seq];
        bool ok = true;
        for (size_t begin = first; ok && begin < end; begin += chunk) {
            const size_t n = std::min(chunk, end - begin);
            for (size_t i = 0; i < n; ++i) {
                batch.token[i] = tokens[begin + i];
                batch.pos[i] = pos + static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = (begin + i + 1 == end);
            }
            batch.n_tokens = static_cast<int32_t>(n);
            ok = llama_decode(ctx, batch) == 0;
            pos += static_cast<llama_pos>(n);
        }
        llama_batch_free(batch);
        if (ok) {
            next_pos_[seq] = pos;
            if (reusable_[seq])
                cached_[seq].insert(cached_[seq].end(), tokens.begin() + static_cast<std::ptrdiff_t>(first),
                                    tokens.begin() + static_cast<std::ptrdiff_t>(end));
        } else {
            reusable_[seq] = false;
            cached_[seq].clear();
            ckpt_.erase(seq);
        }
        return ok;
    }

    // Saves the recurrent part of the sequence's state (the attention keys stay in the cache). mu_ must be held.
    void save_checkpoint_locked(llama_seq_id seq) {
        const size_t n = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n == 0) {
            ckpt_.erase(seq);
            return;
        }
        Checkpoint &c = ckpt_[seq];
        c.data.resize(n);
        if (llama_state_seq_get_data_ext(ctx, c.data.data(), n, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != n) {
            ckpt_.erase(seq);
            return;
        }
        c.pos = cached_[seq].size();
    }

    // log softmax(logits)[token], computed in double.
    static double token_logprob(const float *logits, int32_t n_vocab, llama_token token) {
        float max = logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) max = std::max(max, logits[i]);
        double sum = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) sum += std::exp(static_cast<double>(logits[i]) - max);
        return static_cast<double>(logits[token]) - max - std::log(sum);
    }

    static constexpr std::chrono::milliseconds kDebounceWindow{2};

    struct RoundReq {
        llama_seq_id seq;
        llama_token token;
        common_sampler *sampler;
    };
    struct RoundResult {
        llama_token token = -1;
        bool ok = false;
    };

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<llama_seq_id> free_seq_ids_;
    std::unordered_map<llama_seq_id, llama_pos> next_pos_;
    std::unordered_map<llama_seq_id, std::vector<llama_token>> cached_; // tokens whose KV is in each slot
    std::unordered_map<llama_seq_id, bool> reusable_;                   // cached_ matches the slot's KV cache
    struct Checkpoint {
        size_t pos = 0; // tokens covered: the state is the recurrent state after tokens[0, pos)
        std::vector<uint8_t> data;
    };
    std::unordered_map<llama_seq_id, Checkpoint> ckpt_;
    bool reuse_ok_ = true;
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
    std::vector<upstream::ToolCall> up_calls; // storage behind `calls` when llama.cpp's engine answered
    agent_part part{};
    // AGENT_CAP_EMBED / AGENT_CAP_RERANK: tensor parts pointing into floats.
    std::vector<float> floats;
    int64_t shape = 0;
    std::vector<agent_part> parts;
};

// What the context was created for (create() option "mode"). Embedding and reranking models need a
// context in embedding mode with pooling, which can't generate, so one instance does one of the three.
enum class Mode { Generate, Embed, Rerank, Tools };

struct Instance {
    std::unique_ptr<upstream::Engine> up; // llama.cpp's own engine, for models whose GGUF has an MTP head
    Batcher batcher;
    Mode mode = Mode::Generate;
    std::mutex pool_mu;     // serializes embed / rerank decodes (they use every sequence slot)
    mtmd::context_ptr mtmd; // projector for image/audio input; null without options.mmproj. Freed before the model.
    common_chat_templates_ptr templates;
    std::string tools_input; // Mode::Tools: the model file the operations work on
    std::string ocr_prompt;  // AGENT_CAP_OCR instruction: options.ocr_prompt or the family's
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

bool trace_grammar() {
    static const bool on = [] {
        const char *v = std::getenv("AGENT_LLAMA_TRACE");
        return v && *v && *v != '0';
    }();
    return on;
}

// Plain greedy / temperature sampling, plus the chat session's tool-call grammar when the request has tools:
// the model's own tool-call format (lazy: it only constrains the output once the call marker has been
// produced), so a call is always well formed.
common_sampler *build_sampler(const llama_model *model, float temperature, const common_chat_session *chat) {
    common_params_sampling params;
    params.temp = temperature;
    if (temperature > 0.0f) { // keep the old behaviour: temperature only, no top-k / top-p / penalties
        params.top_k = 0;
        params.top_p = 1.0f;
        params.min_p = 0.0f;
        params.typ_p = 1.0f;
        params.penalty_repeat = 1.0f;
        params.penalty_freq = 0.0f;
        params.penalty_present = 0.0f;
        params.seed = LLAMA_DEFAULT_SEED;
    } else {
        params.top_k = 1;
    }
    if (chat) chat->apply_sampling(params);
    if (trace_grammar())
        std::fprintf(stderr, "llama: tool-call grammar %s (%zu bytes, lazy=%d, %zu triggers)\n",
                     params.grammar.grammar.empty() ? "off" : "on", params.grammar.grammar.size(), params.grammar_lazy ? 1 : 0,
                     params.grammar_triggers.size());
    common_sampler *s = common_sampler_init(model, params);
    if (!s && !params.grammar.grammar.empty()) { // a grammar this build cannot use: sample without it
        params.grammar = {};
        params.grammar_lazy = false;
        params.grammar_triggers.clear();
        s = common_sampler_init(model, params);
    }
    return s;
}

template <typename T> T json_number(simdjson::ondemand::object &obj, const char *key, T fallback) {
    double d = 0;
    if (obj[key].get_double().get(d)) return fallback;
    return static_cast<T>(d);
}

bool is_media(const agent_part &p) { return p.kind == AGENT_PART_FILE || p.kind == AGENT_PART_BYTES; }

// Thinking is switched off, but several templates (Qwen3 style, LFM, TwIL) still make the model open and close an empty
// reasoning block before the answer. Drops those leading empty blocks from the final text.
void strip_empty_thinking(std::string &text, const common_chat_session &session) {
    std::string start = session.thinking_start_tag();
    std::vector<std::string> ends = session.thinking_end_tags();
    if (start.empty()) start = "<think>";
    if (ends.empty()) ends.push_back("</think>");
    auto skip_space = [&](size_t i) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        return i;
    };
    for (int round = 0; round < 4; ++round) {
        const size_t at = skip_space(0);
        if (text.compare(at, start.size(), start) != 0) return;
        const size_t after_start = skip_space(at + start.size());
        bool removed = false;
        for (const std::string &end : ends) {
            if (end.empty() || text.compare(after_start, end.size(), end) != 0) continue;
            text.erase(0, skip_space(after_start + end.size()));
            removed = true;
            break;
        }
        if (!removed) return; // a block with content in it is real reasoning: leave it alone
    }
}

// A tool call's arguments must be a JSON object. A call cut off by max_tokens, or written wrongly by a small model,
// is not: it must neither run nor be replayed into the next prompt, where the chat template would fail on it.
bool is_json_object(std::string_view text) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::object obj;
    return !parser.parse(text).get(doc) && !doc.get(obj);
}

std::string json_quote(std::string_view s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof(b), "\\u%04x", c);
            out += b;
        } else out += static_cast<char>(c);
    }
    return out + "\"";
}

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
    in.parallel_tool_calls = true; // a model may call several tools in one reply (templates that cannot simply ignore it)
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
        for (size_t c = 0; c < m.n_tool_calls; ++c) {
            std::string arguments(sv(m.tool_calls[c].arguments));
            if (!is_json_object(arguments)) arguments = "{\"unparsed\":" + json_quote(arguments) + "}"; // history from an older turn
            msg.tool_calls.push_back(
                common_chat_tool_call{std::string(sv(m.tool_calls[c].name)), std::move(arguments), std::string(sv(m.tool_calls[c].id))});
        }
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
    } catch (const std::exception &e) { error = std::string("llama: chat template failed: ") + e.what(); } catch (...) {
        error = "llama: chat template failed";
    }
    return false;
}

bool finish_session(common_chat_session &session, const common_chat_input &generated, common_chat_msg &out, std::string &error) {
    try {
        out = session.finish(generated);
        return true;
    } catch (const std::exception &e) { error = std::string("llama: could not parse model output: ") + e.what(); } catch (...) {
        error = "llama: could not parse model output";
    }
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
        if (end == std::string_view::npos) continue; // unbalanced here ("{{...}" copied from the template): retry from the next '{'
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
    agent_response_reset(resp);
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

// Embedding with media (image / audio next to text, for projector-equipped embedding models such as Qwen3-VL-Embedding): one
// embedding per message, made of the message's parts in order. The projector (options.mmproj) turns media into embeddings
// that go through the model like prompt tokens; the vector is the model's pooled output after the last of them. Whether the
// model can make sense of the media is for the model to say: nothing here rejects it ahead of time.
int32_t run_embed_media(Instance &inst, const agent_request *req, agent_response *resp) {
    if (!inst.mtmd)
        return inst.fail(AGENT_ERR_UNSUPPORTED,
                         "llama: the request has images or audio, but this instance has no projector (set options.mmproj)");
    llama_context *ctx = inst.batcher.ctx;
    const auto n_embd = static_cast<size_t>(llama_model_n_embd_out(inst.batcher.model));
    std::vector<float> floats;
    int32_t n_prompt = 0;
    size_t rows = 0;
    for (size_t mi = 0; mi < req->n_messages; ++mi) {
        const agent_message &m = req->messages[mi];
        if (m.n_parts == 0) continue;
        agent_request one = *req;
        one.messages = &m;
        one.n_messages = 1;
        std::string error;
        std::vector<mtmd::bitmap> bitmaps;
        if (int32_t rc = load_media(inst, &one, bitmaps, error); rc != AGENT_OK) return inst.fail(rc, error);

        std::vector<mtmd_input_text> texts;
        std::vector<std::pair<bool, size_t>> order;
        texts.reserve(m.n_parts);
        size_t n_bitmaps = 0;
        for (size_t p = 0; p < m.n_parts; ++p) {
            if (m.parts[p].kind == AGENT_PART_TEXT) {
                const std::string_view t = sv(m.parts[p].text);
                if (t.empty()) continue;
                texts.push_back(mtmd_input_text{t.data(), t.size(), false, false});
                order.emplace_back(false, texts.size() - 1);
            } else if (is_media(m.parts[p])) {
                order.emplace_back(true, n_bitmaps++);
            }
        }
        if (order.empty()) continue;
        std::vector<mtmd_input_part> parts(order.size());
        std::vector<const mtmd_input_part *> ptrs(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i].first) parts[i] = mtmd_input_part{nullptr, bitmaps[order[i].second].ptr.get()};
            else parts[i] = mtmd_input_part{&texts[order[i].second], nullptr};
            ptrs[i] = &parts[i];
        }
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        if (mtmd_tokenize_from_parts(inst.mtmd.get(), chunks.ptr.get(), ptrs.data(), ptrs.size(), true) != 0)
            return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: media tokenization failed");
        n_prompt += static_cast<int32_t>(mtmd_helper_get_n_tokens(chunks.ptr.get()));

        std::lock_guard<std::mutex> lock(inst.pool_mu);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos n_past = 0;
        // The pooled vector comes from the last decode only. A message that is nothing but media is evaluated chunk by chunk without
        // mtmd's marker tokens around it, so the pooling covers the media's own tokens (a mean-pooling model such as EmbeddingGemma 2
        // would otherwise return the vector of the closing marker); a message with text keeps the whole sequence in order, which is
        // right for last-token pooling (Qwen3-VL-Embedding).
        bool media_only = true;
        for (size_t p = 0; p < m.n_parts; ++p)
            if (m.parts[p].kind == AGENT_PART_TEXT && !sv(m.parts[p].text).empty()) media_only = false;
        int32_t rc = 0;
        if (media_only) {
            for (size_t ci = 0; ci < mtmd_input_chunks_size(chunks.ptr.get()) && rc == 0; ++ci) {
                const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks.ptr.get(), ci);
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
                rc = mtmd_helper_eval_chunk_single(inst.mtmd.get(), ctx, chunk, n_past, 0, static_cast<int32_t>(llama_n_batch(ctx)), true,
                                                   &n_past);
            }
        } else {
            rc = mtmd_helper_eval_chunks(inst.mtmd.get(), ctx, chunks.ptr.get(), 0, 0, static_cast<int32_t>(llama_n_batch(ctx)), true,
                                         &n_past);
        }
        if (rc != 0)
            return inst.fail(AGENT_ERR_DECODE_FAILED,
                             "llama: the model could not process the media (does this embedding model take images or audio?)");
        const float *e = llama_get_embeddings_seq(ctx, 0);
        if (!e) e = llama_get_embeddings_ith(ctx, -1);
        if (!e) return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: the context produced no pooled output for the media");
        floats.insert(floats.end(), e, e + n_embd);
        llama_memory_clear(llama_get_memory(ctx), true);
        ++rows;
    }
    if (rows == 0) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: nothing to embed");
    for (size_t r = 0; r < rows; ++r) {
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
    tensor_parts(out, "embedding", rows, resp);
    resp->usage = agent_usage{n_prompt, 0};
    return AGENT_OK;
}

bool request_has_media(const agent_request *req) {
    for (size_t i = 0; i < req->n_messages; ++i)
        for (size_t p = 0; p < req->messages[i].n_parts; ++p)
            if (is_media(req->messages[i].parts[p])) return true;
    return false;
}

// AGENT_CAP_EMBED: one pooled, L2-normalized "embedding" tensor (f32 [n_embd]) per TEXT part, in order; with media in the
// request, one per message (see run_embed_media).
int32_t run_embed(Instance &inst, const agent_request *req, agent_response *resp) {
    if (request_has_media(req)) return run_embed_media(inst, req, resp);
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

// ------------------------------------------------------------------ decisions (choice / score / yes-no)

void append_json_string(std::string &out, std::string_view s) {
    out += '"';
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    out += '"';
}

void append_json_number(std::string &out, double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", std::isfinite(v) ? v : -1e30);
    out += buf;
}

// Native decision models (llama-server's /v1/systemone, "<arch>.decision.type" in the GGUF) read their
// answers from model-specific heads and prompt layouts, which generic option scoring doesn't reproduce.
std::string native_decision_type(const llama_model *model) {
    char arch[64], type[64];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) <= 0) return {};
    const std::string key = std::string(arch) + ".decision.type";
    if (llama_model_meta_val_str(model, key.c_str(), type, sizeof(type)) <= 0) return {};
    return type;
}

enum class DecideMode { Choice, Score, YesNo };

struct DecideParams {
    DecideMode mode = DecideMode::Choice;
    std::vector<std::string> options;
    int scale_min = 1, scale_max = 5;
    bool instructions = true; // append the answer-format instruction to the last user message
    bool end = true;          // score "<option><end of turn>", so "1" doesn't also count "10"
};

int32_t parse_decide_params(const agent_request *req, DecideParams &p, std::string &error) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(sv(req->params_json));
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!req->params_json.size || parser.iterate(json).get(doc) || doc.get_object().get(obj)) {
        error = "llama: DECIDE needs params_json, a JSON object with \"mode\" and \"options\" or \"scale\"";
        return AGENT_ERR_INVALID_CONFIG;
    }
    for (auto field : obj) {
        std::string_view key, s;
        if (field.unescaped_key().get(key)) continue;
        auto value = field.value();
        if (key == "mode") {
            if (value.get_string().get(s)) s = {};
            if (s == "choice") p.mode = DecideMode::Choice;
            else if (s == "score") p.mode = DecideMode::Score;
            else if (s == "yesno") p.mode = DecideMode::YesNo;
            else {
                error = "llama: params_json.mode must be \"choice\", \"score\" or \"yesno\"";
                return AGENT_ERR_INVALID_CONFIG;
            }
        } else if (key == "options") {
            simdjson::ondemand::array arr;
            if (value.get_array().get(arr)) {
                error = "llama: params_json.options must be an array of strings";
                return AGENT_ERR_INVALID_CONFIG;
            }
            for (auto el : arr) {
                if (el.get_string().get(s) || s.empty()) {
                    error = "llama: params_json.options must be non-empty strings";
                    return AGENT_ERR_INVALID_CONFIG;
                }
                p.options.emplace_back(s);
            }
        } else if (key == "scale") {
            simdjson::ondemand::array arr;
            int64_t v[2] = {0, 0};
            size_t n = 0;
            if (!value.get_array().get(arr))
                for (auto el : arr) {
                    int64_t x = 0;
                    if (n < 2 && !el.get_int64().get(x)) v[n] = x;
                    ++n;
                }
            if (n != 2 || v[0] >= v[1] || v[0] < -1000 || v[1] > 1000 || v[1] - v[0] > 100) {
                error = "llama: params_json.scale must be [min, max] integers with min < max and at most 101 levels";
                return AGENT_ERR_INVALID_CONFIG;
            }
            p.scale_min = static_cast<int>(v[0]);
            p.scale_max = static_cast<int>(v[1]);
        } else if (key == "instructions") {
            bool b = true;
            if (!value.get_bool().get(b)) p.instructions = b;
        } else if (key == "end") {
            bool b = true;
            if (!value.get_bool().get(b)) p.end = b;
        }
    }
    if (p.mode == DecideMode::YesNo) {
        p.options = {"Yes", "No"};
    } else if (p.mode == DecideMode::Score) {
        p.options.clear();
        for (int v = p.scale_min; v <= p.scale_max; ++v) p.options.push_back(std::to_string(v));
    } else if (p.options.size() < 2) {
        error = "llama: DECIDE choice needs at least two params_json.options";
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (p.options.size() > 256) {
        error = "llama: DECIDE takes at most 256 options";
        return AGENT_ERR_INVALID_CONFIG;
    }
    return AGENT_OK;
}

std::string decide_instruction(const DecideParams &p) {
    switch (p.mode) {
    case DecideMode::YesNo: return "\n\nAnswer with Yes or No.";
    case DecideMode::Score:
        return "\n\nAnswer with a single whole number from " + std::to_string(p.scale_min) + " to " + std::to_string(p.scale_max) + ".";
    case DecideMode::Choice: break;
    }
    std::string s = "\n\nOptions:\n";
    for (const std::string &o : p.options) s += "- " + o + "\n";
    s += "\nAnswer with exactly one of the options, word for word.";
    return s;
}

// The rendered prompt of a non-chat task (DECIDE, OCR): text tokens, or mtmd chunks when the request
// has media. in must come from build_inputs(); its tools are dropped.
struct TaskPrompt {
    std::vector<mtmd::bitmap> bitmaps;
    std::vector<llama_token> tokens;
    mtmd::input_chunks chunks;
    int n_tokens = 0;
    bool has_media() const { return chunks.ptr != nullptr; }
};

int32_t render_task_prompt(Instance &inst, const agent_request *req, common_chat_templates_inputs in, TaskPrompt &out,
                           common_chat_session &session, std::string &error) {
    bool has_media = false;
    for (size_t i = 0; i < req->n_messages && !has_media; ++i)
        for (size_t k = 0; k < req->messages[i].n_parts && !has_media; ++k) has_media = is_media(req->messages[i].parts[k]);
    if (has_media && !inst.mtmd) {
        error = "llama: image/audio input needs a multimodal projector (engine option \"mmproj\")";
        return AGENT_ERR_UNSUPPORTED;
    }
    if (has_media)
        if (int32_t rc = load_media(inst, req, out.bitmaps, error); rc != AGENT_OK) return rc;
    in.tools.clear();
    const llama_vocab *vocab = llama_model_get_vocab(inst.batcher.model);
    if (!start_session(inst, vocab, in, session, error)) return AGENT_ERR_INVALID_CONFIG;
    if (has_media) {
        out.chunks.ptr.reset(mtmd_input_chunks_init());
        if (!tokenize_media_prompt(inst, session.prompt(), out.bitmaps, out.chunks.ptr.get(), error)) return AGENT_ERR_DECODE_FAILED;
        out.n_tokens = static_cast<int>(mtmd_helper_get_n_tokens(out.chunks.ptr.get()));
    } else {
        if (!tokenize_prompt(vocab, session.prompt(), out.tokens)) {
            error = "llama: tokenization failed";
            return AGENT_ERR_DECODE_FAILED;
        }
        out.n_tokens = static_cast<int>(out.tokens.size());
    }
    return AGENT_OK;
}

common_chat_msg *last_user(common_chat_templates_inputs &in) {
    for (auto it = in.messages.rbegin(); it != in.messages.rend(); ++it)
        if (it->role == "user") return &*it;
    return nullptr;
}

// Shapes the answer of a DECIDE request from the options' log-probabilities (softmax-normalized here).
int32_t decide_finish(Instance &inst, const DecideParams &p, const std::vector<double> &logprobs, int n_prompt, int completion,
                      agent_response *resp) {
    // Softmax over the options' sequence log-probabilities.
    const double max = *std::max_element(logprobs.begin(), logprobs.end());
    std::vector<double> probs(logprobs.size());
    double sum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) sum += probs[i] = std::exp(logprobs[i] - max);
    for (double &v : probs) v /= sum;
    const auto best = static_cast<size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());

    CallOutput &out = inst.output();
    out = CallOutput{};
    static constexpr const char *kDecideModeName[] = {"choice", "score", "yesno"};
    std::string &j = out.content;
    j = "{\"mode\":\"";
    j += kDecideModeName[static_cast<size_t>(p.mode)];
    j += "\",\"choice\":";
    append_json_string(j, p.options[best]);
    j += ",\"index\":" + std::to_string(best) + ",\"confidence\":";
    append_json_number(j, probs[best]);
    if (p.mode == DecideMode::Score) {
        double expected = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) expected += probs[i] * (p.scale_min + static_cast<double>(i));
        j += ",\"score\":";
        append_json_number(j, expected);
        j += ",\"scale\":[" + std::to_string(p.scale_min) + "," + std::to_string(p.scale_max) + "]";
    } else if (p.mode == DecideMode::YesNo) {
        j += ",\"yes\":";
        append_json_number(j, probs[0]);
    }
    auto append_array = [&j](const char *key, auto &&values, auto &&append) {
        j += ",\"";
        j += key;
        j += "\":[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) j += ',';
            append(j, values[i]);
        }
        j += ']';
    };
    append_array("options", p.options, [](std::string &o, const std::string &v) { append_json_string(o, v); });
    append_array("probs", probs, [](std::string &o, double v) { append_json_number(o, v); });
    append_array("logprobs", logprobs, [](std::string &o, double v) { append_json_number(o, v); });
    j += '}';

    out.floats.assign(probs.begin(), probs.end());
    out.shape = static_cast<int64_t>(probs.size());
    tensor_parts(out, "probs", 1, resp);
    agent_part text{};
    text.kind = AGENT_PART_TEXT;
    text.text = agent_str{out.content.data(), out.content.size()};
    out.parts.insert(out.parts.begin(), text);
    resp->message.parts = out.parts.data();
    resp->message.n_parts = out.parts.size();
    resp->usage = agent_usage{n_prompt, completion};
    return AGENT_OK;
}

// DECIDE on a native decision model (System One): the question goes to llama.cpp's /v1/systemone handler in process. The last
// user message is the state; the options become a choice question (yes/no too) or a score question.
int32_t run_decide_native(Instance &inst, const agent_request *req, agent_response *resp) {
    DecideParams p;
    std::string error;
    if (int32_t rc = parse_decide_params(req, p, error); rc != AGENT_OK) return inst.fail(rc, error);
    std::string state;
    for (size_t i = req->n_messages; i-- > 0 && state.empty();)
        if (sv(req->messages[i].role) == "user")
            for (size_t k = 0; k < req->messages[i].n_parts; ++k)
                if (req->messages[i].parts[k].kind == AGENT_PART_TEXT) state += std::string(sv(req->messages[i].parts[k].text));
    if (state.empty()) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: DECIDE needs a user message with the question");
    // params.instructions is the question the model answers about the state (the System One "instructions").
    std::string instructions = p.mode == DecideMode::Score ? "Rate the state." : "Which option best describes the state?";
    {
        simdjson::ondemand::parser ip;
        simdjson::padded_string ij(sv(req->params_json));
        simdjson::ondemand::document idoc;
        std::string_view text;
        if (!ip.iterate(ij).get(idoc) && !idoc["instructions"].get_string().get(text) && !text.empty()) instructions = std::string(text);
    }
    std::string ins_json;
    append_json_string(ins_json, instructions);
    std::string q = "{\"state\":";
    append_json_string(q, state);
    q += ",\"questions\":{\"q\":{";
    if (p.mode == DecideMode::Score) {
        q += "\"type\":\"score\",\"instructions\":" + ins_json + ",\"criteria\":[";
        for (size_t i = 0; i < p.options.size(); ++i) {
            if (i) q += ',';
            append_json_string(q, p.options[i]);
        }
        q += "]";
    } else {
        q += "\"type\":\"choice\",\"instructions\":" + ins_json + ",\"criteria\":{";
        for (size_t i = 0; i < p.options.size(); ++i) {
            if (i) q += ',';
            append_json_string(q, p.options[i]);
            q += ":null";
        }
        q += "}";
    }
    q += "}}}";
    std::string out;
    if (!inst.up->systemone(q, out, error)) return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: " + error);
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(out);
    simdjson::ondemand::document doc;
    std::vector<double> logprobs;
    int n_prompt = 0;
    if (!parser.iterate(json).get(doc)) {
        simdjson::ondemand::object probs;
        if (!doc["answers"]["q"]["probabilities"].get_object().get(probs)) {
            std::map<std::string, double> by_key;
            for (auto field : probs) {
                std::string_view key;
                double v = 0;
                if (!field.unescaped_key().get(key) && !field.value().get_double().get(v)) by_key[std::string(key)] = v;
            }
            for (size_t i = 0; i < p.options.size(); ++i) {
                const std::string key = p.mode == DecideMode::Score ? std::to_string(i) : p.options[i];
                auto it = by_key.find(key);
                logprobs.push_back(std::log(std::max(it == by_key.end() ? 0.0 : it->second, 1e-12)));
            }
        }
        int64_t t = 0;
        if (!doc["usage"]["input_tokens"].get_int64().get(t)) n_prompt = static_cast<int>(t);
    }
    if (logprobs.size() != p.options.size())
        return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: the decision model's answer has no probabilities: " + out.substr(0, 200));
    return decide_finish(inst, p, logprobs, n_prompt, 0, resp);
}

// AGENT_CAP_DECIDE: the request's messages hold the question (and optional images, with an mmproj);
// params_json picks the answers: {"mode":"choice","options":[...]}, {"mode":"score","scale":[1,5]} or
// {"mode":"yesno"}. Each answer is scored as the reply to the rendered chat by the sum of its tokens'
// log-probabilities (plus the end-of-turn token), and the scores are normalized with a softmax.
// Returns a TEXT part with a JSON summary and a "probs" tensor (f32 [n_options]) in option order.
int32_t run_decide(Instance &inst, const agent_request *req, agent_response *resp) {
    Batcher &b = inst.batcher;
    if (const std::string native = native_decision_type(b.model); !native.empty())
        return inst.fail(AGENT_ERR_UNSUPPORTED, "llama: this is a native decision model (" + native +
                                                    ", System One format); serve it with llama-server's /v1/systemone. DECIDE here "
                                                    "scores options with an instruct model's LM head");
    DecideParams p;
    std::string error;
    if (int32_t rc = parse_decide_params(req, p, error); rc != AGENT_OK) return inst.fail(rc, error);

    common_chat_templates_inputs in = build_inputs(req, inst.mtmd ? mtmd_default_marker() : nullptr);
    if (p.instructions) {
        common_chat_msg *user = last_user(in);
        if (!user) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: DECIDE needs a user message with the question");
        user->content += decide_instruction(p);
    }
    TaskPrompt prompt;
    common_chat_session session;
    if (int32_t rc = render_task_prompt(inst, req, std::move(in), prompt, session, error); rc != AGENT_OK) return inst.fail(rc, error);

    const llama_vocab *vocab = llama_model_get_vocab(b.model);
    llama_token end = llama_vocab_eot(vocab);
    if (end == LLAMA_TOKEN_NULL) end = llama_vocab_eos(vocab);
    std::vector<std::vector<llama_token>> option_tokens;
    size_t longest = 0;
    for (const std::string &o : p.options) {
        std::vector<llama_token> t = tokenize_text(vocab, o, false, false);
        if (t.empty()) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: option \"" + o + "\" has no tokens");
        if (p.end && end != LLAMA_TOKEN_NULL) t.push_back(end);
        longest = std::max(longest, t.size());
        option_tokens.push_back(std::move(t));
    }
    if (longest > llama_n_batch(b.ctx)) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: a DECIDE option is longer than n_batch tokens");
    if (prompt.n_tokens + static_cast<int>(longest) >= b.seq_ctx)
        return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: prompt is " + std::to_string(prompt.n_tokens) + " tokens (+" +
                                                       std::to_string(longest) + " for the longest option) but context_window is " +
                                                       std::to_string(b.seq_ctx));
    if (req->cancel && *req->cancel) return inst.fail(AGENT_ERR_CANCELLED, "llama: cancelled");

    llama_seq_id seq = 0;
    if (!b.acquire_seq(seq)) return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: no free sequence slot (raise max_parallel_subagents)");
    std::vector<double> logprobs;
    const bool scored = b.score_options(seq, prompt.tokens, inst.mtmd.get(), prompt.chunks.ptr.get(), option_tokens, logprobs);
    b.release_seq(seq);
    if (!scored) return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: decode failed while scoring the options");

    int completion = 0;
    for (const auto &t : option_tokens) completion += static_cast<int>(t.size());
    return decide_finish(inst, p, logprobs, prompt.n_tokens, completion, resp);
}

// AGENT_LLAMA_TRACE=1 prints how many prompt tokens each request found already in the KV cache.
bool trace_prefix_cache() {
    static const bool on = [] {
        const char *v = std::getenv("AGENT_LLAMA_TRACE");
        return v && *v && *v != '0';
    }();
    return on;
}

// Prefills the prompt (text tokens, or mtmd chunks when chunks is set) on a free sequence and samples
// up to max_tokens tokens into generated, streaming them through on_chunk, until an end-of-generation
// token or one of stops. Generation steps join the batcher's shared rounds.
int32_t generate(Instance &inst, const agent_request *req, const std::vector<llama_token> &tokens, const mtmd_input_chunks *chunks,
                 int max_tokens, float temperature, const std::vector<std::string> &stops, common_chat_input &generated, int &completion,
                 agent_run_stats *stats = nullptr, const common_chat_session *chat = nullptr) {
    Batcher &b = inst.batcher;
    const llama_vocab *vocab = llama_model_get_vocab(b.model);
    llama_seq_id seq = 0;
    size_t reused = 0;
    const bool acquired = chunks ? b.acquire_seq(seq) : b.acquire_seq_for(tokens, seq, reused);
    if (!acquired) return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: no free sequence slot (raise max_parallel_subagents)");
    if (trace_prefix_cache())
        std::fprintf(stderr, "llama: prefix cache reused %zu of %zu prompt tokens (slot %d%s)\n", reused, tokens.size(),
                     static_cast<int>(seq), b.use_checkpoints ? ", checkpoint" : "");
    common_sampler *sampler = build_sampler(b.model, temperature, chat);
    if (!sampler) {
        b.release_seq(seq, false);
        return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: could not create the sampler");
    }
    int32_t status = AGENT_OK;
    llama_token tok = 0;
    // Prompt-lookup speculation: greedy, no tools (no grammar), text only, and a model whose cache can be cut back.
    const bool can_speculate = b.spec_enabled && temperature <= 0.0f && !chunks && !req->tools_json.size;
    std::vector<llama_token> history;
    if (can_speculate) history = tokens;
    std::deque<llama_token> ahead; // tokens already sampled (accepted drafts and the one after them), to be processed in order
    uint64_t spec_rounds = 0, spec_accepted = 0;
    const auto t_prefill = std::chrono::steady_clock::now();
    const bool prefilled =
        chunks ? b.prefill_media(seq, inst.mtmd.get(), chunks, sampler, tok) : b.prefill(seq, tokens, reused, sampler, tok);
    const auto t_first_token = std::chrono::steady_clock::now(); // the prompt is decoded and the first token sampled
    if (!prefilled) {
        status = inst.fail(AGENT_ERR_DECODE_FAILED, "llama: prompt decode failed");
    } else {
        char piece[256];
        for (int i = 0; i < max_tokens; ++i) {
            if (llama_vocab_is_eog(vocab, tok)) break;
            if (req->cancel && *req->cancel) {
                status = inst.fail(AGENT_ERR_CANCELLED, "llama: cancelled");
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
            if (can_speculate) history.push_back(tok);
            if (!ahead.empty()) { // sampled earlier by a speculative round: no decode needed
                tok = ahead.front();
                ahead.pop_front();
                continue;
            }
            if (can_speculate) {
                const size_t room = static_cast<size_t>(max_tokens - i - 1);
                std::vector<llama_token> draft = ngram_draft(history, 3, std::min<size_t>(8, room));
                if (!draft.empty()) {
                    std::vector<llama_token> accepted;
                    llama_token next = 0;
                    if (!b.speculate(seq, tok, draft, sampler, accepted, next)) {
                        status = inst.fail(AGENT_ERR_DECODE_FAILED, "llama: decode failed");
                        break;
                    }
                    ++spec_rounds;
                    spec_accepted += accepted.size();
                    for (llama_token t : accepted) ahead.push_back(t);
                    ahead.push_back(next);
                    tok = ahead.front();
                    ahead.pop_front();
                    continue;
                }
            }
            if (!b.next_token(seq, tok, sampler, tok)) {
                status = inst.fail(AGENT_ERR_DECODE_FAILED, "llama: decode failed");
                break;
            }
        }
    }
    if (spec_rounds && trace_prefix_cache())
        std::fprintf(stderr, "llama: speculation %llu rounds, %llu draft tokens accepted\n", static_cast<unsigned long long>(spec_rounds),
                     static_cast<unsigned long long>(spec_accepted));
    if (stats) {
        const auto t_done = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b2) { return static_cast<float>(std::chrono::duration<double, std::milli>(b2 - a).count()); };
        stats->cached_tokens = static_cast<int32_t>(reused);
        stats->kv_tokens = b.position(seq);
        stats->kv_max_tokens = b.seq_ctx;
        stats->kv_type = b.kv_type_id;
        stats->kv_bytes = b.kv_bytes;
        stats->prefill_ms = ms(t_prefill, t_first_token);
        stats->decode_ms = ms(t_first_token, t_done);
    }
    common_sampler_free(sampler);
    b.release_seq(seq, status == AGENT_OK);
    return status;
}

// ------------------------------------------------------------------ OCR

// OCR-tuned VLMs are trained on one fixed instruction each (see vendor/llama.cpp/docs/multimodal.md and
// the model cards). A family matches when all its needles occur in the model's identity: its
// general.architecture / general.name / general.basename and the model and mmproj file names,
// lowercased. Engine option "ocr_prompt" or params_json "prompt" override the prompt.
struct OcrFamily {
    const char *name;
    const char *needles[2];
    const char *prompt;
};
// clang-format off
constexpr OcrFamily kOcrFamilies[] = {
    {"glm-ocr", {"glm", "ocr"}, "Text Recognition:"},
};
// clang-format on
constexpr const char *kGenericOcrPrompt =
    "Transcribe all the text in this image exactly as written, line by line. Output only the text, nothing else.";

std::string lower_ascii(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// The OCR prompt of the model's family, or the generic one.
const char *ocr_family_prompt(const llama_model *model, const std::string &model_path, const std::string &mmproj) {
    std::string id;
    char buf[256];
    for (const char *key : {"general.architecture", "general.name", "general.basename"})
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) (id += buf) += ' ';
    auto file_name = [](const std::string &p) { return p.substr(p.find_last_of("/\\") + 1); };
    id += file_name(model_path) + ' ' + file_name(mmproj);
    id = lower_ascii(std::move(id));
    for (const OcrFamily &f : kOcrFamilies) {
        bool all = true;
        for (const char *n : f.needles) all = all && (!n || id.find(n) != std::string::npos);
        if (all) return f.prompt;
    }
    return kGenericOcrPrompt;
}

// AGENT_CAP_OCR: the request's image parts (FILE / BYTES, any message) are transcribed with the
// family's OCR prompt; text parts in the request are ignored. Generation is greedy and stops at the
// end of turn or max_tokens. Returns one TEXT part with the extracted text.
int32_t run_ocr(Instance &inst, const agent_request *req, agent_response *resp) {
    if (!inst.mtmd || !mtmd_support_vision(inst.mtmd.get()))
        return inst.fail(AGENT_ERR_UNSUPPORTED, "llama: OCR needs a vision model and its projector (engine option \"mmproj\")");
    size_t n_images = 0;
    for (size_t i = 0; i < req->n_messages; ++i)
        for (size_t k = 0; k < req->messages[i].n_parts; ++k) n_images += is_media(req->messages[i].parts[k]) ? 1 : 0;
    if (n_images == 0) return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: OCR needs an image FILE or BYTES part");

    int max_tokens = 1024;
    std::string prompt = inst.ocr_prompt;
    if (req->params_json.size) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(sv(req->params_json));
        simdjson::ondemand::document doc;
        simdjson::ondemand::object params;
        if (!parser.iterate(json).get(doc) && !doc.get_object().get(params)) {
            max_tokens = json_number<int>(params, "max_tokens", max_tokens);
            std::string_view p;
            if (!params["prompt"].get_string().get(p) && !p.empty()) prompt = std::string(p);
        }
    }

    // One user message: the images first (the order OCR models are trained on), then the prompt.
    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.add_generation_prompt = true;
    in.parallel_tool_calls = true; // a model may call several tools in one reply (templates that cannot simply ignore it)
    in.enable_thinking = false;
    common_chat_msg msg;
    msg.role = "user";
    for (size_t i = 0; i < n_images; ++i) msg.content += mtmd_default_marker();
    msg.content += prompt;
    in.messages.push_back(std::move(msg));

    std::string error;
    TaskPrompt task;
    common_chat_session session;
    if (int32_t rc = render_task_prompt(inst, req, std::move(in), task, session, error); rc != AGENT_OK) return inst.fail(rc, error);
    Batcher &b = inst.batcher;
    if (task.n_tokens >= b.seq_ctx)
        return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: prompt is " + std::to_string(task.n_tokens) +
                                                       " tokens (the image) but context_window is " + std::to_string(b.seq_ctx));
    max_tokens = std::clamp(max_tokens, 1, b.seq_ctx - task.n_tokens);

    common_chat_input generated;
    int completion = 0;
    if (int32_t rc =
            generate(inst, req, task.tokens, task.chunks.ptr.get(), max_tokens, 0.0f, session.additional_stops(), generated, completion);
        rc != AGENT_OK)
        return rc;

    CallOutput &out = inst.output();
    out = CallOutput{};
    if (finish_session(session, generated, out.parsed, error)) out.content = out.parsed.content;
    else out.content = generated.text;
    strip_empty_thinking(out.content, session);
    while (!out.content.empty() && std::isspace(static_cast<unsigned char>(out.content.back()))) out.content.pop_back();
    const size_t lead = out.content.find_first_not_of(" \t\r\n");
    out.content.erase(0, lead == std::string::npos ? out.content.size() : lead);
    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    resp->usage = agent_usage{task.n_tokens, completion};
    out.part = agent_part{};
    out.part.kind = AGENT_PART_TEXT;
    out.part.text = agent_str{out.content.data(), out.content.size()};
    resp->message.parts = &out.part;
    resp->message.n_parts = 1;
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
    bool prefix_cache = true;
    // KV cache precision (llama.cpp's own quantized cache types). "auto" (the default) picks q4_0 when the model's
    // keys are wide and q8_0 when they are narrow: measured on Qwen2.5-0.5B (2 KV heads x 64) a q4_0 cache
    // produced garbage while f16 and q8_0 were fine, and LFM3-1.7B was fine at q4_0.
    ggml_type kv_type = GGML_TYPE_Q4_0;
    bool kv_auto = true;
    std::string chat_template, mmproj, ocr_prompt;
    std::vector<std::pair<std::string, float>> loras; // options.lora: adapter file and scale
    bool use_mtp = true;                              // options.mtp: drive the GGUF's MTP head when it has one
    bool speculative = true;                          // options.speculative: prompt-lookup speculation (greedy requests only)
    int n_batch_opt = 0, n_ubatch_opt = 0;            // options.batch / options.ubatch (0 = llama.cpp default)
    Mode mode = Mode::Generate;
    enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        gpu_layers = json_number<int>(options, "gpu_layers", -1);
        threads = json_number<int>(options, "threads", 0);
        std::string_view t;
        if (!options["chat_template"].get_string().get(t)) chat_template = std::string(t);
        if (!options["mmproj"].get_string().get(t)) mmproj = std::string(t);
        if (!options["ocr_prompt"].get_string().get(t)) ocr_prompt = std::string(t);
        simdjson::ondemand::array lora_list;
        if (!options["lora"].get_array().get(lora_list)) {
            for (auto item : lora_list) {
                simdjson::ondemand::object lo;
                std::string_view lp;
                if (item.get_object().get(lo) || lo["path"].get_string().get(lp) || lp.empty()) {
                    std::snprintf(err, err_cap, "llama: each options.lora entry needs a \"path\"");
                    return AGENT_ERR_INVALID_CONFIG;
                }
                loras.push_back({std::string(lp), json_number<float>(lo, "scale", 1.0f)});
            }
        }
        // KV cache precision, using llama.cpp's own quantized cache types (needs flash attention).
        if (!options["kv_cache"].get_string().get(t)) {
            kv_auto = (t == "auto");
            if (t == "auto") {
            } else if (t == "f16") kv_type = GGML_TYPE_F16;
            else if (t == "q8_0") kv_type = GGML_TYPE_Q8_0;
            else if (t == "q4_0") kv_type = GGML_TYPE_Q4_0;
            else {
                std::snprintf(err, err_cap, "llama: options.kv_cache must be \"auto\", \"f16\", \"q8_0\" or \"q4_0\"");
                return AGENT_ERR_INVALID_CONFIG;
            }
        }
        if (bool flag = true; !options["prefix_cache"].get_bool().get(flag)) prefix_cache = flag;
        if (bool flag = true; !options["mtp"].get_bool().get(flag)) use_mtp = flag;
        if (bool flag = true; !options["speculative"].get_bool().get(flag)) speculative = flag;
        n_batch_opt = json_number<int>(options, "batch", 0);   // tokens decoded per call; smaller = smaller compute buffer
        n_ubatch_opt = json_number<int>(options, "ubatch", 0); // tokens per micro-batch (the compute buffer scales with it)
        if (!options["mode"].get_string().get(t)) {
            if (t == "embed") mode = Mode::Embed;
            else if (t == "rerank") mode = Mode::Rerank;
            else if (t == "tools") mode = Mode::Tools;
            else if (t != "generate") {
                std::snprintf(err, err_cap, "llama: options.mode must be \"generate\", \"embed\", \"rerank\" or \"tools\"");
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
    if (mode == Mode::Tools) { // file operations (quantize): the model is not loaded
        auto *tools = new Instance();
        tools->mode = Mode::Tools;
        tools->tools_input = model_str;
        *out_instance = tools;
        return AGENT_OK;
    }

    // A model whose file carries a multi-token-prediction head runs on llama.cpp's own engine, which drives the head.
    // Nothing names a model: llama.cpp looks for the head's tensors in the file.
    const bool native_decision =
        mode == Mode::Generate && upstream::gguf_is_decision(model_str); // System One model: only llama.cpp's engine runs it
    if (mode == Mode::Generate && ((use_mtp && upstream::gguf_has_mtp(model_str)) || native_decision)) {
        upstream::Settings us;
        us.model = model_str;
        us.mmproj = mmproj;
        us.context = n_ctx;
        us.parallel = n_seq;
        us.threads = threads;
        us.gpu_layers = gpu_layers;
        us.loras = loras;
        if (!kv_auto) us.kv_type = kv_type == GGML_TYPE_F16 ? "f16" : kv_type == GGML_TYPE_Q8_0 ? "q8_0" : "q4_0";
        std::string uerr;
        auto up = upstream::Engine::create(us, uerr);
        if (up) {
            auto *inst = new Instance();
            inst->mode = Mode::Generate;
            inst->up = std::move(up);
            *out_instance = inst;
            return AGENT_OK;
        }
        // llama.cpp's engine could not start: fall back to the plugin's own (without the MTP head).
        std::fprintf(stderr, "llama: MTP engine unavailable (%s); using the built-in engine\n", uerr.c_str());
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
    if (n_batch_opt > 0) cparams.n_batch = static_cast<uint32_t>(n_batch_opt);
    if (n_ubatch_opt > 0) cparams.n_ubatch = static_cast<uint32_t>(std::min(n_ubatch_opt, static_cast<int>(cparams.n_batch)));
    if (kv_auto) {
        // Width of one layer's keys: kv heads x head size. Narrow keys lose too much at 4 bits.
        const int n_head = llama_model_n_head(model), n_head_kv = llama_model_n_head_kv(model);
        const double width = n_head > 0 ? static_cast<double>(n_head_kv) * llama_model_n_embd(model) / n_head : 0.0;
        kv_type = width >= 512.0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
    }
    if (mode != Mode::Generate) kv_type = GGML_TYPE_F16; // pooled outputs keep no useful cache
    if (kv_type != GGML_TYPE_F16) {
        cparams.type_k = kv_type;
        cparams.type_v = kv_type;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // a quantized V cache requires it
    }
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
    if (!ctx && kv_type != GGML_TYPE_F16) {
        // Some models (odd head sizes, no flash-attention support on the backend) cannot use a quantized
        // cache: retry with the default f16 cache instead of failing the load.
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        ctx = llama_init_from_model(model, cparams);
    }
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
    inst->batcher.prefix_cache = prefix_cache;
    {
        // The KV cache type actually in use (the f16 fallback may have replaced a quantized one) and its size:
        // layers x 2 (K and V) x kv heads x head size x tokens x bytes per element. An estimate for hybrid models.
        const ggml_type used = cparams.type_k;
        inst->batcher.kv_type_id = used == GGML_TYPE_Q4_0   ? AGENT_KV_Q4_0
                                   : used == GGML_TYPE_Q8_0 ? AGENT_KV_Q8_0
                                   : used == GGML_TYPE_F32  ? AGENT_KV_F32
                                                            : AGENT_KV_F16;
        const double bytes_per_elem = used == GGML_TYPE_Q4_0   ? 18.0 / 32.0
                                      : used == GGML_TYPE_Q8_0 ? 34.0 / 32.0
                                      : used == GGML_TYPE_F32  ? 4.0
                                                               : 2.0;
        const int n_head = llama_model_n_head(model), n_head_kv = llama_model_n_head_kv(model);
        if (n_head > 0 && n_head_kv > 0)
            inst->batcher.kv_bytes = static_cast<uint64_t>(static_cast<double>(llama_model_n_layer(model)) * 2.0 * n_head_kv *
                                                           (static_cast<double>(llama_model_n_embd(model)) / n_head) *
                                                           static_cast<double>(llama_n_ctx(ctx)) * bytes_per_elem);
    }
    if (!loras.empty()) {
        // Adapters belong to the model and are freed with it. They apply to every sequence of this instance.
        std::vector<llama_adapter_lora *> adapters;
        std::vector<float> scales;
        for (const auto &[path, scale] : loras) {
            llama_adapter_lora *adapter = llama_adapter_lora_init(model, path.c_str());
            if (!adapter) {
                std::snprintf(err, err_cap, "llama: cannot load the LoRA adapter %s (a GGUF adapter made for this base model is needed)",
                              path.c_str());
                delete inst;
                return AGENT_ERR_MODEL_LOAD_FAILED;
            }
            adapters.push_back(adapter);
            scales.push_back(scale);
        }
        if (llama_set_adapters_lora(inst->batcher.ctx, adapters.data(), adapters.size(), scales.data()) != 0) {
            std::snprintf(err, err_cap, "llama: the LoRA adapter does not fit this model");
            delete inst;
            return AGENT_ERR_MODEL_LOAD_FAILED;
        }
    }
    inst->batcher.use_checkpoints = llama_model_is_hybrid(model) || llama_model_is_recurrent(model);
    if (llama_model_n_swa(model) > 0 && !inst->batcher.use_checkpoints) inst->batcher.disable_prefix_reuse();
    inst->batcher.spec_enabled =
        speculative && mode == Mode::Generate && llama_model_n_swa(model) == 0; // hybrid models too (state snapshots)
    // What the file itself says: layers of a multi-token-prediction head. They are read from the GGUF, not assumed from a model name.
    if (trace_prefix_cache())
        std::fprintf(stderr, "llama: speculation %s (n_swa=%d, hybrid=%d, mtp_layers=%d%s)\n", inst->batcher.spec_enabled ? "on" : "off",
                     llama_model_n_swa(model), inst->batcher.use_checkpoints ? 1 : 0, llama_model_n_layer_nextn(model),
                     llama_model_n_layer_nextn(model) > 0 ? ": the file has an MTP head, drafting uses prompt lookup" : "");
    inst->mode = mode;
    if (mode != Mode::Generate) {     // no chat template or sequence slots needed
        inst->mtmd = std::move(mctx); // an embedding model with a projector takes images / audio too
        *out_instance = inst;
        return AGENT_OK;
    }
    inst->batcher.seed_free_list(n_seq);
    inst->mtmd = std::move(mctx);
    const char *family_prompt = ocr_family_prompt(model, model_str, mmproj);
    inst->ocr_prompt = ocr_prompt.empty() ? family_prompt : ocr_prompt;
    // Exception firewall (see start_session): template loading parses Jinja and may throw.
    try {
        inst->templates = common_chat_templates_init(model, chat_template);
    } catch (const std::exception &e) { std::snprintf(err, err_cap, "llama: invalid chat template: %s", e.what()); } catch (...) {
        std::snprintf(err, err_cap, "llama: invalid chat template");
    }
    if (!inst->templates) {
        delete inst;
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    *out_instance = inst;
    return AGENT_OK;
}

void llama_destroy(void *instance) { delete static_cast<Instance *>(instance); }

// ------------------------------------------------------------------ file operations (options.mode "tools")

struct QuantType {
    const char *name;
    llama_ftype type;
};
constexpr QuantType kQuantTypes[] = {
    {"F16", LLAMA_FTYPE_MOSTLY_F16},       {"BF16", LLAMA_FTYPE_MOSTLY_BF16},     {"Q8_0", LLAMA_FTYPE_MOSTLY_Q8_0},
    {"Q6_K", LLAMA_FTYPE_MOSTLY_Q6_K},     {"Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M}, {"Q5_K_S", LLAMA_FTYPE_MOSTLY_Q5_K_S},
    {"Q5_0", LLAMA_FTYPE_MOSTLY_Q5_0},     {"Q5_1", LLAMA_FTYPE_MOSTLY_Q5_1},     {"Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M},
    {"Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S}, {"Q4_0", LLAMA_FTYPE_MOSTLY_Q4_0},     {"Q4_1", LLAMA_FTYPE_MOSTLY_Q4_1},
    {"Q3_K_L", LLAMA_FTYPE_MOSTLY_Q3_K_L}, {"Q3_K_M", LLAMA_FTYPE_MOSTLY_Q3_K_M}, {"Q3_K_S", LLAMA_FTYPE_MOSTLY_Q3_K_S},
    {"Q2_K", LLAMA_FTYPE_MOSTLY_Q2_K},     {"IQ4_NL", LLAMA_FTYPE_MOSTLY_IQ4_NL}, {"IQ4_XS", LLAMA_FTYPE_MOSTLY_IQ4_XS},
};

// AGENT_CAP_GENERIC with params {"op":"quantize","output":"out.gguf","type":"Q4_K_M"}: rewrites the GGUF named by the
// instance (config.model) at a lower precision. Needs no context and loads no weights into memory beyond what the
// quantizer streams. {"threads":N,"allow_requantize":true,"overwrite":true} are optional.
int32_t run_tools(Instance &inst, const agent_request *req, agent_response *resp) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(sv(req->params_json));
    simdjson::ondemand::document doc;
    simdjson::ondemand::object params;
    if (!req->params_json.size || parser.iterate(json).get(doc) || doc.get_object().get(params))
        return inst.fail(AGENT_ERR_INVALID_CONFIG,
                         "llama: tools mode needs params like {\"op\":\"quantize\",\"output\":\"out.gguf\",\"type\":\"Q4_K_M\"}");
    std::string_view op, output, type;
    if (params["op"].get_string().get(op) || op != "quantize")
        return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: tools mode supports op \"quantize\"");
    if (params["output"].get_string().get(output) || output.empty())
        return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: quantize needs params.output (the new GGUF path)");
    if (params["type"].get_string().get(type)) type = "Q4_K_M";

    llama_ftype ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    bool known = false;
    for (const QuantType &q : kQuantTypes)
        if (type == q.name) {
            ftype = q.type;
            known = true;
        }
    if (!known) {
        std::string list;
        for (const QuantType &q : kQuantTypes) list += std::string(list.empty() ? "" : ", ") + q.name;
        return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: unknown quantization type \"" + std::string(type) + "\" (" + list + ")");
    }
    std::error_code ec;
    const std::string out_path(output);
    bool overwrite = false;
    if (params["overwrite"].get_bool().get(overwrite)) overwrite = false;
    if (std::filesystem::equivalent(out_path, inst.tools_input, ec))
        return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: params.output is the input file");
    if (!overwrite && std::filesystem::exists(out_path, ec))
        return inst.fail(AGENT_ERR_INVALID_CONFIG, "llama: " + out_path + " exists (set params.overwrite)");

    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.ftype = ftype;
    qp.nthread = json_number<int>(params, "threads", 0);
    bool requantize = false;
    if (!params["allow_requantize"].get_bool().get(requantize)) qp.allow_requantize = requantize;
    const uint64_t in_bytes = std::filesystem::file_size(inst.tools_input, ec);
    if (llama_model_quantize(inst.tools_input.c_str(), out_path.c_str(), &qp) != 0)
        return inst.fail(AGENT_ERR_DECODE_FAILED, "llama: quantization failed. Possible causes: the output folder is not writable by the "
                                                  "sandboxed engine (use <workspace>/artifacts/), the disk is full, or the input is "
                                                  "already quantized (set params.allow_requantize, or start from an F16/BF16 file)");
    const uint64_t out_bytes = std::filesystem::file_size(out_path, ec);

    CallOutput &out = inst.output();
    out = CallOutput{};
    out.content = "quantized " + inst.tools_input + " to " + std::string(type) + ": " + out_path + " (" + std::to_string(in_bytes) +
                  " -> " + std::to_string(out_bytes) + " bytes)";
    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    out.part = agent_part{};
    out.part.kind = AGENT_PART_TEXT;
    out.part.text = agent_str{out.content.data(), out.content.size()};
    resp->message.parts = &out.part;
    resp->message.n_parts = 1;
    return AGENT_OK;
}

int32_t llama_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    if (inst->mode == Mode::Tools) {
        if (req->task != AGENT_CAP_GENERIC)
            return inst->fail(AGENT_ERR_UNSUPPORTED, "llama: an instance created with mode \"tools\" only runs GENERIC requests");
        return run_tools(*inst, req, resp);
    }
    if (inst->up && req->task == AGENT_CAP_DECIDE) return run_decide_native(*inst, req, resp);
    if (inst->up) {
        if (req->task != AGENT_CAP_TEXT_GEN)
            return inst->fail(AGENT_ERR_UNSUPPORTED, "llama: this model runs on llama.cpp's MTP engine, which does text generation only "
                                                     "(set options.mtp=false for other tasks)");
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
        std::string error;
        const bool stream = req->on_chunk != nullptr;
        const std::string body = upstream::chat_body_from_request(req, max_tokens, temperature, stream, error);
        if (body.empty()) return inst->fail(AGENT_ERR_INVALID_CONFIG, error);
        upstream::Result r;
        auto on_text = [&](const std::string &piece) {
            if (stream) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, piece.data(), piece.size());
        };
        if (!inst->up->chat(body, req->cancel, on_text, r, error)) return inst->fail(AGENT_ERR_DECODE_FAILED, "llama: " + error);
        CallOutput &out = inst->output();
        out = CallOutput{};
        out.content = r.content;
        out.up_calls = std::move(r.calls);
        for (size_t i = 0; i < out.up_calls.size(); ++i) {
            auto &c = out.up_calls[i];
            if (c.id.empty()) c.id = "call_" + std::to_string(i);
            if (c.arguments.empty()) c.arguments = "{}";
            out.calls.push_back(agent_tool_call{agent_str{c.id.data(), c.id.size()}, agent_str{c.name.data(), c.name.size()},
                                                agent_str{c.arguments.data(), c.arguments.size()}});
        }
        agent_response_reset(resp);
        resp->message.role = agent_str{"assistant", 9};
        resp->usage = agent_usage{r.prompt_tokens, r.completion_tokens};
        out.part = agent_part{};
        out.part.kind = AGENT_PART_TEXT;
        out.part.text = agent_str{out.content.data(), out.content.size()};
        resp->message.parts = &out.part;
        resp->message.n_parts = 1;
        resp->message.tool_calls = out.calls.data();
        resp->message.n_tool_calls = out.calls.size();
        return AGENT_OK;
    }
    if (inst->mode == Mode::Embed && req->task == AGENT_CAP_EMBED) return run_embed(*inst, req, resp);
    if (inst->mode == Mode::Rerank && req->task == AGENT_CAP_RERANK) return run_rerank(*inst, req, resp);
    if (inst->mode == Mode::Generate && req->task == AGENT_CAP_DECIDE) return run_decide(*inst, req, resp);
    if (inst->mode == Mode::Generate && req->task == AGENT_CAP_OCR) return run_ocr(*inst, req, resp);
    if (inst->mode != Mode::Generate || req->task != AGENT_CAP_TEXT_GEN) {
        static constexpr const char *kModeTask[] = {"TEXT_GEN / DECIDE / OCR", "EMBED", "RERANK", "GENERIC"};
        static constexpr const char *kModeName[] = {"generate", "embed", "rerank", "tools"};
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
        return inst->fail(AGENT_ERR_INVALID_CONFIG,
                          "llama: prompt is " + std::to_string(n_prompt) + " tokens but context_window is " + std::to_string(b.seq_ctx));
    max_tokens = std::min(max_tokens, b.seq_ctx - n_prompt);

    common_chat_input generated;
    int completion = 0;
    agent_run_stats run_stats{};
    if (int32_t rc = generate(*inst, req, tokens, has_media ? chunks.ptr.get() : nullptr, max_tokens, temperature,
                              session.additional_stops(), generated, completion, &run_stats, &session);
        rc != AGENT_OK)
        return rc;

    CallOutput &out = inst->output();
    out = CallOutput{};
    if (finish_session(session, generated, out.parsed, error)) out.content = out.parsed.content;
    else out.content = generated.text; // unparseable output is returned verbatim rather than failing the turn
    strip_empty_thinking(out.content, session);

    common_chat_tool_call loose;
    if (out.parsed.tool_calls.empty() && req->tools_json.size && find_loose_tool_call(generated.text, loose)) {
        out.parsed.tool_calls.push_back(std::move(loose));
        out.content.clear();
    }
    // Calls whose arguments are not a JSON object (cut off at max_tokens, or mis-written) are not run. The model's own text
    // is returned instead, so the caller sees what happened.
    {
        auto &calls = out.parsed.tool_calls;
        const size_t before = calls.size();
        calls.erase(std::remove_if(calls.begin(), calls.end(),
                                   [](const common_chat_tool_call &c) { return !c.arguments.empty() && !is_json_object(c.arguments); }),
                    calls.end());
        if (calls.size() < before && calls.empty() && out.content.empty()) out.content = generated.text;
    }
    out.calls.reserve(out.parsed.tool_calls.size());
    for (size_t i = 0; i < out.parsed.tool_calls.size(); ++i) {
        common_chat_tool_call &c = out.parsed.tool_calls[i];
        if (c.id.empty()) c.id = "call_" + std::to_string(i);
        if (c.arguments.empty()) c.arguments = "{}";
        out.calls.push_back(agent_tool_call{agent_str{c.id.data(), c.id.size()}, agent_str{c.name.data(), c.name.size()},
                                            agent_str{c.arguments.data(), c.arguments.size()}});
    }

    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    resp->usage = agent_usage{n_prompt, completion};
    if (AGENT_HAS_FIELD(resp, agent_response, stats)) resp->stats = run_stats;
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
    agent_response_reset(resp);
}

const char *llama_last_error(void *instance) {
    auto *inst = static_cast<Instance *>(instance);
    std::lock_guard<std::mutex> lk(inst->out_mu);
    auto it = inst->errors.find(std::this_thread::get_id());
    return it == inst->errors.end() ? inst->idle_error.c_str() : it->second.c_str();
}

int32_t llama_count_tokens(void *instance, agent_str text, int32_t *out_count) {
    auto *inst = static_cast<Instance *>(instance);
    if (inst->up) {
        *out_count = inst->up->count_tokens(std::string(sv(text)));
        return AGENT_OK;
    }
    const llama_vocab *vocab = llama_model_get_vocab(inst->batcher.model);
    int32_t n = llama_tokenize(vocab, text.data, static_cast<int32_t>(text.size), nullptr, 0, false, false);
    *out_count = n < 0 ? -n : n;
    return AGENT_OK;
}

int32_t llama_reset_cache(void *instance) {
    auto *inst = static_cast<Instance *>(instance);
    if (!inst->up) inst->batcher.clear();
    return AGENT_OK;
}

const agent_engine_v1 kLlamaEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "llama",
    "0.4",
    // Static: what the plugin can do. Image/audio input also needs options.mmproj (and audio a
    // projector that supports it); otherwise run() returns AGENT_ERR_UNSUPPORTED for media parts.
    // EMBED / RERANK need an instance created with options.mode "embed" / "rerank"; an instance runs
    // only its own mode's task.
    AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL | AGENT_CAP_VISION_IN | AGENT_CAP_AUDIO_IN | AGENT_CAP_EMBED | AGENT_CAP_RERANK |
        AGENT_CAP_DECIDE | AGENT_CAP_OCR | AGENT_CAP_GENERIC,
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
