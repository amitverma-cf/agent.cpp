// embed() / rerank(): drive a session's engine with AGENT_CAP_EMBED / AGENT_CAP_RERANK through
// providers::infer_task. Requests are built and responses copied in a per-thread arena (not the
// session's per-turn arena), so these can run from inside a tool call of another session's turn.

#include "../providers/provider_ops.hpp"
#include "rag_internal.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <atomic>
#include <cmath>
#include <cstring>

namespace agent {

namespace {

constexpr size_t kBatch = 16;               // texts per engine call
constexpr size_t kArenaPerText = 64 * 1024; // request parts + one output vector (up to ~16k f32 dims)
constexpr size_t kArenaBase = 64 * 1024;

// Grown on demand, then reused: steady-state calls don't allocate for the engine round trip.
Arena &task_arena(size_t n_texts) {
    thread_local Arena arena;
    thread_local size_t capacity = 0;
    const size_t need = kArenaBase + n_texts * kArenaPerText;
    if (capacity < need) {
        arena = Arena(need);
        capacity = need;
    }
    arena.reset();
    return arena;
}

bool cancelled(const Session &session) { return std::atomic_ref<int32_t>(*session.cancel_flag).load(std::memory_order_relaxed) != 0; }

InferRequest task_request(Session &session, std::span<const MessageView> messages) {
    InferRequest req;
    req.messages = messages;
    req.cancel = session.cancel_flag.get();
    return req;
}

std::span<const float> as_floats(const TensorView &t) {
    return std::span<const float>(reinterpret_cast<const float *>(t.data.data()), t.data.size() / sizeof(float));
}

} // namespace

namespace rag {

void l2_normalize(std::span<float> v) {
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * static_cast<double>(x);
    if (sum <= 0.0) return;
    const auto inv = static_cast<float>(1.0 / std::sqrt(sum));
    for (float &x : v) x *= inv;
}

} // namespace rag

Result<std::vector<std::vector<float>>> embed(Session &session, std::span<const std::string_view> texts) {
    using R = std::vector<std::vector<float>>;
    for (std::string_view t : texts)
        if (t.empty()) return fail<R>(ErrorCode::InvalidConfig, "embed: texts must not be empty.");

    R out;
    out.reserve(texts.size());
    MessageView msgs[kBatch];
    for (size_t begin = 0; begin < texts.size(); begin += kBatch) {
        if (cancelled(session)) return fail<R>(ErrorCode::Cancelled, "embed: cancelled.");
        const size_t n = std::min(kBatch, texts.size() - begin);
        for (size_t i = 0; i < n; ++i) msgs[i] = MessageView{.role = "user", .content = texts[begin + i]};

        auto res =
            providers::infer_task(session, AGENT_CAP_EMBED, task_request(session, std::span<const MessageView>(msgs, n)), task_arena(n));
        if (!res.ok) return fail<R>(res.error.code, res.error.message);

        size_t got = 0;
        for (const TensorView &t : res.value.output_tensors) {
            if (t.name != "embedding") continue;
            if (t.dtype != DType::Float32 || t.data.size() % sizeof(float) != 0 || t.data.empty())
                return fail<R>(ErrorCode::DecodeFailed, "embed: engine returned a malformed embedding tensor.");
            std::span<const float> f = as_floats(t);
            if (!out.empty() && f.size() != out.front().size())
                return fail<R>(ErrorCode::DecodeFailed, "embed: engine returned embeddings of different sizes.");
            std::vector<float> &v = out.emplace_back(f.begin(), f.end());
            rag::l2_normalize(v);
            ++got;
        }
        if (got != n)
            return fail<R>(ErrorCode::DecodeFailed,
                           "embed: engine returned " + std::to_string(got) + " embeddings for " + std::to_string(n) + " texts.");
    }
    return ok(std::move(out));
}

Result<std::vector<float>> rerank(Session &session, std::string_view query, std::span<const std::string_view> docs) {
    using R = std::vector<float>;
    if (query.empty()) return fail<R>(ErrorCode::InvalidConfig, "rerank: query must not be empty.");

    R out;
    out.reserve(docs.size());
    MessageView msgs[kBatch + 1];
    msgs[0] = MessageView{.role = "user", .content = query};
    for (size_t begin = 0; begin < docs.size(); begin += kBatch) {
        if (cancelled(session)) return fail<R>(ErrorCode::Cancelled, "rerank: cancelled.");
        const size_t n = std::min(kBatch, docs.size() - begin);
        // An empty document would produce no text part and shift every later score.
        for (size_t i = 0; i < n; ++i)
            msgs[i + 1] = MessageView{.role = "user", .content = docs[begin + i].empty() ? " " : docs[begin + i]};

        auto res = providers::infer_task(session, AGENT_CAP_RERANK, task_request(session, std::span<const MessageView>(msgs, n + 1)),
                                         task_arena(n));
        if (!res.ok) return fail<R>(res.error.code, res.error.message);

        bool found = false;
        for (const TensorView &t : res.value.output_tensors) {
            if (t.name != "scores") continue;
            std::span<const float> f = as_floats(t);
            if (t.dtype != DType::Float32 || f.size() != n)
                return fail<R>(ErrorCode::DecodeFailed,
                               "rerank: engine returned " + std::to_string(f.size()) + " scores for " + std::to_string(n) + " documents.");
            out.insert(out.end(), f.begin(), f.end());
            found = true;
            break;
        }
        if (!found) return fail<R>(ErrorCode::DecodeFailed, "rerank: engine returned no \"scores\" tensor.");
    }
    return ok(std::move(out));
}

} // namespace agent
