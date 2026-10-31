// Built-in VectorStoreOps: brute-force cosine similarity over unit vectors kept in memory.

#include "rag_internal.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cmath>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agent {

namespace {

struct Entry {
    std::string id;
    std::string text;
    std::string metadata_json;
};

struct InMemoryVectorState {
    size_t dims = 0;
    std::vector<Entry> entries;
    std::vector<float> vectors; // entries.size() * dims, unit length, row i = entries[i]
    std::unordered_map<std::string, size_t> by_id;
    std::shared_mutex mutex;
};

Result<std::shared_ptr<void>> mem_open(const VectorStoreConfig &config) {
    if (config.dimensions <= 0) return fail<std::shared_ptr<void>>(ErrorCode::InvalidConfig, "vector store: dimensions must be > 0.");
    auto state = std::make_shared<InMemoryVectorState>();
    state->dims = static_cast<size_t>(config.dimensions);
    return ok(std::shared_ptr<void>(std::move(state)));
}

Result<void> check_dims(const InMemoryVectorState &s, std::span<const float> v) {
    if (v.size() != s.dims)
        return fail(ErrorCode::InvalidConfig,
                    "vector store: embedding has " + std::to_string(v.size()) + " dimensions, the store " + std::to_string(s.dims) + ".");
    for (float x : v)
        if (!std::isfinite(x)) return fail(ErrorCode::InvalidConfig, "vector store: embedding contains a non-finite value.");
    return ok();
}

Result<void> mem_upsert(void *state, std::string_view id, std::string_view text, std::span<const float> embedding,
                        std::string_view metadata_json) {
    auto *s = static_cast<InMemoryVectorState *>(state);
    if (auto r = check_dims(*s, embedding); !r.ok) return r;
    if (id.empty()) return fail(ErrorCode::InvalidConfig, "vector store: id must not be empty.");

    std::unique_lock lock(s->mutex);
    size_t row = 0;
    if (auto it = s->by_id.find(std::string(id)); it != s->by_id.end()) {
        row = it->second;
        s->entries[row].text.assign(text);
        s->entries[row].metadata_json.assign(metadata_json);
    } else {
        row = s->entries.size();
        s->entries.push_back(Entry{std::string(id), std::string(text), std::string(metadata_json)});
        s->vectors.resize(s->vectors.size() + s->dims);
        s->by_id.emplace(std::string(id), row);
    }
    std::span<float> dst(s->vectors.data() + row * s->dims, s->dims);
    std::copy(embedding.begin(), embedding.end(), dst.begin());
    rag::l2_normalize(dst);
    return ok();
}

Result<std::vector<VectorHit>> mem_query(void *state, std::span<const float> embedding, int k) {
    using R = std::vector<VectorHit>;
    auto *s = static_cast<InMemoryVectorState *>(state);
    if (auto r = check_dims(*s, embedding); !r.ok) return fail<R>(r.error.code, r.error.message);
    if (k <= 0) return ok(R{});

    std::vector<float> q(embedding.begin(), embedding.end());
    rag::l2_normalize(q);

    std::shared_lock lock(s->mutex);
    std::vector<std::pair<float, size_t>> scored(s->entries.size());
    for (size_t row = 0; row < s->entries.size(); ++row) {
        const float *v = s->vectors.data() + row * s->dims;
        float dot = 0.0f;
        for (size_t d = 0; d < s->dims; ++d) dot += v[d] * q[d];
        scored[row] = {dot, row};
    }
    const size_t n = std::min(scored.size(), static_cast<size_t>(k));
    std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(n), scored.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    R hits;
    hits.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const Entry &e = s->entries[scored[i].second];
        hits.push_back(VectorHit{.id = e.id, .text = e.text, .metadata_json = e.metadata_json, .score = scored[i].first});
    }
    return ok(std::move(hits));
}

Result<bool> mem_erase(void *state, std::string_view id) {
    auto *s = static_cast<InMemoryVectorState *>(state);
    std::unique_lock lock(s->mutex);
    auto it = s->by_id.find(std::string(id));
    if (it == s->by_id.end()) return ok(false);
    const size_t row = it->second, last = s->entries.size() - 1;
    s->by_id.erase(it);
    if (row != last) { // move the last row into the hole
        s->entries[row] = std::move(s->entries[last]);
        std::copy_n(s->vectors.begin() + static_cast<std::ptrdiff_t>(last * s->dims), s->dims,
                    s->vectors.begin() + static_cast<std::ptrdiff_t>(row * s->dims));
        s->by_id[s->entries[row].id] = row;
    }
    s->entries.pop_back();
    s->vectors.resize(s->entries.size() * s->dims);
    return ok(true);
}

Result<void> mem_clear(void *state) {
    auto *s = static_cast<InMemoryVectorState *>(state);
    std::unique_lock lock(s->mutex);
    s->entries.clear();
    s->vectors.clear();
    s->by_id.clear();
    return ok();
}

constexpr VectorStoreOps kInMemoryVectorStore{
    .open = mem_open, .upsert = mem_upsert, .query_topk = mem_query, .erase = mem_erase, .clear = mem_clear};

} // namespace

const VectorStoreOps *in_memory_vector_store() { return &kInMemoryVectorStore; }

} // namespace agent
