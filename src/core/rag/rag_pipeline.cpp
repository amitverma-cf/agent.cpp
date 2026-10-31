// RAG pipeline: chunk -> embed -> VectorStoreOps -> (rerank) -> hits, plus the search_knowledge tool.

#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <simdjson.h>
#include <string>

namespace agent {

namespace {

constexpr int kMaxToolHits = 50;

const VectorStoreOps *ops_of(const RagIndex &index) { return index.store ? index.store : in_memory_vector_store(); }

Result<void> ensure_open(RagIndex &index, size_t dimensions) {
    if (index.state) return ok();
    return open_rag_index(index, static_cast<int>(dimensions));
}

std::string chunk_id(std::string_view doc_id, size_t n) {
    std::string id(doc_id);
    id += '#';
    id += std::to_string(n);
    return id;
}

// Erases "<doc_id>#<n>" for n = first, first + 1, ... until one is missing.
Result<void> erase_chunks_from(RagIndex &index, std::string_view doc_id, size_t first) {
    if (!index.state) return ok();
    const VectorStoreOps *ops = ops_of(index);
    for (size_t n = first;; ++n) {
        auto r = ops->erase(index.state.get(), chunk_id(doc_id, n));
        if (!r.ok) return fail(r.error.code, r.error.message);
        if (!r.value) return ok();
    }
}

} // namespace

Result<void> open_rag_index(RagIndex &index, int dimensions) {
    const VectorStoreOps *ops = ops_of(index);
    if (!ops->open || !ops->upsert || !ops->query_topk || !ops->erase || !ops->clear)
        return fail(ErrorCode::InvalidConfig, "RagIndex::store is missing one of open/upsert/query_topk/erase/clear.");
    VectorStoreConfig config = index.config;
    if (dimensions > 0) {
        if (config.dimensions > 0 && config.dimensions != dimensions)
            return fail(ErrorCode::InvalidConfig, "RagIndex: embeddings have " + std::to_string(dimensions) +
                                                      " dimensions but config.dimensions is " + std::to_string(config.dimensions) + ".");
        config.dimensions = dimensions;
    }
    if (config.dimensions <= 0) return fail(ErrorCode::InvalidConfig, "RagIndex: dimensions must be > 0.");
    auto res = ops->open(config);
    if (!res.ok) return fail(res.error.code, res.error.message);
    index.state = std::move(res.value);
    index.config.dimensions = config.dimensions;
    return ok();
}

Result<void> rag_index(Session &embed_session, RagIndex &index, std::string_view doc_id, std::string_view text,
                       const ChunkOptions &options) {
    if (doc_id.empty()) return fail(ErrorCode::InvalidConfig, "rag_index: doc_id must not be empty.");
    auto chunks = chunk_text(text, options);
    if (!chunks.ok) return fail(chunks.error.code, chunks.error.message);
    if (chunks.value.empty()) return erase_chunks_from(index, doc_id, 0);

    auto vectors = embed(embed_session, chunks.value);
    if (!vectors.ok) return fail(vectors.error.code, vectors.error.message);
    if (auto r = ensure_open(index, vectors.value.front().size()); !r.ok) return r;

    const VectorStoreOps *ops = ops_of(index);
    std::string meta;
    for (size_t n = 0; n < chunks.value.size(); ++n) {
        const std::string_view chunk = chunks.value[n];
        meta = "{\"doc_id\":\"";
        utils::escape_json_string(doc_id, meta);
        meta += "\",\"chunk\":" + std::to_string(n) + ",\"offset\":" + std::to_string(chunk.data() - text.data()) + "}";
        if (auto r = ops->upsert(index.state.get(), chunk_id(doc_id, n), chunk, vectors.value[n], meta); !r.ok) return r;
    }
    return erase_chunks_from(index, doc_id, chunks.value.size());
}

Result<void> rag_erase(RagIndex &index, std::string_view doc_id) { return erase_chunks_from(index, doc_id, 0); }

Result<std::vector<RagHit>> rag_query(Session &embed_session, RagIndex &index, std::string_view query, int k, Session *rerank_session) {
    using R = std::vector<RagHit>;
    if (query.empty()) return fail<R>(ErrorCode::InvalidConfig, "rag_query: query must not be empty.");
    if (k <= 0) return ok(R{});

    const std::string_view texts[] = {query};
    auto vectors = embed(embed_session, texts);
    if (!vectors.ok) return fail<R>(vectors.error.code, vectors.error.message);
    if (auto r = ensure_open(index, vectors.value.front().size()); !r.ok) return fail<R>(r.error.code, r.error.message);

    const int fetch = rerank_session ? std::max(4 * k, 20) : k;
    auto found = ops_of(index)->query_topk(index.state.get(), vectors.value.front(), fetch);
    if (!found.ok) return fail<R>(found.error.code, found.error.message);

    R hits;
    hits.reserve(found.value.size());
    for (VectorHit &h : found.value) {
        const size_t hash = h.id.rfind('#');
        std::string doc_id = hash == std::string::npos ? h.id : h.id.substr(0, hash);
        hits.push_back(RagHit{.id = std::move(h.id),
                              .doc_id = std::move(doc_id),
                              .text = std::move(h.text),
                              .metadata_json = std::move(h.metadata_json),
                              .similarity = h.score,
                              .score = h.score});
    }

    if (rerank_session && !hits.empty()) {
        std::vector<std::string_view> docs;
        docs.reserve(hits.size());
        for (const RagHit &h : hits) docs.push_back(h.text);
        auto scores = rerank(*rerank_session, query, docs);
        if (!scores.ok) return fail<R>(scores.error.code, scores.error.message);
        for (size_t i = 0; i < hits.size(); ++i) hits[i].score = scores.value[i];
        std::stable_sort(hits.begin(), hits.end(), [](const RagHit &a, const RagHit &b) { return a.score > b.score; });
    }
    if (hits.size() > static_cast<size_t>(k)) hits.resize(static_cast<size_t>(k));
    return ok(std::move(hits));
}

// -------------------------- search_knowledge tool --------------------------

namespace {

struct RagToolState {
    Session *embed_session = nullptr;
    RagIndex *index = nullptr;
    Session *rerank_session = nullptr;
    int k = 4;
    std::mutex mutex; // one embed / rerank session must not be driven from two threads at once
};

constexpr std::string_view kRagToolDescription =
    "Search the knowledge base for passages relevant to a question. Returns the best matching text passages "
    "with their source ids, most relevant first.";
constexpr std::string_view kRagToolSchema = R"({"type":"object","properties":{"query":{"type":"string","description":"What to look up"},)"
                                            R"("k":{"type":"integer","description":"How many passages to return"}},"required":["query"]})";

// arguments: {"query": "...", "k": n}. A non-JSON argument string is taken as the query itself.
Result<std::string> rag_tool_query(void *state_ptr, std::string_view arguments, void *) {
    auto *state = static_cast<RagToolState *>(state_ptr);
    thread_local simdjson::ondemand::parser parser;
    std::string query;
    int k = state->k;
    simdjson::padded_string json(arguments);
    simdjson::ondemand::document doc;
    simdjson::ondemand::object obj;
    if (!parser.iterate(json).get(doc) && !doc.get_object().get(obj)) {
        for (auto field : obj) {
            std::string_view key, s;
            if (field.unescaped_key().get(key)) continue;
            int64_t n = 0;
            if (key == "query" && !field.value().get_string().get(s)) query.assign(s);
            else if (key == "k" && !field.value().get_int64().get(n) && n > 0) k = static_cast<int>(std::min<int64_t>(n, kMaxToolHits));
        }
    } else {
        query.assign(arguments);
    }
    if (query.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "search_knowledge: \"query\" is required.");

    std::lock_guard<std::mutex> lock(state->mutex);
    auto hits = rag_query(*state->embed_session, *state->index, query, k, state->rerank_session);
    if (!hits.ok) return fail<std::string>(hits.error.code, hits.error.message);

    std::string out = "[";
    for (size_t i = 0; i < hits.value.size(); ++i) {
        const RagHit &h = hits.value[i];
        if (i) out += ',';
        out += "{\"id\":\"";
        utils::escape_json_string(h.id, out);
        char score[32];
        std::snprintf(score, sizeof(score), "%.4f", static_cast<double>(h.score));
        out += "\",\"score\":";
        out += score;
        out += ",\"text\":\"";
        utils::escape_json_string(h.text, out);
        out += "\"}";
    }
    out += ']';
    return ok(std::move(out));
}

} // namespace

Tool make_rag_tool(Session &chat_session, Session &embed_session, RagIndex &index, Session *rerank_session, int k) {
    auto state = std::make_shared<RagToolState>();
    state->embed_session = &embed_session;
    state->index = &index;
    state->rerank_session = rerank_session;
    state->k = std::clamp(k, 1, kMaxToolHits);

    auto source = std::make_shared<DataSource>();
    source->name = "search_knowledge";
    source->state = std::move(state);
    source->query = rag_tool_query;
    return bind_data_source_tool(chat_session, std::move(source), kRagToolDescription, kRagToolSchema);
}

} // namespace agent
