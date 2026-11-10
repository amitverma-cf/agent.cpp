# RAG: embeddings, rerank, vector stores

`#include <agent-cpp/agent.hpp>` (sqlite-vec store: `#include <agent-cpp/store_sqlite.hpp>`, link `agent_store_sqlite`)

A local retrieval pipeline: **chunk -> embed -> vector store -> (rerank) -> hits**, plus a
`search_knowledge` tool that lets an agent query it. Everything runs on engine plugins, so it is
fully offline. `libagent` gains no dependencies: the in-memory vector store is part of the core;
the sqlite-vec store is in the optional `agent_store_sqlite` library.

Embedding, reranking and chat normally use **three different models**, so each runs in its own
`Session`:

```cpp
agent::Config base;
base.provider      = agent::AiProvider::Engine;
base.engine        = "build/engines/llama/agent-engine-llama.so";
base.workspace_dir = ".workspace";

agent::Config embed_cfg = base;
embed_cfg.model          = ".workspace/models/bge-small-en-v1.5-q8_0.gguf";
embed_cfg.context_window = 512;
embed_cfg.engine_options = R"({"mode":"embed"})";

agent::Config rerank_cfg = base;
rerank_cfg.model          = ".workspace/models/jina-reranker-v1-tiny-en-Q8_0.gguf";
rerank_cfg.context_window = 512;
rerank_cfg.engine_options = R"({"mode":"rerank"})";

agent::Config chat_cfg = base;
chat_cfg.model = ".workspace/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

auto embedder = agent::init(embed_cfg);
auto reranker = agent::init(rerank_cfg);
auto chat     = agent::init(chat_cfg);

agent::RagIndex index; // in-memory store, dimensions taken from the first embedding
agent::rag_index(embedder.value, index, "handbook", handbook_text, {.size = 800, .overlap = 100});

auto hits = agent::rag_query(embedder.value, index, "How many vacation days do I get?", 3, &reranker.value);
for (const agent::RagHit &h : hits.value) printf("%s (%.2f): %s\n", h.id.c_str(), h.score, h.text.c_str());

// Or let the chat model search on its own:
chat.value.config.tools.push_back(agent::make_rag_tool(chat.value, embedder.value, index, &reranker.value));
agent::rebuild_tool_index(chat.value);
```

See [engines.md](engines.md#embeddings-and-reranking) for the llama engine's `embed` / `rerank` modes.

---

## embed / rerank

```cpp
Result<std::vector<std::vector<float>>> embed(Session &session, std::span<const std::string_view> texts);
Result<std::vector<float>> rerank(Session &session, std::string_view query, std::span<const std::string_view> docs);
```

Both send a non-chat task to the session's engine (`AGENT_CAP_EMBED` / `AGENT_CAP_RERANK` in
`agent_request::task`); `infer()` / `run_turn()` keep sending the engine's default task.

- `embed` returns one embedding per text, in order, each L2-normalized (the core re-normalizes
  whatever the engine returns). Empty texts are rejected.
- `rerank` returns one score per document, in order. Higher is more relevant; the scale is the
  model's (the llama engine returns the raw classifier logit). Empty documents score as `" "`.
- Texts are sent in batches of 16 per engine call. The request and response go through a
  per-thread arena owned by the RAG code, not the session's per-turn arena, so `embed` / `rerank`
  can be called from a tool callback in the middle of another session's turn.
- Fails with `UnsupportedProvider` when the session is not an `AiProvider::Engine` session or the
  engine does not declare the capability, and with the engine's error otherwise (the llama
  engine refuses a task that doesn't match the instance's `mode`).
- Don't drive one session from several threads at once (the `search_knowledge` tool serializes its
  own calls).

Engine contract, for plugin authors: for `AGENT_CAP_EMBED`, return one `AGENT_PART_TENSOR` part named
`"embedding"` (f32, shape `[n_dim]`) per TEXT part of the request, in order. For
`AGENT_CAP_RERANK`, the first TEXT part is the query and every later TEXT part a document; return
one tensor named `"scores"` (f32, shape `[n_docs]`).

---

## VectorStoreOps

```cpp
struct VectorStoreConfig {
    std::string path;          // backend-specific; sqlite-vec: file path, or empty / ":memory:"
    std::string name = "rag";  // collection (table) name: [A-Za-z_][A-Za-z0-9_]{0,63}
    int dimensions = 0;
};

struct VectorHit {
    std::string id, text, metadata_json;
    float score = 0.0f; // cosine similarity
};

struct VectorStoreOps {
    Result<std::shared_ptr<void>> (*open)(const VectorStoreConfig &config);
    Result<void> (*upsert)(void *state, std::string_view id, std::string_view text,
                           std::span<const float> embedding, std::string_view metadata_json);
    Result<std::vector<VectorHit>> (*query_topk)(void *state, std::span<const float> embedding, int k);
    Result<bool> (*erase)(void *state, std::string_view id); // true if something was removed
    Result<void> (*clear)(void *state);
    // Optional: words and meaning together, merged by Reciprocal Rank Fusion. score is the fused score.
    Result<std::vector<VectorHit>> (*query_hybrid)(void *state, std::string_view text,
                                                   std::span<const float> embedding, int k);
};

const VectorStoreOps *in_memory_vector_store();            // libagent
const VectorStoreOps *agent::store::sqlite_vector_store();  // agent_store_sqlite
```

A function-pointer table like `StoreOps` (no vtables). `open()` returns the state passed to every
other call; implementations must be safe to call from several threads on one state. Scores are
cosine similarities (1 = same direction), most similar first.

| Store | Where | Notes |
|---|---|---|
| `in_memory_vector_store()` | `libagent` | Brute-force cosine over unit vectors in a flat array, `std::shared_mutex` (parallel queries). Nothing persists. |
| `store::sqlite_vector_store()` | `agent_store_sqlite` | sqlite-vec `vec0(embedding float[d] distance_metric=cosine)` table `<name>_vectors` plus `<name>_chunks(id UNIQUE, text, metadata)`; KNN via `MATCH ... AND k = ?` (k capped at 4096); score = 1 - cosine distance. An existing table keeps the dimensions it was created with. |

**Hybrid search.** `RagIndex::hybrid` (default `true`; set `false` for pure vector ranking) makes `rag_query` call `query_hybrid` when the store has it (the sqlite
store does). The sqlite store keeps an FTS5 index over the chunk text (an external-content table `<name>_fts`, kept in
step by triggers, built once for databases that predate it). A query returns the top candidates of two rankings, the
embedding's nearest neighbours and the BM25 matches of the query's words, and scores each chunk with
`sum 1 / (60 + rank)` over the rankings it appears in. Exact names, codes and rare words that an embedding blurs are
found by the text side; paraphrases by the vector side. The words of the query are quoted, so punctuation or FTS5
operators in a question cannot make the query fail. `RagHit::similarity` and `score` then hold the fused score, not a
cosine similarity. A store without `query_hybrid`, or an SQLite built without FTS5, answers with the vector ranking.

The older [`make_sqlite_vec_data_source`](data-sources.md#sqlite-vec-rag--vector-search-in-agent_store_sqlite)
`DataSource` is unchanged and independent (it uses different tables).

---

## chunk_text

```cpp
struct ChunkOptions {
    size_t size = 1000;   // maximum bytes per chunk
    size_t overlap = 150; // bytes repeated from the end of the previous chunk; must be < size
};
Result<std::vector<std::string_view>> chunk_text(std::string_view text, const ChunkOptions &options = {});
```

Chunks are at most `size` bytes. Within the back half of each window the chunker breaks at the
strongest boundary it finds: a paragraph break (blank line), then a line end, then a sentence end
(`. ` / `! ` / `? `), then whitespace; with no boundary it cuts, but never inside a UTF-8 sequence.
The next chunk starts up to `overlap` bytes earlier, moved forward to the start of a word. The
returned views point into `text`, trimmed of surrounding whitespace; empty chunks are dropped.

---

## RagIndex, rag_index, rag_query

```cpp
struct RagIndex {
    const VectorStoreOps *store = nullptr; // nullptr = in_memory_vector_store()
    VectorStoreConfig config;
    std::shared_ptr<void> state;           // opened lazily
};

Result<void> open_rag_index(RagIndex &index, int dimensions = 0);
Result<void> rag_index(Session &embed_session, RagIndex &index, std::string_view doc_id, std::string_view text,
                       const ChunkOptions &options = {});
Result<void> rag_erase(RagIndex &index, std::string_view doc_id);
Result<std::vector<RagHit>> rag_query(Session &embed_session, RagIndex &index, std::string_view query, int k,
                                      Session *rerank_session = nullptr);

struct RagHit {
    std::string id;            // "<doc_id>#<n>"
    std::string doc_id;
    std::string text;
    std::string metadata_json; // {"doc_id": ..., "chunk": n, "offset": byte offset in the document}
    float similarity = 0.0f;   // cosine similarity from the store
    float score = 0.0f;        // rerank score when reranked, else similarity
};
```

- The store is opened by the first `rag_index` / `rag_query`, with `config.dimensions` taken from
  that embedding when it is 0 (or call `open_rag_index` yourself). A mismatch between a non-zero
  `config.dimensions` and the embedder fails. Opening is not thread-safe; afterwards the index is as
  thread-safe as its store.
- `rag_index` chunks `text`, embeds every chunk, and upserts them as `<doc_id>#0`, `<doc_id>#1`, ...
  Re-indexing a `doc_id` overwrites its chunks and erases leftover higher-numbered ones, so a
  document can be updated in place. `rag_erase` removes all of a document's chunks.
- `rag_query` embeds the query and returns the `k` most similar chunks. With a `rerank_session` it
  fetches `max(4k, 20)` candidates, scores them with `rerank`, sorts by that score, and returns the
  top `k`.
- To persist an index, use the sqlite-vec store with a file path; a later `RagIndex` with the same
  `path` / `name` sees the same chunks:

```cpp
agent::RagIndex index{.store = agent::store::sqlite_vector_store(),
                      .config = {.path = ".workspace/memory/rag.sqlite3", .name = "handbook"}};
```

---

## make_rag_tool

```cpp
Tool make_rag_tool(Session &chat_session, Session &embed_session, RagIndex &index,
                   Session *rerank_session = nullptr, int k = 4);
```

A tool named `search_knowledge` with parameters `{"query": string, "k": integer (optional)}`. It runs
`rag_query` and returns `[{"id": "...", "score": 0.83, "text": "..."}, ...]` (JSON, most relevant
first). A non-JSON argument string is taken as the query.

The tool's state lives in a `DataSource` retained by `chat_session` (via `bind_data_source_tool`).
`embed_session`, `index` and `rerank_session` are referenced, so they must outlive the chat session
and must not be moved after the call. Calls through one tool are serialized by a mutex, so parallel
subagents can share it. Register it like any other tool:

```cpp
chat.config.tools.push_back(agent::make_rag_tool(chat, embedder, index, &reranker));
agent::rebuild_tool_index(chat);
```

---

## Tests and models

`tests/unit/test_rag.cpp` covers the stores, the chunker and the pipeline with a fake in-process
engine. Real-model tests run when these are set:

| Variable | Tested with (Apache-2.0 / MIT) |
|---|---|
| `AGENT_TEST_EMBED_MODEL` | `ggml-org/bge-small-en-v1.5-Q8_0-GGUF` (`bge-small-en-v1.5-q8_0.gguf`, 384 dims, MIT) |
| `AGENT_TEST_RERANK_MODEL` | `gpustack/jina-reranker-v1-tiny-en-GGUF` (`jina-reranker-v1-tiny-en-Q8_0.gguf`, Apache-2.0); also `gpustack/bge-reranker-v2-m3-GGUF` Q4_K_M |
