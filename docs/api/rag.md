# RAG: embeddings, rerank, vector stores

`#include <agent-cpp/agent.hpp>`. The sqlite-vec store is in `#include <agent-cpp/store_sqlite.hpp>` and links `agent-cpp::store_sqlite`.

A local retrieval pipeline: **chunk, embed, store, (rerank), hits**. It also provides a `search_knowledge` tool that lets a chat model
query the index itself. Everything runs on engine plugins, so it works offline. The in-memory vector store is part of the core and adds no
dependencies. The sqlite-vec store is in the optional `agent_store_sqlite` library.

This part can be left out of a build with the CMake option `AGENT_WITH_RAG=OFF`.

Embedding, reranking and chat normally use three different models, so each runs in its own session. Use `engine_config` for the
engine modes (`{"mode":"embed"}`, `{"mode":"rerank"}`), and `open_session` for chat:

```cpp
#include <agent-cpp/agent.hpp>

agent::Config base;
base.workspace_dir = ".workspace";
base.context_window = 512;

const agent::ModelSlot embed_files[]  = {{"model", ".workspace/models/bge-small-en-v1.5-q8_0.gguf"}};
const agent::ModelSlot rerank_files[] = {{"model", ".workspace/models/jina-reranker-v1-tiny-en-Q8_0.gguf"}};

auto embed_cfg  = agent::engine_config(agent::engines::llama, embed_files,  R"({"mode":"embed"})",  base);
auto rerank_cfg = agent::engine_config(agent::engines::llama, rerank_files, R"({"mode":"rerank"})", base);
auto embedder = agent::init(embed_cfg.value);
auto reranker = agent::init(rerank_cfg.value);

agent::Config chat_base = base;
chat_base.context_window = 2048;
auto chat = agent::open_session(agent::engines::llama, ".workspace/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", chat_base);

agent::RagIndex index;   // in-memory store; dimensions taken from the first embedding
agent::rag_index(embedder.value, index, "handbook", handbook_text, {.size = 800, .overlap = 100});

auto hits = agent::rag_query(embedder.value, index, "How many vacation days do I get?", 3, &reranker.value);
for (const agent::RagHit &h : hits.value) printf("%s (%.2f): %s\n", h.id.c_str(), h.score, h.text.c_str());

// Or let the chat model search on its own:
chat.value.config.tools.push_back(agent::make_rag_tool(chat.value, embedder.value, index, &reranker.value));
agent::rebuild_tool_index(chat.value);
```

Error checks are left out. Check `.ok` on each result. The engine's own page covers the modes: [engines.md](engines.md#embeddings-and-reranking).

---

## embed / rerank

```cpp
Result<std::vector<std::vector<float>>> embed(Session &session, std::span<const std::string_view> texts);
Result<std::vector<float>> rerank(Session &session, std::string_view query, std::span<const std::string_view> docs);
```

Both send a non-chat task to the session's engine (`AGENT_CAP_EMBED` or `AGENT_CAP_RERANK`). `infer` and `run_turn` keep sending the
engine's default task.

- `embed` returns one embedding per text, in order. Each is L2-normalized, because the core normalizes whatever the engine returns.
  Empty texts are rejected with `InvalidConfig`.
- `rerank` returns one score per document, in order. Higher is more relevant. The scale is the model's: the llama engine returns the raw
  classifier logit. Empty documents are scored as `" "`, so that every later score stays in place.
- Texts go to the engine in batches of 16 per call.
- The request and response use a per-thread arena owned by the RAG code, not the session's per-turn arena. So `embed` and `rerank` can be
  called from a tool callback in the middle of another session's turn.
- Fails with `UnsupportedProvider` when the session is not an engine session or the engine does not declare the capability. The llama
  engine refuses a task that does not match the instance's `mode`.
- Do not drive one session from several threads at once. The `search_knowledge` tool serializes its own calls.

```cpp
const std::string_view texts[] = {"How many vacation days?", "Vacation policy: 25 days a year."};
auto vectors = agent::embed(embedder.value, texts);
```

### Multimodal embed

```cpp
struct EmbedInput { std::string_view text; std::span<const Media> media; };
Result<std::vector<std::vector<float>>> embed(Session &session, std::span<const EmbedInput> inputs);
```

Each input is optional text plus images or audio, and gets one embedding, in order. The model needs a projector, so open the session
with the `mmproj` option. Text-only models reject media. Verified with Qwen3-VL-Embedding-2B.

```cpp
// vl_embedder: a session opened on an embedding model with its projector, e.g. with engine_config(..., {"mmproj": ...}).
agent::Media photo[] = {{"photos/invoice.png", "image/png"}};
agent::EmbedInput inputs[] = {{"invoice from March", photo}};
auto vectors = agent::embed(vl_embedder.value, inputs);
```

Engine contract, for plugin authors: for `AGENT_CAP_EMBED`, return one `AGENT_PART_TENSOR` part named `"embedding"` (f32, shape `[n_dim]`)
per TEXT part of the request, in order. For `AGENT_CAP_RERANK`, the first TEXT part is the query and every later TEXT part a document.
Return one tensor named `"scores"` (f32, shape `[n_docs]`).

---

## VectorStoreOps

```cpp
struct VectorStoreConfig {
    std::string path;          // backend-specific; sqlite-vec: file path, or empty / ":memory:"
    std::string name = "rag";  // collection (table) name: [A-Za-z_][A-Za-z0-9_]{0,63}
    int dimensions = 0;        // 0 = taken from the first embedding
};

struct VectorHit {
    std::string id, text, metadata_json;
    float score = 0.0f;        // cosine similarity; with hybrid search, the fused score
};

struct VectorStoreOps {
    Result<std::shared_ptr<void>> (*open)(const VectorStoreConfig &config);
    Result<void> (*upsert)(void *state, std::string_view id, std::string_view text,
                           std::span<const float> embedding, std::string_view metadata_json);
    Result<std::vector<VectorHit>> (*query_topk)(void *state, std::span<const float> embedding, int k);
    Result<bool> (*erase)(void *state, std::string_view id);   // true if an entry was removed
    Result<void> (*clear)(void *state);
    Result<std::vector<VectorHit>> (*query_hybrid)(void *state, std::string_view text,
                                                   std::span<const float> embedding, int k);   // optional
};

const VectorStoreOps *in_memory_vector_store();             // libagent
const VectorStoreOps *agent::store::sqlite_vector_store();  // agent_store_sqlite
```

A function-pointer table, like `StoreOps`. `open` returns the state passed to every other call. Implementations must be safe to call from
several threads on one state. Results are sorted most similar first.

| Store | Where | Notes |
|---|---|---|
| `in_memory_vector_store()` | `libagent` | Brute-force cosine search over unit vectors in a flat array, under a shared mutex, so queries run in parallel. Nothing persists. |
| `store::sqlite_vector_store()` | `agent_store_sqlite` | sqlite-vec `vec0(embedding float[d] distance_metric=cosine)` in a table `<name>_vectors`, with `<name>_chunks(id UNIQUE, text, metadata)`. KNN uses `MATCH ... AND k = ?`, with k capped at 4096. The score is `1 - cosine distance`. An existing table keeps the dimensions it was created with. |

### Hybrid search

`RagIndex::hybrid` (default `true`) makes `rag_query` call `query_hybrid` when the store has it. The sqlite store has it. The store keeps an
FTS5 index over the chunk text, kept in step by triggers. A query takes the top candidates of two rankings, the embedding's nearest
neighbours and the BM25 matches of the query's words. Each chunk is scored with `sum 1 / (60 + rank)` over the rankings it appears in. Exact
names, codes and rare words, which embeddings blur, are found by the text side. Paraphrases are found by the vector side.

With hybrid search, `RagHit::similarity` and `score` hold the fused score, not a cosine similarity. A store without `query_hybrid`, or an
SQLite build without FTS5, answers with the vector ranking. Set `hybrid = false` for pure vector ranking.

The older [`make_sqlite_vec_data_source`](data-sources.md#sqlite-vec-rag--vector-search-in-agent_store_sqlite) is a separate `DataSource`. It uses different tables and does
not change with this API.

---

## chunk_text

```cpp
struct ChunkOptions {
    size_t size = 1000;   // maximum bytes per chunk
    size_t overlap = 150; // bytes repeated from the end of the previous chunk; must be < size
};
Result<std::vector<std::string_view>> chunk_text(std::string_view text, const ChunkOptions &options = {});
```

Chunks are at most `size` bytes. In the back half of each window, the chunker breaks at the strongest boundary it finds: a paragraph break
(blank line), then a line end, then a sentence end (`. `, `! `, `? `), then whitespace. With no boundary it cuts, but never inside a UTF-8
sequence. The next chunk starts up to `overlap` bytes earlier, moved forward to a word start. The returned views point into `text`,
trimmed of surrounding whitespace. Empty chunks are dropped.

---

## RagIndex, rag_index, rag_query

```cpp
struct RagIndex {
    const VectorStoreOps *store = nullptr;   // nullptr = in_memory_vector_store()
    VectorStoreConfig config;
    bool hybrid = true;                      // see Hybrid search
    std::shared_ptr<void> state;             // opened lazily
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
    float similarity = 0.0f;   // cosine similarity from the store (fused score with hybrid search)
    float score = 0.0f;        // rerank score when reranked, else similarity
};
```

- The store opens at the first `rag_index` or `rag_query`. `config.dimensions` is taken from that embedding when it is 0. You can also call
  `open_rag_index` yourself. A non-zero `config.dimensions` that does not match the embedder fails. Opening is not thread-safe. After that,
  the index is as thread-safe as its store.
- `rag_index` chunks `text`, embeds every chunk, and stores the chunks as `<doc_id>#0`, `<doc_id>#1`, and so on. Re-indexing a `doc_id`
  replaces its chunks and erases leftover higher-numbered ones, so a document can be updated in place. `rag_erase` removes all of a
  document's chunks.
- `rag_query` embeds the query and returns the `k` most similar chunks. With a `rerank_session`, it fetches `max(4k, 20)` candidates, scores
  them with `rerank`, sorts by that score, and returns the top `k`.

```cpp
agent::RagIndex index;
agent::open_rag_index(index, 384);                                      // optional: fix the dimensions up front
agent::rag_index(embedder.value, index, "handbook", handbook_text);     // default chunking
agent::rag_index(embedder.value, index, "handbook", new_text);          // replaces the chunks of "handbook"
agent::rag_erase(index, "handbook");
```

To keep an index on disk, use the sqlite-vec store with a file path. A later `RagIndex` with the same `path` and `name` sees the same chunks:

```cpp
#include <agent-cpp/store_sqlite.hpp>

agent::RagIndex index{.store = agent::store::sqlite_vector_store(),
                      .config = {.path = ".workspace/memory/rag.sqlite3", .name = "handbook"}};
```

---

## make_rag_tool

```cpp
Tool make_rag_tool(Session &chat_session, Session &embed_session, RagIndex &index,
                   Session *rerank_session = nullptr, int k = 4);
```

Creates a tool named `search_knowledge` with parameters `{"query": string, "k": integer (optional)}`. It runs `rag_query` and returns
`[{"id": "...", "score": 0.83, "text": "..."}, ...]` as JSON, most relevant first. An argument that is not JSON is taken as the query.

The tool's state is held by a `DataSource` that `chat_session` retains. `embed_session`, `index` and `rerank_session` are referenced, so they
must outlive the chat session and must not move after the call. Calls through one tool are serialized, so parallel subagents can share it.

```cpp
chat.config.tools.push_back(agent::make_rag_tool(chat, embedder, index, &reranker));
agent::rebuild_tool_index(chat);
```

---

## Tests and models

`tests/unit/test_rag.cpp` covers the stores, the chunker and the pipeline, with a fake in-process engine. Real-model tests run when these
variables are set:

| Variable | Tested with (Apache-2.0 / MIT) |
|---|---|
| `AGENT_TEST_EMBED_MODEL` | `ggml-org/bge-small-en-v1.5-Q8_0-GGUF` (`bge-small-en-v1.5-q8_0.gguf`, 384 dimensions, MIT) |
| `AGENT_TEST_RERANK_MODEL` | `gpustack/jina-reranker-v1-tiny-en-GGUF` (`jina-reranker-v1-tiny-en-Q8_0.gguf`, Apache-2.0); also `gpustack/bge-reranker-v2-m3-GGUF` Q4_K_M |
