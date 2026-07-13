# Data Sources

`#include <agent-cpp/agent.hpp>`

`DataSource` is a pluggable, open-ended registry for backends a `Flow` needs beyond the library's own conversation-history persistence — RAG/vector search, SQLite, GraphRAG, arbitrary files, or anything else. It is deliberately separate from `MemoryConfig`/`MemoryProvider` (see [memory.md](memory.md)), which stays a plain key-value store for the library's internal use.

---

## Why not just use `MemoryConfig`?

`MemoryConfig`'s `store_memory`/`retrieve_memory` interface is a `get(key)`/`set(key, value)` key-value store. That is exactly right for persisting conversation history (`history:<id>`), but wrong for:

- **SQL** — needs structured queries, not key lookup.
- **Vector search (RAG)** — needs similarity search over embeddings.
- **GraphRAG** — needs graph traversal.

Bending a KV interface to also cover these would produce a lowest-common-denominator API that fits none of them well. `DataSource` instead wraps each backend behind one opaque query function, and stays out of the way of whatever query language or protocol that backend actually needs internally.

---

## DataSource

```cpp
struct DataSource {
    std::string name;
    std::shared_ptr<void> state;

    using QueryFn = Result<std::string> (*)(void *state, std::string_view query, void *user_data);
    QueryFn query = nullptr;

    void *user_data = nullptr;
};
```

| Field | Description |
|---|---|
| `name` | Identifier used to look the source up via `query_data_source` and as the default tool name from `bind_data_source_tool`. |
| `state` | Opaque backend handle (a `sqlite3*`, a vector index, a graph store, ...), kept alive via `shared_ptr<void>` — same ownership pattern as `Session::provider_state`. |
| `query` | Backend-specific query entrypoint. The return format (JSON, plain text, ...) is entirely up to the backend. |
| `user_data` | Extra context passed through to `query`, if needed beyond `state`. |

---

## register_data_source / query_data_source

```cpp
Result<void> register_data_source(Session &session, DataSource ds);
Result<std::string> query_data_source(Session &session, std::string_view name, std::string_view query);
```

`register_data_source` stores a `DataSource` in `session.data_sources`, keyed by `ds.name`. `query_data_source` looks it up by name and calls its `query` function. Use these directly when you want to query a data source from your own application code (e.g. from a `FlowState::context_provider`, see [fsm.md](fsm.md#context_provider)).

```cpp
struct FakeVectorStore {
    std::vector<std::pair<std::string, std::string>> docs; // (text, embedding-ish key)
};

agent::Result<std::string> query_fake_vectors(void *state, std::string_view query, void *) {
    auto *store = static_cast<FakeVectorStore *>(state);
    for (const auto &[text, key] : store->docs)
        if (key.find(query) != std::string_view::npos)
            return agent::ok(text);
    return agent::ok(std::string("no match"));
}

auto store = std::make_shared<FakeVectorStore>();
store->docs.push_back({"To install, run `cmake -B build`.", "install"});

agent::register_data_source(session, agent::DataSource{
    .name  = "product_docs",
    .state = store,
    .query = query_fake_vectors,
});

auto r = agent::query_data_source(session, "product_docs", "install");
if (r.ok) printf("%s\n", r.value.c_str());
```

---

## bind_data_source_tool

```cpp
Tool bind_data_source_tool(Session &session, std::shared_ptr<DataSource> source,
                           std::string_view description, std::string_view parameter_schema);
```

Wraps a `DataSource` as a regular `Tool`, so the model can call it through the exact same tool-calling path as everything else — no new concept on the LLM side. The tool's `name` is `source->name`. It retains the `shared_ptr<DataSource>` on `Session::retained_data_sources`, so the tool's `user_data` can't outlive the data it points to.

```cpp
auto source = std::make_shared<agent::DataSource>(agent::DataSource{
    .name  = "product_docs",
    .state = store,
    .query = query_fake_vectors,
});

agent::Tool docs_tool = agent::bind_data_source_tool(
    session, source,
    "Search product documentation for relevant text.",
    R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})");

session.config.tools.push_back(docs_tool);
agent::rebuild_tool_index(session);
```

`FlowState::allowed_tools` (see [fsm.md](fsm.md#allowed_tools)) scopes which data-source tools are visible in which state, for free — the same allowlist mechanism used for every other tool.

---

## Automatic context injection (classic RAG)

For the "retrieve relevant chunks, stuff them into context" pattern — without requiring the model to make an explicit tool call — use `FlowState::context_provider` instead. It's called before `on_transition`, and its return value is folded into that state's system prompt for that turn only. See [fsm.md](fsm.md#context_provider) for the full example.

```cpp
std::string_view inject_docs_context(agent::Session &session, agent::FlowMemory &, void *) {
    static thread_local std::string buf;
    auto res = agent::query_data_source(session, "product_docs", "installation steps");
    if (!res.ok) return {};
    buf = "Relevant docs:\n" + res.value;
    return buf;
}

agent::FlowState answer_state{
    .state_id         = "answer",
    .context_provider = inject_docs_context,
    .on_transition    = answer_fn,
};
```

---

## Choosing model-initiated vs. automatic

| | `bind_data_source_tool` (model-initiated) | `context_provider` (automatic) |
|---|---|---|
| Who decides to query | The model, via a tool call | You, unconditionally every time the state runs |
| Best for | Backends the model should reason about ("should I look this up?") | Always-relevant context the model shouldn't have to ask for |
| Scoping | `FlowState::allowed_tools` | Per-state, since `context_provider` is a `FlowState` field |

Both can be used together — e.g. always inject a short summary via `context_provider`, and also expose a tool for deeper on-demand queries.

---

## Built-in: sqlite-vec (RAG / vector search)

The `FakeVectorStore` above was illustrative — for real vector search, agent.cpp ships a `sqlite-vec`-backed `DataSource` out of the box (`sqlite3` + the `sqlite-vec` extension, statically compiled in, vendored under `vendor/sqlite`/`vendor/sqlite-vec`). It's queried directly on disk every time via `vec0`'s own nearest-neighbor search — **never** routed through the bounded write-back cache that `MemoryConfig`/`store_memory` uses (see [memory.md](memory.md)), since that cache is sized and designed for small conversation-history-shaped data, not a growing vector index.

```cpp
Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(
    std::string name, std::string db_path, int dimensions);

Result<void> sqlite_vec_insert(DataSource &source, std::string_view text,
                               std::span<const float> embedding);
```

**The library does not compute embeddings itself.** You run your own embedding model (local or via an API) to turn text into a vector; agent.cpp only stores and searches those vectors.

```cpp
// One-time setup: dimensions must match your embedding model's output size.
auto source_res = agent::make_sqlite_vec_data_source(
    "product_docs", ".workspace/memory/vec.sqlite3", /*dimensions=*/384);
if (!source_res.ok) { /* handle error */ }
auto source = source_res.value;

// Ingest: compute an embedding yourself, then insert (text, vector) pairs.
std::vector<float> embedding = my_embed_function("To install, run cmake -B build.");
agent::sqlite_vec_insert(*source, "To install, run cmake -B build.", embedding);
```

`DataSource::query`'s payload for this backend is a small JSON object: `{"vector": [f32, ...], "top_k": N}`, matching the dimensionality the source was created with. It returns a JSON array ordered nearest-first: `[{"text": "...", "distance": 0.12}, ...]`.

```cpp
std::vector<float> query_vec = my_embed_function("how do I build this?");
std::string payload = R"({"vector": [)" + join_floats(query_vec) + R"(], "top_k": 3})";
auto res = agent::query_data_source(session, "product_docs", payload);
```

Wire it up to a `Flow` exactly like any other `DataSource` — `bind_data_source_tool` for model-initiated lookups, or a `context_provider` that embeds the user's message and injects the top matches automatically (the classic RAG pattern). Both need you to call your embedding model first; agent.cpp's part starts at "here's a vector, find the nearest ones."
