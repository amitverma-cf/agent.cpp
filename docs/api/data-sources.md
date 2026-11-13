# Data sources

`#include <agent-cpp/agent.hpp>`

A `DataSource` is a named, pluggable retrieval backend: a vector index, a SQL database, a graph store, a set of files, or anything else
the agent needs beyond its own conversation history. Each backend is wrapped behind one query function, and the agent reaches it in one of
two ways: as a tool the model calls, or as context injected into a graph node before its turn.

Data sources are separate from `MemoryConfig` / `StoreOps` ([memory.md](memory.md)). The memory store is a plain key-value store that the
library uses for conversation history. It is not meant for SQL, vector search or graph traversal, and a key-value interface would fit none of
them well. A `DataSource` leaves the query language to the backend.

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
| `name` | Identifier used by `query_data_source`. It is also the tool name when the source is bound with `bind_data_source_tool`. |
| `state` | The backend handle (a `sqlite3*`, an index, a graph store, ...), kept alive by `shared_ptr<void>`. |
| `query` | The backend's query function. The return format (JSON, plain text, ...) is up to the backend. |
| `user_data` | Extra context passed to `query`, when `state` is not enough. |

## register_data_source / query_data_source

```cpp
Result<void> register_data_source(Session &session, DataSource ds);
Result<std::string> query_data_source(Session &session, std::string_view name, std::string_view query);
```

`register_data_source` stores the source in `session.data_sources`, keyed by `ds.name`. `query_data_source` looks it up and calls its
`query` function. Use these from your own code, for example in a graph node's `context_provider` ([graph.md](graph.md#per-run-context-context_provider)).

Like `session.config.tools`, `session.data_sources` is not mutex-guarded. Register sources during setup, not while a turn runs on another
thread. `query_data_source` is read-only and safe from any thread once registration is done.

```cpp
struct FakeVectorStore {
    std::vector<std::pair<std::string, std::string>> docs; // (text, key)
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

## bind_data_source_tool

```cpp
Tool bind_data_source_tool(Session &session, std::shared_ptr<DataSource> source,
                           std::string_view description, std::string_view parameter_schema);
```

Wraps a source as a regular `Tool`, so the model calls it through the same tool path as every other tool. The tool's name is
`source->name`. The session keeps the `shared_ptr` in `Session::retained_data_sources`, so the tool cannot outlive its data.

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

A graph Agent node's `allowed_tools` ([graph.md](graph.md#system-prompt-and-tool-allowlist)) decides which nodes can see the tool, the same
as for any other tool.

## Automatic context injection

For retrieval that should always run, without a tool call, set a graph Agent node's `context_provider`. It runs before each turn of that
node, and its text is added to the node's system prompt for that turn only. See [graph.md](graph.md#per-run-context-context_provider).

```cpp
agent::Result<std::string> inject_docs_context(agent::Session &session, const agent::GraphState &state, void *) {
    auto res = agent::query_data_source(session, "product_docs", agent::get_state(state, "question"));
    if (!res.ok) return agent::ok(std::string());   // answer without the extra context
    return agent::ok("Relevant docs:\n" + res.value);
}

agent::GraphNode answer{.name = "answer", .kind = agent::GraphNodeKind::Agent,
                        .system_prompt = "Answer using the provided docs context.",
                        .input_key = "question", .output_key = "answer",
                        .context_provider = inject_docs_context};
```

## Choosing between a tool and context injection

| | `bind_data_source_tool` (model decides) | `context_provider` (always runs) |
|---|---|---|
| Who decides to query | The model, through a tool call | Your code, on every run of the node |
| Best for | Backends the model should reason about ("should I look this up?") | Context the model should always have |
| Scope | `GraphNode::allowed_tools` | The node that sets `context_provider` |

The two can be combined: inject a short summary every time, and also expose a tool for deeper lookups.

## sqlite-vec (RAG / vector search), in `agent_store_sqlite`

The `FakeVectorStore` above is only an illustration. For real vector search, the optional `agent_store_sqlite` library provides a
sqlite-vec backed source: `#include <agent-cpp/store_sqlite.hpp>` and link `agent-cpp::store_sqlite`. It uses the `sqlite3` and
`sqlite-vec` code vendored in the repository. `libagent` has no SQLite dependency.

```cpp
namespace agent {
Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(std::string name, std::string db_path, int dimensions);
Result<void> sqlite_vec_insert(DataSource &source, std::string_view text, std::span<const float> embedding);
}
```

- The source is queried from disk on every call with `vec0`'s nearest-neighbour search. It does not go through the bounded write-back cache
  that `store_memory` uses ([memory.md](memory.md)), since that cache is sized for conversation-history data, not a growing index.
- `dimensions` must match the output size of your embedding model.
- **The library does not compute embeddings.** Run your own embedding model, locally or through an API, and pass the vectors in.

```cpp
// One-time setup.
auto source_res = agent::make_sqlite_vec_data_source("product_docs", ".workspace/memory/vec.sqlite3", /*dimensions=*/384);
if (!source_res.ok) { /* handle error */ }
auto source = source_res.value;

// Ingest: embed each chunk yourself, then insert the (text, vector) pair.
std::vector<float> embedding = my_embed_function("To install, run cmake -B build.");
agent::sqlite_vec_insert(*source, "To install, run cmake -B build.", embedding);
```

The query payload is a JSON object, `{"vector": [f32, ...], "top_k": N}`, with as many floats as the source has dimensions. `top_k`
defaults to 5. The result is a JSON array, nearest first: `[{"text": "...", "distance": 0.12}, ...]`.

```cpp
// Builds {"vector":[...],"top_k":3}.
std::string vector_query(std::span<const float> v, int top_k) {
    std::string out = R"({"vector":[)";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ',';
        out += std::to_string(v[i]);
    }
    return out + "],\"top_k\":" + std::to_string(top_k) + "}";
}

std::vector<float> query_vec = my_embed_function("how do I build this?");
auto res = agent::query_data_source(session, "product_docs", vector_query(query_vec, 3));
```

Once registered, wire the source up like any other: `bind_data_source_tool` for model-initiated lookups, or a `context_provider` that
embeds the user's message and injects the best matches. Either way, your code calls the embedding model first. The library's part starts
at "here is a vector, find the nearest ones."

For retrieval with chunking, embeddings, reranking and hybrid search in one pipeline, use the RAG API instead: [rag.md](rag.md).
