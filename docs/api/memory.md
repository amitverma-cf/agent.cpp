# Memory

`#include <agent-cpp/agent.hpp>`

`libagent` has a pluggable key-value store, selected through `MemoryConfig::store` (a `StoreOps` function-pointer table). It persists conversation history after every turn, and you can use it directly to store any application data. (For RAG/vector search, see [data-sources.md](data-sources.md); that's a separate mechanism, not part of this KV store.)

The core library ships only an **in-memory** store, which is the default and has no dependencies; nothing persists past the `Session`. Durable storage comes from the optional **`agent_store_sqlite`** library (used by the agent CLI), or from your own `StoreOps`.

---

## MemoryConfig

```cpp
struct MemoryConfig {
    const StoreOps *store = nullptr;   // nullptr = built-in in-memory store
    std::string     db_path;           // backend-specific; empty = backend default

    // Write-back cache tuning (used by the SQLite store)
    size_t max_cache_bytes             = 16 * 1024 * 1024;  // 16 MiB
    size_t flush_dirty_threshold_bytes = 4 * 1024 * 1024;   // 4 MiB
    int    flush_interval_ms           = 500;
};
```

## StoreOps

```cpp
struct StoreOps {
    using OpenFn  = Result<std::shared_ptr<void>> (*)(const MemoryConfig &config, std::string_view workspace_dir);
    using PutFn   = Result<void> (*)(void *state, std::string_view key, std::string_view value);
    using GetFn   = Result<std::string> (*)(void *state, std::string_view key);   // KeyNotFound if missing
    using ClearFn = Result<void> (*)(void *state);

    OpenFn open; PutFn put; GetFn get; ClearFn clear;
};
```

`open` runs once from `init()`. The state it returns is kept in `Session::memory_state` and passed to every other call. All four functions must be set, and they must be safe to call from multiple threads (parallel graph branches run turns concurrently).

---

## SQLite store (`agent_store_sqlite`)

```cpp
#include <agent-cpp/store_sqlite.hpp>   // link agent_store_sqlite

cfg.memory = {
    .store   = agent::store::sqlite_store(),
    .db_path = ".workspace/memory/db.sqlite3",   // empty = <workspace>/memory/db.sqlite3
};
```

`db_path = ":memory:"` gives an ephemeral SQLite database through the same code path.

### How it works: a bounded write-back cache in front of SQLite

`store_memory`/`retrieve_memory` never talk to SQLite directly on the calling thread. Instead:

- **All reads and writes hit an in-memory cache first.** The cache is the single source of truth from the caller's perspective — a `store_memory` immediately followed by a `retrieve_memory` for the same key always sees the value you just stored, with no round-trip to disk in between.
- **Writes mark entries dirty.** A single background writer thread flushes dirty entries to SQLite whenever `flush_dirty_threshold_bytes` of dirty data has accumulated, or every `flush_interval_ms`, whichever comes first — batching many pending writes into one SQLite transaction instead of committing on every single `store_memory` call, which is the actual throughput win here.
- **On a cache miss**, `retrieve_memory` reads through to SQLite and populates the cache with the result.
- **Once the cache exceeds `max_cache_bytes`**, least-recently-used *clean* (already-flushed) entries are evicted to bring it back under budget. Dirty entries are never evicted — eviction can't drop data that hasn't been durably written yet.
- **On `Session` destruction**, the writer thread performs one final flush of everything still dirty and joins before the connections close, so a `store_memory` followed immediately by process exit doesn't lose that write.

The durability guarantee underneath all of this is standard SQLite: the database is opened in WAL journal mode with `synchronous=NORMAL`, which is the standard safe-and-fast combination — once a flush commits, that data survives a crash. The cache is a performance layer in front of that guarantee, not a replacement for it.

```cpp
cfg.memory = {
    .store   = agent::store::sqlite_store(),
    .db_path = ".workspace/memory/db.sqlite3",   // parent directory created automatically
    .max_cache_bytes = 32 * 1024 * 1024,         // tune for your working-set size
};
```

**Concurrency**: the cache is protected by a single mutex, so `store_memory`/`retrieve_memory` are safe to call concurrently from multiple threads — including from parallel graph branches (see [graph.md](graph.md#parallel-execution)), whose turns already call `store_memory` after every turn via the conversation-history auto-persist below. This is built thread-safe uniformly rather than assuming a particular access pattern. That single mutex is also a shared serialization point across every concurrent `store_memory`/`retrieve_memory` call — fine at the default `Config::max_parallel_subagents = 4`, but worth keeping in mind as a scaling ceiling if that value is raised substantially for a larger deployment.

---

## API

### init_memory

```cpp
Result<void> init_memory(Session &session);
```

Initialises the memory backend. Called automatically by `init()`. You do not need to call this manually.

### store_memory

```cpp
Result<void> store_memory(Session &session, std::string_view key, std::string_view value);
```

Write (upsert) a key-value pair. Keys and values are arbitrary UTF-8 strings.

```cpp
agent::store_memory(session, "user:theme",       "dark");
agent::store_memory(session, "agent:last_task",  "fibonacci computation");
agent::store_memory(session, "cache:fib_result", "0,1,1,2,3,5,8,13,21,34");
```

### retrieve_memory

```cpp
Result<std::string> retrieve_memory(Session &session, std::string_view key);
```

Read a value by key. Returns `ErrorCode::KeyNotFound` if the key does not exist.

```cpp
auto r = agent::retrieve_memory(session, "user:theme");
if (r.ok)
    printf("theme: %s\n", r.value.c_str());
else if (r.error.code == agent::ErrorCode::KeyNotFound)
    printf("no theme set\n");
```

### clear_memory

```cpp
Result<void> clear_memory(Session &session);
```

Delete all key-value pairs — clears the in-memory cache immediately, and hands the actual `DELETE FROM kv` off to the background writer thread, blocking until it completes. When `clear_memory` returns, the data is actually gone (not "eventually will be").

---

## Conversation history storage

After every successful `run_turn`, the conversation history and stats are serialised to JSON via `save_flow_memory` and stored under `"history:<FlowMemory::id>"` (history) and `"history_stats:<FlowMemory::id>"` (`FlowMemory::stats`). This allows histories to be inspected, exported, or resumed.

`run_turn` only ever writes these keys — it never reads them back. To resume a conversation after a process restart, call `load_flow_memory(session, memory)` explicitly (see [conversation.md](conversation.md#history-persistence)); `save_flow_memory`/`load_flow_memory` can also be called directly outside `run_turn`.

The serialised format is a JSON array of objects:

```json
[
  {"role":"system","content":"You are helpful.","tool_call_id":"","tool_calls":[]},
  {"role":"user","content":"Hello","tool_call_id":"","tool_calls":[]},
  {"role":"assistant","content":"Hi there!","tool_call_id":"","tool_calls":[]}
]
```

You can read it back at any time:

```cpp
auto r = agent::retrieve_memory(session, "history:" + conv.id);
if (r.ok) printf("history JSON: %s\n", r.value.c_str());
```

This persisted JSON is separate from the in-memory `FlowMemory::history`. The live history is the authoritative source for what the model sees; the persisted JSON (now durable via the write-back cache + SQLite above) is a snapshot for observability and recovery.

---

## Custom storage keys

Use any string key that does not start with `"history:"` or `"history_stats:"` (reserved for auto-persist):

```cpp
// Store agent configuration
agent::store_memory(session, "cfg:max_depth", "5");

// Store computed results
agent::store_memory(session, "result:plan_v3", plan_text);

// Tag a completed milestone
agent::store_memory(session, "milestone:research_done", "1");

// Read back
auto cfg_r = agent::retrieve_memory(session, "cfg:max_depth");
auto plan_r = agent::retrieve_memory(session, "result:plan_v3");
```
