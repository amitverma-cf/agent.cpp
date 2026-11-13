# Memory

`#include <agent-cpp/agent.hpp>`

Memory is the session's key-value store. The library uses it to save conversation history after each turn, and you can use it for your own data. Memory is a plain string-to-string store; it is not vector search (for that, see [rag.md](rag.md)).

Two backends exist:

- **In-memory** (the default): built into the library, no dependencies, nothing survives the session.
- **SQLite**: durable, from the optional `agent_store_sqlite` library. The agent CLI uses it.

You can also supply your own backend with a `StoreOps` table.

---

## MemoryConfig

Set on `Config::memory` before `init` (or passed through `open_session`'s `base` argument).

```cpp
struct MemoryConfig {
    const StoreOps *store = nullptr;   // nullptr = the built-in in-memory store
    std::string     db_path;           // backend-specific location; empty = the backend's default

    // Write-back cache tuning (used by the SQLite store; ignored by the in-memory store)
    size_t max_cache_bytes             = 16 * 1024 * 1024;   // 16 MiB
    size_t flush_dirty_threshold_bytes = 4 * 1024 * 1024;    // 4 MiB
    int    flush_interval_ms           = 500;
};
```

---

## StoreOps

A backend is a table of four functions. The library does not check the implementation; `init` fails with `InvalidConfig` when one of them is missing.

```cpp
struct StoreOps {
    using OpenFn  = Result<std::shared_ptr<void>> (*)(const MemoryConfig &config, std::string_view workspace_dir);
    using PutFn   = Result<void> (*)(void *state, std::string_view key, std::string_view value);
    using GetFn   = Result<std::string> (*)(void *state, std::string_view key);   // KeyNotFound if the key is missing
    using ClearFn = Result<void> (*)(void *state);

    OpenFn  open  = nullptr;
    PutFn   put   = nullptr;
    GetFn   get   = nullptr;
    ClearFn clear = nullptr;
};
```

- `open` runs once, from `init`. The state it returns is kept in `Session::memory_state` and passed to the other three functions.
- All four must be safe to call from several threads. Parallel graph branches run turns at the same time, and each turn writes history.

---

## In-memory store

The default when `MemoryConfig::store` is `nullptr`. It is a hash map behind a mutex. Everything is lost when the session is destroyed. It needs no configuration.

---

## SQLite store

Link the `agent-cpp::store_sqlite` target. With the installed package:

```cmake
find_package(agent-cpp 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE agent-cpp::agent agent-cpp::store_sqlite)
```

Then include its header:

```cpp
#include <agent-cpp/store_sqlite.hpp>

agent::Config base;
base.workspace_dir = "work";
base.memory.store   = agent::store::sqlite_store();
base.memory.db_path = "work/memory/db.sqlite3";   // empty = <workspace>/memory/db.sqlite3

auto session = agent::open_session(agent::engines::llama, "model.gguf", base);
```

`db_path` can be a file path, `":memory:"` (an ephemeral database that goes through the same code path), or empty. The parent folder of a file path is created when needed.

### How it works

Reads and writes go through an in-memory cache, and SQLite is updated in the background:

- **The cache is the source of truth for the caller.** A `store_memory` followed by `retrieve_memory` on the same key returns the new value at once, without touching the disk.
- **Writes are marked dirty and flushed in batches.** A background thread writes them in one transaction when `flush_dirty_threshold_bytes` of dirty data has built up, or every `flush_interval_ms`, whichever comes first.
- **A read miss loads the value from SQLite** and puts it in the cache.
- **The cache stays within `max_cache_bytes`.** Least recently used clean entries are evicted when it grows past that. Dirty entries are never evicted before they are written.
- **Closing the session flushes everything** that is still dirty before the connections close, so a write just before exit is kept.
- **Durability** comes from SQLite itself: the database uses WAL journal mode with `synchronous=NORMAL`. Once a flush has committed, the data survives a crash.

`clear_memory` clears the cache at once and runs `DELETE FROM kv` on the writer thread. It returns after the delete has finished.

A single mutex guards the cache. It is safe to call from several threads, but every call takes that lock. For large numbers of parallel graph branches this can become a bottleneck.

### Tuning

```cpp
base.memory.max_cache_bytes             = 32 * 1024 * 1024;  // a larger working set
base.memory.flush_dirty_threshold_bytes = 1 * 1024 * 1024;   // flush sooner
base.memory.flush_interval_ms           = 200;
```

---

## API

These work on a `Session`. The store is initialised by `init`, so you do not call `init_memory` yourself.

```cpp
Result<void>        init_memory(Session &session);                                       // called by init
Result<void>        store_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_memory(Session &session, std::string_view key);
Result<void>        clear_memory(Session &session);
```

### store_memory

Writes a key, replacing any previous value. Keys and values are UTF-8 strings of any length the backend accepts.

```cpp
agent::store_memory(session, "user:theme",      "dark");
agent::store_memory(session, "agent:last_task", "fibonacci computation");
```

### retrieve_memory

Reads a key. A missing key returns `ErrorCode::KeyNotFound`.

```cpp
auto r = agent::retrieve_memory(session, "user:theme");
if (r.ok)
    printf("theme: %s\n", r.value.c_str());
else if (r.error.code == agent::ErrorCode::KeyNotFound)
    printf("no theme set\n");
```

### clear_memory

Deletes every key in the store, including conversation history. With the SQLite store the data is gone from the database when the call returns.

---

## Conversation history keys

After each successful `run_turn`, the conversation is saved as JSON under two keys (see [conversation.md](conversation.md#history-persistence)):

| Key | Value |
|---|---|
| `history:<Conversation::id>` | The messages as a JSON array |
| `history_stats:<Conversation::id>` | The saved statistics (a JSON object) |

The messages look like this:

```json
[
  {"role":"system","content":"You are helpful.","tool_call_id":"","tool_calls":[]},
  {"role":"user","content":"Hello","tool_call_id":"","tool_calls":[]},
  {"role":"assistant","content":"Hi there!","tool_call_id":"","tool_calls":[]}
]
```

Tool calls appear as objects with `id`, `name` and `arguments`, and media appears as `media` entries with `path` and `mime`. Tensors are not saved.

You can read the raw JSON at any time:

```cpp
auto r = agent::retrieve_memory(session, "history:" + conv.id);
if (r.ok) printf("history: %s\n", r.value.c_str());
```

The live `Conversation::history` is what the model sees. The stored JSON is a copy, kept for inspection and recovery. `run_turn` writes it but never reads it back. To resume a conversation, call `load_conversation`.

---

## Your own keys

Any other key is yours. Do not use keys that start with `history:` or `history_stats:`; those belong to the conversation saves.

```cpp
agent::store_memory(session, "cfg:max_depth", "5");
agent::store_memory(session, "result:plan_v3", plan_text);
agent::store_memory(session, "milestone:research_done", "1");

auto depth = agent::retrieve_memory(session, "cfg:max_depth");
```

Values are plain strings. Store JSON text yourself if you need structure.

---

## Related pages

- [session.md](session.md): `Config::memory` and session setup
- [conversation.md](conversation.md): history persistence and `load_conversation`
- [rag.md](rag.md): vector search, a separate mechanism from this store
- [data-sources.md](data-sources.md): other sources of data for the model
