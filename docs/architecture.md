# Architecture

agent.cpp is structured around three core principles that inform every design decision: no vtable dispatch, no hot-path heap allocation, and no exceptions. These constraints make it suitable for resource-constrained edge hardware where a GC pause or unexpected allocation would be fatal.

## Core principles

- **No vtables.** Provider and memory backends are selected via data-oriented function-pointer tables (`ProviderOps`, `StoreOps`). Zero overhead beyond a single pointer dereference.
- **No hot-path allocation.** A bump-pointer `Arena` owns all per-turn storage. `MessageView`, `TensorView`, `ToolCallView` are zero-copy views into it. The arena resets in O(1) at the start of each turn.
- **No exceptions.** All fallible operations return `Result<T>`. Callers branch on `.ok`.
- **Single header.** `agent.hpp` is the entire public API surface. Backend headers (`llama.h`, `httplib.h`, `sqlite3.h`) are confined to their implementation files.

---

## System layers

```
+---------------------------------------------------------------------+
|                        User Application                              |
|   Graph  GraphRun  FlowMemory  EventHooks                            |
+-----------------------------+---------------------------------------+
                              | calls
+-----------------------------v---------------------------------------+
|                      Orchestration Layer                             |
|   run_turn                  run_graph / resume_graph                 |
|   compress_context          parallel branches, checkpoints           |
+------+-----------------------------------------------+--------------+
       | infer                                          | execute_tool
+------v------------------+         +------------------v-----------+
|     Provider Layer       |         |       Tools Layer            |
|  ProviderOps[]           |         |  ToolsOps  tool_by_name      |
|  Mock  Engine (plugins)  |         |  filesystem  terminal        |
|  llama / onnx / ...      |         |  compressor  skills          |
+--------------------------+         |  permission gate (Allow/     |
                                     |  Ask/Deny, checked pre-call) |
                                     +------------------------------+
         |                                        |
+--------v----------------------------------------v------------------+
|                           Session                                    |
|  Config  worker_arenas  provider_state  memory_state  data_sources   |
|  tool_by_name  tools_system_prompt  skills  log_file  stats          |
+---------------------------------------------------------------------+
```

---

## Memory model

### Arena allocator

Every `Session` owns one `Arena`, a bump allocator whose first block is allocated at `init()` (default `context_window * 8 + 2 MiB`, override with `Config::arena_capacity`). It grows in chunks if a turn needs more, so it never fails for size; the next `reset()` replaces the chunks with one block sized to the peak, and later turns of that size allocate nothing.

The arena is used only inside `run_turn` to construct zero-copy view spans before passing them to the provider. It resets at the top of every inference iteration, so no turn data leaks into the next. The reset is O(1) while the turn fit in the first block (a single integer assignment).

```
Session::Arena slab  (fixed, allocated once at init)
  |
  +-- [reset each loop iteration: offset = 0]
  |
  +-- MessageView[N]     (zero-copy into history strings)
  +-- ToolCallView[M]    (zero-copy into ToolCall strings)
  +-- TensorView[K]      (zero-copy into TensorData buffers)
```

Owned data — `Message::content`, `ToolCall::arguments`, history entries — live in `FlowMemory::history` on the heap. They are never placed in the arena and persist across turns.

### Conversation history lifetime

`FlowMemory::history` is a `std::vector<Message>` whose elements are stable. `std::string` members inside `Message` do not move when new messages are pushed (because `Message` owns its strings). Each `Message::token_count` is a mutable cache seeded to -1 and populated lazily by `count_message_tokens`.

### Session self-reference cell

The context compressor tool needs a `Session*` at callback time, but tools store `void* user_data` set at registration. If the session ever moves after registration, a stored raw pointer would dangle. The solution is an indirection cell:

```
session.session_ptr_cell  -->  shared_ptr<Session*>  (heap block)
                                        |
tool.user_data  ------------------------+  (Session** into the same block)

Before every tool dispatch:
  *session.session_ptr_cell = &session;   // refresh -- safe after any move
```

---

## Dispatch tables

Provider, memory, and tools subsystems are resolved through `constexpr` arrays of function pointers:

```cpp
ProviderOps kProviderOps[] = {
    { /* Mock     */ init_mock,    infer_mock,    ... },
    { /* Engine   */ init_engine,  infer_engine,  ... },   // adapts engine_abi.h plugins
};

// Memory: Config::memory.store points at a StoreOps table (nullptr = in-memory default).
constexpr StoreOps kInMemoryStore{ open_in_memory, put_in_memory, get_in_memory, clear_in_memory };
```

`find_provider_ops(AiProvider::Engine)` does a 2-element linear scan. Branch predictor handles it in a single cycle. No heap, no map, no virtual dispatch.

---

## Conversation turn data flow

```
run_turn(session, memory, user_prompt)
 |
 +-- append user Message to memory.history
 |
 +-- auto-compress if total_tokens > 70% * context_window  [optional]
 |
 +-- loop:
      |
      +-- prune_history_state()          O(n) single pass + single erase
      |
      +-- current_arena(session).reset()      O(1)  (thread's own arena slot)
      |
      +-- build MessageView[] from history   zero-copy
      |
      +-- trigger_event(OnInferenceSubmit)
      |
      +-- infer()  -->  ProviderOps::infer
      |
      +-- trigger_event(OnInferenceComplete)
      |
      +-- copy response into owned Message, push to history
      |
      +-- if tool_calls present:
      |    |
      |    +-- guard: tool_rounds < max_tool_call_rounds
      |    |
      |    +-- for each tool call:
      |         +-- trigger_event(OnToolCall)
      |         +-- fire ToolHook::Pre callbacks
      |         +-- execute_tool()  -->  O(1) tool_by_name lookup
      |         |     -->  permission_check (Allow/Ask/Deny), if Config::permission_check is set
      |         |     -->  callback
      |         +-- fire ToolHook::Post callbacks
      |         +-- trigger_event(OnToolResult)
      |         +-- push tool Message to history
      |
      +-- if no tool calls  -->  break
 |
 +-- save_flow_memory()  -->  store_memory("history:<id>", JSON) + store_memory("history_stats:<id>", JSON)
 |
 +-- return owned std::string   (safe: copied before arena reset)
```

---

## Graph runtime

Multi-step workflows are graphs (`src/core/graph/`, [api/graph.md](api/graph.md)): named nodes (Agent, Function, Tool, Parallel, Loop) joined by static or routed edges, with a `GraphState` of scoped key/value pairs plus one `FlowMemory` per Agent conversation.

```
run_graph(session, graph, run)
  load checkpoint:<thread>:latest (per-thread values + Agent histories), user:/app: keys, caller seed
  node = graph.entry
  loop:
    trigger_event(OnStateTransition, node.name)
    Agent node:
      memory = state.histories[node.history_key or node.name]
      1. system prompt = context_provider(state) + "\n" + node.system_prompt
         installed into memory.history[0] for this run only (saved / restored)
      2. ambient::allowed_tools = node.allowed_tools   (thread_local; restored after)
         ambient::active_hooks  = [event-log Pre] + node.hooks + [event-log Post]
      3. run_turn(session, memory, state[input_key] or prompt)
      4. state[output_key] = reply
    Function / Tool / Loop node: call it
    Parallel node: run each branch on a copy of the state (threads on concurrent engines),
                   join (distinct keys / histories), trigger_event(OnSubAgentComplete) per branch
    log writes, persist user:/app: keys, checkpoint (per GraphDurability)
    node = edge / router target (kGraphEnd -> exit_to or done)
```

The save/restore pattern guarantees that each Agent node gets exactly the tools, hooks and system prompt it declared, with no global mutation surviving into the next node; nodes sharing a `history_key` continue one conversation under their own prompts.

---

## Concurrency

A graph `Parallel` node fans out to its branches. How they run depends on the provider:

```
run_parallel_node()
  if providers::supports_parallel(session) && top-level fan-out:   // engine flag AGENT_ENGINE_FLAG_CONCURRENT
    for each batch of up to max_parallel_subagents branches:
      spawn one std::thread per branch (ambient::worker_arena_index = slot), each running the branch to its end
      join all threads
  else:
    run each branch to its end, one after another
  join: merge branch writes and histories; fail on a key written by two branches
```

All branches share the same `Session`. Two different mechanisms are used, because each provider family has a genuinely different constraint:

| Provider | Mechanism | Why |
|---|---|---|
| `Mock`, non-concurrent engines | Sequential | No batching support / not worth the complexity for these. |
| Concurrent engines (llama) | Threads + continuous batching inside the plugin (see below) | A single `llama_context` can only run one `llama_decode` at a time regardless of thread count — threads alone buy nothing; the actual parallelism has to come from batching multiple sequences into one decode call. |

Making any of this safe required removing all of `Session`'s ambient mutable "current turn" state and replacing it with `thread_local` variables in the `agent::ambient` namespace (`src/core/executor/ambient_turn.cpp`): which `FlowMemory`/hooks/tool-allowlist the current thread's `execute_tool` call is running under (`ambient::active_memory`, `ambient::active_hooks`, `ambient::allowed_tools`), which `Arena` slot to use (`ambient::worker_arena_index`, resolved via `ambient::current_arena(session)`), and the cached `simdjson::ondemand::parser` (`ambient::json_parser`). A single thread behaves identically to a shared-field design; concurrent threads get automatic isolation for free, no locking needed on any of it. What *is* still shared and mutex-guarded: `Session::stats` (`Session::stats_mutex`), the log file (`Session::log_mutex`), and event hook dispatch (`Session::event_mutex`), since these genuinely accumulate/fire across every concurrently-running turn.

### Continuous batching

`engines/llama/llama_engine.cpp`'s `Batcher` owns the single `llama_model`/`llama_context` for an engine instance and multiplexes every concurrently-active generation onto it via llama.cpp's native multi-sequence batching (`llama_batch` entries tagged with distinct `llama_seq_id`s, context created with `n_seq_max = Config::max_parallel_subagents`):

- Each generation's **prompt prefill** is decoded on its own (split into `n_batch`-sized `llama_decode` calls) when its sequence is acquired (not batched across sequences — prefill is comparatively rare and cheap relative to the token-by-token loop, and batching variable-length prompts together is a lot more complexity for little benefit).
- Each generation's **next-token step**, from then on, goes through `Batcher::next_token()`: the first caller in a round becomes its leader, briefly releases the lock so other concurrently-running generations can join (a ~2ms debounce window), then performs **one** `llama_decode` covering every joined sequence's next token and wakes every waiter with its own sampled result. A round with only one active sequence degenerates to exactly one decode call per token — today's behavior — so continuous batching only kicks in when there's actually something to batch.
- Each generation owns its own `llama_sampler` chain (its own temperature, its own RNG) built fresh per call — nothing sampling-related is shared, so no locking is needed there; only the decode call itself is serialized through the `Batcher`'s mutex. (An earlier version also installed a GBNF grammar sampler here to force tool-call output into valid JSON; it was removed after a real-model crash traced the vendored `llama_sampler_init_grammar_lazy_patterns`/`llama_grammar_init_impl` path to an uncaught `std::regex_error` inside llama.cpp itself. Tool-call JSON is extracted unconstrained via bracket-matching instead, as it always was before that attempt.)
- Sequence slots are a fixed-size free list of `[0, n_seq_max)`; `acquire_seq()`/`release_seq()` check them out/in per generation, failing cleanly with an error if the pool is exhausted rather than corrupting state.

This is deliberately not built on C++20 coroutines: an earlier design considered making `run_turn`/`infer` themselves suspendable so parallel branches could interleave partial generations directly, but that would have meant changing the public `ProviderInferFn` contract for every provider for a benefit only the llama engine needs. The thread-plus-shared-engine design gets the same real batching benefit — one shared `llama_decode` per round across every active sequence — without touching any public signature.

---

## Context management

### Sliding-window pruning

Runs automatically before every `infer` call inside `run_turn`. Target budget: `context_window - max_tokens - 100`. Algorithm:

1. Single forward pass accumulates token totals for all messages.
2. Starting from `fixed_start` (index 1 if system message present, else 0), advance a drop window.
3. Tool call groups (one `assistant` message + its consecutive `tool` replies) are always dropped atomically — partial drops would leave orphaned tool results.
4. A single `history.erase(begin+drop_from, begin+drop_to)` removes the selected range.

Token counts come from `ProviderOps::count_tokens` when available, otherwise fall back to `(chars + 3) / 4`.

### LLM compression

When history token total exceeds 70% of `context_window` and history has more than 6 messages, `compress_context` is called before the turn:

1. Builds a summarization prompt from all messages older than `keep_recent` (default 6).
2. Sends the prompt to the same provider for a summary.
3. Replaces the old messages with one `[Compressed history]: <summary>` message.
4. Falls back gracefully if the summary call fails — the pruner handles any remaining overflow.

### KV cache

`clear_provider_kv_cache(session)` delegates to `ProviderOps::clear_kv_cache`. For llama.cpp this clears the KV cache for every active sequence in the shared `llama_context`, which discards all cached key-value pairs. Useful before switching to work that represents a significant context shift, or to reclaim VRAM between long sessions.

---

## Provider implementations

| File | AiProvider | Key dependency |
|---|---|---|
| `provider_ops.cpp` | Mock | — |
| `engine_provider.cpp` | Engine | `engine_abi.h` (plugins such as `engines/llama/` and `engines/onnx/` link their own backends; the core never links llama.cpp or ONNX Runtime) |

All provider state is heap-allocated and type-erased as `shared_ptr<void>` in `Session::provider_state`. The provider's destructor handles cleanup.

---

## Memory backend

`MemoryConfig::store` selects a `StoreOps` table. The core ships only `in_memory_store.cpp` (the default). The optional `agent_store_sqlite` library (`tools/store-sqlite/`) provides `sqlite_store()`: a bounded write-back cache (`unordered_map` + LRU list, protected by one mutex) in front of a SQLite database (WAL journal mode, `synchronous=NORMAL`). Reads and writes always hit the cache first; a background writer thread flushes dirty entries to SQLite on a size-or-time trigger and evicts LRU clean entries once over the configured byte budget. `db_path = ":memory:"` (opened with SQLite's URI shared-cache form so both the writer and reader connections see the same data) gives the old standalone InMemory provider's ephemeral behavior through the same code path. See [docs/api/memory.md](api/memory.md) for the full design.

Keys are arbitrary UTF-8 strings. Conversation history is serialised as a JSON array and stored under `"history:<FlowMemory::id>"` after every turn.

RAG/vector search is a separate concern, never routed through this cache — see `tools/store-sqlite/src/sqlite_vec_data_source.cpp` and [docs/api/data-sources.md](api/data-sources.md).

---

## Security model

### Sandbox

When `Config::sandbox_filesystem = true` (the default), filesystem tools resolve the target path through `hal::resolve_under_workspace` (`src/hal/path.cpp`): `std::filesystem::weakly_canonical` followed by a prefix-path comparison against the canonicalized `workspace_dir`. Any path that resolves outside — whether via absolute path, `..` traversal, or a symlink that escapes the workspace — is rejected with `ErrorCode::SandboxViolation` before any filesystem operation occurs. Because canonicalization follows symlinks, this is stronger than a purely lexical check.

The workspace root itself is additionally protected: `delete_path` compares canonical paths and rejects the workspace root regardless of sandbox mode. SQL identifiers used internally (e.g. building queries) are checked with `hal::is_safe_identifier`.

### Shell injection

`workspace_dir` is validated at `init()` against a set of shell metacharacters: `"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, `\n`, `\r`. If any are present, `init()` returns `InvalidConfig` without creating any workspace directories.

On POSIX, the terminal tool embeds `workspace_dir` inside single-quotes with proper escape for embedded single-quotes (`'` → `'\''`). On Windows it embeds in double-quotes (safe because `"` is blocked by the metacharacter check).

### Permission gate

`tools_ops.cpp`'s `default_execute` checks `Config::permission_check` immediately before invoking any `Tool::callback` — native tools, `bind_data_source_tool`-backed tools, and the `use_skill` tool alike. A `Deny`, or an `Ask` that's declined (or has no `Config::permission_prompt` set to resolve it), returns `ErrorCode::PermissionDenied` without the callback ever running. `permission_check = nullptr` (the default) allows everything, unchanged from before this existed.

This is a policy gate, not sandboxing: it decides whether a call is *attempted*, and composes with the filesystem sandbox above (which constrains *what paths* a filesystem tool can touch once it runs). Neither mechanism can stop a tool's own native code from doing something outside either check's view — a `Tool::callback` is a plain function pointer with the full privileges of the host process. There is no additional runtime isolation layer beneath the permission gate; if a tool's code isn't fully trusted, the gate (and reviewing that code before registering it) is the mitigation this library provides.

---

## File logging

When `Config::enable_file_logging = true`, `init()` creates `workspace_dir/logs/agent_YYYY-MM-DD_HH-MM-SS.log` using UTC time. Every `utils::log()` call writes:

```
[2025-01-15T14:32:01Z] [INFO]  Session initialised. workspace=.workspace
[2025-01-15T14:32:01Z] [DEBUG] infer: submitting to provider.
[2025-01-15T14:32:02Z] [DEBUG] Pruned 4 messages from history.
[2025-01-15T14:32:05Z] [WARN]  Tool call round limit (10) reached; stopping tool loop.
```

`Session::log_file` is a `shared_ptr<FILE>` with a `fclose` deleter, so the file is closed when the session is destroyed. Log lines are written through `hal::write_log_line` (`src/hal/log_file.cpp`), guarded by `Session::log_mutex` since parallel graph branch threads can log concurrently.
