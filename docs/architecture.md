# Architecture

agent.cpp is structured around three core principles that inform every design decision: no vtable dispatch, no hot-path heap allocation, and no exceptions. These constraints make it suitable for resource-constrained edge hardware where a GC pause or unexpected allocation would be fatal.

## Core principles

- **No vtables.** Provider and memory backends are selected via data-oriented function-pointer tables (`ProviderOps`, `MemoryOps`). Zero overhead beyond a single pointer dereference.
- **No hot-path allocation.** A bump-pointer `Arena` owns all per-turn storage. `MessageView`, `TensorView`, `ToolCallView` are zero-copy views into it. The arena resets in O(1) at the start of each turn.
- **No exceptions.** All fallible operations return `Result<T>`. Callers branch on `.ok`.
- **Single header.** `agent.hpp` is the entire public API surface. Backend headers (`llama.h`, `httplib.h`, `sqlite3.h`) are confined to their implementation files.

---

## System layers

```
+---------------------------------------------------------------------+
|                        User Application                              |
|   Flow  AgentScheduler  FlowMemory  EventHooks         |
+-----------------------------+---------------------------------------+
                              | calls
+-----------------------------v---------------------------------------+
|                      Orchestration Layer                             |
|   run_turn     step_flow / run_flow                       |
|   compress_context          AgentScheduler::pump                     |
+------+-----------------------------------------------+--------------+
       | infer                                          | execute_tool
+------v------------------+         +------------------v-----------+
|     Provider Layer       |         |       Tools Layer            |
|  ProviderOps[]           |         |  ToolsOps  tool_by_name      |
|  Mock  LlamaCpp  OpenAI  |         |  filesystem  terminal        |
|  OnnxRuntime             |         |  compressor  skills          |
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

Every `Session` owns one `Arena` — a contiguous slab allocated once at `init()`. Its default size is `context_window * 8 + 2 MiB` (override with `Config::arena_capacity`).

The arena is used only inside `run_turn` to construct zero-copy view spans before passing them to the provider. It resets at the top of every inference iteration, so no turn data leaks into the next. The reset is a single integer assignment — O(1) regardless of how much was allocated.

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
    { /* LlamaCpp */ init_llama,   infer_llama,   ... },
    { /* OpenAI   */ init_openai,  infer_openai_compatible,  ... },
    { /* Onnx     */ init_onnx,    infer_onnx,    ... },
};

MemoryOps kMemoryOps[] = {
    { /* Sqlite */ init_sqlite,  store_sqlite,  retrieve_sqlite,  clear_sqlite  },
};
```

`find_provider_ops(AiProvider::LlamaCpp)` does a 4-element linear scan. Branch predictor handles it in a single cycle. No heap, no map, no virtual dispatch.

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
      |         +-- fire FlowHook::Pre callbacks
      |         +-- execute_tool()  -->  O(1) tool_by_name lookup
      |         |     -->  permission_check (Allow/Ask/Deny), if Config::permission_check is set
      |         |     -->  callback
      |         +-- fire FlowHook::Post callbacks
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

## FSM executor

The FSM is a named-state machine where each state declares its own isolated context: system prompt, tool allowlist, and pre/post hooks. The `Flow` owns a single `FlowMemory` that persists across all state transitions — the conversation accumulates as the FSM traverses states.

```
Flow
 +-- states:  [StateA, StateB, StateC, ...]
 +-- memory:  FlowMemory   (shared across all states, unless a state sets isolated_memory)
 +-- isolated_memories: unordered_map<string, FlowMemory>   (per-state, when isolated_memory = true)
 +-- current_state_index, has_started, has_finished

step_flow():
  current = states[current_state_index]
  active_memory = current.isolated_memory ? isolated_memories[current.state_id] : memory

  1. Save & inject system_prompt into active_memory.history[0]

  2. Save session.config.tools
     Filter to current.allowed_tools
     Rebuild tool_by_name

  3. Save ambient::active_hooks (thread_local)
     Set to current.hooks

  4. If current.context_provider set, call it and fold the result into the
     effective system prompt for this turn only

  5. Call current.on_transition(session, active_memory, context)
     Returns next_state_id  ("exit" / "" = done)

  6. Restore hooks, tools, tool_by_name, system message

  7. Look up next state --> update current_state_index
```

The save/restore pattern guarantees that each state receives exactly the tools and system prompt it declared, with no global mutation surviving into the next state.

---

## Scheduler concurrency

`AgentScheduler` interleaves multiple `Flow`s and cron tasks. `pump()` fires all due cron tasks, then advances pending subagents — how it advances them depends on `session.config.provider`:

```
pump()
  now = steady_clock::now()

  for each CronEntry where now >= next_fire and !done:
    task.callback(session, user_data)
    if recurring:  next_fire += interval  (skip-ahead if still behind)
    if one-shot:   done = true

  if provider == OpenAICompatible or provider == LlamaCpp:
    batch = up to max_parallel_subagents not-done SubEntry's
    spawn one std::thread per batch entry, each calling step_flow(session, ctx, flow)
    join all threads
    (finish each: done=true + result on error or FSM-finished)
  else:
    for each SubEntry where !done:
      result = step_flow(session, context, flow)
      (finish each: done=true + result on error or FSM-finished)

run_until_done()
  while !all_done():  pump()
  pump()              (one final pass for last-moment crons)
```

All subagents share the same `Session`. Three different concurrency mechanisms are used, one per provider family, because each has a genuinely different constraint:

| Provider | Mechanism | Why |
|---|---|---|
| `Mock`, `OnnxRuntime` | Sequential round-robin (unchanged) | No batching support / not worth the complexity for these. |
| `OpenAICompatible` | Real OS threads | Plain HTTP calls — safe and effective to run concurrently. |
| `LlamaCpp` | Threads + continuous batching (see below) | A single `llama_context` can only run one `llama_decode` at a time regardless of thread count — threads alone buy nothing; the actual parallelism has to come from batching multiple sequences into one decode call. |

Making any of this safe required removing all of `Session`'s ambient mutable "current turn" state and replacing it with `thread_local` variables in the `agent::ambient` namespace (`src/core/executor/ambient_turn.cpp`): which `FlowMemory`/hooks/tool-allowlist the current thread's `execute_tool` call is running under (`ambient::active_memory`, `ambient::active_hooks`, `ambient::allowed_tools`), which `Arena` slot to use (`ambient::worker_arena_index`, resolved via `ambient::current_arena(session)`), and the cached `simdjson::ondemand::parser` (`ambient::json_parser`). A single thread behaves identically to the old shared-field design (nothing changes for callers who never touch `AgentScheduler`); concurrent threads get automatic isolation for free, no locking needed on any of it. What *is* still shared and mutex-guarded: `Session::stats` (`Session::stats_mutex`), the log file (`Session::log_mutex`), and event hook dispatch (`Session::event_mutex`), since these genuinely accumulate/fire across every concurrently-running turn.

### Continuous batching

`llama_provider.cpp`'s `LlamaEngine` owns the single `llama_model`/`llama_context` for a Session and multiplexes every concurrently-active `LlamaCpp` generation onto it via llama.cpp's native multi-sequence batching (`llama_batch` entries tagged with distinct `llama_seq_id`s, context created with `n_seq_max = Config::max_parallel_subagents`):

- Each generation's **prompt prefill** is decoded as its own `llama_decode` call when its sequence is acquired (not batched across sequences — prefill is comparatively rare and cheap relative to the token-by-token loop, and batching variable-length prompts together is a lot more complexity for little benefit).
- Each generation's **next-token step**, from then on, goes through `LlamaEngine::next_token()`: the first caller in a round becomes its leader, briefly releases the lock so other concurrently-running generations can join (a ~2ms debounce window), then performs **one** `llama_decode` covering every joined sequence's next token and wakes every waiter with its own sampled result. A round with only one active sequence degenerates to exactly one decode call per token — today's behavior — so continuous batching only kicks in when there's actually something to batch.
- Each generation owns its own `llama_sampler` chain (its own temperature, its own RNG) built fresh per call — nothing sampling-related is shared, so no locking is needed there; only the decode call itself is serialized through `LlamaEngine`'s mutex. (An earlier version also installed a GBNF grammar sampler here to force tool-call output into valid JSON; it was removed after a real-model crash traced the vendored `llama_sampler_init_grammar_lazy_patterns`/`llama_grammar_init_impl` path to an uncaught `std::regex_error` inside llama.cpp itself. Tool-call JSON is extracted unconstrained via bracket-matching instead, as it always was before that attempt.)
- Sequence slots are a fixed-size free list of `[0, n_seq_max)`; `acquire_seq()`/`release_seq()` check them out/in per generation, failing cleanly with `ProviderInitFailed` if the pool is exhausted rather than corrupting state.

This is deliberately not built on C++20 coroutines: an earlier design considered making `run_turn`/`infer` themselves suspendable so the scheduler could interleave partial generations directly, but that would have meant changing the public `ProviderInferFn` contract for every provider (not just `LlamaCpp`) for a benefit only `LlamaCpp` needs. The thread-plus-shared-engine design gets the same real batching benefit — one shared `llama_decode` per round across every active sequence — without touching any public signature.

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

`clear_provider_kv_cache(session)` delegates to `ProviderOps::clear_kv_cache`. For llama.cpp this clears the KV cache for every active sequence in the shared `llama_context`, which discards all cached key-value pairs. Useful before FSM state transitions that represent a significant context shift, or to reclaim VRAM between long sessions.

---

## Provider implementations

| File | AiProvider | Key dependency |
|---|---|---|
| `provider_ops.cpp` | Mock | — |
| `llama_provider.cpp` | LlamaCpp | `llama.h` |
| `openai_compatible_provider.cpp` | OpenAICompatible | `httplib.h`, `simdjson` |
| `onnx_provider.cpp` | OnnxRuntime | `onnxruntime_c_api.h` |

All provider state is heap-allocated and type-erased as `shared_ptr<void>` in `Session::provider_state`. The provider's destructor handles cleanup.

The OpenAI provider includes retry logic with exponential backoff: it retries on connection errors and HTTP 429/500/502/503/504 responses, up to `Config::retry_max_attempts` times, with initial delay `Config::retry_base_delay_ms` that doubles each attempt. Streaming responses are never retried (the stream is consumed on the first attempt).

---

## Memory backend

`sqlite_provider.cpp` implements the single `MemoryProvider::Sqlite` backend: a bounded write-back cache (`unordered_map` + LRU list, protected by one mutex) in front of a SQLite database (WAL journal mode, `synchronous=NORMAL`). Reads and writes always hit the cache first; a background writer thread flushes dirty entries to SQLite on a size-or-time trigger and evicts LRU clean entries once over the configured byte budget. `db_path = ":memory:"` (opened with SQLite's URI shared-cache form so both the writer and reader connections see the same data) gives the old standalone InMemory provider's ephemeral behavior through the same code path. See [docs/api/memory.md](api/memory.md) for the full design.

Keys are arbitrary UTF-8 strings. Conversation history is serialised as a JSON array and stored under `"history:<FlowMemory::id>"` after every turn.

RAG/vector search is a separate concern, never routed through this cache — see `sqlite_vec_data_source.cpp` and [docs/api/data-sources.md](api/data-sources.md).

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
[2025-01-15T14:32:05Z] [INFO]  AgentScheduler: subagent 'worker' completed.
```

`Session::log_file` is a `shared_ptr<FILE>` with a `fclose` deleter, so the file is closed when the session is destroyed. Log lines are written through `hal::write_log_line` (`src/hal/log_file.cpp`), guarded by `Session::log_mutex` since multiple scheduler worker threads can log concurrently.
