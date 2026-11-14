# Architecture

agent.cpp is a C++20 library and CLI for running language, speech and image models on your own machine. The design
follows three rules, chosen for small edge hardware where an unexpected allocation or pause is a real problem:

- **No virtual dispatch.** Providers, memory stores and tool backends are chosen from `constexpr` tables of function
  pointers (`ProviderOps`, `StoreOps`, `ToolsOps`).
- **No allocation on the turn path.** Each session has a per-turn `Arena`. The views handed to a provider point into it.
  Data that must survive a turn (conversation history, stored memory) owns its storage.
- **No exceptions in the library.** Fallible calls return `agent::Result<T>`; callers check `.ok`. Engine plugins may use
  exceptions internally, but they must catch them before the plugin ABI boundary.

The public C++ API is one header, `include/agent-cpp/agent.hpp`. The engine plugin ABI is `include/agent-cpp/engine_abi.h`
and uses only C types. `include/agent-cpp/agent_c.h` is the C API. Backend headers (llama.cpp, ONNX Runtime, SQLite) are
never part of a public header.

## Source map

| Path | What it is |
|---|---|
| `include/agent-cpp/agent.hpp` | The public API: `Config`, `Session`, `Conversation`, `run_turn`, graphs, RAG, engines. |
| `include/agent-cpp/engine_abi.h` | The C ABI between the core and engine plugins (`agent_engine_v1`, requests, responses, capability bits). |
| `src/agent/agent.cpp` | Lifecycle: `engine_config`, `init`, `open_session`, the one-shot task helpers. |
| `src/agent/executor/` | `turn_executor.cpp` (`run_turn`: history, compression, pruning, tool rounds); `ambient_turn.cpp` (per-thread turn state). |
| `src/agent/providers/` | `provider_ops.cpp` (the `ProviderOps` table: Mock and Engine); `engine_provider.cpp` (builds requests for an engine). |
| `src/agent/engines/` | Engine registry (finds plugins by name or path, reads `engines.json`), in-process and host-backed engine clients, shared memory for large tensors. |
| `src/agent/tools/` | Tool table (`tools_ops.cpp`), filesystem and terminal tools, path gate, context compressor, lazy tool loading. |
| `src/agent/sandbox/` | Sandbox policy for child processes: what the engine host and `run_command` may read, write and reach. |
| `src/agent/memory/` | The `StoreOps` interface and the default in-memory store. |
| `src/agent/events.cpp` | Event hooks (`EventType`, `register_event_hook`). |
| `src/agent/skills/` | `SKILL.md` directories and the `use_skill` tool. |
| `src/agent/graph/` | Graph runtime: nodes, routes, parallel branches, checkpoints, `agent.json` (optional). |
| `src/agent/mcp/` | MCP stdio client (optional). |
| `src/agent/rag/` | Chunking, embedding and rerank through engines, in-memory vector store (optional). |
| `src/agent/models/` | `ModelRegistry`: several models under one memory budget (optional). |
| `src/agent/tasks/` | Helpers for one-shot tasks (`decide`, `ocr`). |
| `src/hal/` | The only layer that calls the operating system: processes, dynamic loading, paths, sandbox, logging, time. |
| `src/engines/<name>/` | Engine plugins: `llama`, `onnx`, `sdcpp`, `trellis`, `audiocpp`, and `mock` (used by the tests). |
| `apps/engine-host/` | `agent-engine-host`: loads one plugin and serves it over pipes (process isolation). |
| `apps/store-sqlite/` | `agent_store_sqlite`: SQLite store, checkpointer and sqlite-vec vector source. Separate library. |
| `apps/hub/` | `agent_hub`: Hugging Face and GitHub downloads and the model catalogue. Separate library. |
| `apps/cli/` | The `agent` program: setup, chat, voice, tasks, engine, model, skill and MCP management. |
| `bench/`, `examples/`, `tests/` | Benchmark harness, example programs, unit and integration tests. |

---

## Layers

```
+-----------------------------------------------------------------------+
|  Your program or the agent CLI                                         |
|  Config, Conversation, run_turn, run_graph, event hooks, tools         |
+------------------------------------+----------------------------------+
                                     | calls
+------------------------------------v----------------------------------+
|  Turn loop (turn_executor)              Graph runtime (optional)       |
|  compress -> prune -> infer -> tool calls -> save                      |
+----------------+----------------------------------------+-------------+
                 | infer                                  | execute tool
+----------------v-------------------+   +----------------v-------------+
|  Provider (ProviderOps)            |   |  Tools (ToolsOps)            |
|    Mock                            |   |  allowed_tools check         |
|    Engine: a plugin loaded in this |   |  permission gate             |
|    process, or run by              |   |  path gate, sandbox          |
|    agent-engine-host               |   |  Tool::callback              |
+----------------+-------------------+   +----------------+-------------+
                 |                                        |
+----------------v----------------------------------------v-------------+
|  Session: Config, Arena and worker arenas, provider and memory state,  |
|  tool table, data sources, skills, stats, log file                     |
+-----------------------------------------------------------------------+
```

---

## Turn flow

`run_turn(session, conversation, prompt)` runs one user turn, which may take several model calls when the model asks for
tools:

1. Append the user message to `conversation.history` and fire `OnTurnStart`.
2. Compress the history if the prompt would use more than 70% of `context_window` (and there are more than six messages).
3. Loop, once per model call:
   1. Compress again if needed (checked before every call after the first) and prune the history to the budget. Pruning
      keeps the leading system messages and drops the oldest exchanges; a tool call and its results are dropped together.
      The budget is `max(200, context_window - max_tokens - 100 - tools text)`.
   2. Reset the turn arena and build the message views. Leading system messages and the injected tool list are merged
      into one system message, because many chat templates accept a system message only first.
   3. Fire `OnInferenceSubmit`, call the provider, fire `OnInferenceComplete`. If a request that carried tool-produced
      media fails, it is retried once without that media.
   4. Append the assistant message and add its usage to the stats.
   5. No tool calls: leave the loop. A graph Agent node may pause here before any call runs (see the graph section).
   6. Tool calls: stop with `ToolCallLimitExceeded` when `max_tool_call_rounds` is reached. Otherwise, for each call:
      fire `OnToolCall` and the `ToolHook::Pre` callbacks, execute the tool, fire `ToolHook::Post` and `OnToolResult`,
      and append a `tool` message. Then go round the loop again.
4. Save the conversation: its history and stats go to the memory store under `history:<id>` and `history_stats:<id>`.
5. Return the final reply as an owned `std::string`. The arena may be reset after this point, so the copy is made first.

Tool execution (`execute_for_message`) checks, in order: the graph node's `allowed_tools` list (if any), the permission gate
(`Config::permission_check`), and the path and OS sandbox rules that the tool itself applies. Only then does the callback run.

---

## Memory model

### Per-turn arena

Each `Session` owns an `Arena`, a bump allocator. Its first block is allocated in `init()`, by default
`context_window * 8 + 2 MiB`; `Config::arena_capacity` overrides it. If a turn needs more, the arena adds chunks, each at
least twice the size of the previous one, so a long turn never fails for lack of size. `reset()` runs before every model
call and is O(1) while the turn fits in the first block. When a turn outgrew the first block, the next reset replaces the
chunks with one block sized to the peak, so later turns of that size allocate nothing.

What the arena holds is the set of views passed to one provider call: `MessageView`, `ToolCallView`, `TensorView`,
`MediaView`. They point at the owned strings in the conversation history, so nothing is copied.

```
Session::arena  (first block allocated at init, grows in chunks, reset before each model call)
  |
  +-- MessageView[N]     zero-copy into the history strings
  +-- ToolCallView[M]    zero-copy into the ToolCall strings
  +-- TensorView[K]      zero-copy into TensorData buffers
  +-- MediaView[P]       zero-copy into Media paths
```

### Owned state

`Message::content`, `ToolCall::arguments` and the rest of `Conversation::history` are `std::string` and `std::vector`
members. They are never placed in the arena, and they survive across turns. `Conversation::history` elements are stable
once pushed, so a view taken from one is valid until that message is removed. `Message::token_count` caches a token count,
computed on first use (-1 means not yet computed).

### Session self-reference cell

The context compressor is a tool, and a tool's callback gets a `void *user_data` fixed at registration. The session can be
moved after that, so the tool cannot hold a raw `Session *`. Instead the session holds a heap cell that the tool points at:

```
session.session_ptr_cell  -->  shared_ptr<Session*>  (heap block)
                                        |
tool.user_data  ------------------------+  (Session** into the same block)

Before every tool dispatch:
  *session.session_ptr_cell = &session;   // refresh, so a moved session is still found
```

---

## Dispatch tables

Providers and memory stores are chosen from `constexpr` tables. Two examples, as they are in the source:

```cpp
constexpr ProviderOps kProviderOps[] = {
    ProviderOps{.provider = AiProvider::Mock, .init = init_mock, .infer = infer_mock, ...},
    ProviderOps{.provider = AiProvider::Engine, .init = init_engine, .infer = infer_engine, ...},
};

constexpr StoreOps kInMemoryStore{.open = open_in_memory, .put = put_in_memory, .get = get_in_memory, .clear = clear_in_memory};
```

`find_provider_ops()` is a linear scan of a two-element table. `Config::memory.store` points at a `StoreOps` table; the
default is the in-memory store.

---

## Engines and the engine host

### What an engine is

An engine is a shared library named `agent-engine-<name>` (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows). It exports
one function, `agent_engine_get_v1`, which returns a table: the engine name, the ABI version, the capability bits it
serves (`AGENT_CAP_TEXT_GEN`, `AGENT_CAP_ASR`, `AGENT_CAP_IMAGE_GEN`, ...), flags, and the `create`, `run`, `destroy`
functions. Everything crossing the boundary is a C type. The host and the plugin may be built with different compilers
and C++ runtimes.

The shipped engines are `llama` (text, vision, embeddings, rerank, decision models), `onnx`, `sdcpp` (images and video),
`trellis` (image to 3D), `audiocpp` (speech recognition, speech synthesis, VAD, diarization) and `mock`. A session finds
an engine by name or by path:

- By name: the registry (`engines::add`, `load_engine_manifest`, or a plugin already loaded), then the folders in this
  order: `engines::search_path()` folders, `$AGENT_ENGINES_DIR`, `engines/` next to the program, `<program>/../lib/agent/engines`,
  the program's own folder, `<program>/../lib`, and the working directory.
- By path: `Config::engine` contains a `/` or ends in a library extension.

### Why each engine is its own file

Each engine carries its own copy of its inference library: llama.cpp and stable-diffusion.cpp each bring a ggml,
TRELLIS.2 brings a ggml fork, ONNX Runtime is its own runtime, and audio.cpp builds its own ggml. These copies are not
compatible with each other, and their symbols and CMake targets clash, so they cannot be linked into one executable.
Keeping each engine in its own file, with hidden symbol visibility and one exported entry point, avoids the clash.
Plugins can share a process when their symbols stay private (llama and sdcpp have been loaded together), but the design does
not depend on it.

A separate file also gives two more things. A crash, `abort()` or out-of-memory kill in one engine does not have to take the
application down (see process isolation below). And an engine can be left out of a build or a release without touching the
others.

One concrete constraint of this kind: ggml's AMX kernels can break when another engine in the same process touches the AMX
tile state. `AGENT_GGML_AMX` is therefore off by default, and turning it on is only safe with process-isolated engines
(`cmake/ggml_no_amx.cmake`).

### Isolation

`Config::engine_isolation` decides where an engine runs:

| Mode | How it runs | Cost | Failure |
|---|---|---|---|
| `InProcess` | `dlopen`ed into the application; called directly. | About 0.7 us per turn (measured). | A crash, `abort()` or `exit()` in the engine ends the application. |
| `Process` | `agent-engine-host <plugin>` loads it and serves the same ABI over pipes. Tensors and media of 64 KiB and more go through shared memory. | About 40 us per turn (measured), plus the engine's own time. | The call fails with `ErrorCode::EngineCrashed`; the next call restarts the host (limited by `engine_max_restarts`). |

The library default is `InProcess`. The `agent` CLI uses `Process`. Under `Process`, the host runs inside an OS sandbox
(`Config::engine_sandbox`, on by default): the plugin, the model and the workspace are readable, only the artifacts folder,
the workspace scratch folder and `/dev/shm` are writable, and there is no network and no `/tmp`. On Linux the sandbox uses
Landlock and seccomp; on Windows it is partial. `engine_timeout_ms` is a per-request watchdog, and `engine_sandbox.memory_limit_bytes`
becomes the host's address-space limit.

Details and the host protocol: [api/engines.md](api/engines.md), [security-sandbox.md](security-sandbox.md).

### Engine lookup in the program

`engines::llama` and the other shipped names are plain strings. `agent_deploy_engines()` in CMake copies named plugins into
an `engines/` folder next to the program, where the lookup above finds them. See [packaging.md](packaging.md).

---

## Context management

### Pruning

Pruning runs before every model call (see the turn flow). It is a single pass that counts tokens and then one `erase` of
the selected range. Token counts come from the engine's `count_tokens` when it has one; otherwise they are estimated as
`(chars + 3) / 4`.

### Compression

When the prompt would take more than 70% of `context_window` and the history has more than six messages,
`compress_context` replaces the older messages with one `[Compressed history]` summary. It keeps the most recent
`keep_recent` messages (default 6), asks the same provider for the summary, and falls back to pruning if the summary call
fails. The check runs before the first model call of a turn and again inside the tool loop, because tool results grow the
history.

Compression can also be triggered by the model: the `compress_context` tool is a registered tool that takes `keep_recent`.

### KV cache

`clear_provider_kv_cache(session)` calls `ProviderOps::clear_kv_cache`. For llama.cpp this clears the cached state of every
sequence in the shared context. Use it before a large change of topic, or to free memory between long sessions.

---

## Concurrency

A graph `Parallel` node fans out to its branches. How they run depends on the engine's flags:

```
run_parallel_node()
  if the engine sets AGENT_ENGINE_FLAG_CONCURRENT and this is a top-level fan-out:
    for each batch of up to max_parallel_subagents branches:
      one std::thread per branch (ambient::worker_arena_index = its slot); join all
  else:
    run each branch to its end, one after another
  join: merge branch writes and histories; fail if two branches wrote the same key
```

All branches share one `Session`. What each thread must not share is kept in `thread_local` variables in the `agent::ambient`
namespace (`src/agent/executor/ambient_turn.cpp`): the conversation and hooks of the current turn (`active_memory`,
`active_hooks`), the tool allowlist (`allowed_tools`), the arena slot (`worker_arena_index`), a simdjson parser, and the
graph pause hook. A single thread behaves as if these were ordinary fields, and concurrent threads are isolated without locks.

Shared state that is still mutex-guarded: `Session::stats` (`stats_mutex`), the log file (`log_mutex`) and event hook dispatch
(`event_mutex`).

The llama engine is the one that runs concurrent generations. It holds one model and one context per engine instance, and a
batcher inside the plugin decodes the next token of every active sequence in one `llama_decode` call per round. Prompt prefill
is decoded per sequence. A round with one active sequence is one decode per token, as before. Sequence slots come from a fixed
pool of `max_parallel_subagents` entries; when the pool is empty the call fails rather than sharing a slot.

---

## Tools, permissions and sandboxes

- **Tool table.** `Config::tools` and the built-in tools are registered into `Session::tool_by_name`. Lookup is O(1).
- **Permission gate.** `Config::permission_check` returns `Allow`, `Ask` or `Deny` before each call. `Ask` is resolved by
  `Config::permission_prompt`; without one it is treated as `Deny`. The default (`nullptr`) allows everything.
- **Filesystem sandbox.** With `Config::sandbox_filesystem` (default `true`), file tools resolve each path with
  `hal::resolve_under_workspace`: `weakly_canonical`, then a prefix check against the canonical workspace. Absolute paths,
  `..` and symlinks that leave the workspace are rejected with `SandboxViolation`. The workspace root itself cannot be deleted.
- **Path grants.** `Config::path_grants` and `Config::path_gate` allow specific paths outside the workspace; the gate is asked
  about any other path.
- **OS sandbox for commands.** `run_command` runs under `Config::tool_sandbox` (workspace read-write, system directories
  read-only, network allowed unless set to `Deny`). Engine hosts run under `Config::engine_sandbox`.
- **Shell metacharacters.** `init()` rejects a `workspace_dir` containing `"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, or a newline,
  before creating any folder.
- **Native tools are not sandboxed.** A `Tool::callback` is a plain function pointer that runs in the application with the
  application's privileges. The permission gate decides whether it is called; the sandboxes do not apply to its own code.
  Review a tool's code before you register it.

---

## Graph runtime

Multi-step workflows are graphs (`src/agent/graph/`, [api/graph.md](api/graph.md)): named nodes (Agent, Function, Tool,
Parallel, Loop) joined by static or routed edges, with a `GraphState` of scoped key/value pairs and one `Conversation` per
Agent conversation.

```
run_graph(session, graph, run)
  load the thread's last checkpoint (per-thread values and Agent histories), user: and app: keys, caller seed
  node = graph.entry
  loop:
    trigger_event(OnStateTransition, node.name)
    Agent node:
      memory = state.histories[node.history_key or node.name]
      1. system prompt = context_provider(state) + "\n" + node.system_prompt
         installed in memory.history[0] for this run only (saved and restored)
      2. ambient::allowed_tools = node.allowed_tools   (restored afterwards)
         ambient::active_hooks  = [event-log Pre] + node.hooks + [event-log Post]
      3. run_turn(session, memory, state[input_key] or prompt)
         (a pause hook can stop the turn before its tool calls run: human approval, see resume_graph)
      4. state[output_key] = reply
    Function / Tool / Loop node: call it
    Parallel node: run each branch on a copy of the state, join (distinct keys and histories),
                   trigger_event(OnSubAgentComplete) per branch
    write the log, persist user: and app: keys, checkpoint (per GraphDurability)
    node = edge or router target (kGraphEnd -> exit_to or done)
```

Saving and restoring the system prompt, allowlist and hooks means each Agent node gets exactly what it declared, and nothing
leaks into the next node. Nodes that share a `history_key` continue one conversation under their own prompts.

---

## Memory backend

`MemoryConfig::store` selects a `StoreOps` table. The core ships only the in-memory store (the default). The optional
`agent_store_sqlite` library provides `agent::sqlite_store()`: a bounded write-back cache (a hash map with an LRU list,
behind one mutex) in front of a SQLite database in WAL mode with `synchronous=NORMAL`. A background thread flushes dirty
entries on a size or time trigger and evicts clean entries over the byte budget. `db_path = ":memory:"` gives an ephemeral
store through the same code. See [api/memory.md](api/memory.md).

Keys are UTF-8 strings. Conversation history is stored as a JSON array under `history:<Conversation::id>`. RAG vector search
does not go through this cache; see [api/data-sources.md](api/data-sources.md).

---

## Build-time options

The library parts that can be left out are compile-time switches. Each one removes its sources, and the header hides its
API when the matching macro is 0 (`AGENT_WITH_GRAPH`, `AGENT_WITH_MCP`, `AGENT_WITH_RAG`, `AGENT_WITH_MODELS`):

| Option | Default | Removes when OFF |
|---|---|---|
| `AGENT_WITH_GRAPH` | `ON` | The graph runtime. `agent.json` loading also needs `AGENT_WITH_MCP`. |
| `AGENT_WITH_MCP` | `ON` | The MCP stdio client. |
| `AGENT_WITH_RAG` | `ON` | Embeddings, rerank, chunking and the in-memory vector store. |
| `AGENT_WITH_MODELS` | `ON` | `ModelRegistry`. |

Engines are plugins, so each is a separate switch that only adds or leaves out one file: `AGENT_ENGINE_LLAMA` (`ON`),
`AGENT_ENGINE_ONNX` (`ON`), `AGENT_ENGINE_SDCPP`, `AGENT_ENGINE_AUDIOCPP` and `AGENT_ENGINE_TRELLIS`. The last three are on by
default only when their sources are checked out (TRELLIS runs on the CPU and needs no CUDA). The full list is in
[getting-started.md](getting-started.md#build-options).

The other pieces are separate targets: `agent_store_sqlite` (`apps/store-sqlite`), `agent_hub` (`apps/hub`, needs OpenSSL,
`AGENT_BUILD_HUB`) and the `agent` CLI (`AGENT_BUILD_CLI`, which needs the hub). `BUILD_SHARED_LIBS` chooses static or shared
for the three libraries; engine plugins are always shared modules. Packaging is in [packaging.md](packaging.md).

---

## File logging

When `Config::enable_file_logging` is `true`, `init()` creates `<workspace_dir>/logs/agent_YYYY-MM-DD_HH-MM-SS.log` (UTC time).
Every log line has a UTC timestamp and a level:

```
[2025-01-15T14:32:01Z] [INFO]  Session initialised. workspace=.workspace
[2025-01-15T14:32:02Z] [DEBUG] infer: submitting to provider.
[2025-01-15T14:32:05Z] [WARN]  Tool call round limit (10) reached; stopping tool loop.
```

`Session::log_file` is a `shared_ptr<FILE>` that closes the file when the session is destroyed. Writes are serialized by
`Session::log_mutex`, because graph branches can log from several threads.
