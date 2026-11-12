# Model registry

`#include <agent-cpp/agent.hpp>` · part of `libagent`

A `ModelRegistry` keeps named model templates and loads them into Sessions on demand, under a RAM
(and best-effort VRAM) budget. When a model has to be loaded and does not fit, the registry unloads
the least recently used models that are idle and not pinned. A loaded model is an ordinary
`Session` created with `agent::init`. The registry sits on top of Session creation and does not
change how engines work.

The registry is thread-safe. Graph parallel branches, or any other threads, can acquire models at
the same time.

## Example

```cpp
agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 6ull << 30});   // 6 GiB

agent::ModelSpec chat;
chat.name = "qwen";
chat.config.provider = agent::AiProvider::Engine;
chat.config.engine = "llama";
chat.config.model = "/models/qwen2.5-3b-instruct-q4_k_m.gguf";
chat.config.workspace_dir = ".workspace";
chat.config.context_window = 4096;
agent::register_model(registry, chat);

agent::ModelSpec coder = chat;
coder.name = "coder";
coder.config.model = "/models/qwen2.5-coder-7b-q4_k_m.gguf";
agent::register_model(registry, coder);

{
    auto lease = agent::acquire_model(registry, "qwen");     // loads qwen
    if (!lease.ok) { /* lease.error.code == ErrorCode::OutOfMemory, ... */ }
    agent::Conversation memory;
    agent::run_turn(*lease.value, memory, "Hello");
}                                                            // lease released: qwen stays loaded, idle

auto lease = agent::acquire_model(registry, "coder");        // evicts qwen if both do not fit
```

## API

| | |
|---|---|
| `ModelSpec` | `name`, `config` (the `Config` template passed to `init`: provider, engine name or plugin path, model path, `engine_options`, `engine_isolation`, `context_window`, `max_tokens`, workspace, tools, permission gate, store, ...), `estimated_bytes` (0 = estimate from disk), `pinned`, `on_load` + `on_load_user_data` |
| `ModelBudget` | `ram_bytes` and `vram_bytes` (0 = unlimited), `allow_overcommit`, `wait_ms` |
| `ModelRegistry(budget)` | A handle. Copies share the same registry. |
| `register_model(registry, spec)` | Adds a model. If the name exists, the spec is replaced: an idle loaded model is unloaded first, and a leased one fails with `InvalidConfig`. |
| `has_model(registry, name)` | |
| `acquire_model(registry, name) -> Result<ModelLease>` | Loads the model if needed, then returns a lease. |
| `release_model(lease)` | Returns the lease, the same as destroying it. Safe to call twice. |
| `unload_model(registry, name)` | Unloads the model now (pinned models too). A model that is not loaded is fine; a leased one fails with `InvalidConfig`. |
| `loaded_models(registry)` | `LoadedModelInfo` for each loaded model, most recently used first: name, RAM/VRAM estimate, pinned, outstanding leases, loaded-at and last-used times, load time |
| `set_budget` / `get_budget` | `set_budget` replaces the budget and evicts idle, unpinned models (LRU first) until the rest fit. |
| `estimate_model_memory(spec)` | What the registry will charge for the spec (see below) |
| `process_rss_bytes()` | Resident set size of this process: `/proc/self/statm` on Linux, `GetProcessMemoryInfo` on Windows. 0 if unavailable. |

### Leases

`ModelLease` has the `Session *session`, the model `name` and a `turn_mutex`. It is copyable.
The model counts as in use while any copy exists, and **a model in use is never evicted**.
Every lease on a model points at the same Session, so the usual Session rules apply: one
`run_turn` at a time per Session. Lock `*lease.turn_mutex` around turns if several threads share a
model. Graph Agent nodes do this for you. Keep leases short, for example one per turn or per graph
node, so idle models can be evicted.

### `acquire_model`

1. The model is loaded: the lease count goes up, and the model becomes the most recently used.
2. Another thread is loading it: wait for that load, then go to step 1.
3. Otherwise, estimate the memory and check it against the budget. If it does not fit, pick
   victims among the idle, unpinned loaded models, least recently used first, until it fits. If it
   still does not fit:
   - if evicted models are still being torn down, wait for that and re-check;
   - with `wait_ms > 0`, wait up to that long for leases to be released;
   - with `allow_overcommit`, evict every idle, unpinned model and load anyway (over budget);
   - otherwise fail with **`ErrorCode::OutOfMemory`**. The message lists what is loaded and why
     each model cannot be evicted (in use, pinned, loading).
4. Reserve the estimate, unload the victims, and run `agent::init(spec.config)` and then
   `spec.on_load`. Neither runs under the registry lock, so other models stay usable meanwhile. If
   either one fails, the reservation is released and the error is returned. The message is
   prefixed with the model name.

`KeyNotFound` means the name is not registered. The reservation counts against the budget from
the moment loading starts. Bytes of evicted models count until their Sessions are destroyed.

`on_load` runs once per load, before the first lease is handed out. Use it to register event
hooks, add tools, or point the Session at a shared store (see the CLI below).

## Memory estimate

`ModelSpec::estimated_bytes` overrides everything. When it is 0, the registry estimates the
memory from the files on disk:

- **Files**: the model path. A directory is summed recursively. For a split GGUF
  (`-00001-of-0000N`), every shard is added. On top of that, each string value anywhere in the
  `engine_options` JSON that names an existing regular file is added (`mmproj`, a vocoder, a tokens
  file, ...). A missing model path fails with `FilesystemError` (Engine provider only).
- **Per-engine factor**: resident bytes ≈ file bytes × factor + overhead. The engine kind is taken
  from `Config::engine`, a registered name or a plugin file name that contains it.

| Engine | Factor | Overhead | Why |
|---|---|---|---|
| `llama` | 1.0 | 64 MiB | mmap'ed weights, plus compute buffers. The KV cache is added separately (below). |
| `onnx` | 1.5 | 32 MiB | ORT keeps a copy of the initializers, plus its arena |
| `sdcpp` | 1.2 | 256 MiB | weights, plus large diffusion compute buffers |
| `trellis` | 1.2 | 512 MiB | several sub-models, plus 3D decoding buffers |
| `mock` | 1.0 | 0 | test engine: exactly the file size |
| other | 1.25 | 64 MiB | |

- **llama KV cache**: `context_window × max_parallel_subagents × max(file_bytes / 32768, 64)`
  bytes. The llama engine allocates `context_window` tokens per sequence slot, with
  `max_parallel_subagents` slots. KV bytes per token grow with model size. `file / 32768` matches
  current GQA models to within about 2×: Qwen2.5-0.5B Q4 is about 12 KiB per token, Llama-3-8B Q4
  about 128 KiB. Older multi-head-attention models (Llama-2) need about 4× more. Set
  `estimated_bytes` for those.
- **VRAM (best effort)**: when `engine_options` has a top-level `"gpu_layers"` greater than 0, the
  whole estimate counts against `vram_bytes` and only the overhead counts against `ram_bytes`.
  Partial offload is not modelled. Without `gpu_layers > 0`, nothing is charged to VRAM.

These are rough numbers. They are meant to keep a machine from running out of memory, not to
account for every byte. `loaded_models()` reports the estimates, and `process_rss_bytes()` gives a
measured figure for in-process engines. Models running under `EngineIsolation::Process` live in
`agent-engine-host` children, so the RSS of this process does not include them. The budget still
applies to them, because the estimate does not depend on where the model runs.

## Graph Agent nodes

`GraphNode::model` names a registry model, and `GraphRun::models` points at the registry. An
Agent node with a model acquires it for the whole node, runs the turn on that model's Session (with
the lease's `turn_mutex` held), and releases it when the node ends. So a graph with several models
fits in a budget that only holds one of them at a time:

```cpp
agent::GraphNode write{.name = "write", .kind = agent::GraphNodeKind::Agent};
write.model = "coder";
write.prompt = "Write a haiku about caches.";
write.output_key = "draft";
agent::GraphNode review{.name = "review", .kind = agent::GraphNodeKind::Agent};
review.model = "qwen";
review.input_key = "draft";
review.output_key = "review";

agent::GraphRun run;
run.models = &registry;        // must outlive the run
agent::run_graph(session, agent::make_sequential("pipeline", {agent::make_graph(write), agent::make_graph(review)}), run);
```

In `agent.json`, use `"model": "<name>"` on an agent node.

- The turn runs on the **model's Session**. Its tools, permission gate, store and event hooks are
  whatever the model's `ModelSpec::config` (and `on_load`) gave it. The graph's state, checkpoints
  and `user:`/`app:` keys stay on the run's Session. The CLI shares the run's store and MCP tools
  with every model through `on_load`.
- Interrupts (`interrupt_before_tools`) and resumes work as usual. A resumed node acquires its
  model again.
- Nodes without a model use the run's Session, as before. A node with a model fails with
  `InvalidConfig` when `GraphRun::models` is null. It fails with `OutOfMemory` when the model
  cannot be loaded.
- Parallel branches can acquire models concurrently. Turns on the *same* model are serialized, so
  same-model branches lose llama's continuous batching (nodes without a model keep it). With
  different models, every branch's model must fit at the same time. Set `ModelBudget::wait_ms` to
  let branches wait for each other instead of failing.

You can also acquire a model yourself, for example from a Function node:

```cpp
agent::Result<void> summarize(agent::Session &, agent::GraphState &state, void *user_data) {
    auto &registry = *static_cast<agent::ModelRegistry *>(user_data);
    auto lease = agent::acquire_model(registry, "qwen");
    if (!lease.ok) return agent::fail(lease.error.code, lease.error.message);
    std::lock_guard<std::mutex> turn(*lease.value.turn_mutex);
    agent::Conversation memory;
    auto reply = agent::run_turn(*lease.value, memory, std::string(agent::get_state(state, "text")));
    if (!reply.ok) return agent::fail(reply.error.code, reply.error.message);
    agent::set_state(state, "summary", std::move(reply.value));
    return agent::ok();
}
```

## Specs from `models.json`

The hub (CLI layer, `agent_hub`) builds specs from catalogue entries:

```cpp
#include <agent-cpp/hub.hpp>
auto entries = agent::hub::load_catalog(catalog_path);
agent::Config base;  base.workspace_dir = ws;  base.context_window = 4096;
for (auto &spec : agent::hub::model_specs_from_catalog(entries.value, base)) agent::register_model(registry, std::move(spec));
```

`model_spec_from_entry(entry, base)` copies `base`, then sets `provider = Engine` and
`model = entry.path`. If `base.engine` is empty, it also sets the engine from the file type:
`.gguf` → llama, `.onnx` → onnx, `.safetensors`/`.ckpt`/`.sft` → sdcpp. When the file is missing,
the catalogued `size` becomes `estimated_bytes`.

The file format is described in [hub.md](hub.md#modelsjson), and a JSON Schema for it is in
[`schemas/models.schema.json`](../../schemas/models.schema.json).

## CLI

`config.json` takes `"memory_budget_mb"` (default: 85% of the installed RAM for `chat` and `run`):

- **`agent chat`**: every model is loaded through a registry. `/model NAME` keeps the previous
  model loaded, idle, while the budget allows. Switching back is then instant, and the first
  model keeps its `--engine-options`. Without a budget, the previous model is unloaded before the
  next one loads, as before. All models share the workspace's SQLite store. `/stats` lists the
  loaded models with their estimates, in-use/idle state, time since last use and load time, plus
  the RSS of the CLI process.
- **`agent run`**: agent nodes with `"model"` use the registry. The graph's default model
  (`--model` / the `"model"` hint / config) is loaded, and leased for the whole run, only if some
  agent node has no `"model"`. Otherwise the run's Session holds just the store and the MCP tools.
  Each node model is resolved like `--model`: a catalogue name or a file path, with its engine
  chosen from the file type.

## Limitations

- The estimates are heuristics. Prefer `estimated_bytes` once you have measured a model.
- VRAM accounting is all-or-nothing on `gpu_layers > 0`, and no GPU query is made.
- `process_rss_bytes()` does not include `agent-engine-host` children.
- The registry is process-local. There is no cross-process budget.
