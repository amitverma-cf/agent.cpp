# Model registry

`#include <agent-cpp/agent.hpp>` · part of `libagent`

A `ModelRegistry` keeps named model templates and loads them into sessions on demand, under a RAM and (best effort) VRAM budget. When a
model has to be loaded and does not fit, the registry unloads the least recently used models that are idle and not pinned. A loaded model
is an ordinary `Session` created with `agent::init`. The registry changes nothing about how engines work.

The registry is thread-safe. Graph parallel branches, or any other threads, can acquire models at the same time.

The registry can be left out of a build with the CMake option `AGENT_WITH_MODELS=OFF`, which removes these declarations.

## Example

```cpp
agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 6ull << 30});   // 6 GiB

agent::ModelSpec chat;
chat.name = "qwen";
chat.config.provider = agent::AiProvider::Engine;
chat.config.engine = agent::engines::llama;
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

Error checks are left out of the other examples. Check `.ok` on each result.

## API

```cpp
using ModelLoadFn = Result<void> (*)(Session &session, const ModelSpec &spec, void *user_data);

struct ModelSpec {
    std::string name;                 // unique per registry
    Config config;                    // the template passed to init(): provider, engine, model, engine_options, ...
    uint64_t estimated_bytes = 0;     // 0 = estimate from the files on disk (see Memory estimate)
    bool pinned = false;              // never evicted once loaded (unload_model still unloads it)
    ModelLoadFn on_load = nullptr;    // runs once per load, before the first lease is handed out
    void *on_load_user_data = nullptr;
};

struct ModelBudget {
    uint64_t ram_bytes = 0;           // 0 = unlimited
    uint64_t vram_bytes = 0;          // 0 = unlimited; counts only models with engine option gpu_layers > 0
    bool allow_overcommit = false;    // load anyway, over budget, when evicting every idle model is not enough
    int wait_ms = 0;                  // how long acquire_model waits for leases to be released; 0 = fail at once
};

struct ModelMemoryEstimate {
    uint64_t file_bytes = 0;          // model file(s) on disk
    uint64_t ram_bytes = 0;           // counted against ModelBudget::ram_bytes
    uint64_t vram_bytes = 0;          // counted against ModelBudget::vram_bytes
};

struct ModelLease {
    Session *session = nullptr;
    std::string name;
    std::shared_ptr<std::mutex> turn_mutex;   // shared by every lease on this loaded model
    std::shared_ptr<void> hold;               // internal: the lease count
    Session &operator*() const;
    Session *operator->() const;
    explicit operator bool() const;
};

struct LoadedModelInfo {
    std::string name;
    uint64_t ram_bytes = 0;
    uint64_t vram_bytes = 0;
    bool pinned = false;
    int leases = 0;                   // outstanding leases (in use when > 0)
    int64_t loaded_at_ms = 0;         // wall clock, ms since the Unix epoch
    int64_t last_used_ms = 0;         // last acquire or release
    int64_t load_time_ms = 0;         // how long agent::init took
};

struct ModelRegistry {
    explicit ModelRegistry(ModelBudget budget = {});   // a handle: copies share one registry
    std::shared_ptr<void> state;
};
```

`on_load` is the place to register event hooks, add tools, or point the session at a shared store. An error from `on_load` unloads the
model and fails the acquire.

## Functions

| Function | What it does |
|---|---|
| `register_model(registry, spec)` | Adds a model. An existing name is replaced: an idle loaded model is unloaded first, and a leased one fails with `InvalidConfig`. An empty name is `InvalidConfig`. |
| `has_model(registry, name)` | Whether the name is registered. |
| `acquire_model(registry, name) -> Result<ModelLease>` | Loads the model if needed and returns a lease. `KeyNotFound` for an unregistered name. |
| `release_model(lease)` | Returns the lease early. The same as destroying it. Safe to call twice. |
| `unload_model(registry, name)` | Unloads a loaded model now, pinned ones too. Not loaded is `ok`. A leased model fails with `InvalidConfig`. |
| `loaded_models(registry)` | `LoadedModelInfo` for each loaded or loading model, most recently used first. |
| `set_budget(registry, budget)` / `get_budget(registry)` | `set_budget` replaces the budget and evicts idle, unpinned models (least recently used first) until the rest fit. |
| `estimate_model_memory(spec)` | What the registry will charge for the spec. Fails with `FilesystemError` when the model path is missing. |
| `process_rss_bytes()` | Resident set size of this process, or 0 when unavailable. Models under `EngineIsolation::Process` are not included. |

## Leases

A lease is the way to use a model. While any copy of a `ModelLease` exists, the model counts as in use, and **a model in use is never
evicted**. All leases on one model share a single `Session`, so the usual session rule applies: one `run_turn` at a time. Lock
`*lease.turn_mutex` around turns when several threads share a model. Graph Agent nodes do this for you.

Keep leases short, for example one per turn or per graph node, so that idle models can be evicted.

## acquire_model

1. If the model is loaded, its lease count goes up and it becomes the most recently used.
2. If another thread is loading it, wait for that load, then go to step 1.
3. Otherwise, estimate the memory and check it against the budget. If it does not fit, choose victims among the idle, unpinned loaded
   models, least recently used first, until it fits. If it still does not fit:
   - if evicted models are still being torn down, wait for that and check again;
   - with `wait_ms > 0`, wait up to that long for leases to be released;
   - with `allow_overcommit`, evict every idle, unpinned model and load anyway;
   - otherwise fail with `ErrorCode::OutOfMemory`. The message lists what is loaded and why each model cannot be evicted (in use,
     pinned, loading).
4. Reserve the estimate, unload the victims, then run `agent::init(spec.config)` and `spec.on_load`. Neither runs under the registry lock,
   so other models stay usable meanwhile. If either fails, the reservation is released and the error is returned, prefixed with the
   model name.

The reservation counts against the budget from the moment loading starts. Bytes of evicted models count until their sessions are destroyed.

## Memory estimate

`ModelSpec::estimated_bytes` overrides everything. When it is 0, the registry estimates from the files on disk:

- **Files.** The model path. A directory is summed recursively. For a split GGUF (`-00001-of-0000N`), every shard is added. Each string
  value in `engine_options` that names an existing file is added as well (`mmproj`, a vocoder, a tokens file, ...).
- **Per-engine factor.** Resident bytes ≈ file bytes × factor + overhead. The engine kind comes from `Config::engine`: a registered name,
  or a plugin file name that contains it.

| Engine | Factor | Overhead | Why |
|---|---|---|---|
| `llama` | 1.0 | 64 MiB | mmap'ed weights and compute buffers. The KV cache is added separately. |
| `onnx` | 1.5 | 32 MiB | ORT keeps a copy of the initializers and an arena |
| `sdcpp` | 1.2 | 256 MiB | weights and large diffusion compute buffers |
| `trellis` | 1.2 | 512 MiB | several sub-models and 3D decoding buffers |
| `audiocpp` | 1.5 | 32 MiB | ggml audio models |
| `mock` | 1.0 | 0 | test engine: exactly the file size |
| other | 1.25 | 64 MiB | any other engine |

- **llama KV cache.** `context_window × max_parallel_subagents × max(file_bytes / 32768, 64)` bytes. The llama engine allocates
  `context_window` tokens per sequence slot, with `max_parallel_subagents` slots. Bytes per token grow with model size, and `file / 32768`
  is within about 2× for current grouped-query models. Older multi-head-attention models need about 4× more. Set `estimated_bytes` for them.
- **VRAM (best effort).** When `engine_options` has a top-level `"gpu_layers"` above 0, the whole estimate counts against `vram_bytes`, and
  only the overhead counts against `ram_bytes`. Partial offload is not modelled. Without `gpu_layers > 0`, nothing is charged to VRAM.

These are rough numbers, meant to keep a machine from running out of memory. Once you have measured a model, set `estimated_bytes`.
`loaded_models()` reports the estimates, and `process_rss_bytes()` gives a measured figure for in-process engines.

## Graph Agent nodes

`GraphNode::model` names a registry model, and `GraphRun::models` points at the registry. An Agent node with a model acquires it for the whole
node, runs the turn on that model's session with the lease's `turn_mutex` held, and releases it when the node ends. A graph with several
models can therefore run in a budget that holds only one of them at a time:

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

- The turn runs on the model's session. Its tools, permission gate, store and event hooks are whatever the model's `ModelSpec::config`
  and `on_load` gave it. The graph's state, checkpoints and `user:` / `app:` keys stay on the run's session. The CLI shares the store and
  MCP tools with every model through `on_load`.
- Interrupts (`interrupt_before_tools`) and resumes work as usual. A resumed node acquires its model again.
- Nodes without a model use the run's session. A node with a model fails with `InvalidConfig` when `GraphRun::models` is null, and with
  `OutOfMemory` when the model cannot be loaded.
- Parallel branches can acquire models concurrently. Turns on the same model are serialized, so same-model branches do not get llama's
  continuous batching. Nodes without a model keep it. With different models, every branch's model must fit at the same time. Set
  `ModelBudget::wait_ms` to let branches wait for each other instead of failing.

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

In `agent.json`, set `"model": "<name>"` on an agent node. See [agent-json.md](agent-json.md).

## Specs from models.json

The hub builds specs from catalogue entries. It is a separate library: link `agent-cpp::hub` ([hub.md](hub.md)).

```cpp
#include <agent-cpp/hub.hpp>

auto entries = agent::hub::load_catalog(catalog_path);
agent::Config base;
base.workspace_dir = ws;
base.context_window = 4096;
for (auto &spec : agent::hub::model_specs_from_catalog(entries.value, base))
    agent::register_model(registry, std::move(spec));
```

`model_spec_from_entry(entry, base)` copies `base`, then sets `provider = Engine` and `model = entry.path`. When `base.engine` is empty, it
sets the engine from the file type: `.gguf` uses llama, `.onnx` uses onnx, and `.safetensors`, `.ckpt` and `.sft` use sdcpp. When the file
is missing, the catalogued `size` becomes `estimated_bytes`.

## CLI

`config.json` takes `"memory_budget_mb"` (default: 85% of installed RAM, for `chat` and `run`):

- **`agent chat`** loads every model through a registry. `/model NAME` keeps the previous model loaded and idle while the budget allows,
  so switching back is instant. Without a budget, the previous model is unloaded before the next one loads. All models share the
  workspace's SQLite store. `/stats` lists the loaded models with their estimates, in-use or idle state, time since last use and load
  time, plus the RSS of the CLI process.
- **`agent run`** uses the registry for agent nodes with `"model"`. The graph's default model is loaded and leased for the whole run only
  when some agent node has no `"model"`. Each node model is resolved like `--model`: a catalogue name or a file path, with its engine
  chosen from the file type.

## Limitations

- The estimates are heuristics. Prefer `estimated_bytes` once you have measured a model.
- VRAM accounting is all-or-nothing on `gpu_layers > 0`, and no GPU query is made.
- `process_rss_bytes()` does not include `agent-engine-host` children.
- The registry is local to one process. There is no budget shared between processes.
