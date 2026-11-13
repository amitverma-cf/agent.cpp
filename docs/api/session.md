# Session

`#include <agent-cpp/agent.hpp>`

A `Session` is the object everything else works through. It holds one loaded engine with its models, the workspace, the tool registry, the memory store and the statistics. Create one with `open_session` for the common case, or with `engine_config` and `init` when you need control over the configuration.

Sessions are **move-only**. Do not use one from two threads at once.

---

## Open a session

### The shortest way

```cpp
#include <agent-cpp/agent.hpp>

int main() {
    agent::AgentRuntime runtime;

    auto res = agent::open_session(agent::engines::llama, "model.gguf");
    if (!res.ok) {
        fprintf(stderr, "open failed: %s\n", res.error.message.c_str());
        return 1;
    }
    agent::Session session = std::move(res.value);

    agent::Conversation conv;
    conv.id = "chat-1";
    auto reply = agent::run_turn(session, conv, "Hello");   // see conversation.md
}
```

`open_session` uses `.agent` in the current folder as the workspace, and every other setting has its default value. Pass a `Config` as the third argument to change any of them:

```cpp
agent::Config base;
base.workspace_dir = "work";
base.enable_file_logging = true;

auto res = agent::open_session(agent::engines::llama, "model.gguf", base);
```

### Several model files

Some engines take more than one model file. Name each file by its role:

```cpp
auto res = agent::open_session(agent::engines::sdcpp,
                               {{"model", "sd.gguf"}, {"vae", "vae.gguf"}});
```

The roles belong to the engine. The ones the shipped engines use are listed below; [engines.md](engines.md) has the full list.

| Engine | Roles |
|---|---|
| `engines::llama` | `model`, `mmproj` (vision projector) |
| `engines::sdcpp` | `model` (one-file checkpoint), or split-model components `diffusion_model`, `vae`, `clip_l`, `clip_g`, `t5xxl`, `llm` |
| `engines::audiocpp` | `tts.model`, `asr.model`, `vad.model`, `diarize.model` |

A dotted role such as `tts.model` is placed inside a section of the engine options (`{"tts": {"model": ...}}`). A role of `model` sets `Config::model`. Every other role goes into the engine options.

---

## One-shot tasks

Engines that do something other than chat (speech, image generation, and so on) can be called with one function per task. Each returns a `TaskOutput`:

```cpp
struct TaskOutput {
    std::string              text;    // text the engine produced, if any
    std::vector<std::string> files;   // files it wrote, under <workspace>/artifacts
};

Result<TaskOutput> transcribe(Session &session, std::string_view audio_path);        // speech to text
Result<TaskOutput> speak(Session &session, std::string_view text);                   // text to speech: files[0] is a WAV
Result<TaskOutput> draw(Session &session, std::string_view prompt, std::string_view params_json = {});  // image generation
Result<TaskOutput> perform(Session &session, uint32_t task, std::string_view text,
                           std::string_view media_path = {}, std::string_view params_json = {});       // any AGENT_CAP_* task
```

The engine must support the task, or the call fails with `UnsupportedProvider`. `perform` covers the tasks without a named function. `params_json` is a JSON object of engine parameters, for example the generation settings of the sdcpp engine.

```cpp
auto voice = agent::open_session(agent::engines::audiocpp, {{"tts.model", "pocket-tts.gguf"}});
if (!voice.ok) return 1;

auto wav = agent::speak(voice.value, "Hello there.");
if (wav.ok) printf("wrote %s\n", wav.value.files[0].c_str());
```

Which parameters each engine accepts is listed in [engines.md](engines.md). The lower-level form is `run_task` in [conversation.md](conversation.md#run_task).

### Full control: engine_config and init

`engine_config` builds a `Config` from an engine, its model files and optional engine options. It does not load anything. Pass the result to `init`:

```cpp
agent::ModelSlot models[] = {{"model", "llama.gguf"}, {"mmproj", "mmproj.gguf"}};

agent::Config base;
base.workspace_dir = "work";

auto cfg = agent::engine_config(agent::engines::llama, models, R"({"gpu_layers": 99})", base);
if (!cfg.ok) return 1;

cfg.value.context_window = 8192;
cfg.value.max_tokens = 1024;

auto res = agent::init(cfg.value);
```

`engine_config(engine, models, extra_options_json, base)`:

- `engine` is one of the `engines::` constants, or the name you gave an engine with `engines::add`.
- `models` is a span of `ModelSlot{role, path}`.
- `extra_options_json` is a JSON object merged into the engine options. Pass `{}` for none.
- `base` supplies everything else. If `base.engine_options` is already set, pass the options through `extra_options_json` instead; setting both is an error.

### Engines

| Constant | Engine | Use |
|---|---|---|
| `agent::engines::llama` | `llama` | Text, vision, embeddings, rerank and decisions (GGUF models) |
| `agent::engines::audiocpp` | `audiocpp` | Speech recognition, text to speech, voice activity detection, diarization |
| `agent::engines::sdcpp` | `sdcpp` | Images and video |
| `agent::engines::trellis` | `trellis` | Image to 3D mesh |
| `agent::engines::onnx` | `onnx` | Any ONNX model |

Engine plugins are found at run time in an `engines/` folder next to your program. `agent_deploy_engines()` in CMake copies only the engines your application uses into that folder. To look in other places, see [engines.md](engines.md).

---

## AgentRuntime

Create one `AgentRuntime` per process, before any session. It starts the global inference backends and stops them at exit.

```cpp
struct AgentRuntime {
    AgentRuntime();    // calls init_backend()
    ~AgentRuntime();   // calls free_backend()
    // copy and move are deleted
};

int main() {
    agent::AgentRuntime runtime;
    // open sessions and run turns here
}
```

The `AgentRuntime` must outlive every `Session`.

---

## Config

`Config` holds every setting `init` reads. Set the fields you need and leave the rest at their defaults. After `init`, `session.config` is readable. Change only `session.config.tools` afterwards, and call `rebuild_tool_index` after you do (see [Tools after init](#tools-after-init)).

```cpp
struct Config {
    AiProvider       provider = AiProvider::Mock;
    std::string      engine;
    std::string      engine_options;
    std::string      model;
    std::string      workspace_dir;
    MemoryConfig     memory;
    std::vector<Tool> tools;
    // ... see the tables below
};
```

### Provider and workspace

| Field | Default | Meaning |
|---|---|---|
| `provider` | `AiProvider::Mock` | `Mock` is the built-in provider: no engine, no model, and text replies only (it returns a fixed reply). `Engine` runs the engine named in `engine`. |
| `engine` | `""` | An `engines::` constant, a name added with `engines::add`, or a plugin path. Required for `Engine`. |
| `engine_options` | `""` | JSON object passed to the engine. Prefer `engine_config`, which builds it from the model slots. |
| `model` | `""` | The main model file. `engine_config` sets it from the role `model`. |
| `workspace_dir` | `""` | **Required.** Folder for models, memory, logs and temporary files. See [Required values](#required-values). |
| `memory` | `{}` | The key-value store: in-memory by default, or SQLite. See [memory.md](memory.md#memoryconfig). |
| `enable_file_logging` | `false` | Write `logs/agent_<timestamp>.log` under the workspace. |
| `skills_dir` | `""` | Folder of skill subfolders, each with a `SKILL.md`. Empty turns skills off. See [guide.md](../guide.md#skills). |

### Inference

| Field | Default | Meaning |
|---|---|---|
| `context_window` | `2048` | Total context size in tokens. Must be greater than 0. |
| `max_tokens` | `512` | Maximum tokens generated per model call. Must be greater than 0. |
| `temperature` | `0.7` | Sampling temperature. Must be finite and not negative. |
| `default_task` | `0` | The `AGENT_CAP_*` task used when a request does not name one. `0` means automatic. |
| `arena_capacity` | `0` | Size of the first arena block in bytes. `0` = `context_window * 8 + 2 MiB`. The arena grows past it. |
| `max_parallel_subagents` | `4` | Worker arenas for parallel graph branches (see [graph.md](graph.md#parallel-execution)), and the llama engine's parallel sequence count. `0` creates no worker arenas. |

### Tools and permissions

| Field | Default | Meaning |
|---|---|---|
| `tools` | `{}` | Your tools. Default tools are added after these during `init`. See [tools.md](tools.md). |
| `auto_default_tools` | `true` | Register the default tools (filesystem, terminal, context compressor, and so on). |
| `lazy_tools` | `false` | Start with only `tool_search`, `load_tool` and `use_skill`; other tools are loaded when the model asks. |
| `max_tool_call_rounds` | `10` | Tool-call rounds allowed in one `run_turn`. |
| `sandbox_filesystem` | `true` | Keep file tools inside the workspace. |
| `tool_sandbox` | `{enabled = true}` | OS sandbox for `run_command`. See [security-sandbox.md](../security-sandbox.md). |
| `path_grants` | `{}` | Paths outside the workspace that sandboxed tools may use. |
| `path_gate` | `nullptr` | Called for a path outside the workspace that no grant covers. `path_gate_user_data` is passed to it. |
| `permission_check` | `nullptr` | Called before each tool call. `nullptr` allows everything. |
| `permission_prompt` | `nullptr` | Resolves an `Ask` from `permission_check`. If `nullptr`, `Ask` is treated as `Deny`. |
| `permission_user_data` | `nullptr` | Passed to `permission_check` and `permission_prompt`. |
| `secret_resolver` | `nullptr` | Resolves `${secret:NAME}` in MCP server environment variables. `secret_user_data` is passed to it. |
| `mcp_launcher` | `nullptr` | Rewrites an MCP server's command just before it starts. `mcp_launcher_user_data` is passed to it. |

Permissions are covered in [guide.md](../guide.md#permissions); MCP in [mcp.md](mcp.md).

### Engine isolation

These apply when `provider` is `Engine`. See [engines.md](engines.md#isolation).

| Field | Default | Meaning |
|---|---|---|
| `engine_isolation` | `EngineIsolation::Default` | `InProcess` loads the engine into your process. `Process` runs it in an `agent-engine-host` child. `Default` uses the isolation the engine was added with (`engines::add`, or its entry in an `engines.json`), otherwise `InProcess`. |
| `engine_host_path` | `""` | Path of the `agent-engine-host` executable for `Process`. Empty = `AGENT_ENGINE_HOST`, then next to your program, then `../bin`, then `PATH`. |
| `engine_timeout_ms` | `0` | For `Process`: a request running longer kills the host and fails with `EngineCrashed`. `0` = no limit. |
| `engine_cancel_grace_ms` | `2000` | For `Process`: time an engine gets to stop after `request_cancel` before the host is killed. |
| `engine_max_restarts` | `3` | For `Process`: automatic host restarts allowed per minute. |
| `engine_sandbox` | `{enabled = true, allow_tmp = false, network = Deny}` | OS sandbox for the engine host. |
| `gpu_cache_dir` | `""` | Where GPU drivers keep compiled kernels when `engine_sandbox.gpu` is set. Empty = `<workspace>/cache/gpu`. See [gpu.md](../gpu.md). |

### Logging and progress

| Field | Default | Meaning |
|---|---|---|
| `logger` | `nullptr` | Receives log messages. `logger_user_data` is passed to it. See [core.md](core.md#callbacks). |
| `on_progress` | `nullptr` | Progress of long engine jobs. `progress_user_data` is passed to it. |

### Required values

`init` returns `InvalidConfig` when:

- `workspace_dir` is empty, or contains any of `"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, a newline or a carriage return. Use a plain path.
- `context_window` or `max_tokens` is 0 or negative.
- `temperature` is negative or not a finite number.

---

## init

```cpp
Result<Session> init(const Config &config);
```

Creates a session from a complete `Config`. On failure it returns an error and leaves nothing behind.

`init` does these steps, in order:

1. Checks `workspace_dir` and the numeric fields (see [Required values](#required-values)).
2. Creates `models`, `memory`, `logs` and `tmp` under `workspace_dir`.
3. Creates the session arena (`arena_capacity`, or its default).
4. Opens the log file when `enable_file_logging` is set.
5. Initialises the provider. For an engine, this loads the engine and the model, so it is the slow step.
6. Creates the worker arenas when the engine runs parallel requests and `max_parallel_subagents` is above 0.
7. Opens the memory store (`memory.store`, or the in-memory store).
8. Registers the default tools when `auto_default_tools` is set, and builds the tool index.
9. Logs `Session initialised.` at `Info` level.

---

## Session struct

```cpp
struct Session {
    Config       config;   // the configuration the session was created with
    Arena        arena;    // the session's arena (single-threaded turns)
    SessionStats stats;    // statistics since init or the last reset_stats (see stats.md)
    // Other fields are internal.
};
```

Read `config`, `arena` and `stats` directly if you need to. Change `config.tools` only as described in [Tools after init](#tools-after-init). Everything else is managed by the library.

---

## Tools after init

Add or replace tools after `init` by changing `config.tools`, then rebuild the tool index:

```cpp
agent::Tool extra{
    .name             = "send_email",
    .description      = "Send an email.",
    .parameter_schema = R"({"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]})",
    .callback = [](std::string_view args, void *) -> agent::Result<std::string> {
        // parse args, send the mail
        return agent::ok(std::string("sent"));
    },
};

session.config.tools.push_back(extra);
agent::rebuild_tool_index(session);
```

The `string_view` fields of a `Tool` must stay valid for as long as the session exists. String literals and static data are safe. Tool details are in [tools.md](tools.md).

---

## Cancellation

```cpp
void request_cancel(Session &session);
void clear_cancel(Session &session);
```

`request_cancel` stops the engine work in progress and makes every later inference on this session fail with `ErrorCode::Cancelled`, until `clear_cancel` is called. It is safe to call from any thread, for example from a signal handler's worker thread or a UI button.

```cpp
std::thread stopper([&] {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    agent::request_cancel(session);
});

auto r = agent::run_turn(session, conv, "Write a long essay");
stopper.join();
agent::clear_cancel(session);
```

---

## KV cache

```cpp
Result<void> clear_provider_kv_cache(Session &session);
```

Frees the provider's cached attention state for the session's sequences. Use it after a large change of context, such as between unrelated graph runs. Call it only when no inference is running on the session. For providers without a cache, it does nothing.

---

## Statistics

```cpp
SessionStats get_stats(const Session &session);
void         reset_stats(Session &session);
```

Session-wide totals. See [stats.md](stats.md).

---

## Backend lifetime

`AgentRuntime` calls these for you. Call them yourself only if you do not use `AgentRuntime`:

```cpp
void init_backend();   // once, at process start
void free_backend();   // once, at process exit
```

---

## Related pages

- [conversation.md](conversation.md): `run_turn`, `infer`, conversations
- [engines.md](engines.md): engines, model roles and isolation
- [tools.md](tools.md): tools, the default tool set and the sandbox
- [memory.md](memory.md): memory store and `MemoryConfig`
- [stats.md](stats.md): statistics
