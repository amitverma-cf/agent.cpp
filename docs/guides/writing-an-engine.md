# Guide: writing an engine plugin

An engine plugin wraps an inference library (llama.cpp, audio.cpp, stable-diffusion.cpp, ONNX
Runtime, your own) behind the C ABI in [`<agent-cpp/engine_abi.h>`](../../include/agent-cpp/engine_abi.h).
The core loads it at runtime, in process or in a sandboxed `agent-engine-host`, and routes requests to
it by capability. This guide walks through ABI 2 with a complete plugin,
[`examples/04-custom-engine/reverse_engine.cpp`](../../examples/04-custom-engine/reverse_engine.cpp) (`agent-engine-reverse`):

- `AGENT_CAP_TEXT_GEN`: reverses the user's text, streamed one character per chunk;
- `AGENT_CAP_IMAGE_GEN`: writes a small PPM image to the artifacts directory, with progress and cancel.

[`examples/04-custom-engine/host.cpp`](../../examples/04-custom-engine/host.cpp) drives it
through libagent in process and under process isolation; both run as the `example.guide_engine` test.
To start a new engine outside this tree, copy [`examples/08-engine-plugin-template/`](../../examples/08-engine-plugin-template)
(a standalone CMake project against the installed package, see [Packaging](#7-building-and-packaging)).

## 1. The one exported symbol

A plugin is a shared library named `agent-engine-<name>.so` (`.dll`, `.dylib`) exporting one C function:

```c
AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void);
```

It returns a static table describing the engine:

```cpp
const agent_engine_v1 kReverseEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1), // struct_size: the size this plugin was compiled with
    "reverse",
    "1.0",
    AGENT_CAP_TEXT_GEN | AGENT_CAP_IMAGE_GEN,
    0, // flags: run() serialized by the host, plain-text tool instructions
    reverse_create,
    reverse_destroy,
    reverse_run,
    reverse_free_response,
    reverse_last_error,
    reverse_count_tokens,
    nullptr, // reset_cache: no KV cache
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kReverseEngine; }
```

- `name` is how sessions, `engines.json` and the CLI refer to the engine.
- `caps` lists every `AGENT_CAP_*` task the plugin can perform. It is static: it describes the plugin,
  not the model loaded (llama declares vision input whether or not a projector is loaded, and refuses
  image parts at `run()` without one).
- `flags`: `AGENT_ENGINE_FLAG_CONCURRENT` means `run()` may be called on one instance from several
  threads at once (parallel graph branches then really run in parallel; llama uses it for continuous
  batching). Without it the core serializes `run()` per instance, which is what this engine relies on
  for its per-instance output buffers. `AGENT_ENGINE_FLAG_NATIVE_TOOLS` means the engine formats
  `agent_request::tools_json` into the prompt itself; otherwise the core prepends plain-text tool
  instructions to the system message.
- Build with hidden visibility so this is the only exported symbol and your copy of ggml / ONNX
  Runtime cannot clash with another plugin's.

Only C types cross the boundary: the host and the plugin may be built with different compilers and
C++ runtimes. No exception may escape a plugin function (catch at the boundary and return a status;
llama does this around `common_chat`).

## 2. create and destroy

```cpp
int32_t reverse_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    // config_json: {"model": ..., "context_window": ..., "options": {...Config::engine_options...}}
    const std::string_view cfg = view(config_json);
    auto *inst = new ReverseInstance();
    inst->step_delay_ms = json_int(cfg, "step_delay_ms", 0);
    if (inst->step_delay_ms < 0) {
        delete inst;
        std::snprintf(err, err_cap, "reverse: step_delay_ms must be >= 0");
        return AGENT_ERR_INVALID_CONFIG;
    }
    *out_instance = inst;
    return AGENT_OK;
}
```

`create()` loads the model. The core sends `{"model", "context_window", "max_tokens", "temperature",
"parallel", "options"}`, where `options` is `Config::engine_options` verbatim (engine-defined JSON; the
example's `"step_delay_ms"` arrives there). On failure write a NUL-terminated message into `err` and
return an `AGENT_ERR_*` code: `init()` reports it as `ErrorCode::ModelLoadFailed`. `destroy()` frees
the instance. One plugin may have many instances (one per Session).

## 3. run: tasks, input parts, output parts

```cpp
int32_t reverse_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<ReverseInstance *>(instance);
    inst->error.clear();
    // Never `*resp = agent_response{}`: that would clear struct_size, and against an older host with a
    // smaller agent_response it would write past the end. agent_response_reset keeps struct_size.
    agent_response_reset(resp);
    resp->message.role = to_c("assistant");

    // Fields appended after the v2 baseline must be guarded, e.g.:
    //   if (AGENT_HAS_FIELD(req, agent_request, some_future_field)) ... req->some_future_field ...
    // Everything this engine reads is part of the baseline, so no check is needed here.

    switch (req->task) {
    case AGENT_CAP_TEXT_GEN:
        return run_text(inst, req, resp);
    case AGENT_CAP_IMAGE_GEN:
        return run_image(inst, req, resp);
    default:
        inst->error = "reverse: unsupported task " + std::to_string(req->task);
        return AGENT_ERR_UNSUPPORTED;
    }
}
```

**The task.** Every request carries exactly one capability bit in `req->task`. The caller picks it
(`InferRequest::task`, `agent::run_task`, `Config::default_task`); the core refuses a task the engine
does not declare before calling `run()`. When the caller did not choose, the core sends `TEXT_GEN` if
declared, else the lowest declared bit, and sets `AGENT_REQUEST_FLAG_AUTO_TASK` in `req->flags`; an
engine with several tasks may then route by content (audiocpp turns audio into ASR and text into TTS).

**Input.** `req->messages` is the conversation (`role` = `system` / `user` / `assistant` / `tool`). Each
message is a list of parts:

| `agent_part::kind` | Fields | Typical use |
|---|---|---|
| `AGENT_PART_TEXT` | `text` | prompts, chat turns |
| `AGENT_PART_FILE` | `path`, `mime` | images, audio, video on disk (`Media` / `run_turn_with_media`) |
| `AGENT_PART_BYTES` | `data`, `size`, `mime` | media in memory |
| `AGENT_PART_TENSOR` | `name`, `dtype`, `shape`, `ndim`, `data`, `size` | raw tensors (`TensorView` on a `MessageView`) |

Assistant messages carry `tool_calls`, tool messages a `tool_call_id`. `params_json` holds per-call
parameters (`max_tokens`, `temperature`, plus anything engine-specific), `tools_json` the tool
definitions (only for `TEXT_GEN`). Everything is borrowed for the duration of the call: copy what you
keep.

**Output.** Fill `resp->message` with parts (and `tool_calls` for a model that requests tools) and
`resp->usage`. The memory is the engine's: it must stay valid until `free_response()` or the next
`run()` on the instance (per thread, for a concurrent engine). The core maps text parts to the reply
text, FILE parts to `Media` entries on the assistant message, TENSOR parts to `output_tensors`:

```cpp
    inst->summary = "generated image: " + inst->image_path;
    inst->parts[0] = agent_part{};
    inst->parts[0].kind = AGENT_PART_FILE;
    inst->parts[0].path = to_c(inst->image_path);
    inst->parts[0].mime = to_c("image/x-portable-pixmap");
    inst->parts[1] = agent_part{};
    inst->parts[1].kind = AGENT_PART_TEXT;
    inst->parts[1].text = to_c(inst->summary);
    resp->message.parts = inst->parts;
    resp->message.n_parts = 2;
    return AGENT_OK;
```

Write generated files into `req->output_dir` (`<workspace>/artifacts`); it is the one directory a
sandboxed host may write to. On failure return a non-zero `AGENT_ERR_*` status and make `last_error()`
describe it; the core turns it into an `agent::Error`. `AGENT_ERR_CANCELLED` becomes
`ErrorCode::Cancelled`.

Engines that serve the RAG pipeline follow a fixed contract: `AGENT_CAP_EMBED` returns one f32
`"embedding"` tensor per input TEXT part; `AGENT_CAP_RERANK` takes the query and documents as TEXT
parts and returns one f32 `"scores"` tensor ([api/rag.md](../api/rag.md)).

## 4. Streaming, progress, cancel

```cpp
    for (size_t end = text.size(); end > 0;) {
        size_t start = end - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) --start;
        if (req->cancel && *req->cancel) {
            inst->error = "reverse: cancelled";
            return AGENT_ERR_CANCELLED;
        }
        const std::string_view ch(text.data() + start, end - start);
        if (req->on_chunk) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, ch.data(), ch.size());
        inst->reply += ch;
        end = start;
    }
```

- **`on_chunk(user_data, kind, data, size)`** streams output as it is produced: `AGENT_CHUNK_TEXT` (UTF-8
  token text; never split a code point) or `AGENT_CHUNK_AUDIO` (mono f32 PCM at the engine's rate,
  `size` in bytes). Text chunks reach `run_turn`'s token callback; the final response must still
  contain the whole reply. It may be NULL.
- **`on_progress(user_data, step, total)`** reports long jobs (diffusion steps, TTS sentences, mesh
  stages); `total` may be 0 when unknown. It reaches `Config::on_progress` / `InferRequest::on_progress`.
- **`*req->cancel`**: poll it between steps or tokens and return `AGENT_ERR_CANCELLED` when it is non-zero
  (`agent::request_cancel` sets it, from any thread). Under process isolation a cancel the engine
  ignores for `Config::engine_cancel_grace_ms` (2 s) gets the host killed.

The host driver checks all three: the streamed chunks add up to the reply, IMAGE_GEN reports 32/32,
and a progress callback that raises the cancel flag at step 8 makes the call fail with
`ErrorCode::Cancelled`:

```cpp
    int32_t cancel = 0;
    ProgressLog cancelling{.cancel = &cancel, .cancel_at = 8};
    agent::InferRequest creq{.messages = {&msg, 1}, .on_progress = on_progress, .progress_user_data = &cancelling, .cancel = &cancel};
    auto cancelled = agent::run_task(session, AGENT_CAP_IMAGE_GEN, creq);
```

## 5. Struct sizes (ABI 2)

`agent_engine_v1`, `agent_request` and `agent_response` carry a `uint32_t struct_size`: the `sizeof` the
writer was compiled with. Fields are only ever appended, so a plugin built today keeps loading into
later hosts and vice versa. The rules:

1. **Set `struct_size = sizeof(agent_engine_v1)`** in your table (second field, after `abi_version`).
   The host refuses tables smaller than the v2 baseline (`AGENT_ENGINE_V2_SIZE`) and plugins with
   another `abi_version`.
2. **Baseline fields need no check.** Everything in ABI 2 is the v2 baseline (`AGENT_REQUEST_V2_SIZE`,
   `AGENT_RESPONSE_V2_SIZE`); the host always passes requests and responses at least that large.
3. **Check before touching anything newer**: `if (AGENT_HAS_FIELD(req, agent_request, new_field))`. A host
   older than your header passes a smaller struct.
4. **Never write `response->struct_size` and never write past it.** Clear the response with
   `agent_response_reset(resp)`, not `*resp = agent_response{}` (that zeroes `struct_size` and, against
   an older host's smaller struct, writes past its end). Do the same in `free_response()`.
5. `agent_message`, `agent_part` and `agent_tool_call` travel in arrays and have no size field; they only
   change with a new `AGENT_ENGINE_ABI_VERSION`.

ABI 1 plugins (no `struct_size`) are refused at load; rebuild them against the current header.

## 6. Isolation and the sandbox

The same plugin runs in two places, chosen per Session (`Config::engine_isolation`) or per engine
(`"isolation"` in `engines.json`; the `agent` CLI defaults to process isolation):

- **In process**: `dlopen`ed into the application. Fastest; a crash, `abort()` or `exit()` in the
  engine kills the application.
- **Process**: loaded by `agent-engine-host`, which serves the ABI over pipes (tensors and BYTES of
  64 KiB and more through shared memory). Your code does not change, but it runs under these conditions:

| Under process isolation | What it means for the engine |
|---|---|
| OS sandbox (`Config::engine_sandbox`, on by default; enforced on Linux with Landlock + seccomp) | Readable: the plugin's directory, the model's directory, the directory of every path-like string in `engine_options`, system directories, the workspace. Writable: `output_dir` (`<workspace>/artifacts`) and `/dev/shm` only. No `/tmp`, no `$HOME` caches, no network: download nothing at `create()` time, and ask users to name extra files in `engine_options` or `engine_sandbox.read_only_paths` |
| `engine_sandbox.memory_limit_bytes` | becomes the host's `RLIMIT_AS`; reserve virtual memory accordingly (mmap'ed weights count) |
| crash, `abort()`, `exit()`, OOM kill | the call fails with `ErrorCode::EngineCrashed` (message quotes the host's last stderr lines); the next call restarts the host and calls `create()` again with the same config. Keep `create()` idempotent |
| `engine_timeout_ms` watchdog | a `run()` that takes longer gets the host killed |
| stdout | redirected to stderr, so logging with `printf` cannot corrupt the protocol |
| `register_engine` (an engine linked into the application) | cannot run out of process: there is no file for the host to load |

Run your engine both ways in its tests; the guide driver does:

```cpp
    exercise(plugin, workspace, agent::EngineIsolation::InProcess, "in process");
    // The same plugin in an agent-engine-host child, under Config::engine_sandbox (default: plugin dir and
    // model read-only, workspace read-only, workspace/artifacts read-write, no network, no /tmp).
    exercise(plugin, workspace, agent::EngineIsolation::Process, "process isolation");
```

## 7. Building and packaging

**Inside this repository** (a `src/engines/<name>/` directory, enabled by an `AGENT_ENGINE_<NAME>` option),
follow `src/engines/mock/CMakeLists.txt` and install with the top-level `agent_install_engine()`:

```cmake
# src/engines/mine/CMakeLists.txt (examples/CMakeLists.txt builds agent-engine-reverse the same way, minus the install)
add_library(agent-engine-mine SHARED mine_engine.cpp)
target_include_directories(agent-engine-mine PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(agent-engine-mine PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(agent-engine-mine PRIVATE ${AGENT_WARNINGS})
agent_install_engine(agent-engine-mine)   # + runtime libraries loaded through $ORIGIN, if any
```

**Standalone, against an installed agent.cpp** ([`examples/08-engine-plugin-template/`](../../examples/08-engine-plugin-template)):
`find_package(agent-cpp)` provides the header-only target `agent-cpp::engine_abi` and the same two
helpers, so a plugin repository needs nothing from this source tree:

```cmake
find_package(agent-cpp 0.1 REQUIRED)
agent_add_engine(agent-engine-echo src/echo_engine.c)   # SHARED, no "lib" prefix, hidden visibility
agent_install_engine(agent-engine-echo)                 # -> <prefix>/lib/agent/engines/
```

`agent_install_engine(<target> <libs>...)` also copies the runtime libraries a plugin loads through an
`$ORIGIN` RPATH (ONNX Runtime, ...), keeping symlinks whose targets are listed too. Prefer
linking your inference library statically (llama and sdcpp do), so the plugin is one file.

Then register it with the CLI, which loads it once to validate the ABI and prints its capabilities:

```sh
cmake --install build --prefix /opt/agent
/opt/agent/bin/agent engine add /opt/agent/lib/agent/engines/agent-engine-echo.so
```

## Checklist

- [ ] `abi_version = AGENT_ENGINE_ABI_VERSION`, `struct_size = sizeof(agent_engine_v1)`, honest `caps`.
- [ ] Hidden visibility; only `agent_engine_get_v1` exported; no exceptions across the boundary.
- [ ] `run()` switches on `req->task` and returns `AGENT_ERR_UNSUPPORTED` for anything else.
- [ ] `agent_response_reset()` in `run()` and `free_response()`; output memory owned until then.
- [ ] Files only in `output_dir`; nothing written elsewhere, no network at runtime.
- [ ] Streams through `on_chunk`, reports long jobs through `on_progress`, polls `cancel`.
- [ ] `AGENT_ENGINE_FLAG_CONCURRENT` only if `run()` really is thread-safe per instance.
- [ ] Tested in process and under process isolation.

Reference: [api/engines.md](../api/engines.md) (the built-in engines, the host protocol, failure
semantics), [security-sandbox.md](../security-sandbox.md).
