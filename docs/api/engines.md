# Engines

`#include <agent-cpp/agent.hpp>` · plugin authors: `#include <agent-cpp/engine_abi.h>`

An **engine** is a shared library wrapping an inference project (llama.cpp, whisper.cpp,
stable-diffusion.cpp, sherpa-onnx, ...). The core loads engines at runtime, so nothing is
compiled in. Each plugin carries its own copy of ggml or ONNX Runtime.

## Using an engine

```cpp
agent::Config cfg;
cfg.provider       = agent::AiProvider::Engine;
cfg.engine         = "./engines/agent-engine-llama.so";   // or a registered engine name
cfg.model          = ".workspace/models/model.gguf";
cfg.engine_options = R"({"gpu_layers": 99})";            // engine-specific, optional
cfg.workspace_dir  = ".workspace";

auto session = agent::init(cfg);
```

`init()` loads the plugin if `Config::engine` is a path, then calls the engine's `create()` with:

```json
{"model": "...", "context_window": 2048, "max_tokens": 512, "temperature": 0.7,
 "parallel": 4, "options": { ...Config::engine_options... }}
```

Each request carries one task (`agent_request::task`, one `AGENT_CAP_*` bit from `engine_abi.h`).
Pick it per call with `InferRequest::task` or `agent::run_task`, or per session with
`Config::default_task` (which also drives `run_turn` / `run_turn_with_media`):

```cpp
agent::InferRequest req{.messages = {&msg, 1}};
auto speech = agent::run_task(session, AGENT_CAP_TTS, req);   // same as req.task = AGENT_CAP_TTS; infer()

cfg.default_task = AGENT_CAP_IMAGE_GEN;                         // every run_turn of this session draws
```

An explicit task must be a single bit the engine declares (otherwise `InvalidConfig` for zero or
several bits, `UnsupportedProvider` for an undeclared one; `Config::default_task` is checked at
`init()`). With no task set (0) the core picks one automatically: `AGENT_CAP_TEXT_GEN` if the engine
supports it, otherwise the lowest capability bit the engine declares (e.g. `AGENT_CAP_GENERIC` for
the onnx engine). Tool definitions (`tools_json`) are only sent with `AGENT_CAP_TEXT_GEN`.

Tensors on a `MessageView` are sent as `AGENT_PART_TENSOR` parts, and tensor parts in the
reply come back in `InferResponse::output_tensors`.

Files that engines generate (audio, images, video, meshes) are written to `<workspace>/artifacts/` and
come back as `Message::media` entries (`path` + `mime`) on the assistant message; they persist with the
history. To send media to an engine:

```cpp
const agent::Media media[] = {{"photo.png", "image/png"}};
auto reply = agent::run_turn_with_media(session, memory, "What is in this picture?", media);
```

Long jobs report progress through `Config::on_progress`. `agent::request_cancel(session)` (any thread)
stops in-flight engine work with `ErrorCode::Cancelled` and blocks further inference until
`agent::clear_cancel(session)`.

## Registry

```cpp
Result<void>        register_engine(const agent_engine_v1 *engine);   // engine linked into the host
Result<std::string> load_engine_plugin(std::string_view path);        // dlopen/LoadLibrary; returns the name
const agent_engine_v1 *find_engine(std::string_view name);
```

Engines are process-wide and stay loaded until exit. Loading the same plugin twice is a no-op.

`load_engine_manifest("engines.json")` loads every plugin listed in a manifest and returns their names:

```json
{"engines": [{"path": "agent-engine-llama.so"}, {"path": "/opt/agent/engines/agent-engine-sdcpp.so"}]}
```

Relative paths resolve against the manifest's directory. Loading stops at the first plugin that fails.
An entry may also carry `"isolation": "process"` or `"inprocess"` (see [Isolation](#isolation)) and a
`"name"`. `"process"` entries are not loaded into this process at all: the manifest only records the
path, under `"name"` or the name derived from `agent-engine-<name>.<ext>`.

## Isolation

```cpp
enum class EngineIsolation { Default, InProcess, Process };

cfg.engine_isolation       = agent::EngineIsolation::Process;
cfg.engine_host_path       = "";    // default: $AGENT_ENGINE_HOST, next to the executable, <exe>/../bin, PATH
cfg.engine_timeout_ms      = 0;     // per-request watchdog, 0 = none
cfg.engine_cancel_grace_ms = 2000;  // after request_cancel(), then the host is killed
cfg.engine_max_restarts    = 3;     // automatic restarts per rolling minute
```

- **InProcess**: the plugin is `dlopen`ed into this process and called directly. Lowest latency, but a
  segfault, `abort()` (`GGML_ASSERT`), `exit()` (sherpa-onnx on a corrupt model), an escaped C++
  exception or an OOM kill in the engine takes the whole process down.
- **Process**: `init()` starts `agent-engine-host <plugin>`, a small executable that loads the plugin
  and serves the same ABI over its stdin/stdout. The plugin is never loaded into the calling process.
  `infer()`, `count_tokens` and the KV-cache reset go over the pipe; name, version, caps and flags come
  from the host at startup.
- **Default** (the `Config` default): the engines.json `"isolation"` of the engine if it was loaded with
  `load_engine_manifest`, otherwise InProcess, so library users and the test suite keep the old
  behaviour. The `agent` CLI defaults to Process (`--inprocess` or `"isolation": "inprocess"` in its
  engines.json opts out).

Under Process isolation the host also runs in an **OS sandbox** (`Config::engine_sandbox`, on by
default; Linux enforces it): the plugin, the model's directory and paths named in `engine_options`
are read-only, the workspace is read-only, `workspace/artifacts` is read-write, the network and
`/tmp` are off, and `engine_sandbox.memory_limit_bytes` becomes the host's `RLIMIT_AS`. An engine
that needs more must have it listed in `engine_sandbox.read_only_paths` / `read_write_paths`. See
[../security-sandbox.md](../security-sandbox.md).

Failure semantics under Process isolation:

| Event | In-flight call | Next call |
|---|---|---|
| Host dies (signal, `abort()`, `exit()`, OOM kill, pipe closed) | `ErrorCode::EngineCrashed`; the message names the exit status or signal and quotes the last lines of the host's stderr | the host is restarted and the engine re-created (same create config; the KV cache is lost) |
| Request exceeds `engine_timeout_ms` (run, token count, cache reset; not model loading) | host killed, `EngineCrashed` ("... killed by the watchdog") | restart |
| `request_cancel()` | a Cancel message sets the flag the engine polls (`agent_request::cancel`); a cooperative engine returns `Cancelled` and the host lives on | normal |
| Engine ignores the cancel for `engine_cancel_grace_ms` | host killed, `Cancelled` | restart (not counted against the limit) |
| More than `engine_max_restarts` restarts in the last minute | — | `EngineCrashed` ("restarts in the last minute ...") until the window frees up |
| `create()` fails | `init()` returns `ModelLoadFailed` (as in-process) | — |
| Host missing / plugin fails to load | `init()` returns `ProviderInitFailed` | — |

The session itself always survives: memory, history and tools are untouched, and the failed turn can
simply be retried.

The host's stderr (the engine's logs, and anything it prints to stdout, which the host redirects) is
forwarded to this process's stderr, so redirecting fd 2 captures it as it would for an in-process
engine. The host ignores SIGINT (Ctrl-C in a terminal reaches the whole process group; cancellation
goes through the protocol) and exits when its stdin closes.

**Concurrent requests.** For engines with `AGENT_ENGINE_FLAG_CONCURRENT` (llama, onnx) several
requests run at once through one host, exactly as in process: parallel graph branches get their own
threads (`supports_parallel` is true) and llama's continuous batching shares one decode across them.
Every frame carries its request's id. In the host a reader thread queues `Run`s to a pool of worker
threads (created on demand, then reused), each executing one `run()`; `Cancel <id>` flags just that
request. In the core, whichever waiting caller is free reads the pipe and hands the other callers
their frames, so a lone call reads its own replies with no thread hop, and streaming callbacks run on
the calling thread. Engines without the flag still get one `run()` at a time. When the host dies,
every call in flight fails with `EngineCrashed`; an ignored cancel or the watchdog kills the host
the same way (the request that triggered it gets `Cancelled` / the watchdog message, the others
`EngineCrashed`).

**Shared memory.** Tensor and BYTES payloads of at least 64 KiB travel in shared memory (POSIX
`shm_open` + `mmap`; on Windows a named `CreateFileMapping`, written but untested) instead of through
the pipe, in both directions. Segments are mapped once and reused: each core call slot has a request
segment and the host keeps a pool of response segments, handed back with a `Release` frame once the
core has copied the response out, so steady-state calls make no shm syscalls. Names are removed as
soon as both sides have them mapped, so nothing stays in `/dev/shm` when a process dies (names of a
crashed host are swept). Both sides check at startup that they can map segments; otherwise, or with
`AGENT_ENGINE_SHM=0`, payloads go inline through the pipe (max 1 GiB per frame). FILE parts pass paths
only, and the host writes outputs straight to `<workspace>/artifacts/`. A sandboxed host needs
`/dev/shm` (Linux) to use shared memory.

**Protocol** (`src/core/engines/engine_ipc.hpp`, internal; both ends come from the same build):
length-prefixed binary frames (`u32 size | u32 type | u64 request id | payload`, native byte order),
no JSON on the hot path. core → host: `Create`, `Run`, `Cancel` (out of band, handled by the host's
reader thread while `run()` is busy), `CountTokens`, `ResetCache`, `Destroy`, `Shutdown`, `Release`.
host → core: `Ready` (protocol/ABI version, caps, flags, shared-memory support, pid, name, version),
`Created`, `Chunk`, `Progress`, `Response`, `Count`, `Done`, `Error` (status + message). Requests and
responses carry the full `agent_request` / `agent_response`: messages, text / FILE / BYTES / TENSOR
parts, tool calls, `tools_json`, `params_json`, `output_dir`, usage.

Limits:

- Engines linked in with `register_engine` have no plugin file and can't run out of process
  (`InvalidConfig`).
- Windows: written (CreateProcess + pipes via `src/hal/child_process`, file mappings), not yet compiled
  or tested.

Cost (`test_agent "Process isolation per-turn overhead"`, mock engine, Linux, Release build on 4
shared and busy cores): a whole turn takes ~1 µs in process and 10-90 µs out of process (a pipe round trip
and its context switches; the spread is scheduler noise); streaming 65 chunks costs
40-230 µs. Negligible next to real inference, but in-process stays available for the lowest latency.
An 8 MiB tensor in and 8 MiB out per call (`test_agent "Large tensor round trip cost" -s`) costs
~1.3 ms in process, ~2.8 ms through the host with shared memory and 15-28 ms inline through the pipe.
Three parallel graph branches on llama (qwen2.5-0.5b Q4_K_M, 48 tokens each, CPU) take the same time in
process and through the host (3.57 s vs 3.59 s, ~33 tok/s), i.e. continuous batching works out of process.

## Writing an engine

Export one function:

```c
AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void);
```

`agent_engine_v1` declares the ABI version (`AGENT_ENGINE_ABI_VERSION`, currently 2), its own
`struct_size`, a name, capability bits (`AGENT_CAP_TEXT_GEN`, `AGENT_CAP_ASR`, `AGENT_CAP_TTS`,
`AGENT_CAP_IMAGE_GEN`, `AGENT_CAP_MESH_GEN`, ...), and the functions `create`, `destroy`, `run`,
`free_response` and `last_error`, plus the optional `count_tokens` and `reset_cache`:

```c
static const agent_engine_v1 kEngine = {
    AGENT_ENGINE_ABI_VERSION, sizeof(agent_engine_v1), "mine", "1.0", AGENT_CAP_TEXT_GEN, 0,
    my_create, my_destroy, my_run, my_free_response, my_last_error, NULL, NULL,
};
```

Each request carries `agent_request::task` (one capability bit) and `flags`:
`AGENT_REQUEST_FLAG_AUTO_TASK` means the caller did not choose the task and the core picked it
(`TEXT_GEN` if declared, else the lowest declared bit), so an engine with several tasks may route by
content instead (sherpa does).

**Struct sizes (ABI 2).** `agent_engine_v1`, `agent_request` and `agent_response` start with (or, for
the engine table, follow `abi_version` with) a `uint32_t struct_size`: the `sizeof` the writer was
compiled with. Fields are only ever appended. Everything that exists in ABI 2 is the *v2 baseline*
(`AGENT_ENGINE_V2_SIZE`, `AGENT_REQUEST_V2_SIZE`, `AGENT_RESPONSE_V2_SIZE`): the core refuses engine
tables smaller than that and always passes requests and responses at least that large, so baseline
fields need no check. The rule for later growth:

- **Readers must check `struct_size` before touching any field past the v2 baseline**:
  `if (AGENT_HAS_FIELD(req, agent_request, some_new_field)) ...`.
- The host sets `response->struct_size` before `run()`. Engines never write it and never write past
  it: clear the response with `agent_response_reset(resp)`, not `*resp = agent_response{}` (which
  zeroes `struct_size` and, against an older host's smaller struct, would write past its end).
- `agent_message`, `agent_part` and `agent_tool_call` are passed in arrays and have no size field;
  they only change with a new `AGENT_ENGINE_ABI_VERSION`.

An ABI 1 plugin (no `struct_size`) is refused at load with "ABI version 1 does not match"; rebuild it
against the current header (the type and entry-point names kept their historical `_v1` suffix).

Rules:

- Only C types cross the boundary. Build with hidden visibility so the plugin's own ggml/ORT symbols
  stay private.
- Inputs are borrowed for the duration of the call. The response is engine-owned until
  `free_response()`.
- `run()` is serialized by the host unless the engine sets `AGENT_ENGINE_FLAG_CONCURRENT`.
- Honour `agent_request::cancel` and report long jobs through `on_progress`.

`engines/mock/mock_engine.cpp` is a complete minimal example.

## The llama engine

`engines/llama` (`agent-engine-llama`) renders prompts with the model's own Jinja chat template and
parses replies into content + tool calls in the model's native format, using llama.cpp's `common_chat`.
It sets `AGENT_ENGINE_FLAG_NATIVE_TOOLS` (the core only adds the skills list to the system message) and
`AGENT_ENGINE_FLAG_CONCURRENT` (parallel graph branches share one continuously batched decode).

`Config::engine_options`: `gpu_layers`, `threads`, `chat_template` (overrides the template embedded in the GGUF),
`mmproj` (path to a multimodal projector GGUF, enables image / audio input), `mode` (`"generate"`, the
default, `"embed"` or `"rerank"`, see below), `pooling` (embed mode: `"mean"`, `"cls"` or `"last"` to
override the model's pooling).

### Image and audio input

With `mmproj` set, the plugin loads the projector with llama.cpp's libmtmd (linked statically into the
plugin) and accepts `AGENT_PART_FILE` / `AGENT_PART_BYTES` parts in input messages:

- Images: anything stb_image decodes (PNG, JPEG, BMP, GIF, ...). Audio: WAV, MP3, FLAC, if the projector
  supports audio (e.g. Voxtral, Qwen2.5-Omni). The format is detected from the bytes, not the mime type.
- Each media part becomes libmtmd's media marker at its position in the message, so text and media keep
  their order in the rendered chat template. Markers typed by the user are stripped.
- Video (`video/*`) is rejected: libmtmd decodes video through an `ffmpeg` subprocess, which is not
  built into the plugin.

`caps` is a static field, so it lists what the plugin can do (`AGENT_CAP_VISION_IN`,
`AGENT_CAP_AUDIO_IN`), not what the loaded model can do. `run()` returns `AGENT_ERR_UNSUPPORTED` if a
request carries media and no `mmproj` was loaded, or if the projector lacks the needed modality.

```cpp
cfg.model          = ".workspace/models/SmolVLM-256M-Instruct-Q8_0.gguf";
cfg.engine_options = R"({"mmproj": ".workspace/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"})";
```

Prompts with media are prefilled on their own sequence under the batcher's lock (libmtmd encodes the
media and decodes text and embedding chunks in turn); generation then joins the shared batched rounds.

### Embeddings and reranking

Embedding and reranker models need a llama context created in embedding mode with pooling, which
can't generate text, so `options.mode` picks what an instance does:

| `mode` | Context | `run()` serves | Output |
|---|---|---|---|
| `"generate"` (default) | chat (as above) | `AGENT_CAP_TEXT_GEN` | text + tool calls |
| `"embed"` | `embeddings = true`, the model's pooling (mean if it declares none; `options.pooling` overrides) | `AGENT_CAP_EMBED` | one `"embedding"` tensor (f32 `[n_embd]`, L2-normalized) per TEXT part, in order |
| `"rerank"` | `embeddings = true`, `LLAMA_POOLING_TYPE_RANK` | `AGENT_CAP_RERANK` | one `"scores"` tensor (f32 `[n_docs]`): first TEXT part = query, later TEXT parts = documents |

`caps` is static, so it lists `AGENT_CAP_EMBED | AGENT_CAP_RERANK` next to text generation; an
instance returns `AGENT_ERR_UNSUPPORTED` for a task that doesn't match its mode (and for media parts
in embed / rerank mode). `mmproj` is only accepted in generate mode. Loading a model that declares
a non-rank pooling type (an embedding model) in rerank mode fails.

- Rerank inputs are built the way llama-server builds them: the model's `rerank` chat template
  (`{query}` / `{document}` placeholders) when the GGUF has one, otherwise
  `[BOS] query [EOS] [SEP] document [EOS]` following the vocab's add-BOS/EOS/SEP flags. Template text
  may contain special tokens; the query and documents are tokenized as plain text. Scores are the
  classifier's raw output (higher = more relevant).
- Up to 16 inputs are packed into one decode, each on its own sequence, with the batch sized to one
  micro-batch so non-causal (BERT-style) models see each sequence whole. An input is limited to
  `min(context_window, n_ctx_train)` tokens and longer ones are truncated (for rerank, the document
  first); keep RAG chunks well under that.
- Embed / rerank decodes on one instance are serialized; the instance still sets
  `AGENT_ENGINE_FLAG_CONCURRENT`, so concurrent callers queue rather than fail.

Use them through `agent::embed` / `agent::rerank` / the RAG pipeline ([rag.md](rag.md)):

```cpp
cfg.model          = ".workspace/models/bge-small-en-v1.5-q8_0.gguf";
cfg.context_window = 512;
cfg.engine_options = R"({"mode": "embed"})";
```

### Decisions: choice / score / yes-no

In generate mode the instance also serves `AGENT_CAP_DECIDE`: the request's messages hold a question
(plus images with `mmproj`), `params_json` the answers (`{"mode":"choice","options":[...]}`,
`{"mode":"score","scale":[1,5]}` or `{"mode":"yesno"}`). Each answer is scored as the assistant's reply
by the sum of its tokens' log-probabilities plus the end-of-turn token, after one shared prompt
prefill, and the sums are softmax-normalized. The response is a TEXT part with a JSON summary and an
f32 `"probs"` tensor. Native System One decision models (`<arch>.decision.type` in the GGUF) are
detected and rejected with `AGENT_ERR_UNSUPPORTED`. Use it through `agent::decide` /
`decide_score` / `decide_yes_no`; see [decide.md](decide.md) for the protocol and the scoring details.

`common_chat` reports errors by throwing, and exceptions must not cross the plugin ABI. So the
plugin catches at exactly those calls (template load/render, output parsing) and converts them to
status codes. This is the only exception boundary in agent.cpp.

## The onnx engine

`engines/onnx` (`agent-engine-onnx`) runs any ONNX graph through the ONNX Runtime C API. It declares
`AGENT_CAP_GENERIC` (raw named tensors in and out) and `AGENT_ENGINE_FLAG_CONCURRENT`: `OrtSession::Run`
is thread-safe and every call keeps its own buffers.

Build it against a prebuilt ONNX Runtime release (option `AGENT_ENGINE_ONNX`, default `OFF`):

```bash
cmake -B build -G Ninja -DAGENT_ENGINE_ONNX=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.31.1
```

`ONNXRUNTIME_ROOT` is an extracted release from https://github.com/microsoft/onnxruntime/releases
(`include/` + `lib/`). Without it CMake falls back to `find_package(onnxruntime)`. The core never links
ONNX Runtime. **Ship `libonnxruntime.so.1` (`.dylib` / `.dll`) next to the plugin**: the plugin's RPATH
is `$ORIGIN`, and the build copies the library into the plugin's output directory.

`Config::model` is the `.onnx` path. `Config::engine_options`:

| Option | Default | |
|---|---|---|
| `threads` | ORT default | intra-op threads |
| `execution_provider` | `"cpu"` | `"cpu"` or `"cuda"`. `"cuda"` needs an ORT build with the CUDA EP; otherwise `create()` fails with a clear error |
| `device_id` | `0` | CUDA device |
| `graph_optimization` | `"all"` | `"disable"`, `"basic"`, `"extended"` or `"all"` |

Input and output names and input types are read once at load time. Each model input is fed from the
last tensor part with the same name across the request's messages; dtype and byte size must match.
A missing input fails with `AGENT_ERR_INVALID_CONFIG` ("missing required input tensor 'x' ...").
Every model output comes back as a tensor part. Supported dtypes are those of `agent_dtype`
(f32, f16, i8, i32, i64, u8).

```cpp
cfg.provider = agent::AiProvider::Engine;
cfg.engine   = "./engines/onnx/agent-engine-onnx.so";
cfg.model    = "add.onnx";                       // sum = a + b
auto session = agent::init(cfg);

const int64_t shape[] = {2, 2};
const float a[4] = {1, 2, 3, 4}, b[4] = {10, 20, 30, 40};
const agent::TensorView in[] = {
    {"a", agent::DType::Float32, shape, {reinterpret_cast<const uint8_t *>(a), sizeof(a)}},
    {"b", agent::DType::Float32, shape, {reinterpret_cast<const uint8_t *>(b), sizeof(b)}},
};
agent::MessageView msg{.role = "user", .tensors = in};
auto out = agent::infer(session.value, {.messages = {&msg, 1}});
// out.value.output_tensors[0]: "sum", f32 [2,2] = {11, 22, 33, 44}
```

Not yet supported: cancelling inside a running `Run()` (the flag is checked before it starts),
IoBinding / device-resident outputs, and non-tensor (sequence/map) outputs.

## The sdcpp engine

`engines/sdcpp` (`agent-engine-sdcpp`) wraps [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(`vendor/stable-diffusion.cpp`, pinned to a release tag) for image generation: SD 1.x/2.x/XL, SD3,
FLUX, Qwen-Image, Z-Image and whatever else that sd.cpp release loads.

It is opt-in because it builds a second copy of ggml:

```sh
git submodule update --init vendor/stable-diffusion.cpp
git -C vendor/stable-diffusion.cpp submodule update --init --depth 1 ggml
cmake -B build -G Ninja -DAGENT_ENGINE_SDCPP=ON [-DAGENT_SDCPP_BUILD_JOBS=4]
```

sd.cpp's ggml uses the same CMake target names as llama.cpp's, so it is built as a nested CMake project
(`ExternalProject`) and its static archives are linked into the plugin with hidden visibility (on Linux
also `-Wl,--exclude-libs,ALL`). The llama and sdcpp plugins can be loaded into one process.
`AGENT_SDCPP_BUILD_JOBS` caps the nested build's parallelism.

- Name `sdcpp`, caps `AGENT_CAP_IMAGE_GEN | AGENT_CAP_IMAGE_EDIT`, flags `0`.
- `IMAGE_GEN` (txt2img): the prompt is the text of the last user message.
- `IMAGE_EDIT` (img2img): the same, plus the last image `FILE` (or `BYTES`) part of that message as the
  init image.
- The image is written as `<output_dir>/sdcpp-<unix ms>-<n>.png` and returned as a `FILE` part
  (`image/png`) followed by a text part `generated image: <path>`.
- Other tasks fail with `AGENT_ERR_UNSUPPORTED`.

`create()` config: `model` is a single-file checkpoint (`.gguf`, `.safetensors`, `.ckpt`). Split models
(FLUX, SD3, Qwen-Image, ...) leave it empty and set components in `options`:

| Option | Meaning |
|--------|---------|
| `diffusion_model`, `vae`, `taesd`, `clip_l`, `clip_g`, `t5xxl`, `llm` | component file paths (`llm` is the LLM text encoder of Qwen-Image / Z-Image / FLUX.2) |
| `threads` | CPU threads (default: physical cores) |
| `flash_attn`, `eager_load`, `verbose` | booleans; `verbose` forwards sd.cpp info logs to stderr (warnings and errors always are) |
| `prediction` | override the prediction type (`eps`, `v`, `edm_v`, `sd3_flow`, `flux_flow`, ...) |
| `width`, `height`, `steps`, `cfg_scale`, `guidance`, `sampler`, `scheduler`, `seed`, `negative_prompt`, `strength`, `clip_skip`, `vae_tiling` | generation defaults |

`params_json` per request accepts the same generation keys (`width`, `height`, `steps`, `seed`,
`negative_prompt`, `cfg_scale`, `guidance`, `sampler`, `scheduler`, `strength`, `clip_skip`,
`vae_tiling`). Defaults: 512x512 (img2img: the input image size, rounded down to a multiple of 8), 20 steps, `cfg_scale` 7, random seed (`-1`), and the model's own
default sampler and scheduler. Sampler and scheduler names are sd.cpp's (`euler`, `euler_a`, `lcm`, ...;
`discrete`, `karras`, ...).

Progress goes to `on_progress(step, total)`; the counter restarts per phase (sampling, VAE decode, and
lazy weight loading on the first run unless `eager_load` is set). `cancel` is checked before starting
and at every sampler step, then `run()` returns `AGENT_ERR_CANCELLED`.

sd.cpp's log and progress callbacks are process-global, so generations from different sdcpp instances
are serialized by a mutex inside the plugin.

Through the core, pick the task explicitly: `run_task(session, AGENT_CAP_IMAGE_EDIT, req)`, or
`Config::default_task = AGENT_CAP_IMAGE_GEN` so `run_turn` draws (the automatic task is the lowest
declared bit, `IMAGE_GEN`).

## The trellis engine

`engines/trellis` (`agent-engine-trellis`) wraps [trellis2.cpp](https://github.com/localai-org/trellis2cpp)
(`vendor/trellis2cpp`, MIT, pinned to commit `2f3e6e2` of its default `pbr-textures` branch; the repo has no
tags) for [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D. It goes through the port's
flat C API (`trellis2_capi.h`), so none of the port's C++ crosses the plugin boundary.

It is opt-in because it builds a third copy of ggml (the port's own fork):

```sh
git submodule update --init vendor/trellis2cpp
git -C vendor/trellis2cpp submodule update --init --depth 1 ggml
cmake -B build -G Ninja -DAGENT_ENGINE_TRELLIS=ON [-DAGENT_TRELLIS_BUILD_JOBS=4]
```

Like sdcpp it is a nested `ExternalProject` (target names clash with llama.cpp's ggml), linked statically
with hidden visibility; on Linux `-Wl,--exclude-libs,ALL` plus a version script leave `agent_engine_get_v1`
as the only exported symbol. The nested build is CPU-only (CGAL print remeshing and Metal are off).
`AGENT_TRELLIS_BUILD_JOBS` caps its parallelism.

- Name `trellis`, caps `AGENT_CAP_MESH_GEN`, flags `0`.
- `MESH_GEN`: the input is the last image `FILE` (or `BYTES`) part of the last user message (anything
  stb_image decodes: PNG, JPEG, BMP, ...). No image fails with `AGENT_ERR_INVALID_CONFIG`; other tasks
  with `AGENT_ERR_UNSUPPORTED`.
- The mesh is written as `<output_dir>/trellis-<unix ms>-<n>.glb` and returned as a `FILE` part
  (`model/gltf-binary`) followed by a text part `generated 3D model: <path> (<v> vertices, <t> triangles)`.

`create()` config: `model` is a directory with the GGUFs, found by their published names
(`<name>_f16.gguf`, else `_f32.gguf`, else `.gguf`). Which files exist decides what the pipeline can do:

| Option (file) | Needed for | Source |
|---|---|---|
| `dino` (`dino_f16.gguf`, 0.6 GB) | always | [LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF](https://huggingface.co/LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF) (DINOv3 License) |
| `ss_flow` (`ss_flow_f16.gguf`, 2.6 GB) | always | [LocalAI-io/TRELLIS.2-4B-GGUF](https://huggingface.co/LocalAI-io/TRELLIS.2-4B-GGUF) (MIT) |
| `ss_dec` (`ss_dec_f16.gguf`, 0.15 GB) | always | [LocalAI-io/TRELLIS-image-large-GGUF](https://huggingface.co/LocalAI-io/TRELLIS-image-large-GGUF) (MIT) |
| `slat_flow` + `shape_dec` (2.6 + 0.95 GB) | 512³ fine mesh | TRELLIS.2-4B-GGUF |
| `slat_flow_1024` (2.6 GB) | 1024³ cascade | TRELLIS.2-4B-GGUF |
| `shape_enc`, `tex_dec`, `tex_flow` (`tex_slat_flow_512`), `tex_flow_1024` | PBR vertex colours | TRELLIS.2-4B-GGUF |

The three required files alone give the coarse path (64³ occupancy, marching cubes, untextured). Each
option may also be set explicitly (relative paths resolve against `model`); `""` disables auto-discovery
of that file, e.g. `"slat_flow_1024": ""` to skip loading the 1024 model. Missing files fail with
`AGENT_ERR_MODEL_LOAD_FAILED`. There is no learned background-removal model: the port turns a
border-connected near-white/near-black background into alpha (or uses the image's own alpha).

Other options: `threads` (CPU threads; default all cores; applied through the port's
`TRELLIS2_N_THREADS` environment variable while loading/generating) and the generation defaults below.
`params_json` per request accepts the same generation keys:

| Key | Default | Meaning |
|---|---|---|
| `seed` | `-1` (random) | sampler seed |
| `steps` | `0` = 12 | flow sampler steps (all geometry stages) |
| `guidance` | `-1` = 7.5 | classifier-free guidance strength |
| `texture_steps` | `0` = 12 | texture flow steps |
| `pipeline` | `auto` | `auto` (best loaded), `coarse`, `512`, `1024` |
| `background` | `auto` | `auto`, `keep`, `black`, `white` |
| `components` | `remove_tiny` | GLB component cleanup: `remove_tiny`, `largest`, `all` |
| `texture_size` | `0` = 2048 | atlas size; only used when `T2GLB_XATLAS` is set (the default GLB uses vertex colours) |

The port has no mesh simplification on its CPU export path, so there is no simplify-ratio option; the GLB
keeps the generated polygon count.

Progress goes to `on_progress(step, total)` for every flow-sampling step; the counter restarts per
sampling phase (sparse structure, shape SLAT, HR SLAT, texture), and `1/1` is sent when the GLB is
written. trellis2.cpp has no cancellation hook inside a generation: `cancel` is checked before starting
and between generation and GLB export, then `run()` returns `AGENT_ERR_CANCELLED`.

trellis2.cpp itself does not throw (only its optional CGAL path does, and that is compiled out), but it is
C++ underneath: an out-of-memory `std::bad_alloc` inside it is not caught, and ggml asserts abort.
CPU generation is slow: the 512/1024 paths run two or three 1.3B-parameter DiTs over thousands of
tokens and the 1024³ decode needs ~14 GB of host RAM.

## The sherpa engine

`engines/sherpa` (`agent-engine-sherpa`) wraps [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) through
its C API for speech: Kokoro text-to-speech, Whisper / NeMo transducer (Parakeet TDT) speech-to-text, Silero
voice activity detection and pyannote speaker diarization.
Caps `AGENT_CAP_TTS | AGENT_CAP_ASR | AGENT_CAP_VAD | AGENT_CAP_DIARIZE`, flags 0.

Build against the official prebuilt shared release (`sherpa-onnx-v<ver>-linux-x64-shared.tar.bz2` from the
GitHub releases page; tested with v2.0.1):

```sh
cmake -B build -DAGENT_ENGINE_SHERPA=ON -DSHERPA_ONNX_ROOT=/path/to/sherpa-onnx-v2.0.1-linux-x64-shared
```

The plugin's RPATH is `$ORIGIN`: `libsherpa-onnx-c-api.so` and `libonnxruntime.so` ship **next to**
`agent-engine-sherpa.so` (the build copies them there). Keep the three files together when installing.

`Config::model` may be empty. Everything comes from `Config::engine_options`; only the components that are
configured get loaded:

```json
{
  "threads": 2,
  "provider": "cpu",
  "tts": {
    "kokoro": {"model": "kokoro-en-v0_19/model.onnx", "voices": "kokoro-en-v0_19/voices.bin",
               "tokens": "kokoro-en-v0_19/tokens.txt", "lexicon": "lexicon-en-us.txt",
               "data_dir": "kokoro-en-v0_19/espeak-ng-data", "lang": "en-us"},
    "speaker_id": 0,
    "speed": 1.0
  },
  "asr": {
    "whisper": {"encoder": "tiny.en-encoder.int8.onnx", "decoder": "tiny.en-decoder.int8.onnx",
                "tokens": "tiny.en-tokens.txt", "language": "en", "task": "transcribe"}
  },
  "vad": {
    "silero": {"model": "silero_vad.onnx"},
    "threshold": 0.5, "min_silence": 0.5, "min_speech": 0.25, "max_speech": 20, "streaming": false
  },
  "diarize": {
    "pyannote": {"model": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"},
    "embedding": {"model": "nemo_en_titanet_small.onnx"},
    "num_speakers": 0, "threshold": 0.5, "min_duration_on": 0.3, "min_duration_off": 0.5
  }
}
```

`asr` takes exactly one family: `whisper` (`encoder`, `decoder`, `tokens`, optional `language`, `task`),
`nemo_transducer` (Parakeet TDT: `encoder`, `decoder`, `joiner`, `tokens`) or `transducer` (same files, plus an
optional `model_type`). `lexicon`, `data_dir` and `lang` are optional for Kokoro, but sherpa-onnx 2.x removed
espeak-ng and ignores `data_dir`, so English text needs a lexicon
([lexicon-en-us.txt](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt));
words missing from it are skipped. Paths are checked at `create()`; a missing file fails with
`AGENT_ERR_MODEL_LOAD_FAILED`.

`vad` takes the `silero` family ([silero_vad.onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx),
MIT). `threshold` is the speech probability, `min_silence` / `min_speech` / `max_speech` are in seconds (longer
speech is split). `diarize` needs a pyannote segmentation model
([sherpa-onnx-pyannote-segmentation-3-0](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2),
MIT) and a speaker embedding model from the `speaker-recongition-models` release (sic), for example
`nemo_en_titanet_small.onnx` (CC-BY-4.0). `num_speakers` > 0 fixes the number of speakers; otherwise
clusters closer than `threshold` merge. With two Kokoro voices, TitaNet separated the speakers in our test,
while `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` mixed them up, also in sherpa-onnx's
own `sherpa-onnx-offline-speaker-diarization` tool.

Requests are routed by task, and by content when the task was not chosen explicitly. Pick it with
`run_task(session, AGENT_CAP_TTS, req)` / `InferRequest::task`, or `Config::default_task` for
`run_turn`:

| `task` | Last user message | Result |
|---|---|---|
| `TTS` | text | speech: WAV file in `output_dir` (`FILE`, `audio/wav`) + a short `TEXT` summary |
| `ASR` | audio `FILE`/`BYTES` part, `asr` configured | transcript as a `TEXT` part |
| `ASR` | no audio part | `AGENT_ERR_INVALID_CONFIG` |
| automatic (no task set: the core sends `ASR`, the lowest bit, with `AGENT_REQUEST_FLAG_AUTO_TASK`) or `TEXT_GEN` | audio part, `asr` configured | transcript, as for `ASR` |
| automatic or `TEXT_GEN` | audio part, no `asr`, `vad` configured | speech segments, as for `VAD` |
| automatic or `TEXT_GEN` | audio part, only `diarize` configured | speaker segments, as for `DIARIZE` |
| automatic or `TEXT_GEN` | text, no audio | speech, as for `TTS` |
| `VAD` | audio part (+ optional `TEXT` command) | `TEXT` `[{"start":0.52,"end":1.84}, ...]` (seconds) + `TENSOR` `segments` F32 `[n, 2]` |
| `DIARIZE` | audio part | `TEXT` `[{"start":0.03,"end":4.64,"speaker":0}, ...]` + `TENSOR` `segments` F32 `[n, 3]` (start, end, speaker) |

Hosts that do not send an explicit task send the engine's lowest capability bit, `ASR`; give a session
only a `vad` (or only a `diarize`) component and its audio requests are segmented instead of transcribed.
A task whose component is not configured returns `AGENT_ERR_UNSUPPORTED`; `VAD` / `DIARIZE` without an
audio part return `AGENT_ERR_INVALID_CONFIG`.

Audio input is a `FILE` or `BYTES` WAV (16-bit PCM mono; other sample rates are resampled to 16 kHz), or an
`F32` `TENSOR` of mono samples at 16 kHz (any name, shape `[n]`). Through the C++ API that is a
`TensorView` on the message, so audio can go from a microphone to the engine without a file.

**Streaming VAD.** With `"streaming": true` the detector keeps its state between `run()` calls, so a host can
feed microphone audio in small chunks (`agent voice` sends 100 ms). Each call returns the segments that
*finished* during that call, with times counted from the start of the stream, plus one entry
`{"start":s,"end":now,"open":true}` while speech is still going on (the `TENSOR` holds only finished
segments). A `TEXT` part `reset` on the request starts a new stream (the clock goes back to 0), `flush`
ends it and returns whatever was still open. Without `streaming`, every call is a stream of its own. During synthesis each sentence is
streamed through `on_chunk(AGENT_CHUNK_AUDIO, samples, n_samples * sizeof(float))` (mono f32 at the model's
sample rate, which the `TEXT` part reports), progress goes to `on_progress(step, 1000)`, and `cancel` stops
generation at the next sentence. ASR reports progress 0/1 and 1/1 and the transcript as one `TEXT` chunk.

Limitation: sherpa-onnx validates configs and returns NULL for most errors, but some malformed model files
(for example a corrupt tokens file) make it call `exit()` inside `create()`. Only point it at real model files.
Licences: Kokoro (Apache-2.0), Whisper (MIT), Silero VAD (MIT), pyannote segmentation-3.0 (MIT) and the
TitaNet embedding model (CC-BY-4.0) are fine for the catalogue. 3D-Speaker embedding models are Apache-2.0 according to
the 3D-Speaker repository; check the individual model card.
