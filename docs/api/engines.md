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

Each request carries one task (`agent_request::task`): `AGENT_CAP_TEXT_GEN` if the engine supports
it, otherwise the lowest capability bit the engine declares (e.g. `AGENT_CAP_GENERIC` for the onnx
engine). Tensors on a `MessageView` are sent as `AGENT_PART_TENSOR` parts, and tensor parts in the
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

## Writing an engine

Export one function:

```c
AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void);
```

`agent_engine_v1` declares the ABI version, a name, capability bits (`AGENT_CAP_TEXT_GEN`,
`AGENT_CAP_ASR`, `AGENT_CAP_TTS`, `AGENT_CAP_IMAGE_GEN`, `AGENT_CAP_MESH_GEN`, ...), and the functions
`create`, `destroy`, `run`, `free_response` and `last_error`, plus the optional `count_tokens` and
`reset_cache`.

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
`AGENT_ENGINE_FLAG_CONCURRENT` (parallel subagents share one continuously batched decode).

`Config::engine_options`: `gpu_layers`, `threads`, `chat_template` (overrides the template embedded in the GGUF),
`mmproj` (path to a multimodal projector GGUF, enables image / audio input).

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

The core currently sends every request as `AGENT_CAP_TEXT_GEN`, so the sdcpp engine is driven
through the C ABI directly until the core routes image tasks.
