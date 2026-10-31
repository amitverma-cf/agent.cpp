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
