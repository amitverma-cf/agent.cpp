# agent.cpp

A fast, local, offline-first AI agent. It ships as the `agent` command-line app and as `libagent`, a C++20
runtime with a single public header. Models run on your machine through engine plugins (llama.cpp, ONNX
Runtime, audio.cpp, stable-diffusion.cpp, TRELLIS.2). No cloud API and no
global Python or Node install are needed.

Status: under active development. APIs, CLI flags and file formats change without notice until v1.0.0.

---

## Why agent.cpp

- **One local CLI for chat, tasks, graphs and voice.** `agent` opens a full-screen chat (markdown answers, tool-permission prompts, slash commands for
  `/pull`, `/doctor`, `/setup`), `agent task` runs one engine capability once, `agent run` executes declarative graphs, and `agent voice` is a spoken loop.
- **It tells you what will run on your machine.** `agent model fit` and `agent model recommend` read a model's header, compare it with your RAM and GPU, and
  estimate the speed from your memory bandwidth. Nothing is downloaded to find out.
- **Sleep, wake and schedule without a daemon.** A graph `wait` node pauses a run; `agent schedule` registers `agent wake` with cron or Task Scheduler.
- **Any local engine, one interface.** Text and vision with llama.cpp, speech (recognition, synthesis, VAD and
  diarization) with audio.cpp, images and video with stable-diffusion.cpp, 3D meshes with TRELLIS.2, and generic ONNX models.
  See the [engine matrix](#engines).
- **Engines run in their own process.** A crashing engine fails one turn, not the CLI. On Linux the engine host
  and `run_command` run under Landlock and seccomp. On Windows 11 the sandbox uses an AppContainer. Where the OS cannot provide a
  sandbox, or with `--no-sandbox`, things run unsandboxed and a warning says so: no weaker fallbacks.
- **MCP servers and skills.** Connect stdio MCP servers and load `SKILL.md` skills. Private copies of `bun` and
  `uv` run the servers that are published as npm or Python packages.
- **A small, explicit C++ SDK.** You name the engine and each model file (`engine_config`), and responses own their memory. Errors are `Result<T>`,
  with no exceptions, and engines, stores and tools are plain function tables (no virtual calls).

---

## Platforms

v1.0.0 targets Windows 11 and Linux. macOS: not tested, not supported in v1.0.0.

---

## 60-second quickstart

No prebuilt archives are published yet, so build the CLI from source (CMake 3.21+, Ninja, a C++20 compiler; see
[docs/getting-started.md](docs/getting-started.md) for Windows and OpenSSL):

```sh
cmake -B build -G Ninja
cmake --build build
./build/agent setup --yes                                              # home folder, hardware check, engines, config
./build/agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M        # download and smoke-test a model from Hugging Face
./build/agent chat                                                     # interactive chat with streaming and tools
./build/agent task text_gen -e llama -m qwen2.5-0.5b-instruct-q4_k_m --text "Say hi" --max-tokens 16
./build/agent run my-graph.json --model qwen2.5-0.5b-instruct-q4_k_m   # run a declarative agent graph
```

On Windows, run `.\build\agent.exe` in place of `./build/agent` from PowerShell.

Other commands you will use:

| Command | What it does |
|---|---|
| `agent model search [TEXT]` | the curated registry of tested models, by short name |
| `agent` (no arguments, in a terminal) | the full-screen chat, same as `agent chat` |
| `agent model fit SPEC` / `agent model recommend [TEXT]` | will a model run here and about how fast; search Hugging Face for what fits (`--task tts` etc. for the other engines) |
| `agent schedule add|ls|rm` / `agent wake` | run an agent command on a timer with cron or Task Scheduler; resume graph runs paused at a wait node |
| `agent doctor` | what is installed, disk use per area, and what is safe to clean |
| `agent secret set NAME` | store an API key in the OS credential store |
| `agent mcp` / `agent skill` / `agent sandbox` | MCP servers, skills, and folders that sandboxed tools may use |

`agent voice` runs on the audio.cpp engine (build with `AGENT_ENGINE_AUDIOCPP`) with GGUF speech models: an ASR model, a TTS model and optionally a VAD model. See [docs/cli.md](docs/cli.md#agent-voice).

Full CLI reference: [docs/cli.md](docs/cli.md). Setup, layout and runtimes: [docs/workspace.md](docs/workspace.md).

---

## Engines

<a id="engines"></a>

| Engine | Capabilities | Build option (default) |
|---|---|---|
| `llama` (llama.cpp) | text and tool calls, image and audio input (with `mmproj`), OCR, embeddings, rerank, decision models (choice, score, yes/no) | `AGENT_ENGINE_LLAMA` (ON) |
| `audiocpp` (audio.cpp) | text to speech (pocket-tts, supertonic-3), speech recognition (Parakeet TDT, Moonshine, CrisperWhisper), VAD, speaker diarization. GGUF models | `AGENT_ENGINE_AUDIOCPP` (OFF, needs the `vendor/audio.cpp` submodule) |
| `sdcpp` (stable-diffusion.cpp) | image generation, image editing, video generation (Wan 2.1 tested) | `AGENT_ENGINE_SDCPP` (OFF) |
| `trellis` (TRELLIS.2) | image to 3D mesh (GLB) | `AGENT_ENGINE_TRELLIS` (OFF) |
| `onnx` (ONNX Runtime) | generic named-tensor models | `AGENT_ENGINE_ONNX` (OFF) |
| `mock` | test engine, not released | always built for tests |

Build the engines from source with the options above. `agent engine add <path>` registers any plugin you built yourself.

---

## Build

```sh
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Native Windows (MSVC, Ninja, OpenSSL) and Linux instructions, and the install step, are in
[docs/getting-started.md](docs/getting-started.md). The options table is below. The scripts in `scripts/` are run by hand; see [CONTRIBUTING.md](CONTRIBUTING.md).

### CMake options

| Option | Default | Description |
|---|---|---|
| `AGENT_BUILD_CLI` | `ON` | Build the `agent` CLI (`apps/cli`); needs `AGENT_BUILD_HUB` |
| `AGENT_BUILD_HUB` | `ON` | Build `agent_hub`, the Hugging Face and GitHub downloader (needs OpenSSL) |
| `AGENT_CLI_AUDIO_DEVICE` | `ON` | Build microphone and speaker support (miniaudio) into the CLI; `OFF` leaves it out |
| `AGENT_ENGINE_LLAMA` | `ON` | Build the llama.cpp engine plugin (`src/engines/llama`) |
| `AGENT_LLAMA_UPSTREAM` | `ON` | Link llama.cpp's own server handlers in process (MTP models and native System One decision models) |
| `AGENT_ENGINE_ONNX` | `OFF` | Build the ONNX Runtime engine (`src/engines/onnx`). CMake downloads the pinned ONNX Runtime release; `-DONNXRUNTIME_ROOT=<dir>` is needed only where upstream publishes no archive |
| `AGENT_ENGINE_SDCPP` | `OFF` | Build the stable-diffusion.cpp engine for images and video (`src/engines/sdcpp`) |
| `AGENT_ENGINE_AUDIOCPP` | `OFF` | Build the audio.cpp speech engine (`src/engines/audiocpp`: TTS, ASR, VAD, diarization); needs the `vendor/audio.cpp` submodule |
| `AGENT_AUDIOCPP_MODEL_SET` | `custom` | audio.cpp model families compiled in: `custom` builds the families in `AGENT_AUDIOCPP_MODELS`, `full` builds all of them |
| `AGENT_AUDIOCPP_MODELS` | (list in `src/engines/audiocpp/CMakeLists.txt`) | Model families built when `AGENT_AUDIOCPP_MODEL_SET=custom` |
| `AGENT_ENGINE_TRELLIS` | `OFF` | Build the TRELLIS.2 image-to-3D engine (`src/engines/trellis`); needs the `vendor/trellis2cpp` submodule |
| `AGENT_TRELLIS_BUILD_JOBS`, `AGENT_AUDIOCPP_BUILD_JOBS` | (empty) | Parallel jobs for the nested build of the TRELLIS and audio.cpp engines; empty uses the build tool's default |
| `AGENT_GGML_AMX` | `OFF` | Keep Intel AMX kernels in the engines' ggml; only safe when every engine runs out of process |
| `BUILD_SHARED_LIBS` | `OFF` | Build `libagent`, `agent_store_sqlite` and `agent_hub` as shared libraries |
| `AGENT_INSTALL` | `ON` (top level) | Generate install rules and the `agent-cpp` CMake package |
| `AGENT_BUILD_EXAMPLES` | `ON` | Build `examples/` (guide code, image critique, RAG over docs) and their smoke tests (`ctest -L examples`) |
| `AGENT_BUILD_BENCH` | `OFF` | Build `agent_bench`, the benchmark harness (`bench/`); results in [docs/benchmarks.md](docs/benchmarks.md) |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer on GCC and Clang |
| `AGENT_ENABLE_COVERAGE` | `OFF` | gcov coverage instrumentation |

---

## C++ SDK and C API

`libagent` is a C++20 library with one public header, `#include <agent-cpp/agent.hpp>`. Install it with
`cmake --install build --prefix <prefix>` and link it from your own project:

```cmake
find_package(agent-cpp REQUIRED)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
```

You choose the engine and each model file yourself. `engine_config` fills a `Config` from them; a different engine takes
different roles (an image engine takes `diffusion_model`, `llm`, `vae`; the speech engine takes `tts.model`, `asr.model`).
A minimal chat loop:

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    agent::Config base;
    base.workspace_dir = ".workspace";
    base.context_window = 4096;
    base.max_tokens = 512;
    const agent::ModelSlot models[] = {{"model", ".workspace/models/my.gguf"}};
    auto cfg = agent::engine_config("build/src/engines/llama/agent-engine-llama.so", models, {}, base);
    if (!cfg.ok) { fprintf(stderr, "%s
", cfg.error.message.c_str()); return 1; }

    auto sess = agent::init(cfg.value);
    if (!sess.ok) { fprintf(stderr, "%s
", sess.error.message.c_str()); return 1; }
    agent::Session session = std::move(sess.value);

    agent::Conversation conv;
    conv.id = "demo";

    while (true) {
        char buf[512];
        printf("> ");
        if (!fgets(buf, sizeof(buf), stdin)) break;

        auto r = agent::run_turn(session, conv, buf,
            true, [](std::string_view tok, void *){ fwrite(tok.data(), 1, tok.size(), stdout); });
        if (r.ok) printf("
");
    }

    auto st = agent::get_stats(session);
    printf("turns=%d  tokens=%d
", st.total_turns,
           st.total_prompt_tokens + st.total_completion_tokens);
    return 0;
}
```

From C, or from any language that can call C, `include/agent-cpp/agent_c.h` offers sessions, streamed chat turns with tools, embeddings and
one-shot tasks behind opaque handles ([docs/api/c-api.md](docs/api/c-api.md)).

The SDK covers providers, the graph runtime, tools and MCP servers, skills, the permission gate, RAG and the
model registry. Start with [docs/guides/getting-started-library.md](docs/guides/getting-started-library.md).

---

## Providers

| Provider | Enum | Notes |
|---|---|---|
| Mock | `AiProvider::Mock` | Fixed response; always available; for testing |
| Engine plugin | `AiProvider::Engine` | Any engine shared library built against `engine_abi.h`, such as `agent-engine-llama` (GGUF) or `agent-engine-onnx` (`.onnx`). See [docs/api/engines.md](docs/api/engines.md). |

---

## Workspace layout

The CLI keeps one global home (`.agent.cpp-workspace`, chosen at `agent setup`) and an optional `.agents/`
folder per project. The SDK needs a `workspace_dir`, which `init()` creates as:

```
.workspace/
  models/    <-- GGUF / ONNX model files
  memory/    <-- SQLite database (db.sqlite3)
  logs/      <-- log files (when enable_file_logging = true)
  tmp/       <-- scratch space for tools
```

Details: [docs/workspace.md](docs/workspace.md).

---

## Documentation

- [Documentation index](docs/index.md): the full map
- [Getting started](docs/getting-started.md): build from source, native Windows, first program
- [The `agent` CLI](docs/cli.md): every command and flag
- [Packaging and install](docs/packaging.md): `cmake --install`, `find_package(agent-cpp)`, the Linux tarball script
- [Architecture](docs/architecture.md) and [API reference](docs/api.md)
- Guides: [embedding the library](docs/guides/getting-started-library.md) ·
  [writing an engine plugin](docs/guides/writing-an-engine.md) ([template](examples/08-engine-plugin-template)) ·
  [writing a tool](docs/guides/writing-a-tool.md) · [writing a graph](docs/guides/writing-a-graph.md) ·
  [a local voice assistant](docs/guides/voice-assistant.md)
- Examples, one folder per use case: [`examples/`](examples/README.md)
- Library: [C++ API](docs/api.md) · [C API](docs/api/c-api.md)
- Security: [OS sandbox](docs/security-sandbox.md) and [SECURITY.md](SECURITY.md)
- Status: [benchmarks](docs/benchmarks.md)

---

## Project layout

```
include/agent-cpp/        agent.hpp (single public header), engine_abi.h (plugin ABI v2)
src/agent/                libagent: agent, arena, events, executor/, graph/, mcp/, models/ (ModelRegistry),
                          providers/, engines/, memory/, tools/, rag/, tasks/ (decide, ocr), skills/,
                          data/, sandbox/, utils/
src/hal/                  OS abstraction: child_process, process, dylib, sandbox, path, time, log_file
src/engines/              plugins: common/ (shared helpers), llama/, onnx/, sdcpp/, audiocpp/, trellis/, mock/ (tests only)
apps/
  cli/                    the agent binary (commands, chat, voice, audio)
  engine-host/            agent-engine-host: runs one engine per process
  hub/                    agent_hub: Hugging Face and GitHub downloads, models.json
  store-sqlite/           agent_store_sqlite: SQLite store, checkpointer, sqlite-vec
cmake/                    package config, engine helpers, ggml AMX switch

examples/                 numbered examples (01-hello-agent ... 08-engine-plugin-template)
bench/                    agent_bench benchmark harness
tests/
  unit/                   Catch2 unit tests
  integration/            session, turn and graph tests
  consumer/               an out-of-tree project that uses the installed package
  helpers/                fake MCP server and test helpers
schemas/                  JSON schemas for engines.json, models.json and agent.json
scripts/                  update-deps.sh, test.sh, package.sh, format.sh, setup-hooks.sh (all run by hand)
docs/                     documentation
vendor/                   llama.cpp, audio.cpp, stable-diffusion.cpp, trellis2cpp,
                          simdjson, cpp-httplib, sqlite, sqlite-vec, miniaudio, catch2 (ONNX Runtime is downloaded at configure time)
```

---

## License

[MIT License](LICENSE)
