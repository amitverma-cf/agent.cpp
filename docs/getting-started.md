# Getting started

Two ways in: run the **`agent`** command, or embed **`libagent`** in your own C++ program. Both start from one build.

## What you need

| | Linux | Windows 11 |
|---|---|---|
| Compiler | GCC 12+ or Clang 15+ | MSVC 2022 (C++ workload) |
| CMake, build tool | CMake 3.21+, Ninja | CMake 3.21+, Ninja |
| OpenSSL 3 | `libssl-dev` | `vcpkg install openssl:x64-windows` |

OpenSSL is used for model downloads (`agent_hub`); without it (`-DAGENT_BUILD_HUB=OFF`) the CLI is not built. macOS is not tested
and not supported in v1.0.0.

## Build

```sh
git clone --recurse-submodules https://github.com/amitverma-cf/agent.cpp
cd agent.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows, run the same from a "Developer PowerShell for VS 2022" and add
`-DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows`.

The build puts `agent`, `agent-engine-host` and the tests at the top of `build/`, and each engine plugin under
`build/src/engines/<name>/`. The default build has the llama engine only. Add the others when you need them:

```sh
cmake -B build -DAGENT_ENGINE_AUDIOCPP=ON -DAGENT_ENGINE_SDCPP=ON -DAGENT_ENGINE_ONNX=ON -DAGENT_ENGINE_TRELLIS=ON
```

audio.cpp and TRELLIS.2 come from submodules (`git submodule update --init vendor/audio.cpp vendor/trellis2cpp`). The full list of
options is [at the end of this page](#build-options).

## Use the command

```sh
./build/agent setup                                            # home folder, hardware check, engines, first model
./build/agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
./build/agent                                                  # chat
```

On Windows use `.\build\agent.exe`. Setup records the engines it finds, so `-e llama` works afterwards. The commands are
described in [cli.md](cli.md); `agent --help` lists them and `agent <command> --help` explains one.

## Use the library

Install, then link:

```sh
cmake --install build --prefix /opt/agent
```

```cmake
find_package(agent-cpp REQUIRED)                  # -DCMAKE_PREFIX_PATH=/opt/agent if it is not on the default path
target_link_libraries(my_app PRIVATE agent-cpp::agent)
# optional: agent-cpp::store_sqlite (SQLite store, sqlite-vec), agent-cpp::hub (Hugging Face downloads)
```

You can also `add_subdirectory` the repository and link the same targets. The install puts `bin/agent`, `bin/agent-engine-host`, the
engine plugins in `lib/agent/engines/`, `libagent`, the headers and the CMake package; see [packaging.md](packaging.md).

### A conversation

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;                       // keep it alive as long as any session

    agent::Config base;
    base.workspace_dir = ".workspace";
    const agent::ModelSlot models[] = {{"model", ".workspace/models/my.gguf"}};
    auto cfg = agent::engine_config("llama", models, {}, base);
    if (!cfg.ok) return std::fprintf(stderr, "%s\n", cfg.error.message.c_str()), 1;

    auto session = agent::init(cfg.value);
    if (!session.ok) return std::fprintf(stderr, "%s\n", session.error.message.c_str()), 1;

    agent::Conversation chat;
    chat.id = "my_chat";
    auto r1 = agent::run_turn(session.value, chat, "Hello! What can you do?");
    if (r1.ok) std::printf("Agent: %s\n", r1.value.c_str());
    auto r2 = agent::run_turn(session.value, chat, "List the files in my workspace.");   // tools run inside the turn
    if (r2.ok) std::printf("Agent: %s\n", r2.value.c_str());
}
```

`"llama"` is the engine's name. It is found in `engines.json` (written by `agent setup`), next to your executable, in `engines/`, in
`../lib` or in `$AGENT_ENGINES_DIR`; a full path to a plugin works too. A different engine takes different model roles, for example
`{"tts.model", ...}` and `{"asr.model", ...}` for audio.cpp. See [api/engines.md](api/engines.md).

### A graph

```cpp
agent::GraphNode plan{.name = "plan", .kind = agent::GraphNodeKind::Agent,
                      .system_prompt = "You are a careful planner.", .prompt = "Make a plan.", .history_key = "demo"};
agent::GraphNode execute{.name = "execute", .kind = agent::GraphNodeKind::Agent,
                         .prompt = "Execute the plan step by step.", .output_key = "result", .history_key = "demo"};
agent::Graph g = agent::make_sequential("plan_execute", {agent::make_graph(plan), agent::make_graph(execute)});

agent::GraphRun run;
run.thread_id = "graph_demo";
auto done = agent::run_graph(session.value, g, run);
if (done.ok) std::printf("Result: %s\n", std::string(agent::get_state(run.state, "result")).c_str());
```

Routing, parallel branches, loops, checkpoints and human-in-the-loop interrupts: [api/graph.md](api/graph.md). From C:
[api/c-api.md](api/c-api.md). More examples: [../examples/README.md](../examples/README.md).

## Build options

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
| `AGENT_BUILD_BENCH` | `OFF` | Build `agent_bench`, the benchmark harness (`bench/`); results in [docs/benchmarks.md](benchmarks.md) |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer on GCC and Clang |
| `AGENT_ENABLE_COVERAGE` | `OFF` | gcov coverage instrumentation |

