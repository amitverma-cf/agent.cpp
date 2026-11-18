# Getting started

Two ways in. **The `agent` command** is a ready-to-use terminal assistant: install it, run `agent`, and chat. **`libagent`** is the C++ library behind it, for embedding in a program. The command needs no build; the library needs the install tree or a source build.

## Requirements

| | Linux | Windows 11 |
|---|---|---|
| Compiler | GCC 12+ or Clang 15+ (GCC 13+ or Clang 18+ for the full-screen chat) | MSVC 2022 (C++ workload) |
| CMake, build tool | CMake 3.21+, Ninja | CMake 3.21+, Ninja |
| OpenSSL 3 | `libssl-dev` | `vcpkg install openssl:x64-windows` |

OpenSSL is used for model downloads (`agent_hub`); without it (`-DAGENT_BUILD_HUB=OFF`) the CLI is not built. macOS 14 or newer (Apple
silicon and Intel) builds with Apple Clang and `brew install cmake ninja openssl@3`; releases include macOS archives.

## Install a release

`curl -fsSL https://agent.qozu.fyi/install.sh | sh` (Linux and macOS) or `irm https://agent.qozu.fyi/install.ps1 | iex` (Windows PowerShell) installs the latest release; see the
[README](../README.md#install). Set `AGENT_VERSION=v1.2.3` to install a given tag. The installer is the only thing that installs, changes or removes agent.cpp: `agent setup`, `agent update`, `agent uninstall`, `agent engine pull` and `agent runtime install` run its copy in `<home>/scripts/`.

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
`build/src/engines/<name>/`. Every engine is built by default (llama.cpp, audio.cpp and stable-diffusion.cpp when their submodules are checked out, so clone with
`--recurse-submodules`; trellis needs none). Switch one off with `-DAGENT_ENGINE_<NAME>=OFF`. The full list of options is
[at the end of this page](#build-options).

## Use the command

```sh
agent                                                          # chat; built from source: ./build/agent
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M       # more models
agent engine pull audiocpp                                     # add an engine (runs the installer script)
agent doctor                                                   # what is installed and where
```

On Windows use `.\build\agent.exe`. `agent setup` and the other commands that change the installation need an installed release
(they run `<home>/scripts/install.sh` or `install.ps1`); a source build is used for development and runs the engines built next to
it. The commands are described in [cli.md](cli.md); `agent --help` lists them and `agent <command> --help` explains one.

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

`add_subdirectory` also works with the repository and link the same targets. The install puts `bin/agent`, `bin/agent-engine-host`, the
engine plugins in `lib/agent/engines/`, `libagent`, the headers and the CMake package; see [packaging.md](packaging.md).

### A conversation

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;                       // keep it alive as long as any session

    auto session = agent::open_session(agent::engines::llama, ".workspace/models/my.gguf");
    if (!session.ok) return std::fprintf(stderr, "%s\n", session.error.message.c_str()), 1;

    agent::Conversation chat;
    chat.id = "my_chat";
    auto r1 = agent::run_turn(session.value, chat, "Hello! What can you do?");
    if (r1.ok) std::printf("Agent: %s\n", r1.value.c_str());
    auto r2 = agent::run_turn(session.value, chat, "List the files in my workspace.");   // tools run inside the turn
    if (r2.ok) std::printf("Agent: %s\n", r2.value.c_str());
}
```

`agent::engines::llama` is the engine; `audiocpp`, `sdcpp`, `trellis` and `onnx` are next to it. In CMake, `agent_deploy_engines(my_app llama)` puts the plugin where the program finds it. A different engine takes different
model roles, for example `{"tts.model", ...}` and `{"asr.model", ...}` for audio.cpp. See [api/engines.md](api/engines.md).

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
| `AGENT_CLI_TUI` | `ON` | Build the full-screen chat (RTXUI, C++23, `vendor/RTXUI`) into the CLI; `OFF`, or a compiler older than GCC 13 / Clang 18, keeps only the line-based `agent chat --plain` |
| `AGENT_CLI_AUDIO_DEVICE` | `ON` | Build microphone and speaker support (miniaudio) into the CLI; `OFF` leaves it out |
| `AGENT_ENGINE_LLAMA` | `ON` | Build the llama.cpp engine plugin (`src/engines/llama`) |
| `AGENT_LLAMA_UPSTREAM` | `ON` | Source builds only: link llama.cpp's scheduler (`tools/server`) in process for MTP, draft and native decision models. A prebuilt build always has it |
| `AGENT_LLAMA_PREBUILT` | (empty) | Folder with an upstream llama.cpp release (`scripts/install.sh fetch llama DIR`): only the small plugin is compiled, against its libraries. What the release does; `vendor/llama.cpp` must be at the pinned tag |
| `AGENT_AUDIOCPP_PREBUILT`, `AGENT_SDCPP_PREBUILT` | (empty) | Same for audio.cpp and stable-diffusion.cpp (`install.sh fetch audiocpp DIR`, `fetch sdcpp DIR`); `vendor/stable-diffusion.cpp` must be at the pinned tag |
| `AGENT_ENGINE_API` | `ON` | Build the remote API engine plugin (runs a bun child) |
| `AGENT_BUILD_TESTS` | `ON` (top level) | Build `test_agent`, the unit tests |
| `AGENT_ENGINE_ONNX` | `ON` | Build the ONNX Runtime engine (`src/engines/onnx`). CMake downloads the pinned official ONNX Runtime release (`AGENT_ONNXRUNTIME_FETCH`); `-DONNXRUNTIME_ROOT=<dir>` uses a local copy |
| `AGENT_ENGINE_SDCPP` | `ON` with its submodule | Build the stable-diffusion.cpp engine for images and video (`src/engines/sdcpp`) |
| `AGENT_ENGINE_AUDIOCPP` | `ON` with its submodule | Build the audio.cpp speech engine (`src/engines/audiocpp`: TTS, ASR, VAD, diarization); needs the `vendor/audio.cpp` submodule |
| `AGENT_AUDIOCPP_MODEL_SET` | `custom` | audio.cpp model families compiled in: `custom` builds the families in `AGENT_AUDIOCPP_MODELS`, `full` builds all of them |
| `AGENT_AUDIOCPP_MODELS` | (list in `src/engines/audiocpp/CMakeLists.txt`) | Model families built when `AGENT_AUDIOCPP_MODEL_SET=custom` |
| `AGENT_ENGINE_TRELLIS` | `ON` | Build the TRELLIS.2 plugin (`src/engines/trellis`), which runs the trellis.cpp release program `trellis-cli`; nothing of trellis.cpp is compiled |
| `AGENT_AUDIOCPP_BUILD_JOBS`, `AGENT_SDCPP_BUILD_JOBS` | (empty) | Parallel jobs for the nested build of the audio.cpp and stable-diffusion.cpp engines (source builds); empty uses the build tool's default |
| `AGENT_GGML_AMX` | `OFF` | Keep Intel AMX kernels in the engines' ggml; only safe when every engine runs out of process |
| `BUILD_SHARED_LIBS` | `OFF` | Build `libagent`, `agent_store_sqlite` and `agent_hub` as shared libraries |
| `AGENT_INSTALL` | `ON` (top level) | Generate install rules and the `agent-cpp` CMake package |
| `AGENT_BUILD_EXAMPLES` | `ON` | Build `examples/` (guide code, image critique, RAG over docs) and their smoke tests (`ctest -L examples`) |
| `AGENT_BUILD_BENCH` | `OFF` | Build `agent_bench`, the benchmark harness (`bench/`); results in [docs/benchmarks.md](benchmarks.md) |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer on GCC and Clang |
| `AGENT_ENABLE_COVERAGE` | `OFF` | gcov coverage instrumentation |

