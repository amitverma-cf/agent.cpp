# Getting Started

There are two ways to use agent.cpp. Track A runs the `agent` command-line app. Track B embeds `libagent` in
your own C++20 program. Both start from the same build.

## Prerequisites

| | Linux | Windows |
|---|---|---|
| Compiler | GCC 12+ or Clang 14+ | MSVC 19.30+ (Visual Studio 2022 Build Tools, C++ workload) |
| CMake | 3.21+ | 3.21+ |
| Build tool | Ninja (`sudo apt-get install ninja-build`) | Ninja (`choco install ninja` or the copy in Visual Studio) |
| OpenSSL 3 | `libssl-dev` | `vcpkg install openssl:x64-windows` |

Platforms: v1.0.0 targets Windows 11 and Linux. macOS: not tested, not supported in v1.0.0.

OpenSSL is needed for `agent_hub` (the model downloader). Turn it off with `-DAGENT_BUILD_HUB=OFF`, which also
turns off the CLI.

## Track A: the `agent` CLI

### Build from source

Linux:

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Native Windows (Developer PowerShell for VS 2022, or "x64 Native Tools Command Prompt"):

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build
ctest --test-dir build --output-on-failure
```

The build puts `agent`, `agent-engine-host` and the test binaries at the top of `build/`. The llama engine is built
into `build/src/engines/llama/`. Set `-DAGENT_ENGINE_LLAMA=OFF` to build without it.

### First run

```sh
./build/agent setup --yes                                              # home folder, hardware, engines, config
./build/agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M        # download and smoke-test a model
./build/agent chat
```

On Windows use `.\build\agent.exe` in place of `./build/agent`. `agent setup` asks where the global home
(`.agent.cpp-workspace`) goes unless you pass `--home DIR`. The full command list is in [cli.md](cli.md).

## Track B: embedding libagent

### Install the package

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /opt/agent
```

This installs `bin/agent`, `bin/agent-engine-host`, the engine plugins in `lib/agent/engines/`, `libagent` with its
headers, and the `agent-cpp` CMake package. See [packaging.md](packaging.md) for the full layout.

### Step 1: link the library

```cmake
find_package(agent-cpp 0.1 REQUIRED)        # set -DCMAKE_PREFIX_PATH=/opt/agent when it is not on the default path
target_link_libraries(my_app PRIVATE agent-cpp::agent)
# optional targets: agent-cpp::store_sqlite (SQLite store, sqlite-vec), agent-cpp::hub (Hugging Face downloads)
```

Or add the repository with `add_subdirectory` and link the same targets.

### Step 2: initialise a session

```cpp
#include <agent-cpp/agent.hpp>

agent::AgentRuntime runtime;   // must outlive all sessions

agent::Config cfg;
cfg.provider      = agent::AiProvider::Engine;
cfg.engine        = "build/src/engines/llama/agent-engine-llama.so";
cfg.model         = ".workspace/models/my.gguf";
cfg.workspace_dir = ".workspace";
cfg.context_window = 4096;
cfg.max_tokens    = 512;
cfg.temperature   = 0.7f;

auto res = agent::init(cfg);
if (!res.ok) {
    fprintf(stderr, "init failed: %s\n", res.error.message.c_str());
    return 1;
}
agent::Session session = std::move(res.value);
```

### Step 3: run a conversation

```cpp
agent::Conversation conv;
conv.id = "my_chat";

// Add an optional system message
agent::Message sys;
sys.role = "system";
sys.content = "You are a helpful assistant.";
conv.history.push_back(std::move(sys));

// Each call appends the user message and gets an assistant reply
auto r1 = agent::run_turn(session, conv, "Hello! What can you do?");
if (r1.ok) printf("Agent: %s\n", r1.value.c_str());

auto r2 = agent::run_turn(session, conv, "List the files in my workspace.");
if (r2.ok) printf("Agent: %s\n", r2.value.c_str());
```

### Step 4: use a graph for structured agent flows

```cpp
// Two Agent nodes that continue one conversation (same history_key), each with its own prompt.
agent::GraphNode plan{.name = "plan", .kind = agent::GraphNodeKind::Agent,
                      .system_prompt = "You are a careful planner.", .prompt = "Make a plan.",
                      .history_key = "demo"};
agent::GraphNode execute{.name = "execute", .kind = agent::GraphNodeKind::Agent,
                         .prompt = "Execute the plan step by step.", .output_key = "result",
                         .history_key = "demo"};
agent::Graph g = agent::make_sequential("plan_execute", {agent::make_graph(plan), agent::make_graph(execute)});

agent::GraphRun run;
run.thread_id = "graph_demo";
auto graph_res = agent::run_graph(session, g, run);
if (graph_res.ok) printf("Result: %s\n", std::string(agent::get_state(run.state, "result")).c_str());
```

See [api/graph.md](api/graph.md) for routing, parallel subagents, loops, checkpoints and human-in-the-loop interrupts.

## Workspace layout (SDK)

Every session requires a `workspace_dir`. `init()` creates this structure in it:

```
.workspace/
├── models/      # put your .gguf / .onnx model files here
├── memory/      # SQLite database (db.sqlite3)
├── logs/        # structured log files (if enable_file_logging = true)
└── tmp/         # scratch space for tools
```

The path is validated at `init()` time. It must not contain `"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, newlines, or carriage
returns, because the terminal tool interpolates it into shell commands.

## Build options

| Option | Effect |
|---|---|
| `-DAGENT_ENGINE_ONNX=ON` | Builds the ONNX Runtime engine and copies `libonnxruntime` next to it. CMake downloads the pinned upstream ONNX Runtime release (never built from source); pass `-DONNXRUNTIME_ROOT=<dir>` only where upstream publishes no archive. Ship the two together; the plugin finds ONNX Runtime through its RPATH (`$ORIGIN` on Linux). |
| `-DAGENT_ENGINE_AUDIOCPP=ON` (after `git submodule update --init vendor/audio.cpp`) | Builds the audio.cpp speech engine used by `agent voice`. |
| `-DAGENT_ENABLE_SANITIZERS=ON` | AddressSanitizer + UndefinedBehaviorSanitizer on GCC and Clang. |
| `-DBUILD_SHARED_LIBS=ON` | Shared `libagent`, `agent_store_sqlite` and `agent_hub`. |

The full list is in the [README](../README.md#cmake-options).
