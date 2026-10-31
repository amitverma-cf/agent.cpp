# Getting Started

## Prerequisites

- CMake 3.21+
- C++20-capable compiler (GCC 12+, Clang 14+, MSVC 19.30+)
- Ninja (recommended) or Make

On Windows, build tools must run under WSL (Debian/Ubuntu). Git and file operations run natively on Windows.

## Build

```bash
# Configure (from repo root, run under WSL)
wsl cmake -B build -G Ninja

# Build everything
wsl cmake --build build

# Run tests
cd build && ctest --output-on-failure
```

### Build without llama.cpp

```bash
wsl cmake -B build -G Ninja -DAGENT_ENGINE_LLAMA=OFF
wsl cmake --build build
```

### Build the ONNX Runtime engine plugin

Download a prebuilt release (e.g. `onnxruntime-linux-x64-1.31.1.tgz` from
https://github.com/microsoft/onnxruntime/releases), extract it, and point CMake at it:

```bash
wsl cmake -B build -G Ninja -DAGENT_ENGINE_ONNX=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.31.1
wsl cmake --build build
```

This builds `build/engines/onnx/agent-engine-onnx.so` and copies `libonnxruntime.so.1` next to it.
Ship the two together: the plugin finds ONNX Runtime through its `$ORIGIN` RPATH.

### Enable sanitizers

```bash
wsl cmake -B build -G Ninja -DAGENT_ENABLE_SANITIZERS=ON
wsl cmake --build build
```

## Workspace layout

Every session requires a `workspace_dir`. The library creates this directory structure automatically on `init()`:

```
.workspace/
├── models/      # put your .gguf / .onnx model files here
├── memory/      # SQLite database (db.sqlite3)
├── logs/        # structured log files (if enable_file_logging = true)
└── tmp/         # scratch space for tools
```

Pass the workspace path at session creation:

```cpp
cfg.workspace_dir = ".workspace";   // relative or absolute — no shell metacharacters
```

The path is validated at `init()` time. It must not contain `"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, newlines, or carriage returns, because the terminal tool interpolates it into shell commands.

## Your first program

### Step 1 — link the library

```cmake
find_package(agent REQUIRED)       # or add_subdirectory if vendored
target_link_libraries(my_app PRIVATE agent)
```

### Step 2 — initialise a session

```cpp
#include <agent-cpp/agent.hpp>

agent::AgentRuntime runtime;   // must outlive all sessions

agent::Config cfg;
cfg.provider      = agent::AiProvider::Engine;
cfg.engine        = "build/engines/llama/agent-engine-llama.so";
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

### Step 3 — run a conversation

```cpp
agent::FlowMemory conv;
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

### Step 4 — use a graph for structured agent flows

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


