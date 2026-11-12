# agent.cpp documentation

agent.cpp is a local, offline-first AI agent in two parts:

- **`agent`**, a command-line app for chat, one-shot tasks, declarative graphs and a voice loop. Models run on your
  machine through engine plugins, and the CLI manages the engines, models, MCP servers and skills for you.
- **`libagent`**, a C++20 runtime with one public header (`agent-cpp/agent.hpp`) for embedding the same agent loop,
  tools, graphs and engines in your own program. It has no heap allocations in the per-turn path (a session-owned
  bump-pointer `Arena`), dispatches through `constexpr` function-pointer tables, and reports errors with `Result<T>`.

## Start here

| If you want to | Read |
|---|---|
| chat with a local model in five minutes | [Getting started](getting-started.md), then the quickstart in the [README](../README.md#60-second-quickstart) |
| know every command and flag | [The `agent` CLI](cli.md) |
| install or package the binaries and the CMake package | [Installing and packaging](packaging.md) |
| embed the runtime in a C++ program | [Getting started](getting-started.md#track-b-embedding-libagent), [Guide: embedding the library](guides/getting-started-library.md) |
| add a model engine | [Guide: writing an engine plugin](guides/writing-an-engine.md) |
| write tools, graphs or workflows | [Guide: writing a tool](guides/writing-a-tool.md), [Guide: writing a graph](guides/writing-a-graph.md) |

## Documentation map

### Using the CLI

| Document | What it covers |
|---|---|
| [Getting Started](getting-started.md) | Build from source (Linux, native Windows), install, first program |
| [The `agent` CLI](cli.md) | `setup`, `init`, `doctor`, `clean`, `secret`, `engine`, `model`, `chat`, `run`, `voice`, `task`, `mcp`, `skill`, `runtime`, `sandbox` |
| [Workspace layout](workspace.md) | The global home `.agent.cpp-workspace`, project `.agents/` folders, SQLite files, runtimes |
| [Installing and packaging](packaging.md) | `cmake --install`, `find_package(agent-cpp)`, the Linux tarball script |
| [Examples](../examples/README.md) | Image critique loop, RAG over a folder of docs, graph JSON examples |
| [Guide: a local voice assistant](guides/voice-assistant.md) | `agent voice` with an audio.cpp VAD, ASR and TTS model |
| [GPU engines](gpu.md) | What is built for GPUs, the sandbox cache folders, and the test checklist (untested so far) |

### Embedding the SDK

| Document | What it covers |
|---|---|
| [Architecture](architecture.md) | System diagrams, memory model, data flow |
| [Usage guide](guide.md) | Sessions, turns, tools, permissions, graphs, events, memory, errors |
| [API reference](api.md) | The public types and functions in `agent.hpp` |
| [Guide: embedding the library](guides/getting-started-library.md) | `find_package(agent-cpp)`, a first program with the llama plugin under process isolation |
| [Guide: writing an engine plugin](guides/writing-an-engine.md) | ABI 2 walkthrough: `struct_size` rules, tasks, parts, streaming, progress, cancel, isolation, packaging ([template](../examples/08-engine-plugin-template)) |
| [Guide: writing a tool](guides/writing-a-tool.md) | `Tool`, the permission gate, event hooks and per-node `ToolHook`s, MCP servers |
| [Guide: writing a graph](guides/writing-a-graph.md) | C++ builders and `agent.json`, state scopes, checkpointers, interrupts, fork, a model per node |

### Reference

| Document | What it covers |
|---|---|
| [API — Core Types](api/core.md) | `Result<T>`, `Error`, `ErrorCode`, `Arena` |
| [API — Session](api/session.md) | `Config`, `Session`, `init()`, `AgentRuntime` |
| [API — Engines](api/engines.md) | Engine plugins, `engine_abi.h`, `AiProvider::Engine`, process isolation |
| [API — Conversation](api/conversation.md) | `Conversation`, `run_turn`, `run_turn_with_media`, pruning, compression |
| [API — Graph runtime](api/graph.md) | `Graph`, Agent/Function/Tool nodes, Sequential/Parallel/Loop, scoped state, checkpoints, interrupts |
| [API — agent.json](api/agent-json.md) | Declarative graphs: `load_graph_json`, JSON schema, routes, `register_graph_function` |
| [API — Tools](api/tools.md) | Built-in tools, `Tool` and `ToolResult`, custom tools, skills, permission gate |
| [API — MCP client](api/mcp.md) | `connect_mcp_server`: MCP stdio servers as tools |
| [API — Memory](api/memory.md) | `MemoryConfig`, `store_memory`, `retrieve_memory` |
| [API — Data Sources](api/data-sources.md) | `DataSource`, RAG, SQLite and GraphRAG integration |
| [API — RAG](api/rag.md) | `embed`, `rerank`, `VectorStoreOps` (in-memory, sqlite-vec), chunking, `rag_index` / `rag_query`, `search_knowledge` |
| [API — Engines: OCR](api/engines.md#ocr) | `ocr`: text from an image with an OCR model (`AGENT_CAP_OCR`) |
| [API — Decisions](api/decide.md) | `decide`, `decide_score`, `decide_yes_no` (`AGENT_CAP_DECIDE`) |
| [API — Events](api/events.md) | `EventType`, `EventHook`, `register_event_hook` |
| [API — Stats](api/stats.md) | `TurnStats`, `SessionStats`, `get_stats` |
| [API — Model registry](api/models.md) | `ModelRegistry`: lazy loading under a RAM/VRAM budget, LRU unload, leases |
| [API — Model hub](api/hub.md) | `agent_hub`: Hugging Face downloads, `models.json`, licence policy |

### Operations and status

| Document | What it covers |
|---|---|
| [OS sandbox](security-sandbox.md) | Landlock, seccomp and rlimits on Linux, AppContainer on Windows; `SandboxPolicy` and platform status |
| [Benchmarks](benchmarks.md) | `agent_bench`: tokens/s, time to first token, peak RSS, process isolation overhead, parallel branches |
| [Security policy](../SECURITY.md) | How to report a vulnerability |

## Quick example (SDK)

```cpp
#include <agent-cpp/agent.hpp>
#include <iostream>

int main() {
    agent::AgentRuntime runtime;          // initialises global backends (llama.cpp etc.)

    agent::Config cfg;
    cfg.provider     = agent::AiProvider::Engine;
    cfg.engine       = "build/src/engines/llama/agent-engine-llama.so";
    cfg.model        = ".workspace/models/my-model.gguf";
    cfg.workspace_dir = ".workspace";
    cfg.context_window = 4096;
    cfg.max_tokens   = 512;
    cfg.enable_file_logging = true;

    auto sess = agent::init(cfg);
    if (!sess.ok) { std::cerr << sess.error.message; return 1; }

    agent::Conversation conv;
    conv.id = "demo";

    auto r = agent::run_turn(sess.value, conv, "What files are in my workspace?");
    if (r.ok) std::cout << r.value << "\n";

    return 0;
}
```

## CMake options

The complete table is in the [README](../README.md#cmake-options). The options that change what gets built:

| Option | Default | Description |
|---|---|---|
| `AGENT_BUILD_CLI` | `ON` | Build the `agent` CLI (needs `AGENT_BUILD_HUB`) |
| `AGENT_BUILD_HUB` | `ON` | Build `agent_hub` (needs OpenSSL) |
| `AGENT_ENGINE_LLAMA` | `ON` | llama.cpp engine plugin |
| `AGENT_ENGINE_ONNX` | `OFF` | ONNX Runtime engine plugin (needs `ONNXRUNTIME_ROOT`) |
| `AGENT_ENGINE_SDCPP` | `OFF` | stable-diffusion.cpp engine plugin (images and video) |
| `AGENT_ENGINE_AUDIOCPP` | `OFF` | audio.cpp speech engine plugin (TTS, ASR, VAD, diarization; needs `vendor/audio.cpp`) |
| `AGENT_ENGINE_TRELLIS` | `OFF` | TRELLIS.2 image-to-3D engine plugin |
| `BUILD_SHARED_LIBS` | `OFF` | Shared `libagent`, `agent_store_sqlite` and `agent_hub` |
| `AGENT_INSTALL` | `ON` (top level) | Install rules and the `agent-cpp` CMake package |
| `AGENT_BUILD_EXAMPLES` | `ON` | Build `examples/` and their smoke tests |
| `AGENT_BUILD_BENCH` | `OFF` | Build the benchmark harness |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer (GCC and Clang) |
| `AGENT_ENABLE_COVERAGE` | `OFF` | gcov coverage instrumentation |
