# agent.cpp

A modular, lightweight C++20 AI agent orchestration library built for edge devices, offline inference, and ultra-low-latency deployments.

## What is agent.cpp?

agent.cpp gives you a complete runtime for building AI agents on top of local inference backends (llama.cpp, ONNX Runtime, ... loaded as engine plugins), fully offline. It handles the hard parts: context management, tool calling, graph workflows with parallel subagents, memory persistence, and observability — all without heap allocations in the hot path.

The library is designed around three principles:

- **Zero hot-path allocations.** A session-owned bump-pointer `Arena` allocator handles all inference-time memory. Only conversation history (owned `Message` objects) lives on the heap. The arena resets between turns in O(1).
- **Data-oriented dispatch.** Providers, memory backends, and tool registries are resolved through constexpr function-pointer tables (`ProviderOps`, `StoreOps`, `ToolsOps`) — no virtual dispatch, no RTTI.
- **Monadic error handling.** Every fallible function returns `Result<T>`. There are no exceptions anywhere in the library. Callers check `.ok` and inspect `.error.code` / `.error.message`.

## Documentation Map

| Document | What it covers |
|---|---|
| [Getting Started](getting-started.md) | Build instructions, first program |
| [Installing and packaging](packaging.md) | `cmake --install`, `find_package(agent-cpp)`, static/shared, Linux release tarball (`scripts/package.sh`) |
| [The `agent` CLI](cli.md) | `agent setup`, `engine`, `model`, `chat`, `run`, `voice`: config files, permissions, slash commands |
| [Guide: embedding the library](guides/getting-started-library.md) | `cmake --install`, `find_package(agent-cpp)`, a first program with the llama plugin under process isolation, sandbox defaults |
| [Guide: writing an engine plugin](guides/writing-an-engine.md) | ABI 2 walkthrough: `struct_size` rules, tasks, parts, streaming / progress / cancel, isolation and sandbox, packaging (`agent_add_engine`, `agent_install_engine`, [template](../templates/engine-plugin)) |
| [Guide: writing a tool](guides/writing-a-tool.md) | `Tool`, the permission gate, event hooks and per-node `ToolHook`s, MCP servers |
| [Guide: writing a graph](guides/writing-a-graph.md) | C++ builders and `agent.json`, state scopes, checkpointers, interrupts, fork, a model per node |
| [Examples](../examples/README.md) | Image critique loop (LLM → stable-diffusion.cpp → VLM), RAG over a folder of docs, graph JSON examples (router, review loop) |
| [Benchmarks](benchmarks.md) | `agent_bench` (`AGENT_BUILD_BENCH`): tokens/s, time to first token, peak RSS, process isolation overhead, parallel branches; machine details and caveats |
| [Guide: a local voice assistant](guides/voice-assistant.md) | `agent voice` with Silero VAD, Whisper, a llama.cpp model and Kokoro; file backend for headless runs |
| [Architecture](architecture.md) | System diagrams, memory model, data flow |
| [Limitations](LIMITATIONS.md) | Known limitations and planned fixes |
| [OS sandbox](security-sandbox.md) | Landlock / seccomp / rlimits for the engine host, `run_command` and MCP servers; `SandboxPolicy`, defaults, platform status |
| [API — Core Types](api/core.md) | `Result<T>`, `Error`, `ErrorCode`, `Arena` |
| [API — Session](api/session.md) | `Config`, `Session`, `init()`, `AgentRuntime` |
| [API — Engines](api/engines.md) | Engine plugins, `engine_abi.h`, `AiProvider::Engine` |
| [API — Conversation](api/conversation.md) | `FlowMemory`, `run_turn`, pruning, compression |
| [API — Graph runtime](api/graph.md) | `Graph`, Agent/Function/Tool nodes (system prompt, tool allowlist, `ToolHook`s, shared histories), Sequential/Parallel/Loop, parallel subagents, scoped state, checkpoints, interrupts, migrating from `Flow`/`AgentScheduler` |
| [API — agent.json](api/agent-json.md) | Declarative graphs: `load_graph_json`, JSON schema, routes, workflow nodes, `register_graph_function` |
| [API — Tools](api/tools.md) | Built-in tools, `Tool` struct, custom tools, Skills, permission gate |
| [API — MCP client](api/mcp.md) | `connect_mcp_server`: MCP stdio servers as tools |
| [API — Memory](api/memory.md) | `MemoryConfig`, `store_memory`, `retrieve_memory` |
| [API — Data Sources](api/data-sources.md) | `DataSource`, RAG/SQLite/GraphRAG integration |
| [API — RAG](api/rag.md) | `embed`, `rerank`, `VectorStoreOps` (in-memory, sqlite-vec), chunking, `rag_index` / `rag_query`, `search_knowledge` tool |
| [API — Engines: OCR](api/engines.md#ocr) | `ocr`: text from an image with an OCR VLM (GLM-OCR, PaddleOCR-VL, ...) through the llama engine (`AGENT_CAP_OCR`) |
| [API — Decisions](api/decide.md) | `decide`, `decide_score`, `decide_yes_no`: choice / score / yes-no probabilities from model log-probabilities (`AGENT_CAP_DECIDE`) |
| [API — Events](api/events.md) | `EventType`, `EventHook`, `register_event_hook` |
| [API — Stats](api/stats.md) | `TurnStats`, `SessionStats`, `get_stats` |
| [API — Model registry](api/models.md) | `ModelRegistry`: lazy model loading under a RAM/VRAM budget, LRU unload, leases, `GraphNode::model`, memory estimates |
| [API — Model hub](api/hub.md) | CLI-side `agent_hub`: Hugging Face downloads, `models.json`, licence policy |

## Quick Example

```cpp
#include <agent-cpp/agent.hpp>
#include <iostream>

int main() {
    agent::AgentRuntime runtime;          // initialises global backends (llama.cpp etc.)

    agent::Config cfg;
    cfg.provider     = agent::AiProvider::Engine;
    cfg.engine       = "build/engines/llama/agent-engine-llama.so";
    cfg.model        = ".workspace/models/my-model.gguf";
    cfg.workspace_dir = ".workspace";
    cfg.context_window = 4096;
    cfg.max_tokens   = 512;
    cfg.enable_file_logging = true;

    auto sess = agent::init(cfg);
    if (!sess.ok) { std::cerr << sess.error.message; return 1; }

    agent::FlowMemory conv;
    conv.id = "demo";

    auto r = agent::run_turn(sess.value, conv, "What files are in my workspace?");
    if (r.ok) std::cout << r.value << "\n";

    return 0;
}
```

## CMake Options

| Option | Default | Description |
|---|---|---|
| `AGENT_ENGINE_LLAMA` | `ON` | Build the llama.cpp engine plugin |
| `AGENT_ENGINE_ONNX` | `OFF` | Build the ONNX Runtime engine plugin (`engines/onnx`); needs `-DONNXRUNTIME_ROOT` |
| `AGENT_ENGINE_TRELLIS` | `OFF` | Build the TRELLIS.2 image-to-3D engine plugin (`engines/trellis`); `AGENT_TRELLIS_BUILD_JOBS` caps the nested build |
| `AGENT_ENGINE_WHISPER` | `OFF` | Build the whisper.cpp speech-to-text engine plugin (`engines/whisper`); `AGENT_WHISPER_BUILD_JOBS` caps the nested build |
| `AGENT_ENGINE_PARAKEET` | `OFF` | Build the parakeet.cpp speech-to-text engine plugin (`engines/parakeet`); `AGENT_PARAKEET_BUILD_JOBS` caps the nested build |
| `AGENT_BUILD_EXAMPLES` | `ON` | Build `examples/` (guide code, image critique, RAG over docs) and their mock-engine smoke tests (`ctest -L examples`) |
| `AGENT_BUILD_BENCH` | `OFF` | Build `agent_bench`, the benchmark harness (`bench/`, results in [benchmarks.md](benchmarks.md)) |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UBSan |
| `AGENT_ENABLE_COVERAGE` | `OFF` | gcov coverage instrumentation |
