# agent.cpp

A modular, lightweight C++20 AI agent orchestration library built for edge devices, offline inference, and ultra-low-latency deployments.

## What is agent.cpp?

agent.cpp gives you a complete runtime for building AI agents on top of local inference backends (llama.cpp, ONNX Runtime), fully offline. It handles the hard parts: context management, tool calling, multi-agent scheduling, FSM-based agent flows, memory persistence, and observability — all without heap allocations in the hot path.

The library is designed around three principles:

- **Zero hot-path allocations.** A session-owned bump-pointer `Arena` allocator handles all inference-time memory. Only conversation history (owned `Message` objects) lives on the heap. The arena resets between turns in O(1).
- **Data-oriented dispatch.** Providers, memory backends, and tool registries are resolved through constexpr function-pointer tables (`ProviderOps`, `StoreOps`, `ToolsOps`) — no virtual dispatch, no RTTI.
- **Monadic error handling.** Every fallible function returns `Result<T>`. There are no exceptions anywhere in the library. Callers check `.ok` and inspect `.error.code` / `.error.message`.

## Documentation Map

| Document | What it covers |
|---|---|
| [Getting Started](getting-started.md) | Build instructions, first program |
| [Architecture](architecture.md) | System diagrams, memory model, data flow |
| [API — Core Types](api/core.md) | `Result<T>`, `Error`, `ErrorCode`, `Arena` |
| [API — Session](api/session.md) | `Config`, `Session`, `init()`, `AgentRuntime` |
| [API — Engines](api/engines.md) | Engine plugins, `engine_abi.h`, `AiProvider::Engine` |
| [API — Conversation](api/conversation.md) | `FlowMemory`, `run_turn`, pruning, compression |
| [API — FSM](api/fsm.md) | `FlowState`, `Flow`, `FlowHook`, `step_flow`, `run_flow` |
| [API — Scheduler](api/scheduler.md) | `AgentScheduler`, parallel subagents |
| [API — Tools](api/tools.md) | Built-in tools, `Tool` struct, custom tools, Skills, permission gate |
| [API — Memory](api/memory.md) | `MemoryConfig`, `store_memory`, `retrieve_memory` |
| [API — Data Sources](api/data-sources.md) | `DataSource`, RAG/SQLite/GraphRAG integration |
| [API — Events](api/events.md) | `EventType`, `EventHook`, `register_event_hook` |
| [API — Stats](api/stats.md) | `TurnStats`, `SessionStats`, `get_stats` |
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
| `AGENT_ENABLE_ONNX` | `OFF` | ONNX Runtime multimodal inference backend |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UBSan |
| `AGENT_ENABLE_COVERAGE` | `OFF` | gcov coverage instrumentation |
