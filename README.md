# Agent.cpp
A modular, lightweight C/C++ AI agent library for edge devices and ultra-low latency

## Description

The main goal of this project is to enable lightweight, ultra-low latency AI agent orchestration with minimal setup and zero-cost abstractions on a wide range of hardware — from edge devices to high-end servers.

- **Zero-Cost Abstractions**: Leveraging modern language features to provide a modular framework without performance penalties.
- **Zero Hot-Path Allocations**: Deterministic memory management via a session-based arena, preventing fragmentation and OOM on restricted hardware.
- **Universal Portability**: Self-contained core designed for everything from high-end servers to resource-constrained environments (WASM, RISC-V, ESP32, etc.).
- **Stable Binary Interface**: Flat C-ABI boundary for seamless integration with Python, Rust, Go, and other ecosystems.

## Current Features

- **C++20 target** compile validation.
- **Move-only Session Lifetime**: Non-copyable `Session` handles prevent double-frees of local provider/memory handles.
- **Exception-free RAII Guards**: `AgentRuntime` automates global inference backend initialization and teardown safely.
- **Usage Metrics & Token Tracking**: Returns `Usage` stats (`prompt_tokens`, `completion_tokens`, `total_tokens`) in `GenerationResult` for local and network inference.
- **Session Memory Subsystem**: Thread-safe database management with a default `InMemory` provider key-value store.
- **Direct Database Conversation Manager**: `Conversation` instances store history directly in the active session memory backend (JSON serialized), maintaining a stateless context.
- **Automated Sliding Window Pruning**: Automatically prunes older turns during conversation completion to fit in the context budget, while keeping the system prompt preserved at index 0.
- **Comprehensive Test Suite**: Modular, warning-free unit and integration test layouts under `tests/` verified under WSL environments.
- **Optional local llama.cpp inference** and **OpenAI-compatible HTTP/HTTPS** chat completion providers.
- **Robust Validation**: Enforces parameter bounds (finite temperature, positive window context/max tokens) during session configuration.

## Build Configuration

CMake options:

- `AGENT_ENABLE_LLAMACPP=ON`: enable local llama.cpp provider. Default: `ON`.
- `AGENT_ENABLE_NETWORKING=ON`: enable HTTP/HTTPS support. Default: `ON`.
- `AGENT_ENABLE_OPENAI=ON`: enable OpenAI-compatible provider when networking is enabled. Default: `ON`.
- `AGENT_ENABLE_ROCKSDB=ON`: enable RocksDB local key-value store memory provider. Default: `ON`.

Example build:

```bash
cmake -B build -G Ninja -DAGENT_ENABLE_LLAMACPP=OFF
cmake --build build
```

Run tests and example:

```bash
./build/test_agent
./build/basic
```

## API Shape Overview

The core API is C++ only today:

### Backend & Session Lifetimes
- `agent::AgentRuntime runtime;` - RAII initialization and teardown of the backend inference environment.
- `agent::init(config)` - Creates a move-only `Session` handle after validating hyperparameters.

### Inference & Streams
- `agent::generate_text(session, prompt)` - Returns `Result<GenerationResult>` containing text response and token usage stats.
- `agent::stream_text(session, prompt, callback, user_data)` - Streams tokens chunk-by-chunk to the callback.

### Session Memory Store
- `agent::store_memory(session, key, value)` - Persists a key-value pair to session storage.
- `agent::retrieve_memory(session, key)` - Retrieves a value by key. Returns `KeyNotFound` error if missing.
- `agent::clear_memory(session)` - Wipes session storage.

### Stateless Conversations
- `agent::add_message(conv, role, content)` - Appends a message directly into session memory.
- `agent::complete_conversation(conv)` - Evaluates sliding window limits, generates the model response, and appends the assistant reply to database storage.
- `agent::get_conversation_history(conv)` - Restores the database conversation turns back into a `std::vector<Message>`.

## Roadmap

Planned but not implemented:

- Flat C ABI with opaque handles.
- Arena allocator and bounded hot-path buffers.
- Deterministic FSM executor.
- Tool registry and tool-call parser.
- HAL boundaries for filesystem, networking, time, and concurrency.
- Vector search / semantic memory provider.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). This repository restricts AI-generated contributions; contributors must understand and own the code they submit.

## License

[MIT LICENSE](LICENSE)

