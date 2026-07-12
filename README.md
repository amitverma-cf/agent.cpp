# Agent.cpp
A modular, lightweight C/C++ AI agent library for edge devices and ultra-low latency

## Description

The main goal of this project is to enable lightweight, ultra-low latency AI agent orchestration with minimal setup and zero-cost abstractions on a wide range of hardware — from edge devices to high-end servers.

- **Zero-Cost Abstractions**: Leveraging modern language features to provide a modular framework without performance penalties.
- **Zero Hot-Path Allocations**: Deterministic memory management via a session-owned `Arena` bump allocator, preventing fragmentation and OOM on restricted hardware.
- **Stateless & Stateful Decoupled Engines**: Clean separation between a pure stateless engine (`execute_turn`) and a stateful orchestrator (`run_conversation_turn`).
- **simdjson Parsing**: Employs the world's fastest JSON parser (`simdjson` on-demand) for zero-copy, exception-free arguments extraction and network payload decoding.
- **Stable Binary Interface**: Flat C-ABI boundary for seamless integration with Python, Rust, Go, and other ecosystems.

## Current Features

- **C++20 target** compile validation.
- **Move-only Session Lifetime**: Non-copyable `Session` handles prevent double-frees of local provider/memory handles.
- **Exception-free RAII Guards**: `AgentRuntime` automates global inference backend initialization and teardown safely.
- **Session Arena Allocator**: Zero-allocation hot path; text and tool calls are referenced as `std::string_view` views allocated inside the `Arena`.
- **Event Hook Bus**: A decoupled pub/sub hook system (`OnTurnStart`, `OnInferenceSubmit`, `OnInferenceComplete`, `OnToolCall`, `OnStateTransition`) for logging, subagents, and triggers.
- **FSM State Machine Executor**: An orchestrator supporting deterministic Finite State Machine transitions (`run_agent_fsm`) for ReAct, Planning, or custom agent loop flows.
- **Session Memory Subsystem**: Thread-safe database management with a default `InMemory` provider key-value store and high-performance `RocksDB` persistent database option.
- **Automated Sliding Window Pruning**: Automatically prunes older turns during conversation completion to fit in the context budget, while keeping the system prompt preserved at index 0.
- **Tool Calling & Function Registry**: Register typed callback functions using JSON schemas. Supported natively via OpenAI-compatible payloads and locally using `llama.cpp`.
- **Comprehensive Test Suite**: Modular, warning-free unit and integration test layouts under `tests/` verified under WSL environments.

## Build Configuration

CMake options:

- `AGENT_ENABLE_LLAMACPP=ON`: enable local llama.cpp provider. Default: `ON`.
- `AGENT_ENABLE_NETWORKING=ON`: enable HTTP/HTTPS support. Default: `ON`.
- `AGENT_ENABLE_OPENAI=ON`: enable OpenAI-compatible provider when networking is enabled. Default: `ON`.
- `AGENT_ENABLE_ROCKSDB=ON`: enable RocksDB local key-value store memory provider. Default: `ON`.
- `AGENT_ENABLE_SANITIZERS=ON`: enable AddressSanitizer (ASan) target-specifically. Default: `OFF`.

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

### Stateless Inference Engine
- `agent::execute_turn(session, request)` - Evaluates a `ChatRequest` containing a span of zero-copy `MessageView`s, returning a `ChatResponse` allocated in the session's arena. Supports streaming via `TokenCallback`.

### Stateful Conversation Orchestrator
- `agent::run_conversation_turn(session, state, user_prompt, stream, on_token, token_user_data)` - High-level stateful chat entry point: handles history sliding-window pruning, execute_turn, automatic tool execution loops (ReAct), and history memory syncing.

### Event Hook Bus
- `agent::register_event_hook(session, type, callback, user_data)` - Subscribes a callback to events like `OnToolCall`, `OnInferenceSubmit`, etc.
- `agent::trigger_event(session, type, payload)` - Dispatches an event payload to active hooks.

### FSM Executor
- `agent::run_agent_fsm(session, conv_state, executor)` - Executes state transitions utilizing registered `FSMState` configurations and a transition callback function.

### Session Memory Store
- `agent::store_memory(session, key, value)` - Persists a key-value pair to session storage.
- `agent::retrieve_memory(session, key)` - Retrieves a value by key. Returns `KeyNotFound` error if missing.
- `agent::clear_memory(session)` - Wipes session storage.

### Tool Calling & Registry
- `agent::Tool` - Define a tool with `name`, `description`, `parameter_schema`, and callback pointer matching `Result<std::string> (*)(std::string_view arguments, void *user_data)`.
- Tools are registered via the `Config::tools` field when initializing a session.
- Generated tool calls are parsed exception-free using `simdjson` and automatically executed in the conversation loop.

## Roadmap

Planned but not implemented:

- Flat C ABI with opaque handles.
- HAL boundaries for filesystem, networking, time, and concurrency.
- Vector search / semantic memory provider.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). This repository restricts AI-generated contributions; contributors must understand and own the code they submit.

## License

[MIT LICENSE](LICENSE)
