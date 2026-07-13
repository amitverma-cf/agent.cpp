# agent.cpp

A modular, lightweight C++20 AI agent orchestration library for edge devices, offline inference, and ultra-low-latency deployments.

---

## What it does

agent.cpp provides a complete runtime for building AI agents on any LLM backend — local (llama.cpp, ONNX Runtime) or remote (any OpenAI-compatible API). It handles context management, tool calling, multi-agent scheduling, FSM-based agent flows, memory persistence, and observability — all with no heap allocations in the hot path.

---

## Key features

- **Zero hot-path allocations.** A session-owned bump-pointer `Arena` handles all per-turn memory. Resets in O(1) between turns.
- **Data-oriented dispatch.** Providers, memory backends, and tools resolve through `constexpr` function-pointer tables — no virtual dispatch, no RTTI.
- **Monadic error handling.** Every fallible function returns `Result<T>`. No exceptions anywhere.
- **Single public header.** `#include <agent-cpp/agent.hpp>`. Backend headers are behind CMake flags.
- **FSM executor.** Decompose agent workflows into named states, each with its own system prompt, tool allowlist, and pre/post hooks.
- **Cooperative multi-agent scheduler.** Run parallel FSM subagents interleaved round-robin with cron tasks — no threads required.
- **Automatic context management.** Sliding-window pruning and LLM-based compression keep conversations within the context budget automatically.
- **Built-in tools.** Filesystem (8 tools), terminal, and context compressor — registered by default.
- **Sandbox enforcement.** Filesystem tools block paths outside `workspace_dir` when sandbox mode is on.
- **Retry with backoff.** OpenAI provider retries on network errors and HTTP 429/5xx.
- **File logging.** Structured timestamped log files written to `workspace_dir/logs/` when enabled.
- **KV cache management.** Explicit cache clear for llama.cpp to manage VRAM between tasks.
- **Telemetry.** Per-conversation and session-wide token/tool/compression counters.

---

## Quick start

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    auto sess = agent::init({
        .provider       = agent::AiProvider::LlamaCpp,
        .model          = ".workspace/models/my.gguf",
        .workspace_dir  = ".workspace",
        .context_window = 4096,
        .max_tokens     = 512,
        .enable_file_logging = true,
    });
    if (!sess.ok) { fprintf(stderr, "%s\n", sess.error.message.c_str()); return 1; }
    agent::Session session = std::move(sess.value);

    agent::FlowMemory conv;
    conv.id = "demo";

    while (true) {
        char buf[512];
        printf("> ");
        if (!fgets(buf, sizeof(buf), stdin)) break;

        auto r = agent::run_turn(session, conv, buf,
            true, [](std::string_view tok, void *){ fwrite(tok.data(), 1, tok.size(), stdout); });
        if (r.ok) printf("\n");
    }

    auto st = agent::get_stats(session);
    printf("turns=%d  tokens=%d\n", st.total_turns,
           st.total_prompt_tokens + st.total_completion_tokens);
    return 0;
}
```

---

## Build

```bash
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

### CMake options

| Option | Default | Description |
|---|---|---|
| `AGENT_ENABLE_LLAMACPP` | `ON` | Local llama.cpp inference (GGUF models) |
| `AGENT_ENABLE_NETWORKING` | `ON` | HTTP/HTTPS via cpp-httplib + OpenSSL |
| `AGENT_ENABLE_OPENAI` | `ON` | OpenAI-compatible provider |
| `AGENT_ENABLE_ONNX` | `OFF` | ONNX Runtime multimodal inference |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UBSan |

---

## Providers

| Provider | Enum | Notes |
|---|---|---|
| Mock | `AiProvider::Mock` | Fixed response; always available; for testing |
| llama.cpp | `AiProvider::LlamaCpp` | Local GGUF models; offline |
| OpenAI-compatible | `AiProvider::OpenAICompatible` | OpenAI, Ollama, LM Studio, Groq, etc. |
| ONNX Runtime | `AiProvider::OnnxRuntime` | Multimodal; `.onnx` models |

---

## Workspace layout

`workspace_dir` is required. Created automatically by `init()`:

```
.workspace/
  models/    <-- GGUF / ONNX model files
  memory/    <-- SQLite database (db.sqlite3)
  logs/      <-- log files (when enable_file_logging = true)
  tmp/       <-- scratch space for tools
```

---

## Documentation

Full documentation here: [`docs`](docs/index.md)

---

## Project layout

```
include/agent-cpp/agent.hpp    -- single public header
src/core/
  agent.cpp                    -- init(), infer(), public functions
  arena.cpp                    -- bump allocator
  events.cpp                   -- event hook dispatch
  executor/
    turn_executor.cpp          -- run_turn, pruning, save/load_flow_memory
    flow_executor.cpp          -- step_flow, run_flow
    ambient_turn.cpp           -- thread_local turn/hook/tool-allowlist/arena/json_parser state
  data/
    data_source_registry.cpp   -- register_data_source, query_data_source, bind_data_source_tool
    sqlite_vec_data_source.cpp -- sqlite-vec RAG/vector search DataSource
  scheduler/
    scheduler.cpp              -- AgentScheduler, cron
  providers/
    mock / llama / openai / onnx
  memory/
    sqlite_provider.cpp        -- bounded write-back cache in front of SQLite
  tools/
    filesystem_tools.cpp       -- read_file, write_file, ...
    terminal_tools.cpp         -- run_command
    context_compressor.cpp     -- compress_context tool
  utils/
    utils.cpp                  -- logging, JSON escaping
src/hal/                       -- hardware/OS abstraction layer
  platform.hpp                 -- AGENT_HAL_WINDOWS / MACOS / LINUX / POSIX detection
  time.cpp                     -- UTC time formatting (gmtime_s vs gmtime_r)
  path.cpp                     -- sandboxed path resolution (symlink/.. escape hardened)
  process.cpp                  -- shell command execution (CreateProcess / posix_spawn)
  log_file.cpp                 -- session log file open/write
examples/
  basic/basic.cpp              -- minimal example
  agent_cli/agent_cli.cpp      -- interactive CLI
tests/
  unit/                        -- provider, memory, utils, onnx, sqlite-vec tests
  integration/                 -- session lifecycle, conversation, FSM tests
vendor/
  llama.cpp / simdjson / cpp-httplib / sqlite / sqlite-vec / onnxruntime
```

---

## License

[MIT License](LICENSE)
