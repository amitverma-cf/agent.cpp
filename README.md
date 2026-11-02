# agent.cpp

A modular, lightweight C++20 AI agent orchestration library for edge devices, offline inference, and ultra-low-latency deployments.

---

## What it does

agent.cpp provides a complete runtime for building AI agents on local inference backends (llama.cpp, ONNX Runtime, ... loaded as engine plugins), fully offline. It handles context management, tool calling, graph workflows with parallel subagents, memory persistence, and observability — all with no heap allocations in the hot path.

---

## Key features

- **Zero hot-path allocations.** A session-owned bump-pointer `Arena` handles all per-turn memory. Resets in O(1) between turns.
- **Data-oriented dispatch.** Providers, memory backends, and tools resolve through `constexpr` function-pointer tables — no virtual dispatch, no RTTI.
- **Monadic error handling.** Every fallible function returns `Result<T>`. No exceptions anywhere.
- **Single public header.** `#include <agent-cpp/agent.hpp>`. Backend headers are behind CMake flags.
- **Graph workflows.** LangGraph / ADK-style graphs of agent, function and tool nodes: per-node system prompt, tool allowlist and pre/post tool hooks, private or shared conversations, conditional edges, sequential / parallel / loop builders, checkpoints and human-in-the-loop interrupts. Load them from C++ or `agent.json`.
- **Parallel subagents.** Parallel graph branches run concurrently on concurrent engines; llama.cpp branches share one continuously-batched decode.
- **Automatic context management.** Sliding-window pruning and LLM-based compression keep conversations within the context budget automatically.
- **Built-in tools.** Filesystem (8 tools), terminal, and context compressor — registered by default.
- **Skills.** Load [SKILL.md](https://agentskills.io)-format capability packages from a directory; only name/description sit in context until the model calls `use_skill` to load the rest.
- **Permission gate.** Optional per-call Allow/Ask/Deny policy checked before any tool dispatch — native tools, data-source tools, and `use_skill` alike.
- **Sandbox enforcement.** Filesystem tools block paths outside `workspace_dir` when sandbox mode is on.
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
        .provider       = agent::AiProvider::Engine,
        .engine         = "build/engines/llama/agent-engine-llama.so",
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

## The `agent` CLI

The build also produces `build/agent`, the command-line app: a setup wizard, an engine and model
manager (Hugging Face downloads) and a chat TUI with streaming and tool calls.

```bash
./build/agent setup --yes
./build/agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
./build/agent chat
```

See [docs/cli.md](docs/cli.md).

## Install and package

```bash
cmake --install build --prefix /opt/agent   # bin/agent, lib/agent/engines/, libagent + headers + CMake package
scripts/package.sh --engines llama          # dist/agent-<version>-linux-x64.tar.gz
```

Library users: `find_package(agent-cpp)` and link `agent-cpp::agent` (plus `agent-cpp::store_sqlite`,
`agent-cpp::hub`). See [docs/packaging.md](docs/packaging.md); Windows and macOS packaging are still TODO.

### CMake options

| Option | Default | Description |
|---|---|---|
| `AGENT_ENGINE_LLAMA` | `ON` | Build the llama.cpp engine plugin (`engines/llama`, GGUF models) |
| `AGENT_BUILD_CLI` | `ON` | Build the `agent` CLI binary (`tools/cli`); needs `AGENT_BUILD_HUB` |
| `AGENT_BUILD_HUB` | `ON` | Build `agent_hub`, the Hugging Face model downloader (needs OpenSSL) |
| `BUILD_SHARED_LIBS` | `OFF` | Build `libagent`, `agent_store_sqlite`, `agent_hub` as shared libraries |
| `AGENT_INSTALL` | `ON` (top level) | Generate install rules and the `agent-cpp` CMake package |
| `AGENT_ENGINE_ONNX` | `OFF` | Build the ONNX Runtime engine plugin (`engines/onnx`); needs `-DONNXRUNTIME_ROOT=<prebuilt ORT>` |
| `AGENT_ENGINE_TRELLIS` | `OFF` | Build the TRELLIS.2 image-to-3D (GLB) engine plugin (`engines/trellis`); needs the `vendor/trellis2cpp` submodule and its ggml; `AGENT_TRELLIS_BUILD_JOBS` caps the nested build |
| `AGENT_ENGINE_WHISPER` | `OFF` | Build the whisper.cpp speech-to-text + Silero VAD engine plugin (`engines/whisper`); needs the `vendor/whisper.cpp` submodule; `AGENT_WHISPER_BUILD_JOBS` caps the nested build |
| `AGENT_ENGINE_PARAKEET` | `OFF` | Build the parakeet.cpp (NVIDIA Parakeet / Nemotron) speech-to-text engine plugin (`engines/parakeet`); needs the `vendor/parakeet.cpp` submodule and its ggml; `AGENT_PARAKEET_BUILD_JOBS` caps the nested build |
| `AGENT_ENGINE_QWEN3TTS` | `OFF` | Build the Qwen3-TTS text-to-speech + voice cloning engine plugin (`engines/qwen3tts`, qwentts.cpp); needs the `vendor/qwentts.cpp` submodule and its ggml; `AGENT_QWEN3TTS_BUILD_JOBS` caps the nested build |
| `AGENT_ENABLE_SANITIZERS` | `OFF` | AddressSanitizer + UBSan |
| `AGENT_ENABLE_COVERAGE` | `OFF` | gcov coverage instrumentation |

---

## Providers

| Provider | Enum | Notes |
|---|---|---|
| Mock | `AiProvider::Mock` | Fixed response; always available; for testing |
| Engine plugin | `AiProvider::Engine` | Any engine shared library built against `engine_abi.h`: `agent-engine-llama` (GGUF), `agent-engine-onnx` (`.onnx`, raw tensors), ... (see [docs/api/engines.md](docs/api/engines.md)) |

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
    ambient_turn.cpp           -- thread_local turn/hook/tool-allowlist/arena/json_parser state
  data/
    data_source_registry.cpp   -- register_data_source, query_data_source, bind_data_source_tool
  skills/
    skills_registry.cpp        -- load_skills_dir, read_skill_body, use_skill tool
  graph/
    graph_executor.cpp         -- run_graph, resume_graph, parallel branches
    graph_builders.cpp         -- make_graph / make_sequential / make_parallel / make_loop
    graph_checkpoint.cpp       -- checkpoints, scoped state persistence
    graph_json.cpp             -- agent.json loader
  providers/
    provider_ops.cpp           -- Mock + Engine dispatch
    engine_provider.cpp        -- adapter to engine plugins (engine_abi.h)
  memory/
    in_memory_store.cpp        -- default StoreOps (no persistence)
  tools/
    tools_ops.cpp              -- dispatch table, permission gate (Allow/Ask/Deny)
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
engines/
  mock/ llama/ onnx/ sdcpp/ sherpa/ trellis/ whisper/ parakeet/ qwen3tts/  -- engine plugins (agent-engine-<name>)
tools/
  cli/                         -- the `agent` binary (docs/cli.md)
  hub/                         -- agent_hub: Hugging Face downloads, models.json
  store-sqlite/                -- agent_store_sqlite: SQLite KV store, sqlite-vec
tests/
  unit/                        -- Catch2 tests: providers, memory, tools, hal, skills, permissions, ...
  integration/                 -- session lifecycle, conversation tests
vendor/
  llama.cpp / simdjson / cpp-httplib / sqlite / sqlite-vec / catch2
```

---

## License

[MIT License](LICENSE)
