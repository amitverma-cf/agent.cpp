# Session

`#include <agent-cpp/agent.hpp>`

A `Session` is the central object in agent.cpp. It owns the provider connection, arena allocator, tool registry, memory backend, and all accumulated statistics. Sessions are **move-only** — they cannot be copied, and must not be shared between threads.

---

## AgentRuntime

Create one `AgentRuntime` per process before creating any sessions. It initialises and frees global inference backends (llama.cpp backend, etc.).

```cpp
struct AgentRuntime {
    AgentRuntime();    // calls init_backend()
    ~AgentRuntime();   // calls free_backend()

    AgentRuntime(const AgentRuntime &)            = delete;
    AgentRuntime &operator=(const AgentRuntime &) = delete;
    AgentRuntime(AgentRuntime &&)                 = delete;
    AgentRuntime &operator=(AgentRuntime &&)       = delete;
};
```

`AgentRuntime` must outlive all `Session` objects. Typically declared at the top of `main()`:

```cpp
int main() {
    agent::AgentRuntime runtime;   // init
    // ... create sessions, run agents ...
}                                  // free
```

---

## Config

All session parameters are set through `Config` before calling `init()`. After `init()`, `session.config` is accessible but most fields should not be modified (the exception is `session.config.tools`, which can be mutated followed by `rebuild_tool_index`).

```cpp
struct Config {
    // Provider selection
    AiProvider  provider = AiProvider::Mock;
    std::string base_url;       // required for OpenAICompatible: e.g. "https://api.openai.com"
    std::string api_key;        // Bearer token for OpenAICompatible
    std::string model;          // model file path (llama.cpp/ONNX) or model name (OpenAI)

    // Workspace -- REQUIRED
    std::string workspace_dir;  // must not contain shell metacharacters
                                // init() creates {models,memory,logs,tmp} subdirs

    // Memory backend
    MemoryConfig memory;

    // Tools
    std::vector<Tool> tools;             // user tools; default tools appended during init()
    bool auto_default_tools = true;      // auto-register filesystem+terminal+compressor

    // Inference parameters
    int    context_window = 2048;        // total context size in tokens
    int    max_tokens     = 512;         // max tokens to generate per turn
    float  temperature    = 0.7f;        // sampling temperature (>= 0.0)

    // Memory management
    size_t arena_capacity = 0;           // 0 = auto (context_window*8 + 2 MiB)

    // Safety
    int  max_tool_call_rounds = 10;      // max tool-call loop iterations per turn
    bool sandbox_filesystem   = true;    // block filesystem paths outside workspace_dir

    // Logging
    bool  enable_file_logging  = false;  // write to workspace_dir/logs/agent_*.log
    LogFn logger               = nullptr;
    void *logger_user_data     = nullptr;

    // Network retry (OpenAICompatible provider only)
    int retry_max_attempts  = 3;         // attempts before giving up
    int retry_base_delay_ms = 500;       // initial retry delay; doubles each attempt

    // Parallel subagent bound for OpenAICompatible and LlamaCpp scheduler paths
    // (bounds both the OpenAI thread pool size and llama.cpp's n_seq_max). See scheduler.md.
    int max_parallel_subagents = 4;
};
```

### Required fields

- `workspace_dir` — always required. `init()` returns `InvalidConfig` if empty or if it contains shell metacharacters (`"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, `\n`, `\r`).

### Provider-specific required fields

| Provider | Required fields |
|---|---|
| `Mock` | none |
| `LlamaCpp` | `model` (path to `.gguf` file) |
| `OpenAICompatible` | `base_url`, `model`; `api_key` if the endpoint requires it |
| `OnnxRuntime` | `model` (path to `.onnx` file) |

---

## init()

```cpp
Result<Session> init(const Config &config);
```

Creates a `Session` from `config`. On success, returns a fully-initialised session. On failure, returns an error — no partial state is left behind.

`init()` performs the following steps in order:

1. Validates `workspace_dir` is non-empty and free of shell metacharacters.
2. Validates `context_window > 0`, `max_tokens > 0`, `temperature >= 0`.
3. Creates `workspace_dir/{models,memory,logs,tmp}` via `std::filesystem::create_directories`.
4. Allocates the `Arena` slab.
5. Opens the log file if `enable_file_logging = true`.
6. Initialises the provider (`ProviderOps::init`).
7. Adjusts `memory.db_path` if it equals the default `"memory/db.sqlite3"` (prepends `workspace_dir`).
8. Initialises the memory backend (`MemoryOps::init`).
9. Registers default tools if `auto_default_tools = true`.
10. Builds the O(1) `tool_by_name` index.
11. Logs `"Session initialised."` at `Info` level.

**Example:**

```cpp
agent::AgentRuntime runtime;

auto res = agent::init({
    .provider       = agent::AiProvider::LlamaCpp,
    .model          = ".workspace/models/llama-3.1-8b-q4.gguf",
    .workspace_dir  = ".workspace",
    .context_window = 8192,
    .max_tokens     = 1024,
    .temperature    = 0.7f,
    .memory         = { .provider = agent::MemoryProvider::Sqlite },
    .enable_file_logging = true,
});

if (!res.ok) {
    fprintf(stderr, "init failed: %s\n", res.error.message.c_str());
    return 1;
}
agent::Session session = std::move(res.value);
```

---

## Session struct

```cpp
struct Session {
    Config       config;              // frozen configuration (tools may be mutated)
    Arena        arena;               // default arena, for single-threaded turns
    SessionStats stats;                // accumulated telemetry since init or last reset_stats

    // internal fields -- do not access directly
    std::vector<Arena>         worker_arenas;      // one exclusive arena per parallel scheduler worker
    std::vector<EventHook>     event_hooks;
    std::shared_ptr<void>      provider_state;
    std::shared_ptr<void>      memory_state;
    std::shared_ptr<std::string> workspace_path;   // stable owned workspace path; tools hold a pointer into it
    std::shared_ptr<void>      tool_context;        // e.g. WorkspaceContext for filesystem tools
    std::vector<std::shared_ptr<DataSource>> retained_data_sources;  // keeps bind_data_source_tool sources alive
    std::shared_ptr<std::mutex> stats_mutex;
    std::shared_ptr<std::mutex> log_mutex;
    std::shared_ptr<std::mutex> event_mutex;
    std::unordered_map<std::string, size_t> tool_by_name;  // tool name -> index into config.tools
    std::string                 tools_system_prompt;        // schema block injected into local system prompts
    std::unordered_map<std::string, DataSource> data_sources;
    std::shared_ptr<FILE>      log_file;
    std::shared_ptr<Session *> session_ptr_cell;    // lets tools follow the Session after moves

    Session() = default;
    ~Session() = default;
    Session(const Session &)            = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&)                  = default;
    Session &operator=(Session &&)       = default;
};
```

The fields `config`, `arena`, and `stats` are the only ones you should ever read directly. Everything else is managed internally.

`Session` is move-only and must not be called into from multiple threads directly. `AgentScheduler` achieves concurrency across `OpenAICompatible`/`LlamaCpp` provider turns without sharing a single `Arena` or mutable `Session` state unsafely: each worker thread gets its own slot in `worker_arenas`, and `stats_mutex`/`log_mutex`/`event_mutex` guard the shared counters, log file, and event hooks. The "current turn" tracking (which memory/hooks/tool-allowlist a tool call is running under, which arena slot, the cached JSON parser) is `thread_local` state internal to the executor, not `Session` fields. See [scheduler.md](scheduler.md) for the full concurrency model.

---

## Modifying tools after init

You can add or replace tools after `init()`:

```cpp
agent::Tool extra{
    .name        = "send_email",
    .description = "Send an email.",
    .parameter_schema = R"({"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]})",
    .callback = [](std::string_view args, void *) -> agent::Result<std::string> {
        // ...
        return agent::ok(std::string("sent"));
    },
};

session.config.tools.push_back(extra);
agent::rebuild_tool_index(session);    // must call after mutating tools
```

---

## Session statistics

```cpp
SessionStats get_stats(const Session &session);
void         reset_stats(Session &session);
```

`SessionStats` accumulates across all conversations driven through this session. See [Stats](stats.md) for details.

---

## KV cache

```cpp
Result<void> clear_provider_kv_cache(Session &session);
```

Clears the provider's key-value cache for every active sequence. For llama.cpp, this frees cached attention layers in GPU/CPU memory. For other providers it is a no-op. Useful after major context shifts (e.g. between FSM states) to reduce memory footprint. Caller-timed: call it when you know no inference is in flight on that session.

---

## Manual backend management

These are called by `AgentRuntime` automatically. Only use them if you manage the runtime lifecycle yourself:

```cpp
void init_backend();   // call once at process start
void free_backend();   // call once at process exit
```
