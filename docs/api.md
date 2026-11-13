# API reference

The entry point to the agent.cpp C++ API. All symbols are in the `agent` namespace and come from one header:

```cpp
#include <agent-cpp/agent.hpp>
```

Other parts have their own headers: `<agent-cpp/hub.hpp>` (model downloads, `agent::hub`), `<agent-cpp/store_sqlite.hpp>` (SQLite
store, graph checkpointer and sqlite-vec, `agent::store`) and `<agent-cpp/agent_c.h>` (the C API). Every call that can fail returns
`Result<T>`. No function throws.

## Pages

Start with the guides if you are new. Use the reference pages for exact signatures.

| Page | What it covers |
|---|---|
| [Getting started (library)](guides/getting-started-library.md) | Building an app against the library |
| [Choosing parts](guides/choosing-parts.md) | Which optional parts to build, and what each one brings |
| [Writing a tool](guides/writing-a-tool.md) | Writing a tool the model can call |
| [Writing a graph](guides/writing-a-graph.md) | Building a multi-step workflow |
| [Writing an engine](guides/writing-an-engine.md) | Writing an engine plugin |
| [Voice assistant](guides/voice-assistant.md) | Speech in and out, end to end |
| [Core](api/core.md) | Errors, arena, types, `Config`, lifecycle |
| [Session](api/session.md) | Creating and using a session |
| [Engines](api/engines.md) | Shipped engines, plugins, isolation, engine ABI |
| [Conversation](api/conversation.md) | Multi-turn chat, history, compression |
| [Tools](api/tools.md) | Tool definitions, built-in tools, permissions, sandbox |
| [Events](api/events.md) | Event hooks |
| [Stats](api/stats.md) | Token and turn counters |
| [Memory](api/memory.md) | Key-value memory and the SQLite store |
| [Data sources](api/data-sources.md) | Pluggable retrieval backends |
| [RAG](api/rag.md) | Chunking, embeddings, rerank, vector stores |
| [Decisions](api/decide.md) | Choice, score and yes/no answers from log-probabilities |
| [Models](api/models.md) | `ModelRegistry`: several models under a memory budget |
| [Model hub](api/hub.md) | Downloading models from Hugging Face |
| [Graph](api/graph.md) | Graphs, nodes, checkpoints, interrupts |
| [agent.json](api/agent-json.md) | Declarative graphs loaded from JSON |
| [MCP](api/mcp.md) | Model Context Protocol servers as tools |
| [C API](api/c-api.md) | `agent_c.h` for C and other languages |

The [CLI](cli.md), [getting started](getting-started.md), [architecture](architecture.md) and [security sandbox](security-sandbox.md)
pages are outside the reference.

---

## Quick start

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    auto session = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if (!session.ok) { std::fprintf(stderr, "%s\n", session.error.message.c_str()); return 1; }

    agent::Conversation chat;
    auto reply = agent::run_turn(session.value, chat, "Say hi in five words.");
    if (reply.ok) std::printf("%s\n", reply.value.c_str());
}
```

`open_session(engine, model_path)` is the shortest way to a session. The workspace is `.agent` unless you pass a `Config` with
`workspace_dir` set. For several model files, engine options, or full control, build the `Config` with `engine_config()` and call
`init()`. Both are described below.

### Choosing an engine

The shipped engines are named by constants in `agent::engines`:

| Constant | Engine |
|---|---|
| `agent::engines::llama` | text, vision, embeddings, rerank, decision models |
| `agent::engines::audiocpp` | speech recognition, text to speech, VAD, diarization |
| `agent::engines::sdcpp` | images and video |
| `agent::engines::trellis` | image to 3D mesh |
| `agent::engines::onnx` | any ONNX model |

The engine plugin is found at run time, next to your program (in an `engines/` folder), or in a folder added with
`agent::engines::search_path(dir)`. In CMake, `agent_deploy_engines(my_app llama)` copies only the engines the application uses. See
[engines.md](api/engines.md).

### Other tasks

Engines that do a task other than chat are called through one-shot functions: `transcribe`, `speak`, `draw` and `perform`. See
[One-shot tasks](#one-shot-tasks).

---

## Error handling

### `ErrorCode`

```cpp
enum class ErrorCode {
    Ok,
    InvalidConfig,         // bad Config field, or missing workspace_dir
    UnsupportedProvider,   // provider or capability not available, or missing required fields
    FilesystemError,       // I/O error on a file or directory
    ProviderInitFailed,    // model load or backend init failed
    ModelLoadFailed,       // model file not found or corrupt
    DecodeFailed,          // JSON parse or response decode failed
    NetworkError,          // connection or timeout error
    HttpError,             // non-retryable HTTP error
    ParseError,            // tool argument JSON is malformed
    Cancelled,             // operation was cancelled
    KeyNotFound,           // memory key missing, tool not registered, or model name not registered
    ToolCallLimitExceeded, // max_tool_call_rounds reached
    SandboxViolation,      // path escapes workspace_dir (sandbox mode)
    PermissionDenied,      // permission_check returned Deny, an Ask was declined, or a licence refused
    EngineCrashed,         // agent-engine-host died or could not be restarted (process isolation)
    OutOfMemory,           // ModelRegistry could not fit the model in its RAM/VRAM budget
    ToolError,             // a tool reported is_error (string execute_tool; execute_tool_parts returns the ToolResult)
};
```

### `Error` and `Result<T>`

```cpp
struct Error {
    ErrorCode code = ErrorCode::Ok;
    std::string message;          // human-readable description
};

template <typename T> struct Result { bool ok = false; T value{}; Error error{}; };
template <> struct Result<void> { bool ok = true; Error error{}; };
```

```cpp
template <typename T> Result<T>   ok(T value);
                      Result<void> ok();
template <typename T> Result<T>   fail(ErrorCode code, std::string message);
                      Result<void> fail(ErrorCode code, std::string message);
```

```cpp
auto r = agent::init(cfg);
if (!r.ok) {
    fprintf(stderr, "[%d] %s\n", (int)r.error.code, r.error.message.c_str());
    return 1;
}
agent::Session session = std::move(r.value);
```

---

## Arena

A bump allocator for one turn's data. Each session has one (`Session::arena`), and `run_turn` resets it before each turn.

```cpp
class Arena {
public:
    explicit Arena(size_t capacity = 0);   // size of the first block; 0 = allocated on first use

    void *allocate(size_t size, size_t alignment = alignof(std::max_align_t));
    void reset();
    std::string_view allocate_string(std::string_view src);
    template <typename T> std::span<T> allocate_span(size_t count);

    size_t capacity() const;               // bytes reserved across all blocks
    size_t used() const;                   // bytes handed out since the last reset
    size_t high_water() const;             // the largest used() seen so far
};
```

The arena grows in chunks instead of failing, and earlier pointers stay valid. `reset()` is O(1) while a turn fits the first block. After a
turn that grew, the next `reset()` replaces the chunks with one block sized to the peak. `allocate` returns `nullptr` only for size 0 or when
the system is out of memory. Do not call `reset()` yourself during a turn: `run_turn` manages it.

---

## Core types

### `TokenUsage`

Token counts and speeds for one inference call. Fields an engine does not measure are 0.

```cpp
struct TokenUsage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
    int cached_tokens = 0;    // prompt tokens already in the KV cache, not decoded again
    int kv_tokens = 0;        // tokens in the request's KV sequence at the end
    int kv_max_tokens = 0;    // capacity of one sequence
    int kv_type = 0;          // AGENT_KV_* (engine_abi.h)
    uint64_t kv_bytes = 0;    // size of the engine's whole KV cache (an estimate for hybrid models)
    float prefill_ms = 0;     // time decoding the prompt
    float decode_ms = 0;      // time generating
};
```

### `DType`, `TensorView` and `TensorData`

```cpp
enum class DType { Float32, Float16, Int8, Int32, Int64, UInt8 };

struct TensorView {                     // zero-copy, arena-backed
    std::string_view name;
    DType dtype = DType::Float32;
    std::span<const int64_t> shape;
    std::span<const uint8_t> data;
};

struct TensorData {                     // owned
    std::string name;
    DType dtype = DType::Float32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};
```

### `Media` and `MediaView`

Media is referenced by file path: images, audio, video, meshes. Inputs can live anywhere. Files that an engine generates are written to
`<workspace>/artifacts/`.

```cpp
struct MediaView { std::string_view path; std::string_view mime; };   // mime e.g. "image/png", "audio/wav"
struct Media     { std::string path; std::string mime; };
```

Pass media with `run_turn_with_media` (user messages), or return it from a tool through `ToolResult::media`.

### `ToolCallView` and `ToolCall`

```cpp
struct ToolCallView { std::string_view id; std::string_view name; std::string_view arguments; };   // zero-copy, arena-backed
struct ToolCall     { std::string id; std::string name; std::string arguments; };                   // owned, lives in Message
```

`arguments` is a JSON string.

### `ToolResult`

What a tool returns through `Tool::parts_callback`: text, media and tensors.

```cpp
struct ToolResult {
    std::string text;
    std::vector<Media> media;
    std::vector<TensorData> tensors;
    bool is_error = false;
};
```

Inside `run_turn`, media and tensors go on the `tool` message. They stay in the history, and the engine gets them as FILE or TENSOR parts on
the next inference. The message text lists each attachment as `[attached: <path> (<mime>)]`. With `is_error`, the conversation gets
`Error: <text>`, and the string `execute_tool` fails with `ErrorCode::ToolError`. See [Tools](api/tools.md#tools-that-return-parts).

### Callbacks

```cpp
using TokenStreamFn = void (*)(std::string_view token, void *user_data);                  // one call per streamed token
using ProgressFn    = void (*)(uint64_t step, uint64_t total, void *user_data);           // long engine jobs; total 0 = unknown
enum class LogLevel { Debug, Info, Warning, Error };
using LogFn        = void (*)(LogLevel level, std::string_view message, void *user_data);
```

### `Tool`

A function the model can call.

```cpp
struct Tool {
    std::string_view name;
    std::string_view description;
    std::string_view parameter_schema;      // JSON Schema for the arguments

    using CallbackFn = Result<std::string> (*)(std::string_view arguments, void *user_data);
    CallbackFn callback = nullptr;
    void *user_data = nullptr;

    using PartsCallbackFn = Result<ToolResult> (*)(std::string_view arguments, void *user_data);
    PartsCallbackFn parts_callback = nullptr;
};
```

Set exactly one of `callback` and `parts_callback`. `execute_tool` fails with `InvalidConfig` otherwise. Call tools through
`execute_tool` or `execute_tool_parts`, never through the function pointers. Those functions run the allowed-tools check and the permission
gate, and refresh the session pointer that some built-in tools use.

Tool results go back into the conversation as `role = "tool"` messages.

### Path gate

When a sandboxed tool wants a path outside the workspace, `Config::path_gate` decides. Without a gate, only `Config::path_grants` apply.

```cpp
struct PathGrant { std::string path; bool write = false; std::string consumer = "*"; };
struct PathAccessRequest { std::string_view consumer; std::string_view path; bool write; };   // path is absolute
enum class PathGateDecision { Deny, AllowOnce, AllowAlways };
using PathGateFn = PathGateDecision (*)(const PathAccessRequest &request, void *user_data);
```

`consumer` is `files`, `run_command` or `mcp:<server>`. `AllowAlways` adds a grant to the session's list. Persisting it is up to the host.
The CLI keeps grants and lists them with `agent sandbox ls`.

### Secrets

```cpp
struct SecretRequest { std::string_view consumer; std::span<const std::string> identity_parts; };   // "mcp:github", {command, args...}
using SecretResolverFn = Result<std::string> (*)(std::string_view text, const SecretRequest &who,
                                                 std::vector<std::string> &secret_values, void *user_data);
```

When the host starts an MCP server, the resolver expands every `${secret:NAME}` in that server's environment for the consumer. It adds each
inserted value to `secret_values`, so the value stays out of logs and tool output. The model never calls the resolver. Placeholders in
command-line arguments are refused.

### Permission gate

`execute_tool` checks each call against the gate just before the callback runs. This covers native tools, tools bound with
`bind_data_source_tool`, and `use_skill`.

```cpp
enum class PermissionDecision { Allow, Ask, Deny };

using PermissionCheckFn  = PermissionDecision (*)(std::string_view tool_name, std::string_view arguments, void *user_data);
using PermissionPromptFn = bool (*)(std::string_view tool_name, std::string_view arguments, void *user_data);   // true = approved
```

`Config::permission_check` defaults to `nullptr`, which allows everything. `Deny`, or an `Ask` that is declined, returns
`ErrorCode::PermissionDenied`, and the callback does not run. `Ask` with no `Config::permission_prompt` is treated as `Deny`.

`agent::tools::default_permission_policy` is a ready-made example. It asks before `run_command`, `delete_path` and `move_path`, and allows
the rest. It is not wired in automatically. Assign it to `Config::permission_check` to use it.

---

## Message types

### `MessageView`

A view into arena memory, valid until the next `arena.reset()`. Engines and tool code receive these.

```cpp
struct MessageView {
    std::string_view role;                        // "system" | "user" | "assistant" | "tool"
    std::string_view content;
    std::span<const ToolCallView> tool_calls;
    std::string_view tool_call_id;                // non-empty for role "tool"
    std::span<const TensorView> tensors;          // raw named tensors (for example, the onnx engine)
    std::span<const MediaView> media;             // images, audio, video by path
};
```

### `Message`

An owned message, stored in `Conversation::history`.

```cpp
struct Message {
    std::string role;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    std::vector<TensorData> tensors;
    std::vector<Media> media;
    mutable int token_count = -1;                 // -1 = not yet computed
};
```

### `InferRequest`

The input to `infer` and `run_task`.

```cpp
struct InferRequest {
    std::span<const MessageView> messages;
    int max_tokens = 0;                           // 0 = Config::max_tokens
    float temperature = -1.0f;                    // < 0 = Config::temperature
    bool stream = false;
    TokenStreamFn on_token = nullptr;
    void *token_user_data = nullptr;
    ProgressFn on_progress = nullptr;             // long engine jobs (image, video, 3D)
    void *progress_user_data = nullptr;
    const volatile int32_t *cancel = nullptr;     // engines stop with ErrorCode::Cancelled when *cancel != 0
    // Engine sessions: the one AGENT_CAP_* bit this call performs. 0 = Config::default_task, then automatic.
    uint32_t task = 0;
    // Engine sessions: extra engine parameters as a JSON object, merged into the request (for example {"mode":"yesno"}).
    std::string_view params_json;
};
```

### `InferResponse`

The output of `infer` and `run_task`. The response owns the memory its views point into (`storage`), so it stays valid across later calls
and can be moved or kept.

```cpp
struct InferResponse {
    MessageView message;
    TokenUsage usage;
    std::span<const TensorView> output_tensors;   // tensor outputs (for example, the onnx engine)
    std::shared_ptr<Arena> storage;
};
```

---

## Memory backend

Key-value storage used for conversation persistence and `store_memory` / `retrieve_memory`. See [memory.md](api/memory.md).

```cpp
struct StoreOps {
    using OpenFn  = Result<std::shared_ptr<void>> (*)(const MemoryConfig &config, std::string_view workspace_dir);
    using PutFn   = Result<void> (*)(void *state, std::string_view key, std::string_view value);
    using GetFn   = Result<std::string> (*)(void *state, std::string_view key);
    using ClearFn = Result<void> (*)(void *state);
    OpenFn open; PutFn put; GetFn get; ClearFn clear;
};

struct MemoryConfig {
    const StoreOps *store = nullptr;          // nullptr = built-in in-memory store;
                                              // agent::store::sqlite_store() from agent_store_sqlite
    std::string db_path;                      // backend-specific; empty = backend default
    size_t max_cache_bytes = 16 * 1024 * 1024;
    size_t flush_dirty_threshold_bytes = 4 * 1024 * 1024;
    int flush_interval_ms = 500;
};
```

---

## Providers and engine isolation

```cpp
enum class AiProvider { Mock, Engine };   // llama.cpp, ONNX Runtime, ... are engine plugins selected by Config::engine

enum class EngineIsolation { Default, InProcess, Process };
```

- `InProcess`: the plugin is loaded into your process. It is the fastest, and a crash in the engine takes the process down.
- `Process`: the plugin runs in an `agent-engine-host` child process. A crash fails the in-flight call with `EngineCrashed`, and the next call
  restarts the host.
- `Default`: the isolation listed for that engine in its `engines.json` manifest, else `InProcess`.

See [engines.md](api/engines.md).

---

## Events

Event hooks fire on turn, inference, tool, graph and compression steps. See [events.md](api/events.md).

```cpp
enum class EventType {
    OnTurnStart,           // start of run_turn
    OnInferenceSubmit,     // about to call the engine
    OnInferenceComplete,   // the engine returned
    OnToolCall,            // tool call dispatched (payload = tool name bytes)
    OnToolResult,          // tool result received (payload = result bytes)
    OnStateTransition,     // graph node started (payload = node name bytes)
    OnCompression,         // context compression completed
    OnPrune,               // history prune completed
    OnSubAgentComplete,    // graph Parallel branch finished (payload = branch entry node name)
};

struct Event { EventType type; std::span<const uint8_t> payload; };
using EventHookFn = void (*)(const Event &event, void *user_data);
struct EventHook { EventType type; EventHookFn callback = nullptr; void *user_data = nullptr; };
```

---

## Statistics

### `TurnStats`

Counters for one conversation, stored in `Conversation::stats` and updated by `run_turn`.

```cpp
struct TurnStats {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int tool_calls = 0;
    int compressions = 0;
    int prune_cycles = 0;
    int turns = 0;
    int cached_tokens = 0;    // prompt tokens served from the KV cache over the whole conversation
    TokenUsage last;          // the most recent model call: prompt and reused tokens, speeds, KV size
};
```

### `SessionStats`

Counters for the whole session.

```cpp
struct SessionStats {
    int total_prompt_tokens = 0;
    int total_completion_tokens = 0;
    int total_tool_calls = 0;
    int total_turns = 0;
    int total_compressions = 0;
    int total_prune_cycles = 0;
};
```

---

## Conversation

The history and stats of one chat thread.

```cpp
struct Conversation {
    std::string id;                            // the key for saved history ("history:<id>")
    std::vector<Message> history;
    TurnStats stats;
    std::vector<std::string> active_tools;     // tools loaded with load_tool (Config::lazy_tools); not part of the history
};
```

Create one per chat and pass it to `run_turn` by reference. Graph Agent nodes keep theirs in `GraphState::histories`. A conversation must
outlive every call that uses it. `save_conversation` and `load_conversation` persist and restore it through the memory backend.

---

## Tool hooks

`ToolHook` runs before or after tool calls. Attach it to a graph Agent node (`GraphNode::hooks`), where it is active only while that node
runs. See [graph.md](api/graph.md#tool-hooks).

```cpp
struct ToolHook {
    enum class When { Pre, Post };
    When when = When::Pre;

    std::string_view tool_name;                // empty = every tool; otherwise only that tool

    using HookFn = void (*)(Session &session, Conversation &memory, std::string_view tool_name, std::string_view args,
                            std::string_view result, void *user_data);   // result is empty for Pre hooks
    HookFn callback = nullptr;
    void *user_data = nullptr;
};
```

Multi-step workflows, with per-node prompts and tool lists, shared or private conversations, parallel branches, checkpoints and interrupts,
are graphs: `Graph`, `GraphNode`, `run_graph`, `resume_graph` and the builders `make_sequential`, `make_parallel` and `make_loop`. See
[graph.md](api/graph.md) and, for JSON graphs, [agent-json.md](api/agent-json.md).

---

## Data sources

A `DataSource` is a named retrieval backend: a vector index, SQL, a graph, or files. It is separate from `MemoryConfig`. See
[data-sources.md](api/data-sources.md).

```cpp
struct DataSource {
    std::string name;
    std::shared_ptr<void> state;
    using QueryFn = Result<std::string> (*)(void *state, std::string_view query, void *user_data);
    QueryFn query = nullptr;
    void *user_data = nullptr;
};

Result<void> register_data_source(Session &session, DataSource ds);
Result<std::string> query_data_source(Session &session, std::string_view name, std::string_view query);
Tool bind_data_source_tool(Session &session, std::shared_ptr<DataSource> source,
                           std::string_view description, std::string_view parameter_schema);
```

The sqlite-vec backend (`agent_store_sqlite`) is described in [data-sources.md](api/data-sources.md#sqlite-vec-rag--vector-search-in-agent_store_sqlite). For chunking,
embeddings and reranking, see [rag.md](api/rag.md).

---

## Skills

[SKILL.md](https://agentskills.io)-format capability packages. See [Skills](guide.md#skills) in the guide for a walkthrough.

```cpp
struct Skill {
    std::string name;         // from SKILL.md frontmatter
    std::string description;  // from SKILL.md frontmatter; the only part kept in context by default
    std::string path;         // directory containing this skill's SKILL.md
};

// Scans skills_dir for one subdirectory per skill (each with a SKILL.md), fills session.skills, and folds each
// skill's name and description into the tools system prompt. Called by init_tools() when Config::skills_dir is set.
Result<void> load_skills_dir(Session &session, std::string_view skills_dir);

// Full SKILL.md content for a loaded skill, read on demand.
Result<std::string> read_skill_body(Session &session, std::string_view name);
```

---

## Configuration

### `Config`

The fields are grouped by topic here. For designated initializers, use the order of the fields in `agent.hpp`.

```cpp
struct Config {
    AiProvider provider = AiProvider::Mock;
    std::string engine;             // Engine: a registered engine name or a path to an engine plugin
    std::string engine_options;     // Engine: JSON object forwarded to the engine as "options"
    std::string model;              // the model file the engine loads
    std::string workspace_dir;      // required. No quotes, $, &, |, ;, or newlines.

    MemoryConfig memory;
    std::vector<Tool> tools;        // your tools; the default tools are added by init()

    int context_window = 2048;
    int max_tokens = 512;
    float temperature = 0.7f;
    size_t arena_capacity = 0;      // first arena block; 0 = context_window * 8 + 2 MiB; grows past it

    // Tools and safety
    int max_tool_call_rounds = 10;  // tool-call rounds per turn before ToolCallLimitExceeded
    bool sandbox_filesystem = true; // file tools and run_command stay inside workspace_dir (plus path_grants)
    bool auto_default_tools = true; // filesystem tools, run_command, compress_context
    bool lazy_tools = false;        // only tool_search, load_tool and use_skill start in the prompt
    PermissionCheckFn permission_check = nullptr;   // nullptr = allow everything
    PermissionPromptFn permission_prompt = nullptr; // resolves Ask; nullptr = Ask is Deny
    void *permission_user_data = nullptr;
    std::vector<PathGrant> path_grants;
    PathGateFn path_gate = nullptr;
    void *path_gate_user_data = nullptr;
    SandboxPolicy tool_sandbox{.enabled = true};   // run_command
    SandboxPolicy engine_sandbox{.enabled = true, .allow_tmp = false, .network = SandboxNetwork::Deny};   // engine host
    SecretResolverFn secret_resolver = nullptr;     // ${secret:NAME} in MCP server environments
    void *secret_user_data = nullptr;
    using McpLauncherFn = void (*)(McpServerConfig &server, void *user_data);
    McpLauncherFn mcp_launcher = nullptr;           // rewrite an MCP server's command before it starts
    void *mcp_launcher_user_data = nullptr;

    // Skills and logging
    std::string skills_dir;         // empty = skills off
    bool enable_file_logging = false;   // write a log under workspace_dir/logs/
    LogFn logger = nullptr;
    void *logger_user_data = nullptr;

    // Long-running engine jobs and parallel graph branches
    ProgressFn on_progress = nullptr;
    void *progress_user_data = nullptr;
    int max_parallel_subagents = 4; // parallel graph branches; llama continuous-batching slots

    // Engine plugins (Engine provider)
    uint32_t default_task = 0;      // AGENT_CAP_* for requests without InferRequest::task; 0 = automatic
    EngineIsolation engine_isolation = EngineIsolation::Default;
    std::string engine_host_path;   // agent-engine-host; empty = found next to the program, else PATH
    int engine_timeout_ms = 0;      // per-request watchdog for process isolation; 0 = none
    int engine_cancel_grace_ms = 2000;
    int engine_max_restarts = 3;    // automatic host restarts per rolling minute
    std::string gpu_cache_dir;      // GPU driver caches when engine_sandbox.gpu is set; empty = <workspace>/cache/gpu
};
```

Tool-related defaults and the sandbox rules are in [tools.md](api/tools.md) and [security-sandbox.md](security-sandbox.md).

---

## Session

A `Session` holds all state for one engine or mock provider: the config, arena, memory, tools, data sources, skills and stats. It is
move-only. Do not copy it, and do not use it from several threads at once, except where a graph's parallel branches already do that for
you.

```cpp
struct Session {
    Config config;
    Arena arena;                    // per-turn arena
    std::vector<Arena> worker_arenas;   // one per parallel graph branch
    SessionStats stats;
    // Internal fields are not for direct use: data_sources, skills, tool_by_name,
    // tools_system_prompt, retained_data_sources, log_file, cancel_flag, and mutexes.
};
```

---

## AgentRuntime

An RAII wrapper for process-wide backend setup. Create one per process, before the first session, and keep it alive until the last session
is gone.

```cpp
struct AgentRuntime {
    AgentRuntime();    // calls init_backend()
    ~AgentRuntime();   // calls free_backend()
    // non-copyable, non-movable
};
```

---

## Public functions

### Backend

```cpp
void init_backend();   // prefer AgentRuntime
void free_backend();
```

### open_session

The shortest way to a session: one engine, the model files, defaults for everything else. The workspace is `.agent` unless `base` names one.

```cpp
struct ModelSlot { std::string_view role; std::string_view path; };

Result<Session> open_session(std::string_view engine, std::string_view model_path, const Config &base = {});
Result<Session> open_session(std::string_view engine, std::initializer_list<ModelSlot> models, const Config &base = {});
```

```cpp
auto chat  = agent::open_session(agent::engines::llama, "model.gguf");
auto image = agent::open_session(agent::engines::sdcpp, {{"model", "sd.gguf"}, {"vae", "vae.gguf"}});
auto voice = agent::open_session(agent::engines::audiocpp, {{"asr.model", "asr.gguf"}, {"tts.model", "tts.gguf"}, {"tts.voice", "alba"}});
```

Roles name the model files the engine needs: llama `model` and `mmproj`; sdcpp `diffusion_model`, `llm`, `vae`, `clip_l` and `t5xxl`;
audiocpp `tts.model`, `asr.model`, `vad.model` and `diarize.model`. The model is passed as the role named `model`, and dotted roles are
placed in sections of the engine options: `tts.model` becomes `{"tts":{"model":...}}`. Other values, such as `tts.voice`, go into the
options the same way.

### engine_config

Builds a `Config` for an explicit engine from role-named model files, so you do not write the options JSON for model paths. Use it when you
need engine options that `open_session` cannot set, such as `{"mode":"embed"}`:

```cpp
Result<Config> engine_config(std::string_view engine, std::span<const ModelSlot> models,
                             std::string_view extra_options_json = {}, const Config &base = {});
```

- `engine` is an engine constant, another registered name, or a plugin path.
- `extra_options_json` is a JSON object merged into the engine options, for example threads or voices.
- `base` supplies everything else. Set `base.workspace_dir` yourself: unlike `open_session`, `engine_config` does not default it.

```cpp
agent::Config base;
base.workspace_dir = ".workspace";
const agent::ModelSlot files[] = {{"model", ".workspace/models/bge-small-en-v1.5-q8_0.gguf"}};
auto cfg = agent::engine_config(agent::engines::llama, files, R"({"mode":"embed"})", base);
auto session = agent::init(cfg.value);
```

### Session

```cpp
Result<Session> init(const Config &config);
```

Creates the session. It returns `InvalidConfig` when `workspace_dir` is empty or contains shell metacharacters, or when
`context_window`, `max_tokens` or `temperature` is invalid. It creates `workspace_dir/{models,memory,logs,tmp}`. It initializes the
provider and memory, and registers the default tools, `Config::tools`, and skills when `skills_dir` is set. Call `init_memory` or
`init_tools` yourself only in unusual cases.

### One-shot tasks

```cpp
struct TaskOutput { std::string text; std::vector<std::string> files; };   // files: paths under <workspace>/artifacts

Result<TaskOutput> transcribe(Session &session, std::string_view audio_path);                              // AGENT_CAP_ASR
Result<TaskOutput> speak(Session &session, std::string_view text);                                         // AGENT_CAP_TTS; files[0] is a WAV
Result<TaskOutput> draw(Session &session, std::string_view prompt, std::string_view params_json = {});     // AGENT_CAP_IMAGE_GEN
Result<TaskOutput> perform(Session &session, uint32_t task, std::string_view text,
                           std::string_view media_path = {}, std::string_view params_json = {});           // any AGENT_CAP_* bit
Result<std::string> ocr(Session &session, const Media &image, std::string_view prompt = {});                // AGENT_CAP_OCR
```

The session's engine must support the task. `params_json` is a JSON object of engine parameters, for example `{"steps":1,"cfg_scale":1}`.
`ocr` uses an OCR-tuned vision model or any vision model with the `mmproj` option. `prompt` overrides the model family's default. For
streaming, progress callbacks or tensors, use `run_task` with an `InferRequest`.

```cpp
auto voice = agent::open_session(agent::engines::audiocpp, {{"tts.model", "tts.gguf"}});
auto wav = agent::speak(voice.value, "Hello there.");   // wav.value.files[0] is the WAV path
```

### Inference

```cpp
// Stateless single-turn inference. The response owns the memory its views point into.
Result<InferResponse> infer(Session &session, const InferRequest &request);

// infer() with InferRequest::task set to one AGENT_CAP_* bit, for example run_task(session, AGENT_CAP_TTS, request).
Result<InferResponse> run_task(Session &session, uint32_t task, const InferRequest &request);
```

### Conversation turns

```cpp
// The high-level, stateful turn. Appends user_prompt to memory.history, runs the model and any tool calls, and returns
// the assistant's final reply.
//   - Compresses the older history when the prompt would take more than 70% of context_window.
//   - Prunes history to fit the budget. Tool messages that belong together are kept or dropped as a group.
//   - Saves the history through the memory backend after each successful turn. Only save_conversation writes it;
//     load_conversation is the only way back in after a restart.
Result<std::string> run_turn(Session &session, Conversation &memory, std::string_view user_prompt, bool stream = false,
                             TokenStreamFn on_token = nullptr, void *token_user_data = nullptr);

// run_turn with media on the user message (images, audio, ...), for engines that accept them.
Result<std::string> run_turn_with_media(Session &session, Conversation &memory, std::string_view user_prompt,
                                        std::span<const Media> media, bool stream = false,
                                        TokenStreamFn on_token = nullptr, void *token_user_data = nullptr);

// Summarises the oldest messages with the active model and replaces them with one "[Compressed history]: ..." message.
// Non-fatal if the model call fails.
Result<void> compress_context(Session &session, Conversation &memory, size_t keep_recent = 6);

// Persist and restore history and stats through the memory backend, under "history:<id>" and "history_stats:<id>".
Result<void> save_conversation(Session &session, const Conversation &memory);
Result<void> load_conversation(Session &session, Conversation &memory);
```

```cpp
agent::Conversation chat;
agent::run_turn(session, chat, "What is the capital of France?");
agent::run_turn(session, chat, "And of Italy?");          // the second turn sees the first
agent::save_conversation(session, chat);                 // stored under "history:<id>"
```

### Graph runtime

Graphs are described in [graph.md](api/graph.md). The main calls are:

```cpp
Result<void> run_graph(Session &session, const Graph &graph, GraphRun &run);
Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run,
                          std::string_view human_input = {});
Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run,
                          PermissionDecision decision, std::string_view human_input = {});
```

### Memory

```cpp
Result<void> init_memory(Session &session);     // called by init(); call again only after switching backends
Result<void> store_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_memory(Session &session, std::string_view key);   // KeyNotFound if missing
Result<void> clear_memory(Session &session);   // deletes every key in the backend
```

### Tools

```cpp
Result<void> init_tools(Session &session);      // called by init(); loads skills when skills_dir is set
void rebuild_tool_index(Session &session);      // after you change session.config.tools

// Dispatches a tool by name, through the allowed-tools check and the permission gate. KeyNotFound for an unknown name.
// Media of a parts tool are listed as "[attached: <path> (<mime>)]" lines; a ToolResult with is_error fails with ToolError.
Result<std::string> execute_tool(Session &session, std::string_view name, std::string_view arguments);

// The same checks, returning the whole ToolResult (text, media, tensors, is_error). See api/tools.md.
Result<ToolResult> execute_tool_parts(Session &session, std::string_view name, std::string_view arguments);
```

### Event hooks

```cpp
void register_event_hook(Session &session, EventType type, EventHookFn callback, void *user_data = nullptr);
void trigger_event(Session &session, EventType type, std::span<const uint8_t> payload);   // called internally; exposed for tests
```

### Stats

```cpp
SessionStats get_stats(const Session &session);   // a copy of the session-wide counters
void reset_stats(Session &session);
```

### Cancelling

```cpp
void request_cancel(Session &session);   // stops in-flight engine work; later inferences fail with Cancelled
void clear_cancel(Session &session);     // lets the session run again
```

`request_cancel` is safe to call from any thread. Until `clear_cancel` is called, every later inference on the session fails with
`ErrorCode::Cancelled`.

### KV cache

```cpp
// Clears the engine's KV cache. A no-op for engines without cache control. For llama this clears every sequence,
// so call it only when no other turn is running on this session.
Result<void> clear_provider_kv_cache(Session &session);
```

### Engines

```cpp
namespace agent::engines {
void search_path(std::string_view dir);                                              // searched before the defaults
Result<void> add(std::string_view name, std::string_view plugin_path, EngineIsolation isolation = EngineIsolation::Default);
void remove(std::string_view name);
std::vector<std::string> list();
}

Result<void> register_engine(const agent_engine_v1 *engine);                 // an engine linked into the host
Result<std::string> load_engine_plugin(std::string_view path);               // loads a plugin, returns its name
const agent_engine_v1 *find_engine(std::string_view name);
Result<std::vector<std::string>> load_engine_manifest(std::string_view manifest_path);   // engines.json
```

Engine plugins are process-wide and stay loaded until exit. The full picture, including isolation and manifests, is in
[engines.md](api/engines.md).

### Optional parts

These parts can be left out of a build with CMake options. Each page says what it contains.

| CMake option | Default | What it removes | Page |
|---|---|---|---|
| `AGENT_WITH_RAG` | on | `embed`, `rerank`, `chunk_text`, the RAG index and vector stores, `search_knowledge` | [rag.md](api/rag.md) |
| `AGENT_WITH_MODELS` | on | `ModelRegistry`, leases, memory estimates | [models.md](api/models.md) |
| `AGENT_WITH_GRAPH` | on | Graphs, checkpoints, `agent.json` loader | [graph.md](api/graph.md), [agent-json.md](api/agent-json.md) |
| `AGENT_WITH_MCP` | on | MCP client, `connect_mcp_server` | [mcp.md](api/mcp.md) |

The `agent.json` loader needs both graphs and MCP. Without RAG, `agent_c_embed` returns an error.

---

## Tool factory functions (namespace `agent::tools`)

```cpp
namespace agent::tools {

// The eight filesystem tools: read_file, write_file, append_file, list_dir, create_dir, delete_path, move_path, file_info.
// ctx must outlive the returned tools.
std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);

// run_command. The workspace is read-write under the sandbox policy in ctx->command_sandbox.
// ctx must outlive the tool.
Tool get_terminal_tool(std::shared_ptr<WorkspaceContext> ctx);
Tool get_terminal_tool(const std::string *workspace_dir);   // no OS sandbox; prefer the WorkspaceContext overload

// The context is held in session.tool_context. Get a pointer with:
//   static_cast<WorkspaceContext *>(session.tool_context.get())
struct WorkspaceContext {
    const std::string *workspace_dir = nullptr;
    bool sandbox = true;
    SandboxPolicy command_sandbox;                 // for run_command
    const volatile int32_t *cancel = nullptr;      // stops a running command when non-zero
    std::shared_ptr<std::vector<PathGrant>> grants;
    PathGateFn gate = nullptr;
    void *gate_user_data = nullptr;
};

// Ask before run_command, delete_path and move_path; allow the rest. Not wired in automatically.
PermissionDecision default_permission_policy(std::string_view tool_name, std::string_view arguments, void *user_data);

} // namespace agent::tools
```

Sandbox checks resolve paths with `hal::resolve_under_workspace`, which compares canonical paths, so `..` and symlink escapes are blocked,
not only lexical joins. The sandbox rules are in [security-sandbox.md](security-sandbox.md).
