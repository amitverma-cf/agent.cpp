# API Reference

All symbols are in the `agent` namespace. Single public header: `#include <agent-cpp/agent.hpp>`.

---

## Error handling

### `ErrorCode`

```cpp
enum class ErrorCode {
    Ok,
    InvalidConfig,        // bad Config field, or missing workspace_dir
    UnsupportedProvider,  // provider not compiled in, or missing required fields
    FilesystemError,      // I/O error on a file or directory
    ProviderInitFailed,   // model load or backend init failed
    ModelLoadFailed,      // model file not found or corrupt
    DecodeFailed,         // JSON parse or response decode failed
    NetworkError,         // connection/timeout error (OpenAI provider)
    HttpError,            // non-retryable HTTP error
    ParseError,           // tool argument JSON is malformed
    Cancelled,            // operation was cancelled
    KeyNotFound,          // memory key missing, or tool not registered
    ToolCallLimitExceeded,// max_tool_call_rounds reached
    SandboxViolation,     // path escapes workspace_dir (sandbox mode)
};
```

### `Error`

```cpp
struct Error {
    ErrorCode   code    = ErrorCode::Ok;
    std::string message;          // human-readable description
};
```

### `Result<T>`

```cpp
template <typename T>
struct Result {
    bool  ok    = false;
    T     value{};
    Error error{};
};

template <>
struct Result<void> {
    bool  ok = true;
    Error error{};
};
```

### Helper functions

```cpp
// Construct a success result
template <typename T> Result<T>   ok(T value);
                      Result<void> ok();

// Construct a failure result
template <typename T> Result<T>   fail(ErrorCode code, std::string message);
                      Result<void> fail(ErrorCode code, std::string message);
```

**Usage:**

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

Bump-pointer slab allocator. One per session, reset in O(1) before each inference call.

```cpp
class Arena {
public:
    explicit Arena(size_t capacity = 1024 * 1024);

    // Allocate `size` bytes aligned to `alignment`. Returns nullptr on overflow.
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t));

    // Reset offset to zero. All previously allocated memory is invalidated.
    void reset();

    // Copy `src` into the arena and return a string_view into it.
    std::string_view allocate_string(std::string_view src);

    // Allocate a contiguous span of T. Returns empty span on overflow.
    template <typename T>
    std::span<T> allocate_span(size_t count);
};
```

The arena is owned by `Session` and must not be used after the session is destroyed. Do not call `reset()` manually inside tool callbacks or during a turn — it is managed automatically by `run_turn`.

---

## Core types

### `TokenUsage`

Token usage from one inference call.

```cpp
struct TokenUsage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int total_tokens      = 0;
};
```

### `DType`

Data type for tensor elements.

```cpp
enum class DType { Float32, Float16, Int8, Int32, Int64, UInt8 };
```

### `TensorView` and `TensorData`

Zero-copy view (arena-backed) and owned storage (heap) for multimodal tensors.

```cpp
struct TensorView {
    std::string_view         name;
    DType                    dtype = DType::Float32;
    std::span<const int64_t> shape;
    std::span<const uint8_t> data;
};

struct TensorData {
    std::string          name;
    DType                dtype = DType::Float32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};
```

### `TokenStreamFn`

Called once per streaming token.

```cpp
using TokenStreamFn = void (*)(std::string_view token, void *user_data);
```

### `LogLevel` and `LogFn`

```cpp
enum class LogLevel { Debug, Info, Warning, Error };
using LogFn = void (*)(LogLevel level, std::string_view message, void *user_data);
```

### `ToolCallView` and `ToolCall`

```cpp
struct ToolCallView {           // zero-copy, arena-backed
    std::string_view id;
    std::string_view name;
    std::string_view arguments; // JSON string
};

struct ToolCall {               // owned, lives in Message
    std::string id;
    std::string name;
    std::string arguments;
};
```

### `Tool`

A callable function the LLM can invoke.

```cpp
struct Tool {
    std::string_view name;
    std::string_view description;
    std::string_view parameter_schema;   // JSON Schema for arguments

    using CallbackFn = Result<std::string>(*)(std::string_view arguments, void *user_data);
    CallbackFn callback  = nullptr;
    void      *user_data = nullptr;
};
```

The `callback` receives raw JSON arguments and returns either a success string or an error. Tool results are inserted back into the conversation history as `role = "tool"` messages.

---

## Message types

### `MessageView`

Zero-copy view into arena memory. Valid only until the next `arena.reset()`.

```cpp
struct MessageView {
    std::string_view              role;           // "system"|"user"|"assistant"|"tool"
    std::string_view              content;
    std::span<const ToolCallView> tool_calls;
    std::string_view              tool_call_id;   // non-empty for role="tool"
    std::span<const TensorView>   tensors;        // multimodal (ONNX)
};
```

### `Message`

Owned, heap-allocated message stored in `FlowMemory::history`.

```cpp
struct Message {
    std::string             role;
    std::string             content;
    std::vector<ToolCall>   tool_calls;
    std::string             tool_call_id;
    std::vector<TensorData> tensors;
    mutable int             token_count = -1;   // -1 = not yet computed
};
```

### `InferRequest`

Input to `infer`.

```cpp
struct InferRequest {
    std::span<const MessageView> messages;
    int            max_tokens      = 0;       // 0 -> use Config::max_tokens
    float          temperature     = -1.0f;   // <0 -> use Config::temperature
    bool           stream          = false;
    TokenStreamFn  on_token        = nullptr;
    void          *token_user_data = nullptr;
};
```

### `InferResponse`

Output from `infer`. Views are arena-backed and invalidated on the next `infer`/`run_turn` call that reuses the same arena slot.

```cpp
struct InferResponse {
    MessageView                 message;
    TokenUsage                  usage;
    std::span<const TensorView> output_tensors;   // ONNX output only
};
```

---

## Memory backend

### `MemoryProvider`

```cpp
enum class MemoryProvider { Sqlite };
```

### `MemoryConfig`

```cpp
struct MemoryConfig {
    MemoryProvider provider = MemoryProvider::Sqlite;
    std::string    db_path  = "memory/db.sqlite3";    // auto-prefixed with workspace_dir at init;
                                                       // ":memory:" for ephemeral storage
    size_t max_cache_bytes             = 16 * 1024 * 1024;
    size_t flush_dirty_threshold_bytes = 4 * 1024 * 1024;
    int    flush_interval_ms           = 500;
};
```

---

## Providers

```cpp
enum class AiProvider { Mock, LlamaCpp, OpenAICompatible, OnnxRuntime };
```

---

## Events

### `EventType`

```cpp
enum class EventType {
    OnTurnStart,           // start of run_turn
    OnInferenceSubmit,     // about to call infer
    OnInferenceComplete,   // infer returned
    OnToolCall,            // tool call dispatched (payload = tool name bytes)
    OnToolResult,          // tool result received (payload = result bytes)
    OnStateTransition,     // FSM state entered (payload = state_id bytes)
    OnCompression,         // context compression completed
    OnPrune,               // history sliding-window prune completed
    OnCronFire,            // cron task fired
    OnSubAgentComplete,    // subagent FSM finished
};
```

### `Event`

```cpp
struct Event {
    EventType                type;
    std::span<const uint8_t> payload;
};
```

### `EventHook`

```cpp
using EventHookFn = void (*)(const Event &event, void *user_data);

struct EventHook {
    EventType   type;
    EventHookFn callback;
    void       *user_data;
};
```

---

## Statistics

### `TurnStats`

Per-conversation counters, stored on `FlowMemory::stats`. Updated automatically by `run_turn`.

```cpp
struct TurnStats {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int tool_calls        = 0;
    int compressions      = 0;
    int prune_cycles      = 0;
    int turns             = 0;
};
```

### `SessionStats`

Session-wide accumulated counters.

```cpp
struct SessionStats {
    int total_prompt_tokens     = 0;
    int total_completion_tokens = 0;
    int total_tool_calls        = 0;
    int total_turns             = 0;
    int total_compressions      = 0;
    int total_prune_cycles      = 0;
};
```

---

## FlowMemory

Stores the history and stats for one conversation thread.

```cpp
struct FlowMemory {
    std::string          id;        // used as key "history:<id>" in memory
    std::vector<Message> history;   // owned message list
    TurnStats             stats;    // per-conversation telemetry
};
```

You create this on the stack or heap and pass it by reference to `run_turn` and `step_flow`. Its lifetime must exceed every call that uses it. `save_flow_memory`/`load_flow_memory` persist/restore it via the memory backend (see [Memory](#memory-1) below).

---

## FSM types

### `FlowHook`

Pre/post callback around tool calls within an FSM state.

```cpp
struct FlowHook {
    enum class When { Pre, Post };
    When when = When::Pre;

    // Empty tool_name = fire for every tool call.
    // Non-empty = fire only when that specific tool is called.
    std::string_view tool_name;

    using HookFn = void (*)(Session &session,
                            FlowMemory &state,
                            std::string_view tool_name,
                            std::string_view args,
                            std::string_view result,  // empty for Pre hooks
                            void *user_data);
    HookFn callback  = nullptr;
    void  *user_data = nullptr;
};
```

### `FlowState`

One node in the state machine.

```cpp
struct FlowState {
    std::string_view state_id;

    // Injected as memory.history[0] for the duration of on_transition.
    // Empty = leave system message unchanged.
    std::string_view system_prompt;

    // Tool allowlist. Empty span = all session tools are available.
    std::span<const std::string_view> allowed_tools;

    // Pre/post tool-call hooks active while in this state.
    std::span<const FlowHook> hooks;

    // true => this state uses Flow::isolated_memories[state_id] instead of Flow::memory.
    bool isolated_memory = false;

    // Optional. Called before on_transition; its return value is folded into
    // effective_system_prompt for this turn only (see docs/api/data-sources.md).
    using ContextProviderFn = std::string_view (*)(Session &session, FlowMemory &memory,
                                                   void *context);
    ContextProviderFn context_provider = nullptr;

    // Required. Returns the next state_id, or "exit" / "" to terminate the FSM.
    using TransitionFn = Result<std::string_view>(*)
                         (Session &session, FlowMemory &memory, void *context);
    TransitionFn on_transition = nullptr;
};
```

### `Flow`

Container for the FSM runtime state plus the memory it drives. Not safe to step concurrently on the same instance (see [scheduler.md](api/scheduler.md)).

```cpp
struct Flow {
    std::string id;
    std::span<const FlowState> states;  // first entry = entry state; caller keeps storage alive

    FlowMemory memory;                                          // shared by default
    std::unordered_map<std::string, FlowMemory> isolated_memories;  // used by isolated_memory states

    // Internal -- managed by step_flow / run_flow. Do not modify.
    size_t current_state_index = 0;
    bool   has_started         = false;
    bool   has_finished        = false;
};
```

---

## Scheduler types

### `CronTask`

```cpp
struct CronTask {
    std::string id;

    using TaskFn = Result<void> (*)(Session &session, void *user_data);
    TaskFn   callback  = nullptr;
    void    *user_data = nullptr;

    std::chrono::steady_clock::time_point next_fire;
    std::chrono::milliseconds             interval{0};   // 0 = one-shot
};
```

### `SubAgentResult`

Result from a completed subagent.

```cpp
struct SubAgentResult {
    std::string id;
    bool        ok = false;
    Error       error;
    TurnStats   stats;
};
```

---

## Data sources

User-registered retrieval backends (RAG, SQL, GraphRAG, arbitrary files, ...), separate from `MemoryConfig` (the library's own small KV store for conversation persistence). See [docs/api/data-sources.md](api/data-sources.md) for the full design, including the built-in sqlite-vec backend.

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

// Binds a DataSource as a Tool and retains `source` on the Session.
Tool bind_data_source_tool(Session &session, std::shared_ptr<DataSource> source,
                           std::string_view description, std::string_view parameter_schema);

// Built-in sqlite-vec backend for RAG/vector search.
Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(
    std::string name, std::string db_path, int dimensions);
Result<void> sqlite_vec_insert(DataSource &source, std::string_view text,
                               std::span<const float> embedding);
```

---

## Configuration

### `Config`

```cpp
struct Config {
    AiProvider        provider      = AiProvider::Mock;
    std::string       base_url;         // OpenAI-compatible endpoint URL
    std::string       api_key;          // API key (OpenAI provider)
    std::string       model;            // model path (llama.cpp/ONNX) or name (OpenAI)
    std::string       workspace_dir;    // REQUIRED -- no shell metacharacters

    MemoryConfig      memory;
    std::vector<Tool> tools;            // user-defined tools; default tools appended by init()

    int    context_window   = 2048;
    int    max_tokens       = 512;
    float  temperature      = 0.7f;
    size_t arena_capacity   = 0;        // 0 = auto (context_window*8 + 2 MiB)

    // Safety
    int  max_tool_call_rounds = 10;     // tool call iterations per turn before ToolCallLimitExceeded
    bool sandbox_filesystem   = true;   // reject filesystem paths outside workspace_dir

    // Default tool auto-registration
    bool auto_default_tools   = true;   // filesystem + terminal + compressor

    // File logging
    bool enable_file_logging  = false;  // write timestamped log to workspace_dir/logs/

    // Network retry (OpenAI provider)
    int retry_max_attempts    = 3;
    int retry_base_delay_ms   = 500;    // doubles each retry

    // Parallel subagent bound for OpenAICompatible (real threads) and LlamaCpp
    // (continuous-batching sequence slots) scheduler paths.
    int max_parallel_subagents = 4;

    LogFn logger          = nullptr;
    void *logger_user_data = nullptr;
};
```

---

## Session

Move-only handle to all session state. Do not copy. Do not share between threads (aside from the specific concurrency AgentScheduler already provides -- see [scheduler.md](api/scheduler.md)).

```cpp
struct Session {
    Config       config;
    Arena        arena;           // default arena for single-threaded turns
    SessionStats stats;

    // internal fields omitted -- do not access directly. Notably: worker_arenas (one per
    // parallel scheduler worker), retained_data_sources (keeps bind_data_source_tool's
    // DataSources alive), tool_by_name, tools_system_prompt, data_sources, log_file,
    // stats_mutex/log_mutex/event_mutex.
};
```

---

## AgentRuntime

RAII wrapper for global backend initialization. Create one per process; it must outlive all sessions.

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
// Manual backend init/free (prefer AgentRuntime for RAII).
void init_backend();
void free_backend();
```

### Session

```cpp
// Create a session. Returns InvalidConfig if workspace_dir is empty or contains
// shell metacharacters, or if context_window/max_tokens/temperature are invalid.
// Creates workspace_dir/{models,memory,logs,tmp} automatically.
Result<Session> init(const Config &config);
```

### Inference

```cpp
// Stateless single-turn inference. The returned InferResponse contains arena-backed
// views valid only until the next infer()/run_turn() call reusing the same arena slot.
Result<InferResponse> infer(Session &session, const InferRequest &request);
```

### Conversation

```cpp
// High-level stateful multi-turn call.
//   - Appends user_prompt to memory.history (skipped if empty).
//   - Auto-compresses if tokens > 70% of context_window.
//   - Prunes history to fit context budget (orphaned tool-message runs are skipped
//     as a group, not treated as a pruning dead-end).
//   - Calls infer() and resolves all tool call loops.
//   - Persists history to memory (via save_flow_memory) after each successful turn.
//   - Returns the assistant's final reply as an owned std::string.
Result<std::string> run_turn(
    Session          &session,
    FlowMemory       &memory,
    std::string_view  user_prompt,
    bool              stream          = false,
    TokenStreamFn     on_token        = nullptr,
    void             *token_user_data = nullptr);

// Summarise oldest messages with the active LLM and replace them with a
// single "[Compressed history]: ..." message. Non-fatal if LLM call fails.
Result<void> compress_context(
    Session    &session,
    FlowMemory &memory,
    size_t      keep_recent = 6);

// Persist / restore FlowMemory.history (+ stats) via the session's memory backend,
// under keys "history:<id>" and "history_stats:<id>". Use load_flow_memory to resume
// a conversation after a process restart -- run_turn only ever writes, never reads.
Result<void> save_flow_memory(Session &session, const FlowMemory &memory);
Result<void> load_flow_memory(Session &session, FlowMemory &memory);
```

### FSM

```cpp
// Run the FSM to completion (loops step_flow until finished or error).
Result<void> run_flow(Session &session, void *context, Flow &flow);

// Advance exactly one FSM state transition.
// Returns Result<bool>: true = still running, false = finished.
Result<bool> step_flow(Session &session, void *context, Flow &flow);
```

### Memory

```cpp
// Called automatically by init(). Call manually only if you switch memory backends.
Result<void> init_memory(Session &session);

// Store an arbitrary key-value pair.
Result<void> store_memory(Session &session, std::string_view key, std::string_view value);

// Retrieve a value by key. Returns KeyNotFound if missing.
Result<std::string> retrieve_memory(Session &session, std::string_view key);

// Delete all key-value pairs in the current memory backend.
Result<void> clear_memory(Session &session);
```

### Tools

```cpp
// Called automatically by init(). Call manually only in exceptional cases.
Result<void> init_tools(Session &session);

// Dispatch a tool by name. Uses the O(1) tool_by_name index.
// Returns KeyNotFound if name is not registered.
Result<std::string> execute_tool(
    Session         &session,
    std::string_view name,
    std::string_view arguments);

// Rebuild the owned-string tool_by_name index (and tools_system_prompt) after mutating
// session.config.tools.
void rebuild_tool_index(Session &session);
```

### Events

```cpp
// Register a callback for a specific event type.
void register_event_hook(
    Session    &session,
    EventType   type,
    EventHookFn callback,
    void       *user_data = nullptr);

// Fire all hooks registered for `type`. Called internally; exposed for testing.
void trigger_event(
    Session                  &session,
    EventType                 type,
    std::span<const uint8_t>  payload);
```

### Stats

```cpp
// Get a copy of the session-wide accumulated stats.
SessionStats get_stats(const Session &session);

// Zero all session-wide stat counters.
void reset_stats(Session &session);
```

### KV cache

```cpp
// Clear the provider's KV cache. No-op for providers that don't expose cache control.
// For llama.cpp: calls llama_memory_clear on the model context -- clears every active
// sequence's KV, so only call this when you know no other Flow has a generation in
// flight on this Session.
Result<void> clear_provider_kv_cache(Session &session);
```

---

## AgentScheduler

Multi-agent scheduler. `pump()`/`run_until_done()` internally dispatch across real OS threads for `OpenAICompatible` and `LlamaCpp` (bounded by `Config::max_parallel_subagents`), and sequentially otherwise; see [scheduler.md](api/scheduler.md) for the concurrency model per provider. `AgentScheduler`'s own methods (`spawn`, `add_cron`, etc.) are not thread-safe themselves — call them from one thread.

```cpp
class AgentScheduler {
public:
    // Register a new subagent FSM. Returns InvalidConfig if id is already used.
    // context is passed through to each step_flow call for that executor.
    Result<void> spawn(std::string id, Flow flow, void *context = nullptr);

    // Register a cron task.
    //   delay    -- time before the first fire
    //   interval -- repeat period; 0 = one-shot
    Result<void> add_cron(
        std::string                id,
        CronTask::TaskFn           fn,
        void                      *user_data,
        std::chrono::milliseconds  delay,
        std::chrono::milliseconds  interval = std::chrono::milliseconds{0});

    // Mark a cron task as done (stops future fires). Returns KeyNotFound if missing.
    Result<void> cancel_cron(std::string_view id);

    // Advance all pending subagents one FSM step each; fire all due cron tasks.
    Result<void> pump(Session &session);

    // Loop pump() until all subagents are done. Continues firing cron tasks.
    Result<void> run_until_done(Session &session);

    // True when every spawned subagent has finished (success or error).
    bool all_done() const;

    // Return a copy of all subagent results (populated as agents complete).
    std::vector<SubAgentResult> results() const;
};
```

---

## Tool factory functions (namespace `agent::tools`)

```cpp
namespace agent::tools {

// Returns 8 filesystem tools (read_file, write_file, etc.).
// ctx must outlive the returned tools.
std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);

// Returns the run_command terminal tool.
// workspace_dir pointer must outlive the returned tool.
Tool get_terminal_tool(const std::string *workspace_dir);

// WorkspaceContext is held in session.tool_context; get a pointer via:
//   static_cast<WorkspaceContext*>(session.tool_context.get())
// (Sandbox checks resolve through hal::resolve_under_workspace -- weakly_canonical +
// prefix comparison, so symlink and ".." escapes are blocked, not just lexical joins.)
struct WorkspaceContext {
    const std::string *workspace_dir = nullptr;
    bool               sandbox       = true;
};

} // namespace agent::tools
```
