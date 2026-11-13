# Core types

`#include <agent-cpp/agent.hpp>`

This page covers the types every other part of the API uses: the `Result<T>` return type and its errors, the per-turn `Arena`, media and tensor values, token usage, and the callback signatures. Sessions, conversations, tools and the rest have their own pages (see [Related pages](#related-pages)).

agent.cpp does not use C++ exceptions. Every fallible function returns `Result<T>` (or `Result<void>`).

---

## Result\<T\>

```cpp
template <typename T> struct Result {
    bool  ok = false;
    T     value{};
    Error error{};
};

template <> struct Result<void> {
    bool  ok = true;
    Error error{};
};
```

Check `ok` before you read `value`. When `ok` is false, `value` is default-constructed and `error` describes the failure.

```cpp
auto r = agent::run_turn(session, conv, "hello");
if (!r.ok) {
    fprintf(stderr, "failed [%d]: %s\n", (int)r.error.code, r.error.message.c_str());
    return;
}
printf("%s\n", r.value.c_str());
```

### Helper functions

```cpp
template <typename T> Result<T>   ok(T value);
                      Result<void> ok();

template <typename T> Result<T>   fail(ErrorCode code, std::string message);
                      Result<void> fail(ErrorCode code, std::string message);

Error make_error(ErrorCode code, std::string message);
```

These are for code that implements callbacks, tools or engines. A callback that succeeds returns `agent::ok(std::string("..."))`.

---

## Error and ErrorCode

```cpp
struct Error {
    ErrorCode   code    = ErrorCode::Ok;
    std::string message;   // human-readable; not empty when ok is false
};

enum class ErrorCode {
    Ok,
    InvalidConfig,          // a Config field or argument is invalid (for example an empty workspace_dir)
    UnsupportedProvider,    // the provider or engine is not available in this build or is not found
    FilesystemError,        // a file or directory could not be created or read
    ProviderInitFailed,     // the provider or engine failed to start
    ModelLoadFailed,        // the model file is missing or could not be loaded
    DecodeFailed,           // a response or stored value could not be decoded, or memory ran out
    NetworkError,           // a network connection failed
    HttpError,              // an HTTP error status that is not retried
    ParseError,             // JSON is malformed
    Cancelled,              // the work was cancelled with request_cancel()
    KeyNotFound,            // a memory key or a tool name does not exist
    ToolCallLimitExceeded,  // the model asked for more tool-call rounds than max_tool_call_rounds
    SandboxViolation,       // a path or command is outside what the sandbox allows
    PermissionDenied,       // permission_check denied the call, or an Ask was declined
    EngineCrashed,          // an out-of-process engine died, hit its watchdog or ran out of restarts
    OutOfMemory,            // a ModelRegistry cannot fit the model in its RAM or VRAM budget
    ToolError,              // a tool reported a failure in its result (execute_tool)
};
```

`ToolError` comes from [execute_tool](tools.md): the tool ran, but its result says it failed. `execute_tool_parts` returns such a result as a successful `Result<ToolResult>` with `is_error` set, so check that field there.

### Common cases

| Situation | Code |
|---|---|
| `workspace_dir` is empty or contains a shell metacharacter (`"`, `'`, `&`, `\|`, `;`, `` ` ``, `$`, newline) | `InvalidConfig` |
| `context_window` or `max_tokens` is not greater than 0, or `temperature` is negative | `InvalidConfig` |
| A folder under the workspace cannot be created | `FilesystemError` |
| A sandboxed path is outside the workspace and no grant allows it | `SandboxViolation` |
| `permission_check` returned `Deny`, or `Ask` was declined | `PermissionDenied` |
| The model asked for more tool rounds than `max_tool_call_rounds` | `ToolCallLimitExceeded` |
| `retrieve_memory` on a key that was never stored | `KeyNotFound` |
| `request_cancel()` was called and not yet cleared | `Cancelled` |

---

## Arena

A bump allocator for the memory that one turn of inference needs. Each `Session` has one (`session.arena`), and each parallel graph branch has its own (`session.worker_arenas`). Application code rarely uses an `Arena` directly; the views the library returns from `infer()` and `run_turn()` are described in [conversation.md](conversation.md).

```cpp
class Arena {
public:
    explicit Arena(size_t capacity = 0);   // capacity of the first block; 0 = allocated on first use
    Arena(const Arena &) = delete;
    Arena(Arena &&) noexcept;

    void *allocate(size_t size, size_t alignment = alignof(std::max_align_t));
    void  reset();                                     // invalidates everything allocated so far
    std::string_view allocate_string(std::string_view src);
    template <typename T> std::span<T> allocate_span(size_t count);

    size_t capacity() const;    // bytes reserved in all blocks
    size_t used() const;        // bytes handed out since the last reset
    size_t high_water() const;  // the largest used() so far
};
```

### Size and growth

- `init()` gives the session arena `Config::arena_capacity` bytes, or `context_window * 8 + 2 MiB` when that is 0.
- When a block is full, `allocate()` adds a larger block. Pointers handed out earlier stay valid.
- `reset()` is constant time while the turn fits in the first block. After a turn that grew the arena, the next `reset()` replaces all blocks with one block sized to the peak, so that reset does more work. Later turns of that size allocate nothing.
- `allocate()` returns `nullptr` only for a zero size or when the system is out of memory.

### Lifetime rules

- Data allocated from an arena (including `string_view`s into it) is invalid after `reset()`.
- Do not call `reset()` yourself inside a tool callback. `run_turn` manages the arena.
- Do not store arena-backed views in `Conversation::history`. Use owned `Message` values there.

---

## TokenUsage

Token counts and timings reported by an engine for one model call. Fields an engine does not measure are 0.

```cpp
struct TokenUsage {
    int      prompt_tokens     = 0;  // tokens in the prompt
    int      completion_tokens = 0;  // tokens generated
    int      total_tokens      = 0;
    int      cached_tokens     = 0;  // prompt tokens already in the KV cache, not decoded again
    int      kv_tokens         = 0;  // tokens in the request's KV sequence at the end
    int      kv_max_tokens     = 0;  // capacity of one sequence
    int      kv_type           = 0;  // AGENT_KV_* (engine_abi.h)
    uint64_t kv_bytes          = 0;  // size of the engine's whole KV cache (an estimate for hybrid models)
    float    prefill_ms        = 0;  // time spent on the prompt
    float    decode_ms         = 0;  // time spent generating
};
```

`TurnStats::last` and the `usage` field of `InferResponse` hold this type. See [stats.md](stats.md).

---

## Media

Images, audio, video and meshes are passed by file path. Inputs can be anywhere on disk. Files an engine generates are written under `<workspace>/artifacts/`.

```cpp
struct MediaView {                 // non-owning; used in MessageView
    std::string_view path;
    std::string_view mime;         // for example "image/png", "audio/wav"
};

struct Media {                     // owning; used in Message and ToolResult
    std::string path;
    std::string mime;
};
```

Relative paths in a tool result are made absolute.

---

## Tensors

Named tensors for engines that take raw tensor input or produce tensor output (the `onnx` engine, for example).

```cpp
enum class DType { Float32, Float16, Int8, Int32, Int64, UInt8 };

struct TensorView {                    // non-owning, arena-backed
    std::string_view         name;
    DType                    dtype = DType::Float32;
    std::span<const int64_t> shape;
    std::span<const uint8_t> data;
};

struct TensorData {                    // owning; stored in Message::tensors
    std::string          name;
    DType                dtype = DType::Float32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};
```

Example: a float32 input tensor added to a conversation.

```cpp
agent::TensorData input;
input.name  = "pixel_values";
input.dtype = agent::DType::Float32;
input.shape = {1, 3, 224, 224};
input.data.resize(1 * 3 * 224 * 224 * sizeof(float), 0);
// fill input.data with the pixel bytes

agent::Message msg;
msg.role = "user";
msg.tensors.push_back(std::move(input));
conv.history.push_back(std::move(msg));
```

Tensors are kept in the conversation history but are not written by the history persistence (see [memory.md](memory.md)).

---

## Callbacks

All callbacks are plain function pointers. Pass state through the `user_data` pointer. Callbacks must not throw.

```cpp
enum class LogLevel { Debug, Info, Warning, Error };

using TokenStreamFn = void (*)(std::string_view token, void *user_data);
using ProgressFn    = void (*)(uint64_t step, uint64_t total, void *user_data);  // total 0 = unknown
using LogFn         = void (*)(LogLevel level, std::string_view message, void *user_data);
```

- `TokenStreamFn` receives each token as it is generated when streaming is on (see `run_turn`, `InferRequest::on_token`).
- `ProgressFn` reports the progress of long engine jobs such as image or video generation. It is set with `Config::on_progress`.
- `LogFn` receives library log messages. It is set with `Config::logger`.

```cpp
void on_log(agent::LogLevel level, std::string_view msg, void *) {
    if (level == agent::LogLevel::Warning || level == agent::LogLevel::Error)
        fprintf(stderr, "%.*s\n", (int)msg.size(), msg.data());
}

cfg.logger = on_log;   // cfg is an agent::Config, see session.md
```

---

## Tool call values

`ToolCall` is the owned form of a tool call the model made, and `ToolCallView` is the non-owning form used when building an `InferRequest`.

```cpp
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;   // JSON object text
};

struct ToolCallView {
    std::string_view id;
    std::string_view name;
    std::string_view arguments;
};
```

For tools themselves (`Tool`, `ToolResult`, `PermissionDecision`) see [tools.md](tools.md).

---

## Related pages

- [session.md](session.md): `Config`, `init`, `open_session`, sessions and cancellation
- [conversation.md](conversation.md): `Conversation`, `Message`, `infer`, `run_turn`
- [events.md](events.md): event hooks
- [stats.md](stats.md): token and turn statistics
- [memory.md](memory.md): key-value store and history persistence
- [tools.md](tools.md): tools and the permission gate
- [engines.md](engines.md): engines and their model roles
