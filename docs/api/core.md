# Core Types

`#include <agent-cpp/agent.hpp>`

---

## Result\<T\>

Every fallible function in agent.cpp returns a `Result<T>`. There are no exceptions.

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

**Always check `.ok` before accessing `.value`.** Accessing `.value` when `.ok` is `false` is undefined behaviour.

```cpp
auto r = agent::run_turn(session, conv, "hello");
if (!r.ok) {
    fprintf(stderr, "failed [%d]: %s\n", (int)r.error.code, r.error.message.c_str());
    return;
}
printf("%s\n", r.value.c_str());
```

### Helper constructors

```cpp
template <typename T> Result<T>   ok(T value);
                      Result<void> ok();

template <typename T> Result<T>   fail(ErrorCode code, std::string message);
                      Result<void> fail(ErrorCode code, std::string message);

Error make_error(ErrorCode code, std::string message);
```

---

## ErrorCode

```cpp
enum class ErrorCode {
    Ok,                   // no error
    InvalidConfig,        // bad Config field, or missing required workspace_dir
    UnsupportedProvider,  // provider not compiled in, or missing fields (base_url etc.)
    FilesystemError,      // I/O error on file or directory operation
    ProviderInitFailed,   // model load or backend initialisation failed
    ModelLoadFailed,      // model file not found or corrupt
    DecodeFailed,         // JSON parse failure or arena overflow
    NetworkError,         // TCP/TLS connection error
    HttpError,            // non-retryable HTTP error status
    ParseError,           // tool argument JSON is malformed
    Cancelled,            // operation was cancelled mid-flight
    KeyNotFound,          // memory key missing, or tool name not registered
    ToolCallLimitExceeded,// max_tool_call_rounds limit reached in one turn
    SandboxViolation,     // path escapes workspace_dir in sandbox mode
};
```

### Common patterns

| Situation | Expected code |
|---|---|
| `workspace_dir` not set | `InvalidConfig` |
| `workspace_dir` contains `&` or `'` | `InvalidConfig` |
| `context_window` or `max_tokens` <= 0 | `InvalidConfig` |
| llama.cpp model file missing | `ProviderInitFailed` |
| OpenAI API key wrong | `HttpError` |
| Tool argument missing required field | `ParseError` |
| Filesystem path outside workspace | `SandboxViolation` |
| LLM stuck calling tools in a loop | `ToolCallLimitExceeded` |
| Memory key not found | `KeyNotFound` |

---

## Error

```cpp
struct Error {
    ErrorCode   code    = ErrorCode::Ok;
    std::string message;   // human-readable, never empty when ok=false
};
```

---

## Arena

A bump-pointer slab allocator. One instance lives in each `Session`. All arena-backed objects become invalid after the next `arena.reset()` call.

```cpp
class Arena {
public:
    explicit Arena(size_t capacity = 1024 * 1024);

    ~Arena();
    Arena(const Arena &)            = delete;
    Arena &operator=(const Arena &) = delete;
    Arena(Arena &&) noexcept;
    Arena &operator=(Arena &&) noexcept;

    // Allocate `size` bytes with the given alignment.
    // Returns nullptr if the slab is exhausted.
    void *allocate(size_t size, size_t alignment = alignof(std::max_align_t));

    // Reset the offset to zero in O(1). Invalidates all previously allocated memory.
    void reset();

    // Copy src into the arena and return a string_view backed by it.
    std::string_view allocate_string(std::string_view src);

    // Allocate a contiguous array of T. Returns empty span on overflow.
    template <typename T>
    std::span<T> allocate_span(size_t count);
};
```

### Capacity

Default: `context_window * 8 + 2 MiB`. Override via `Config::arena_capacity`. The arena is allocated once at `init()` time and never reallocated.

### Lifetime rules

- Do not call `reset()` manually inside a tool callback or a `TransitionFn`. `run_turn` manages resets.
- Do not store arena-backed `string_view` or `span` values in `FlowMemory::history`. History uses owned `Message` objects.
- Views into the arena from `InferResponse::message` become invalid after the next call to `run_turn` (or any explicit `arena.reset()`).

---

## DType

Element types for tensor data used with the ONNX Runtime provider.

```cpp
enum class DType { Float32, Float16, Int8, Int32, Int64, UInt8 };
```

---

## TensorView and TensorData

```cpp
struct TensorView {                    // zero-copy, arena-backed
    std::string_view         name;
    DType                    dtype = DType::Float32;
    std::span<const int64_t> shape;
    std::span<const uint8_t> data;
};

struct TensorData {                    // owned, heap-allocated
    std::string          name;
    DType                dtype = DType::Float32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};
```

`TensorView` is the zero-copy projection used at inference time. `TensorData` is the owned storage you put in `Message::tensors` for history.

**Example — building a float32 tensor for ONNX inference:**

```cpp
agent::TensorData input;
input.name  = "pixel_values";
input.dtype = agent::DType::Float32;
input.shape = {1, 3, 224, 224};

size_t n = 1 * 3 * 224 * 224;
input.data.resize(n * sizeof(float), 0);
// fill input.data with your image pixels...

agent::Message msg;
msg.role = "user";
msg.tensors.push_back(std::move(input));
conv.history.push_back(std::move(msg));
```

---

## Callbacks

```cpp
using TokenStreamFn = void (*)(std::string_view token, void *user_data);
using LogFn         = void (*)(LogLevel level, std::string_view message, void *user_data);
using EventHookFn   = void (*)(const Event &event, void *user_data);
```

All callbacks are plain C function pointers. Capture state via `user_data`. Callbacks must not throw.
