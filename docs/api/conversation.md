# Conversation

`#include <agent-cpp/agent.hpp>`

agent.cpp separates inference into two levels: a **stateless layer** (`infer`) for raw LLM calls, and a **stateful layer** (`run_turn`) that manages history, tool calls, pruning, compression, and memory persistence.

---

## FlowMemory

Holds the full history and stats for one logical conversation thread. You own this object and can pre-populate it before the first turn.

```cpp
struct FlowMemory {
    std::string          id;       // used as memory key "history:<id>"
    std::vector<Message> history;  // ordered list of messages
    TurnStats             stats;    // per-conversation telemetry (auto-updated)
};
```

`save_flow_memory`/`load_flow_memory` persist/restore a `FlowMemory` (history + stats) via the memory backend — see [History persistence](#history-persistence) below.

Typical setup:

```cpp
agent::FlowMemory conv;
conv.id = "my_chat_1";

// Optional: inject a system message before the first turn
agent::Message sys;
sys.role    = "system";
sys.content = "You are a helpful assistant with access to filesystem tools.";
conv.history.push_back(std::move(sys));
```

The `id` is used as the key when persisting history to the memory backend (`"history:my_chat_1"`). Multiple conversations can coexist in the same session by using distinct IDs.

---

## Message

Owned, heap-allocated message stored in `FlowMemory::history`.

```cpp
struct Message {
    std::string             role;           // "system" | "user" | "assistant" | "tool"
    std::string             content;
    std::vector<ToolCall>   tool_calls;     // non-empty when role="assistant" with tool use
    std::string             tool_call_id;   // non-empty when role="tool"
    std::vector<TensorData> tensors;        // multimodal data (ONNX)
    mutable int             token_count = -1; // -1 = not yet counted; cached lazily
};
```

### Roles

| Role | Meaning |
|---|---|
| `"system"` | Persistent instruction; always preserved at `history[0]` during pruning |
| `"user"` | Human turn; added by `run_turn` |
| `"assistant"` | Model reply; added after each `infer` call |
| `"tool"` | Tool result; added after each tool execution; paired with an `assistant` message |

---

## MessageView

Zero-copy view into arena memory. Valid only until the next `session.arena.reset()`.

```cpp
struct MessageView {
    std::string_view              role;
    std::string_view              content;
    std::span<const ToolCallView> tool_calls;
    std::string_view              tool_call_id;
    std::span<const TensorView>   tensors;
};
```

`MessageView` is used only in `InferRequest` for the low-level `infer`. You never construct or store these manually when using the high-level API.

---

## infer

Stateless, single-shot inference. Takes a span of `MessageView`s, returns an `InferResponse`. Does not touch `FlowMemory`.

```cpp
Result<InferResponse> infer(Session &session, const InferRequest &request);
```

### InferRequest

```cpp
struct InferRequest {
    std::span<const MessageView> messages;
    int           max_tokens      = 0;       // 0 -> Config::max_tokens
    float         temperature     = -1.0f;   // <0 -> Config::temperature
    bool          stream          = false;
    TokenStreamFn on_token        = nullptr;
    void         *token_user_data = nullptr;
};
```

### InferResponse

```cpp
struct InferResponse {
    MessageView                 message;
    TokenUsage                  usage;
    std::span<const TensorView> output_tensors;   // ONNX only
};
```

Response views are arena-backed. They become invalid after the next call to any function that calls `session.arena.reset()`, including subsequent `run_turn` calls.

**Example — direct low-level inference:**

```cpp
agent::MessageView msg;
msg.role    = "user";
msg.content = "Translate 'hello' to French.";

agent::InferRequest req;
req.messages = std::span<const agent::MessageView>(&msg, 1);

auto r = agent::infer(session, req);
if (r.ok)
    printf("%s\n", std::string(r.value.message.content).c_str());
```

**Example — streaming:**

```cpp
req.stream          = true;
req.on_token        = [](std::string_view tok, void *) {
    fwrite(tok.data(), 1, tok.size(), stdout);
};
agent::infer(session, req);
printf("\n");
```

---

## run_turn

The main high-level API. Manages history, tool call loops, pruning, compression, and memory persistence.

```cpp
Result<std::string> run_turn(
    Session           &session,
    FlowMemory        &memory,
    std::string_view   user_prompt,
    bool               stream          = false,
    TokenStreamFn      on_token        = nullptr,
    void              *token_user_data = nullptr);
```

**Parameters:**

| Parameter | Description |
|---|---|
| `session` | The active session |
| `memory` | Conversation history + stats; modified in place |
| `user_prompt` | User input; appended as a `"user"` message. Pass `""` to run without a user message (useful in FSM states where the system prompt sets the context) |
| `stream` | If `true`, tokens are delivered via `on_token` as they arrive |
| `on_token` | Streaming callback; called once per token when `stream = true` |
| `token_user_data` | Passed through to `on_token` |

**Returns:** `Result<std::string>` containing the assistant's final reply text (owned, always valid regardless of arena state). On error, `.ok` is `false` and `.error` describes what went wrong.

### Internal sequence

1. If `user_prompt` is non-empty, push a `"user"` message to `memory.history`.
2. Count total tokens. If > 70% of `context_window` and history is long enough, call `compress_context`.
3. Loop:
   a. `prune_history_state` — drop oldest non-system messages to fit the context budget.
   b. `arena.reset()`.
   c. Build `MessageView[]` from `memory.history` (zero-copy projections).
   d. `infer` — submit to provider.
   e. Append assistant `Message` to history.
   f. If the response contains tool calls, dispatch each one (see below), append results, continue loop.
   g. If no tool calls, exit loop.
4. `save_flow_memory` — serialise `memory.history` (and `memory.stats`) to JSON and store under `"history:<memory.id>"` / `"history_stats:<memory.id>"`. This is a write-only step — `run_turn` never reads history back; call `load_flow_memory` explicitly to resume a conversation after a process restart (see [History persistence](#history-persistence)).
5. Return assistant text as owned `std::string`.

### Tool call loop

Each tool call in the assistant's response is processed in order:

- `trigger_event(OnToolCall)` fires with the tool name as payload.
- All `FlowHook::Pre` callbacks for this tool fire.
- `execute_tool` dispatches the call via `Session::tool_by_name` (O(1)).
- All `FlowHook::Post` callbacks fire with the result.
- `trigger_event(OnToolResult)` fires with the result as payload.
- A `"tool"` message is appended to history.

If the number of tool-call rounds reaches `Config::max_tool_call_rounds` (default 10), the loop halts and `ToolCallLimitExceeded` is returned.

### Error conditions

| Error | Meaning |
|---|---|
| `ToolCallLimitExceeded` | Model called tools more than `max_tool_call_rounds` times in one turn |
| `DecodeFailed` | Arena capacity exhausted building `MessageView` array (increase `arena_capacity`) |
| Provider errors | Any error from `infer` is propagated unchanged |

---

## compress_context

```cpp
Result<void> compress_context(
    Session           &session,
    FlowMemory        &memory,
    size_t             keep_recent = 6);
```

Summarises the oldest messages using the active LLM and replaces them with a single `[Compressed history]: <summary>` message, preserving the system message (if any) and the most recent `keep_recent` messages verbatim.

Called automatically by `run_turn` when total tokens exceed 70% of `context_window`. You can also call it manually to trigger compression early.

**Behaviour:**

1. If `history.size() <= start + keep_recent + 2`, returns `ok()` immediately (nothing to compress).
2. Builds a summary prompt from messages `[start ... size - keep_recent)`.
3. Calls `infer` with a short summarization instruction. Uses stack-local `MessageView`s — does **not** touch the arena.
4. On success, erases old messages and inserts the summary.
5. On failure (network error, provider error), logs a warning and returns `ok()` — the pruner handles overflow as a fallback.

**Example:**

```cpp
// Force compression before a new topic begins
agent::compress_context(session, conv, 4);   // keep 4 most recent messages
```

---

## History persistence

After every successful `run_turn`, the conversation history and stats are serialised to JSON and stored in the memory backend via `save_flow_memory`, under `"history:<conv.id>"` (history) and `"history_stats:<conv.id>"` (`FlowMemory::stats`). This is best-effort — if the memory write fails, the function does not propagate the error (the in-memory history is still correct).

`run_turn` only ever *writes* history — it never reads it back. To resume a conversation after a process restart, load it explicitly:

```cpp
Result<void> save_flow_memory(Session &session, const FlowMemory &memory);
Result<void> load_flow_memory(Session &session, FlowMemory &memory);
```

```cpp
agent::FlowMemory conv;
conv.id = "my_chat_1";
auto load_res = agent::load_flow_memory(session, conv);   // fills conv.history + conv.stats
if (!load_res.ok) {
    // no saved history yet (or it failed to parse) -- start fresh
}
```

`save_flow_memory`/`load_flow_memory` can also be called directly to snapshot or restore a `FlowMemory` outside the normal `run_turn` flow. You can still read the raw stored JSON manually if needed:

```cpp
auto r = agent::retrieve_memory(session, "history:my_chat_1");
if (r.ok) printf("stored: %s\n", r.value.c_str());
```

---

## Context pruning details

Pruning enforces the budget: `context_window - max_tokens - 100` tokens.

The algorithm is O(n) — one forward pass to compute token totals, followed by a single `vector::erase` call:

1. Always preserves `history[0]` if it is a system message.
2. Drops messages starting from the oldest non-system message.
3. Never breaks a tool call pair — an `assistant` message with `tool_calls` and its immediately following `tool` messages are always dropped atomically.
4. Stops if fewer than `min_keep = fixed_start + 1` messages would remain.

Token counts are cached per `Message::token_count`. The first time a message is counted, the result is cached (mutable field). Subsequent prune cycles use the cached value at no cost. If the provider does not implement `count_tokens`, the fallback is `(chars + 3) / 4`.
