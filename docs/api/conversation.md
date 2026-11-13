# Conversation

`#include <agent-cpp/agent.hpp>`

Inference has two layers:

- **`infer`** is stateless. It takes the messages you pass and returns one model reply. It does not keep history and does not update the session statistics.
- **`run_turn`** is the one you normally use. It keeps a `Conversation`, runs the tool-call loop, prunes and compresses history to fit the context window, and saves the history to the memory store.

---

## Conversation

A `Conversation` is one chat thread: its history, its statistics and the tools loaded for it. You own it and can fill its history before the first turn.

```cpp
struct Conversation {
    std::string                id;            // names the saved history: "history:<id>"
    std::vector<Message>       history;       // the messages, oldest first
    TurnStats                  stats;         // telemetry for this conversation (updated by run_turn)
    std::vector<std::string>   active_tools;  // tools loaded with load_tool (Config::lazy_tools); not part of history
};
```

Start a conversation with an id, and optionally a system message:

```cpp
agent::Conversation conv;
conv.id = "support-42";

agent::Message sys;
sys.role    = "system";
sys.content = "You are a helpful assistant with access to filesystem tools.";
conv.history.push_back(std::move(sys));
```

Use a different `id` for each thread. The id is required to save history (see [History persistence](#history-persistence)).

---

## Message

An owned message in `Conversation::history`.

```cpp
struct Message {
    std::string             role;           // "system", "user", "assistant" or "tool"
    std::string             content;
    std::vector<ToolCall>   tool_calls;     // set on an assistant message that calls tools
    std::string             tool_call_id;   // set on a tool message: the call it answers
    std::vector<TensorData> tensors;        // raw tensors (see core.md)
    std::vector<Media>      media;          // images, audio and other files (see core.md)
    mutable int             token_count = -1; // cached by the library; -1 = not counted yet
};
```

### Roles

| Role | Meaning |
|---|---|
| `system` | Instructions. Leading system messages are kept when history is pruned. |
| `user` | A user turn. `run_turn` adds it from `user_prompt`. |
| `assistant` | A model reply. `run_turn` adds it after each model call. |
| `tool` | The result of one tool call. Follows the assistant message that made the call. |

Tool messages always stay with the assistant message that requested them. Pruning and compression never separate the two.

---

## infer

Runs one model call on the messages you give it. It does not change a `Conversation`, does not run tools, and does not fire the `OnInferenceSubmit` and `OnInferenceComplete` events. Use it for one-off prompts where history does not matter.

```cpp
Result<InferResponse> infer(Session &session, const InferRequest &request);
```

### MessageView

Input to `infer`. It does not own its data, so the strings must outlive the call.

```cpp
struct MessageView {
    std::string_view              role;
    std::string_view              content;
    std::span<const ToolCallView> tool_calls;
    std::string_view              tool_call_id;
    std::span<const TensorView>   tensors;
    std::span<const MediaView>    media;
};
```

### InferRequest

```cpp
struct InferRequest {
    std::span<const MessageView> messages;
    int           max_tokens       = 0;      // 0 = Config::max_tokens
    float         temperature      = -1.0f;  // negative = Config::temperature
    bool          stream           = false;
    TokenStreamFn on_token         = nullptr;
    void         *token_user_data  = nullptr;
    ProgressFn    on_progress      = nullptr;
    void         *progress_user_data = nullptr;
    const volatile int32_t *cancel = nullptr; // non-zero stops the call with Cancelled
    uint32_t      task             = 0;      // one AGENT_CAP_* bit; 0 = Config::default_task, then automatic
    std::string_view params_json;            // extra engine parameters, a JSON object; empty = none
};
```

`task` and `params_json` are for engines that do more than text, such as `decide` and speech. See [engines.md](engines.md).

### InferResponse

```cpp
struct InferResponse {
    MessageView                 message;        // the reply
    TokenUsage                  usage;          // token counts and timings (core.md)
    std::span<const TensorView> output_tensors; // tensors the engine produced, if any
    std::shared_ptr<Arena>      storage;        // owns everything the views point to
};
```

The views point into `storage`, so a response stays valid after later calls for as long as you keep it.

### Example: one reply

```cpp
agent::MessageView msg;
msg.role    = "user";
msg.content = "Translate 'hello' to French.";

agent::InferRequest req;
req.messages = std::span<const agent::MessageView>(&msg, 1);

auto r = agent::infer(session, req);
if (r.ok) printf("%.*s\n", (int)r.value.message.content.size(), r.value.message.content.data());
```

### Example: streaming

```cpp
void print_token(std::string_view tok, void *) {
    fwrite(tok.data(), 1, tok.size(), stdout);
}

req.stream      = true;
req.on_token    = print_token;
auto r = agent::infer(session, req);
printf("\n");
```

### run_task

`infer` with `InferRequest::task` set to one `AGENT_CAP_*` bit, for engines that do more than text (see [session.md](session.md#one-shot-tasks) for the one-shot functions):

```cpp
Result<InferResponse> run_task(Session &session, uint32_t task, const InferRequest &request);
```

```cpp
#include <agent-cpp/engine_abi.h>   // AGENT_CAP_* values

auto r = agent::run_task(session, AGENT_CAP_TTS, req);   // same as setting req.task = AGENT_CAP_TTS and calling infer
```

`task` must be exactly one bit; `0` fails with `InvalidConfig`. The engine must declare the task, or the call fails with `UnsupportedProvider`.

---

## run_turn

The main high-level call. It adds the user message to the conversation, runs the model, runs any tools it asks for, and returns the final reply.

```cpp
Result<std::string> run_turn(Session &session, Conversation &memory, std::string_view user_prompt,
                             bool stream = false, TokenStreamFn on_token = nullptr, void *token_user_data = nullptr);

Result<std::string> run_turn_with_media(Session &session, Conversation &memory, std::string_view user_prompt,
                                        std::span<const Media> media, bool stream = false,
                                        TokenStreamFn on_token = nullptr, void *token_user_data = nullptr);
```

| Parameter | Description |
|---|---|
| `session` | The session to run on |
| `memory` | The conversation. Its history and stats are updated in place. |
| `user_prompt` | Added as a `user` message. Pass `""` to add no user message: useful after you have appended tool results yourself, or when the system prompt sets the task. |
| `media` | (`run_turn_with_media` only) Files attached to the user message, for engines that accept them. |
| `stream` | When true, tokens are passed to `on_token` as they arrive. |
| `on_token` | Called for each token when `stream` is true. |
| `token_user_data` | Passed to `on_token`. |

**Returns** the final assistant text as an owned `std::string`. It is valid after later calls.

### Example: a conversation with two turns

```cpp
agent::Conversation conv;
conv.id = "demo";

auto a = agent::run_turn(session, conv, "What files are in my workspace?");
auto b = agent::run_turn(session, conv, "Which of them is largest?");   // sees the first turn
if (!b.ok) fprintf(stderr, "turn failed: %s\n", b.error.message.c_str());
```

### What one turn does

1. Fires `OnTurnStart`.
2. Adds the user message if `user_prompt` or `media` is not empty.
3. Compresses the history if it is too large (see [Compression](#compression)).
4. Loop, once per model call:
   - Prunes the history to the context budget (see [Pruning](#pruning)).
   - Fires `OnInferenceSubmit`, calls the model, fires `OnInferenceComplete`.
   - Adds the assistant message and updates `Conversation::stats` and `SessionStats`.
   - If the model asked for tools, runs each one in order and adds a `tool` message per call, then loops. Otherwise the loop ends.
5. Saves the history (see [History persistence](#history-persistence)).
6. Returns the final assistant text.

### Tool calls inside a turn

For each tool call in an assistant message, in order:

1. `OnToolCall` fires with the tool name.
2. `ToolHook::Pre` callbacks run (set on a graph Agent node; see [graph.md](graph.md)).
3. The tool runs through the permission gate and the sandbox.
4. `ToolHook::Post` callbacks run with the result.
5. `OnToolResult` fires with the result text.
6. A `tool` message holding the result is added to the history.

The number of tool-call rounds is limited by `Config::max_tool_call_rounds` (default 10). When the model asks for a round beyond that limit, the turn fails with `ToolCallLimitExceeded`. The assistant message that asked is already in the history at that point.

### Cancellation

If `request_cancel` was called on the session, the turn fails with `Cancelled` before the next model call. The same happens for a model call that is running when the cancel arrives. See [session.md](session.md#cancellation).

### Errors

| Error | Meaning |
|---|---|
| `ToolCallLimitExceeded` | The model asked for more tool-call rounds than `max_tool_call_rounds` |
| `Cancelled` | `request_cancel` was called |
| `DecodeFailed` | The per-turn arena could not hold the request's message views |
| `PermissionDenied`, `SandboxViolation`, ... | Not returned from `run_turn` directly. A failed tool is reported to the model as a `tool` message, and the turn goes on. |
| Engine errors | Any error from the model call is returned unchanged, for example `EngineCrashed` |

A failed turn does not save its history. Messages added before the failure stay in `conv.history`.

---

## Compression

```cpp
Result<void> compress_context(Session &session, Conversation &memory, size_t keep_recent = 6);
```

Replaces the older messages with one summary, made by the model. The summary becomes a `system` message that starts with `[Compressed history]: `. The leading system message and the `keep_recent` most recent messages are kept as they are.

`run_turn` calls it by itself when the history is longer than six messages and its token count is above 70% of `context_window`. It also checks again during a long tool loop, using the prompt size the engine last reported. Call it yourself to shrink the history early:

```cpp
agent::compress_context(session, conv, 4);   // keep the 4 most recent messages
```

Behaviour:

- Does nothing when there is too little history to compress.
- Never separates a tool result from the assistant message that asked for it.
- If the summary call fails, it logs a warning and returns `ok`. The pruner then handles the overflow.
- On success it increases `stats.compressions` and fires `OnCompression`.

---

## Pruning

Before each model call, the history is cut to fit the context budget:

```
budget = max(200, context_window - max_tokens - 100 - tokens of the injected tool list)
```

The pruner removes the oldest messages after the leading system messages until the history fits:

- Leading system messages are never removed.
- An assistant message that made tool calls and the tool messages after it are removed together.
- At least one message after the leading system messages is always kept.

Each pass that removes something increases `stats.prune_cycles` and fires `OnPrune`. Token counts are cached per message (`Message::token_count`). When the provider cannot count tokens, the library estimates `(characters + 3) / 4`.

---

## History persistence

At the end of every successful `run_turn`, the conversation is saved to the memory store under two keys:

| Key | Contents |
|---|---|
| `history:<id>` | The messages as a JSON array |
| `history_stats:<id>` | `prompt_tokens`, `completion_tokens`, `tool_calls`, `compressions`, `prune_cycles`, `turns` |

Saving is best effort: if it fails, `run_turn` logs a warning and still returns its reply. A conversation with an empty `id` cannot be saved.

`run_turn` never reads these keys back. To resume a conversation after a restart, load it:

```cpp
agent::Conversation conv;
conv.id = "support-42";

auto loaded = agent::load_conversation(session, conv);   // fills conv.history and the saved stats
if (!loaded.ok) {
    // nothing saved yet, or the saved data could not be read: start fresh
}
```

`save_conversation` and `load_conversation` can also be called directly, to snapshot or restore a conversation outside `run_turn`:

```cpp
Result<void> save_conversation(Session &session, const Conversation &memory);
Result<void> load_conversation(Session &session, Conversation &memory);
```

What is kept: messages, including their tool calls, tool call ids and media. Not kept: tensors, `TurnStats::cached_tokens`, `TurnStats::last` and `active_tools`. `load_conversation` replaces `history` and sets the six saved statistics. The persistence store is covered in [memory.md](memory.md).

---

## Related pages

- [session.md](session.md): creating sessions, `Config`, cancellation
- [stats.md](stats.md): what the statistics count
- [events.md](events.md): the events fired during a turn
- [tools.md](tools.md): writing tools and the default tool set
- [memory.md](memory.md): the store behind history persistence
