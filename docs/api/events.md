# Events

`#include <agent-cpp/agent.hpp>`

Event hooks let you observe what a session does (turns, model calls, tool calls, compression, pruning, graph steps) without changing the code that does it. Register a C function pointer on a session; it is called on the thread that triggered the event.

For per-tool auditing inside one graph node, use `ToolHook` instead. See [Event hooks or tool hooks](#event-hooks-or-tool-hooks) below.

---

## EventType

```cpp
enum class EventType {
    OnTurnStart,           // run_turn / run_turn_with_media began
    OnInferenceSubmit,     // a model call in run_turn is about to start
    OnInferenceComplete,   // a model call in run_turn returned successfully
    OnToolCall,            // a tool is about to run (run_turn or a graph Tool / Agent node)
    OnToolResult,          // a tool returned its result
    OnStateTransition,     // a graph node is starting
    OnCompression,         // compress_context replaced part of the history
    OnPrune,               // the pruner removed messages from the history
    OnSubAgentComplete,    // a graph Parallel branch finished (after the join)
};
```

| EventType | Fired | Payload |
|---|---|---|
| `OnTurnStart` | At the start of each `run_turn` | empty |
| `OnInferenceSubmit` | Before each model call made by `run_turn` | empty |
| `OnInferenceComplete` | After each successful model call made by `run_turn` | empty |
| `OnToolCall` | Before each tool call | the tool name (UTF-8) |
| `OnToolResult` | After each tool call | the tool's result text (UTF-8) |
| `OnStateTransition` | When a graph node starts (graph builds, `AGENT_WITH_GRAPH`) | the node name (UTF-8) |
| `OnCompression` | After `compress_context` succeeds | empty |
| `OnPrune` | After the pruner removes messages | empty |
| `OnSubAgentComplete` | Once per finished branch of a graph `Parallel` node (graph builds) | the branch's entry node name (UTF-8) |

`OnInferenceSubmit` and `OnInferenceComplete` fire only around the model calls that `run_turn` makes. A direct call to `infer` fires neither.

---

## Event

```cpp
struct Event {
    EventType                type;
    std::span<const uint8_t> payload;   // valid only during the callback
};
```

---

## EventHook

```cpp
using EventHookFn = void (*)(const Event &event, void *user_data);

struct EventHook {
    EventType   type;
    EventHookFn callback  = nullptr;
    void       *user_data = nullptr;
};
```

`EventHook` is the record the session stores. You do not create one yourself; use `register_event_hook`.

---

## register_event_hook

```cpp
void register_event_hook(Session &session, EventType type, EventHookFn callback, void *user_data = nullptr);
```

Adds a hook for one event type. Hooks for the same type are called in the order they were added. They stay for the life of the session; there is no way to remove one.

A hook is called with `(event, user_data)`. `user_data` is the pointer you gave, so it must stay valid for as long as the session can fire events.

---

## trigger_event

```cpp
void trigger_event(Session &session, EventType type, std::span<const uint8_t> payload);
```

Calls every hook registered for `type` with `payload`. The library calls it itself; it is public so that custom code and tests can fire events too.

---

## Rules for callbacks

- Do not throw. Hooks are C function pointers.
- Keep them fast. They run inside the turn, before the next model call.
- Hooks registered while an event is firing take effect from the next event.
- Inside a graph `Parallel` node, branches run on worker threads, so events can come from several threads at once. Make the hook and its `user_data` thread-safe.

---

## Examples

### Log the turn and model-call boundaries

```cpp
agent::register_event_hook(session, agent::EventType::OnTurnStart,
    [](const agent::Event &, void *) { printf("-- turn start --\n"); });

agent::register_event_hook(session, agent::EventType::OnInferenceSubmit,
    [](const agent::Event &, void *) { printf("  [model] calling...\n"); });

agent::register_event_hook(session, agent::EventType::OnInferenceComplete,
    [](const agent::Event &, void *) { printf("  [model] done\n"); });
```

### Log every tool call and its result

```cpp
agent::register_event_hook(session, agent::EventType::OnToolCall,
    [](const agent::Event &ev, void *) {
        printf("  [tool] %.*s\n", (int)ev.payload.size(), (const char *)ev.payload.data());
    });

agent::register_event_hook(session, agent::EventType::OnToolResult,
    [](const agent::Event &ev, void *) {
        auto text = std::string_view(reinterpret_cast<const char *>(ev.payload.data()), ev.payload.size());
        printf("  [result] %.*s\n", (int)std::min<size_t>(text.size(), 80), text.data());
    });
```

### Count model calls with user data

The hook receives the pointer you pass to `register_event_hook`, so state can live in your own struct:

```cpp
struct Counter {
    int submitted = 0;
    int completed = 0;
};
Counter counter;

agent::register_event_hook(session, agent::EventType::OnInferenceSubmit,
    [](const agent::Event &, void *ud) { static_cast<Counter *>(ud)->submitted++; }, &counter);

agent::register_event_hook(session, agent::EventType::OnInferenceComplete,
    [](const agent::Event &, void *ud) { static_cast<Counter *>(ud)->completed++; }, &counter);
```

For token counts, read `Conversation::stats` or `get_stats` (see [stats.md](stats.md)) rather than the payload, which is empty for these events.

### Track graph node transitions

```cpp
agent::register_event_hook(session, agent::EventType::OnStateTransition,
    [](const agent::Event &ev, void *) {
        printf("graph entered node: %.*s\n", (int)ev.payload.size(), (const char *)ev.payload.data());
    });
```

For a full record of a graph run (node starts and ends, state writes, tool calls, checkpoints), use `GraphRun::events` instead. See [graph.md](graph.md#event-log).

### Observe context management

```cpp
agent::register_event_hook(session, agent::EventType::OnPrune,
    [](const agent::Event &, void *) { printf("context pruned\n"); });

agent::register_event_hook(session, agent::EventType::OnCompression,
    [](const agent::Event &, void *) { printf("context compressed\n"); });
```

### Detect a finished parallel branch

```cpp
agent::register_event_hook(session, agent::EventType::OnSubAgentComplete,
    [](const agent::Event &ev, void *) {
        printf("branch '%.*s' completed\n", (int)ev.payload.size(), (const char *)ev.payload.data());
    });
```

---

## Event hooks or tool hooks

`EventHook` (this page) and `ToolHook` (set on a graph Agent node through `GraphNode::hooks`) serve different purposes:

| | `EventHook` | `ToolHook` |
|---|---|---|
| Scope | Whole session; every conversation and every graph node | Only while its Agent node runs |
| Tools | `OnToolCall` / `OnToolResult` fire for every tool | Can be limited to one `tool_name` |
| Tool result | Payload bytes (`OnToolResult`) | `result` string in `Post`, plus the node's `Conversation` |
| Typical use | Telemetry, logs, dashboards | Auditing one part of a graph |

A typical pairing: an event hook for session-wide timing, and a tool hook on the one node that must log a specific tool:

```cpp
// Session-wide: time each model call
agent::register_event_hook(session, agent::EventType::OnInferenceSubmit, start_timer, &timer);
agent::register_event_hook(session, agent::EventType::OnInferenceComplete, stop_timer, &timer);

// Node-scoped: warn before an agent deletes a path
worker_node.hooks = {
    { .when      = agent::ToolHook::When::Pre,
      .tool_name = "delete_path",
      .callback  = [](agent::Session &, agent::Conversation &, std::string_view, std::string_view args,
                      std::string_view, void *) {
          printf("[warn] deleting: %.*s\n", (int)args.size(), args.data());
      } },
};
```

Tool hooks are described in [graph.md](graph.md#tool-hooks).

---

## Related pages

- [conversation.md](conversation.md): the turn that fires most of these events
- [stats.md](stats.md): counts that do not need hooks
- [graph.md](graph.md): graph nodes, `GraphRun::events` and tool hooks
