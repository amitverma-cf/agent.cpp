# Events

`#include <agent-cpp/agent.hpp>`

The event hook bus provides a decoupled way to observe lifecycle events in agent.cpp without modifying inference logic. Events fire synchronously within the calling thread.

---

## EventType

```cpp
enum class EventType {
    OnTurnStart,           // fired at the start of run_turn
    OnInferenceSubmit,     // fired immediately before infer is called
    OnInferenceComplete,   // fired immediately after infer returns
    OnToolCall,            // fired when a tool is about to be dispatched
    OnToolResult,          // fired after a tool returns its result
    OnStateTransition,     // fired when a new FSM state is entered
    OnCompression,         // fired after context compression completes
    OnPrune,               // fired after a sliding-window prune operation
    OnCronFire,            // fired when a cron task fires
    OnSubAgentComplete,    // fired when a subagent FSM finishes
};
```

---

## Event

The event object passed to every hook callback.

```cpp
struct Event {
    EventType                type;
    std::span<const uint8_t> payload;   // meaning depends on type (see table below)
};
```

### Payload meanings

| EventType | Payload |
|---|---|
| `OnTurnStart` | empty |
| `OnInferenceSubmit` | empty |
| `OnInferenceComplete` | empty |
| `OnToolCall` | UTF-8 bytes of the tool name |
| `OnToolResult` | UTF-8 bytes of the tool result string |
| `OnStateTransition` | UTF-8 bytes of the new state_id |
| `OnCompression` | empty |
| `OnPrune` | empty |
| `OnCronFire` | empty |
| `OnSubAgentComplete` | empty |

---

## EventHook

```cpp
using EventHookFn = void (*)(const Event &event, void *user_data);

struct EventHook {
    EventType   type;
    EventHookFn callback;
    void       *user_data;
};
```

---

## register_event_hook

```cpp
void register_event_hook(
    Session    &session,
    EventType   type,
    EventHookFn callback,
    void       *user_data = nullptr);
```

Registers a callback for a specific event type. Multiple callbacks can be registered for the same type — they fire in registration order. Hooks are stored in `session.event_hooks` and fire for the lifetime of the session.

**Callbacks must not throw** and must not modify the `session.event_hooks` vector (doing so invalidates the iteration).

---

## trigger_event

```cpp
void trigger_event(
    Session                  &session,
    EventType                 type,
    std::span<const uint8_t>  payload);
```

Fires all hooks registered for `type` with the given payload. Called internally by the library; exposed publicly for use in tests or custom extensions.

---

## Examples

### Log all inference boundaries

```cpp
agent::register_event_hook(session, agent::EventType::OnTurnStart,
    [](const agent::Event &, void *) {
        printf("-- turn start --\n");
    });

agent::register_event_hook(session, agent::EventType::OnInferenceSubmit,
    [](const agent::Event &, void *) {
        printf("  [inference] submitting...\n");
    });

agent::register_event_hook(session, agent::EventType::OnInferenceComplete,
    [](const agent::Event &, void *) {
        printf("  [inference] done.\n");
    });
```

### Log every tool call

```cpp
agent::register_event_hook(session, agent::EventType::OnToolCall,
    [](const agent::Event &ev, void *) {
        std::string name(reinterpret_cast<const char*>(ev.payload.data()), ev.payload.size());
        printf("  [tool] called: %s\n", name.c_str());
    });

agent::register_event_hook(session, agent::EventType::OnToolResult,
    [](const agent::Event &ev, void *) {
        std::string result(reinterpret_cast<const char*>(ev.payload.data()), ev.payload.size());
        printf("  [tool] result: %.80s...\n", result.c_str());
    });
```

### Track FSM state transitions

```cpp
agent::register_event_hook(session, agent::EventType::OnStateTransition,
    [](const agent::Event &ev, void *) {
        std::string state(reinterpret_cast<const char*>(ev.payload.data()), ev.payload.size());
        printf("FSM entered state: %s\n", state.c_str());
    });
```

### Measure token usage with a custom accumulator

```cpp
struct TokenStats { int prompt = 0; int completion = 0; };
TokenStats my_stats;

agent::register_event_hook(session, agent::EventType::OnInferenceComplete,
    [](const agent::Event &, void *ud) {
        auto *st = static_cast<TokenStats*>(ud);
        // get_stats gives session-wide totals; subtract to get this turn's delta
        // (or use FlowMemory::stats directly after run_turn)
    },
    &my_stats);
```

### Detect when a subagent finishes

```cpp
agent::register_event_hook(session, agent::EventType::OnSubAgentComplete,
    [](const agent::Event &, void *) {
        printf("A subagent just completed.\n");
    });
```

### Observe context management

```cpp
agent::register_event_hook(session, agent::EventType::OnPrune,
    [](const agent::Event &, void *) {
        printf("Context pruned.\n");
    });

agent::register_event_hook(session, agent::EventType::OnCompression,
    [](const agent::Event &, void *) {
        printf("Context compressed.\n");
    });
```

---

## Combining with FlowHooks

`EventHook` and `FlowHook` serve different purposes:

| | `EventHook` | `FlowHook` |
|---|---|---|
| Scope | Session-wide; fires for all conversations | FSM state-specific; fires only while that state's `on_transition` runs |
| Tool visibility | `OnToolCall` fires for all tools | Can be narrowed to a specific `tool_name` |
| Result access | `OnToolResult` provides result bytes | `FlowHook::Post` receives `std::string_view result` |
| Use case | Telemetry, logging, dashboards | State-scoped auditing, result post-processing, abort logic |

Use both together for comprehensive observability:

```cpp
// Session-wide latency measurement
agent::register_event_hook(session, agent::EventType::OnInferenceSubmit, start_timer, &timer);
agent::register_event_hook(session, agent::EventType::OnInferenceComplete, stop_timer, &timer);

// Per-state tool restriction enforcement (via FlowHook, not EventHook)
static const agent::FlowHook kStateHooks[] = {
    { .when = agent::FlowHook::When::Pre,
      .tool_name = "delete_path",
      .callback = [](auto &, auto &, auto, auto args, auto, void *) {
          printf("[WARN] agent is about to delete: %s\n", std::string(args).c_str());
      }
    },
};
```
