# Statistics

`#include <agent-cpp/agent.hpp>`

agent.cpp tracks token usage, tool calls, context management operations, and turn counts at two granularities: per-conversation and session-wide.

---

## TurnStats

Tracks usage for a single `Conversation`. Updated automatically by `run_turn`.

```cpp
struct TurnStats {
    int prompt_tokens     = 0;   // total prompt tokens submitted across all turns
    int completion_tokens = 0;   // total completion tokens received across all turns
    int tool_calls        = 0;   // total tool calls dispatched
    int compressions      = 0;   // number of times compress_context succeeded
    int prune_cycles      = 0;   // number of sliding-window prune operations
    int turns             = 0;   // number of infer calls (including tool-call loops)
};
```

Access via `Conversation::stats`:

```cpp
agent::Conversation conv;
conv.id = "demo";

agent::run_turn(session, conv, "Hello");
agent::run_turn(session, conv, "What files are in my workspace?");

const auto &s = conv.stats;
printf("prompt tokens:  %d\n", s.prompt_tokens);
printf("tool calls:     %d\n", s.tool_calls);
printf("turns:          %d\n", s.turns);
```

### What counts as a "turn"

Each call to `infer` inside `run_turn` increments `turns` by 1. In a single `run_turn` call, if the model makes 3 sequential tool calls before giving a final answer, `turns` increments by 4 (1 initial + 3 tool-loop iterations).

---

## SessionStats

Session-wide totals accumulated across all conversations driven through the session. Updated after each `infer` call inside `run_turn`.

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

## get_stats

```cpp
SessionStats get_stats(const Session &session);
```

Returns a copy of the current session-wide stats. Thread-safe to read; do not call concurrently with `run_turn` on the same session.

```cpp
auto stats = agent::get_stats(session);
printf("session totals:\n");
printf("  prompt tokens:     %d\n", stats.total_prompt_tokens);
printf("  completion tokens: %d\n", stats.total_completion_tokens);
printf("  tool calls:        %d\n", stats.total_tool_calls);
printf("  turns:             %d\n", stats.total_turns);
printf("  compressions:      %d\n", stats.total_compressions);
printf("  prune cycles:      %d\n", stats.total_prune_cycles);
```

---

## reset_stats

```cpp
void reset_stats(Session &session);
```

Zeroes all fields in `session.stats`. Does not affect individual `Conversation::stats`.

Use this to measure stats for a specific time window:

```cpp
agent::reset_stats(session);

// run some conversations...
agent::run_turn(session, conv_a, "task A");
agent::run_turn(session, conv_b, "task B");

auto delta = agent::get_stats(session);
printf("task A+B used %d tokens\n",
       delta.total_prompt_tokens + delta.total_completion_tokens);
```

---

## Cost estimation

Use token counts to estimate API costs. For example, with per-token prices (approximate):

```cpp
auto st = agent::get_stats(session);
double input_cost  = st.total_prompt_tokens     / 1e6 * 2.50;   // gpt-4o input  ~$2.50/M tokens
double output_cost = st.total_completion_tokens / 1e6 * 10.0;   // gpt-4o output ~$10/M tokens
printf("estimated cost: $%.4f\n", input_cost + output_cost);
```

---

## Graph Agent node stats

After `run_graph`, each Agent conversation in `run.state.histories` (keyed by node name, or by `history_key`) carries the `TurnStats` of its turns in this call (per-node stats are not checkpointed). For parallel subagents, that is one entry per branch:

```cpp
for (const auto &[name, memory] : run.state.histories) {
    printf("agent '%s':\n", name.c_str());
    printf("  turns:             %d\n", memory.stats.turns);
    printf("  tool calls:        %d\n", memory.stats.tool_calls);
    printf("  compressions:      %d\n", memory.stats.compressions);
    printf("  prompt tokens:     %d\n", memory.stats.prompt_tokens);
    printf("  completion tokens: %d\n", memory.stats.completion_tokens);
}
```

---

## Observability with event hooks

For real-time per-turn stats, combine stats reading with event hooks:

```cpp
agent::register_event_hook(session, agent::EventType::OnInferenceComplete,
    [](const agent::Event &, void *ud) {
        auto &session = *static_cast<agent::Session*>(ud);
        auto s = agent::get_stats(session);
        printf("[inference complete] total turns so far: %d\n", s.total_turns);
    },
    &session);
```
