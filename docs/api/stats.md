# Statistics

`#include <agent-cpp/agent.hpp>`

The library counts tokens, turns, tool calls and context operations at two levels: each `Conversation` has `TurnStats`, and each `Session` has `SessionStats`. `run_turn` updates both. Direct calls to `infer` update neither.

---

## TurnStats

Statistics for one `Conversation`, in `Conversation::stats`.

```cpp
struct TurnStats {
    int        prompt_tokens     = 0;  // prompt tokens sent, summed over all model calls
    int        completion_tokens = 0;  // tokens generated, summed over all model calls
    int        tool_calls        = 0;  // tool calls run
    int        compressions      = 0;  // successful compress_context calls
    int        prune_cycles      = 0;  // prune passes that removed messages
    int        turns             = 0;  // model calls (including the calls that follow tool results)
    int        cached_tokens     = 0;  // prompt tokens the engine reused from its KV cache, summed
    TokenUsage last;                   // usage of the most recent model call (core.md)
};
```

### What counts as a turn

Each model call counts as one turn. A `run_turn` in which the model makes three tool calls and then answers makes four model calls, so `turns` goes up by 4.

### Example

```cpp
agent::Conversation conv;
conv.id = "demo";

agent::run_turn(session, conv, "Hello");
agent::run_turn(session, conv, "What files are in my workspace?");

const auto &s = conv.stats;
printf("prompt tokens:  %d\n", s.prompt_tokens);
printf("cached tokens:  %d\n", s.cached_tokens);
printf("tool calls:     %d\n", s.tool_calls);
printf("turns:          %d\n", s.turns);
printf("last call:      prompt %.1f ms, generation %.1f ms\n",
       s.last.prefill_ms, s.last.decode_ms);
```

`TurnStats` is saved with the history and restored by `load_conversation`, except for `cached_tokens` and `last` (see [conversation.md](conversation.md#history-persistence)).

---

## SessionStats

Totals for every conversation on a session, in `Session::stats`.

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

| Field | Updated by |
|---|---|
| `total_prompt_tokens`, `total_completion_tokens`, `total_turns` | Each model call in `run_turn` |
| `total_tool_calls` | Each tool call in `run_turn` |
| `total_compressions` | Each successful `compress_context` |
| `total_prune_cycles` | Times old messages were dropped from a history to fit the context window, across all conversations. |

`cached_tokens` is kept per conversation only; there is no session-wide total.

---

## get_stats

```cpp
SessionStats get_stats(const Session &session);
```

Returns a copy of the session totals, read under the session's stats lock.

```cpp
auto stats = agent::get_stats(session);
printf("session totals:\n");
printf("  prompt tokens:     %d\n", stats.total_prompt_tokens);
printf("  completion tokens: %d\n", stats.total_completion_tokens);
printf("  tool calls:        %d\n", stats.total_tool_calls);
printf("  turns:             %d\n", stats.total_turns);
printf("  compressions:      %d\n", stats.total_compressions);
```

---

## reset_stats

```cpp
void reset_stats(Session &session);
```

Sets all `SessionStats` fields to 0. It does not change any `Conversation::stats`.

To measure one period, reset first and read afterwards:

```cpp
agent::reset_stats(session);

agent::run_turn(session, conv_a, "task A");
agent::run_turn(session, conv_b, "task B");

auto delta = agent::get_stats(session);
printf("tasks A and B used %d tokens\n",
       delta.total_prompt_tokens + delta.total_completion_tokens);
```

---

## Estimating cost

The library counts tokens, not money. Multiply by your provider's prices yourself. The prices below are placeholders:

```cpp
const double price_in_per_million  = 0.0;   // your input price, per million tokens
const double price_out_per_million = 0.0;   // your output price, per million tokens

auto st = agent::get_stats(session);
double cost = st.total_prompt_tokens     / 1e6 * price_in_per_million
            + st.total_completion_tokens / 1e6 * price_out_per_million;
printf("estimated cost: %.4f\n", cost);
```

For a local engine the cost is zero, but the token counts still show how much work a workload does.

---

## Graph Agent node statistics

Graph builds only (`AGENT_WITH_GRAPH`, on by default).

After `run_graph`, each Agent conversation in `run.state.histories` holds the `TurnStats` for the turns of that call. Keys are the node name, or `GraphNode::history_key` when set. Parallel branches each have their own entry. Per-node statistics are not saved in checkpoints.

```cpp
for (const auto &[name, memory] : run.state.histories) {
    printf("agent '%s': turns=%d tool_calls=%d prompt=%d completion=%d\n",
           name.c_str(), memory.stats.turns, memory.stats.tool_calls,
           memory.stats.prompt_tokens, memory.stats.completion_tokens);
}
```

See [graph.md](graph.md) for the graph runtime.

---

## Live numbers during a turn

Read the session totals from an event hook to see progress while a turn is still running:

```cpp
agent::register_event_hook(session, agent::EventType::OnInferenceComplete,
    [](const agent::Event &, void *ud) {
        auto &s = *static_cast<agent::Session *>(ud);
        printf("[model call done] session turns so far: %d\n", agent::get_stats(s).total_turns);
    },
    &session);
```

Hooks are described in [events.md](events.md).

---

## Related pages

- [conversation.md](conversation.md): where `run_turn` updates these counters
- [session.md](session.md): sessions and `reset_stats` in context
- [events.md](events.md): hooks for live values
