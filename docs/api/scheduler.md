# Scheduler

`#include <agent-cpp/agent.hpp>`

`AgentScheduler` runs multiple `Flow`s. All agents share the same `Session`. How subagents advance depends on the provider:

- **`AiProvider::LlamaCpp`** — also dispatched across up to `max_parallel_subagents` threads, but the actual parallelism comes from **continuous batching** inside the provider, not from the threads themselves: one shared `llama_context` (created with `n_seq_max = max_parallel_subagents`) multiplexes every currently-active generation's next token into a single `llama_decode` call per round, tagged by `llama_seq_id`. A single `llama_context` can't run two `llama_decode` calls at once regardless of thread count, so the threads mostly block waiting on that shared batch rather than doing independent decode work — see [architecture.md](../architecture.md#continuous-batching) for the full design.
- **`Mock` / `OnnxRuntime`** — strictly single-flight, cooperative round-robin, unchanged.

---

## Overview

```
AgentScheduler
  +-- subagents:  [WorkerA, WorkerB, WorkerC, ...]

pump(session):
  Advance pending subagents one Flow step each:
       - LlamaCpp: up to max_parallel_subagents at once, on real threads, joined
         before pump() returns
       - otherwise: one at a time, sequentially (today's round-robin, unchanged)

run_until_done(session):
  while !all_done(): pump()
```

For providers other than `LlamaCpp`, each `pump()` call still gives every pending subagent exactly one `Flow` state transition — a subagent that blocks inside `on_transition` (e.g. doing a long local-model call) blocks the rest of that pump. For `LlamaCpp`, up to `max_parallel_subagents` subagents advance concurrently per pump, sharing one batched decode.

---

## Spawning subagents

```cpp
Result<void> AgentScheduler::spawn(std::string id, Flow flow, void *context = nullptr);
```

Registers a new `Flow` as a subagent. Each subagent has a unique string `id` used to identify its result.

**Parameters:**

| Parameter | Description |
|---|---|
| `id` | Unique identifier; returned in `SubAgentResult::id` |
| `flow` | Fully configured `Flow` (`states` + `memory` must be set); moved into the scheduler |
| `context` | Passed to every `step_flow` call for this Flow; can be `nullptr` |

**Returns:** `InvalidConfig` if a subagent with the same `id` was already registered.

**Example:**

```cpp
agent::AgentScheduler scheduler;

auto make_flow = [](std::string conv_id, std::string topic) {
    agent::Flow flow;
    flow.states    = my_states;
    flow.memory.id = conv_id;
    agent::Message seed;
    seed.role    = "user";
    seed.content = "Research: " + topic;
    flow.memory.history.push_back(std::move(seed));
    return flow;
};

scheduler.spawn("research_ai",      make_flow("conv_1", "artificial intelligence"));
scheduler.spawn("research_quantum",  make_flow("conv_2", "quantum computing"));
scheduler.spawn("research_biotech",  make_flow("conv_3", "biotechnology"));
```

---

## Running the scheduler

### pump()

```cpp
Result<void> AgentScheduler::pump(Session &session);
```

One tick of the scheduler. Advances each pending subagent by one FSM step.

Returns `ok()` unless the function itself fails internally (not if a subagent fails — those are recorded in their results and logged).

Use `pump()` when you need to interleave scheduler ticks with other application work:

```cpp
while (!scheduler.all_done()) {
    auto r = scheduler.pump(session);
    if (!r.ok) { fprintf(stderr, "scheduler error\n"); break; }

    // do other application work here
    check_user_input();
    update_ui();
}
```

### run_until_done()

```cpp
Result<void> AgentScheduler::run_until_done(Session &session);
```

Loops `pump()` until `all_done()` returns `true`.

```cpp
auto res = scheduler.run_until_done(session);
if (!res.ok)
    fprintf(stderr, "scheduler failed: %s\n", res.error.message.c_str());
```

### all_done()

```cpp
bool AgentScheduler::all_done() const;
```

Returns `true` when every spawned subagent has finished — either successfully or with an error.

### results()

```cpp
std::vector<SubAgentResult> AgentScheduler::results() const;
```

Returns a copy of all subagent result records. Records for still-running agents have `ok = false` and a zero `stats` — check results only after `all_done()` or `run_until_done()`.

```cpp
for (const auto &r : scheduler.results()) {
    printf("agent '%s': %s\n", r.id.c_str(), r.ok ? "OK" : r.error.message.c_str());
    printf("  turns=%d  tool_calls=%d  compressions=%d\n",
           r.stats.turns, r.stats.tool_calls, r.stats.compressions);
}
```

---

## SubAgentResult

```cpp
struct SubAgentResult {
    std::string id;
    bool        ok = false;
    Error       error;        // populated when ok=false
    TurnStats   stats;        // conversation stats from the Flow's conversation
};
```

---

## Complete example — parallel research agents

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <string>
#include <vector>

agent::Result<std::string_view> research_fn(
        agent::Session &s, agent::FlowMemory &c, void *ctx) {
    const char *topic = static_cast<const char *>(ctx);
    std::string prompt = std::string("Research ") + topic + ". Write a 3-paragraph summary.";
    auto r = agent::run_turn(s, c, prompt);
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("exit"));
}

int main() {
    agent::AgentRuntime runtime;

    auto sess = agent::init({
        .provider      = agent::AiProvider::LlamaCpp,
        .model         = ".workspace/models/my.gguf",
        .workspace_dir = ".workspace",
        .context_window = 4096,
        .max_tokens     = 512,
    });
    if (!sess.ok) return 1;
    agent::Session session = std::move(sess.value);

    static const agent::FlowState kResearch[] = {
        { .state_id = "research", .on_transition = research_fn },
    };

    agent::AgentScheduler scheduler;

    const char *topics[] = { "quantum computing", "machine learning", "climate science" };
    for (const char *topic : topics) {
        agent::Flow flow;
        flow.states    = kResearch;
        flow.memory.id = std::string("research_") + topic;

        scheduler.spawn(std::string("agent_") + topic, std::move(flow),
                        const_cast<char *>(topic));
    }

    auto res = scheduler.run_until_done(session);

    for (const auto &r : scheduler.results()) {
        printf("'%s': %s (turns=%d)\n",
               r.id.c_str(), r.ok ? "done" : r.error.message.c_str(), r.stats.turns);
    }

    return res.ok ? 0 : 1;
}
```
