# Flow (FSM Executor)

`#include <agent-cpp/agent.hpp>`

`Flow` provides structured multi-step agent flows: a finite state machine plus the conversational memory it drives, in one type. Instead of writing a single monolithic prompt loop, you decompose your agent into named states — each with its own system prompt, tool allowlist, and hooks — and declare transitions between them.

`Flow` replaces what used to be two separate types (`ConversationState` + `FSMExecutor`) — one type now owns both "the state machine" and "the memory it drives."

---

## Concepts

### Why an FSM?

A plain `run_turn` loop is flexible but unstructured. The FSM adds:

- **State isolation.** Each state gets its own system prompt and tool allowlist. The model cannot access tools it was not granted.
- **Explicit transitions.** `on_transition` returns the next state ID. The flow is auditable and testable.
- **Shared conversation.** The `Flow` owns a single `FlowMemory` that accumulates across all states. Later states have full context of earlier ones.
- **Composability.** Multiple FSM executors can run concurrently via `AgentScheduler`.

### State machine topology

```
             on_transition returns "execute"
 [plan] ----------------------------------------> [execute]
                                                       |
                                         on_transition returns "exit"
                                                       |
                                                    [done]
```

States are looked up by `state_id` string, so the topology can be non-linear (loops, branches, multiple entry points to the same state).

---

## FlowState

One node in the state machine.

```cpp
struct FlowState {
    std::string_view state_id;

    std::string_view system_prompt;

    std::span<const std::string_view> allowed_tools;

    std::span<const FlowHook> hooks;

    bool isolated_memory = false;

    using ContextProviderFn = std::string_view (*)(Session &session, FlowMemory &memory, void *context);
    ContextProviderFn context_provider = nullptr;

    using TransitionFn = Result<std::string_view>(*)
                         (Session &session, FlowMemory &memory, void *context);
    TransitionFn on_transition = nullptr;
};
```

### `state_id`

Unique identifier for this state. Used as the target of transitions from other states. The first state in `Flow::states` is the entry state.

### `system_prompt`

If non-empty, this string is injected as the system message (`history[0]`) for the **duration of `on_transition` only**. `step_flow` saves the original system message before calling `on_transition` and restores it afterward — no permanent mutation occurs.

If `system_prompt` is empty, the existing system message (if any) is left unchanged.

### `allowed_tools`

A span of tool name strings. If non-empty, only these tools are dispatchable via `execute_tool` while this state's `on_transition` is running — `session.config.tools` itself is never mutated (this is scoped through a thread-local allowlist internally, which is also what makes it safe for multiple `Flow`s to run concurrently under `AgentScheduler`). After `on_transition` returns, the previous state's allowlist (if any) is restored.

If empty, all session tools are available.

```cpp
static const std::string_view kReadOnlyTools[] = { "read_file", "list_dir", "file_info" };

agent::FlowState read_state{
    .state_id      = "read",
    .allowed_tools = kReadOnlyTools,
    .on_transition = read_fn,
};
```

### `hooks`

A span of `FlowHook` values that are active while this state's `on_transition` is running. See [FlowHook](#FlowHook) below.

### `isolated_memory`

By default (`false`), every state shares the same `Flow::memory` — the conversation accumulates across state transitions, so later states see everything earlier states said. Set `isolated_memory = true` to give a state its own isolated `FlowMemory` instead, stored in `Flow::isolated_memories[state_id]` and created empty the first time the state runs. Useful for a state that does throwaway/scratch work (e.g. a sub-task that shouldn't pollute the main conversation) or that intentionally starts with a clean slate each time it's entered.

```cpp
agent::FlowState scratch_state{
    .state_id        = "scratch_lookup",
    .isolated_memory = true,     // isolated from the main conversation
    .on_transition   = scratch_lookup_fn,
};
```

### `context_provider`

Optional. Called before `on_transition`, letting a state pull retrieved context (e.g. from a [`DataSource`](data-sources.md), classic RAG-style) and fold it into that state's effective system prompt for this turn only — without requiring the model to make an explicit tool call. Reuses the same save/restore mechanism `system_prompt` already uses, so nothing is permanently written into history.

```cpp
std::string_view inject_docs_context(agent::Session &session, agent::FlowMemory &, void *) {
    static thread_local std::string buf;
    auto res = agent::query_data_source(session, "product_docs", "installation steps");
    if (!res.ok) return {};
    buf = "Relevant docs:\n" + res.value;
    return buf;
}

agent::FlowState answer_state{
    .state_id         = "answer",
    .system_prompt    = "Answer using the provided docs context.",
    .context_provider = inject_docs_context,
    .on_transition    = answer_fn,
};
```

### `on_transition`

The required function that does the actual work for this state. It should:

1. Call `run_turn` (or `infer` directly) one or more times.
2. Inspect the result.
3. Return the `state_id` of the next state, or `"exit"` / `""` to finish the FSM.

```cpp
agent::Result<std::string_view> my_state_fn(
        agent::Session           &session,
        agent::FlowMemory &conv,
        void                     *context) {

    auto r = agent::run_turn(session, conv, "Analyse the data.");
    if (!r.ok)
        return agent::fail<std::string_view>(r.error.code, r.error.message);

    if (r.value.find("done") != std::string::npos)
        return agent::ok(std::string_view("exit"));

    return agent::ok(std::string_view("analyse_more"));
}
```

The `context` pointer is the value passed to `run_flow` / `step_flow`. Use it to share application state between states without globals.

---

## FlowHook

Callbacks fired before (`Pre`) or after (`Post`) each tool call while a specific state is active.

```cpp
struct FlowHook {
    enum class When { Pre, Post };
    When when = When::Pre;

    std::string_view tool_name;   // empty = fire for every tool; non-empty = specific tool only

    using HookFn = void (*)(Session          &session,
                            FlowMemory &memory,
                            std::string_view   tool_name,
                            std::string_view   args,      // tool input JSON
                            std::string_view   result,    // empty for Pre, populated for Post
                            void              *user_data);
    HookFn callback  = nullptr;
    void  *user_data = nullptr;
};
```

### Hook firing order

For each tool call inside `run_turn`:

1. All `FlowHook::Pre` hooks where `tool_name` matches (or is empty) fire in span order.
2. The tool callback executes.
3. All `FlowHook::Post` hooks where `tool_name` matches (or is empty) fire in span order.

Hooks are scoped to the current FSM state. When `on_transition` returns, hooks from the previous state are no longer active.

### Example — audit all tool calls

```cpp
static void log_tool_pre(agent::Session &, agent::FlowMemory &,
                         std::string_view name, std::string_view args,
                         std::string_view, void *) {
    printf("[TOOL PRE]  %s(%s)\n", std::string(name).c_str(), std::string(args).c_str());
}

static void log_tool_post(agent::Session &, agent::FlowMemory &,
                          std::string_view name, std::string_view,
                          std::string_view result, void *) {
    printf("[TOOL POST] %s -> %s\n", std::string(name).c_str(), std::string(result).c_str());
}

static const agent::FlowHook kAuditHooks[] = {
    { .when = agent::FlowHook::When::Pre,  .tool_name = "", .callback = log_tool_pre  },
    { .when = agent::FlowHook::When::Post, .tool_name = "", .callback = log_tool_post },
};
```

### Example — hook only a specific tool

```cpp
static void on_write_file(agent::Session &, agent::FlowMemory &,
                          std::string_view, std::string_view args,
                          std::string_view, void *) {
    printf("Agent is writing a file: %s\n", std::string(args).c_str());
}

static const agent::FlowHook kWriteHook[] = {
    { .when = agent::FlowHook::When::Pre, .tool_name = "write_file", .callback = on_write_file },
};
```

---

## Flow

Container for the state machine plus the memory it drives. You create one, populate `states` and `memory`, then pass it to `run_flow` or `step_flow`.

```cpp
struct Flow {
    std::string id;
    std::span<const FlowState> states;   // first = entry state

    FlowMemory memory;                                              // shared by default
    std::unordered_map<std::string, FlowMemory> isolated_memories;  // used by isolated_memory states

    // Internal -- managed by step_flow / run_flow. Do not set or read.
    size_t current_state_index = 0;
    bool   has_started         = false;
    bool   has_finished        = false;
};
```

Not safe to step concurrently on the same `Flow` instance — see [scheduler.md](scheduler.md).

### Setup

```cpp
std::vector<agent::FlowState> states = {
    { .state_id = "plan",    .system_prompt = "Make a plan.", .on_transition = plan_fn    },
    { .state_id = "execute", .system_prompt = "Execute it.",  .on_transition = execute_fn },
};

agent::Flow flow;
flow.states    = states;
flow.memory.id = "my_agent";

// Optional: pre-populate history
agent::Message seed;
seed.role    = "user";
seed.content = "Your task: write a sorting algorithm in C++.";
flow.memory.history.push_back(std::move(seed));
```

---

## run_flow

```cpp
Result<void> run_flow(Session &session, void *context, Flow &flow);
```

Runs the FSM from its current state to completion by calling `step_flow` in a loop. Blocks until the FSM finishes or an error occurs.

**Parameters:**

| Parameter | Description |
|---|---|
| `session` | Active session |
| `context` | Passed to every `on_transition` call; can be `nullptr` or a pointer to your application state |
| `flow` | The Flow; `states` and `memory` must be set before calling |

**Returns:** `Result<void>` — `ok()` if the FSM ran to completion, error if any `on_transition` returned an error or a state ID was not found.

**Example:**

```cpp
auto res = agent::run_flow(session, nullptr, flow);
if (!res.ok)
    fprintf(stderr, "FSM error: %s\n", res.error.message.c_str());

printf("Final history length: %zu\n", flow.memory.history.size());
```

---

## step_flow

```cpp
Result<bool> step_flow(Session &session, void *context, Flow &flow);
```

Advances the FSM exactly one state transition.

**Returns:** `Result<bool>`:
- `.ok = true, .value = true` — transition completed, FSM still running.
- `.ok = true, .value = false` — FSM finished (transition returned `"exit"` or `""`).
- `.ok = false` — error in `on_transition`, or unknown next state ID.

Use `step_flow` when you need to inspect or modify the conversation between state transitions, or when integrating the FSM into your own event loop.

**Example — manual step loop with inspection:**

```cpp
while (true) {
    auto step = agent::step_flow(session, &my_ctx, flow);
    if (!step.ok) {
        fprintf(stderr, "FSM error: %s\n", step.error.message.c_str());
        break;
    }
    if (!step.value) {
        printf("FSM finished.\n");
        break;
    }

    // Inspect after each state
    const auto &h = flow.memory.history;
    printf("State done. History has %zu messages.\n", h.size());

    // Optionally: clear KV cache between states
    agent::clear_provider_kv_cache(session);
}
```

---

## Complete example — ReAct agent

A ReAct (Reason + Act) agent pattern with three states: planning, acting, and summarising.

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <vector>

static const std::string_view kPlanTools[] = {};     // no tools during planning
static const std::string_view kActTools[]  = {       // only action tools
    "read_file", "write_file", "run_command"
};

agent::Result<std::string_view> plan_fn(
        agent::Session &s, agent::FlowMemory &c, void *) {
    auto r = agent::run_turn(s, c,
        "Think step by step about how to accomplish the goal. "
        "List the exact steps you will take.");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("act"));
}

agent::Result<std::string_view> act_fn(
        agent::Session &s, agent::FlowMemory &c, void *) {
    auto r = agent::run_turn(s, c,
        "Execute your plan now. Use the available tools.");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("summarise"));
}

agent::Result<std::string_view> summarise_fn(
        agent::Session &s, agent::FlowMemory &c, void *) {
    auto r = agent::run_turn(s, c,
        "Summarise what was accomplished and any issues encountered.");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    printf("Summary: %s\n", r.value.c_str());
    return agent::ok(std::string_view("exit"));
}

int main() {
    agent::AgentRuntime runtime;

    auto sess = agent::init({
        .provider       = agent::AiProvider::LlamaCpp,
        .model          = ".workspace/models/my.gguf",
        .workspace_dir  = ".workspace",
        .context_window = 8192,
        .max_tokens     = 1024,
    });
    if (!sess.ok) return 1;
    agent::Session session = std::move(sess.value);

    std::vector<agent::FlowState> states = {
        { .state_id = "plan",      .system_prompt = "You are a careful planner.",
          .allowed_tools = kPlanTools, .on_transition = plan_fn      },
        { .state_id = "act",       .system_prompt = "You are an executor. Use tools precisely.",
          .allowed_tools = kActTools,  .on_transition = act_fn       },
        { .state_id = "summarise", .system_prompt = "You are a concise reporter.",
          .on_transition = summarise_fn },
    };

    agent::Flow flow;
    flow.states    = states;
    flow.memory.id = "react_agent";

    agent::Message task;
    task.role    = "user";
    task.content = "Compute the first 10 Fibonacci numbers and save them to fibonacci.txt.";
    flow.memory.history.push_back(std::move(task));

    auto res = agent::run_flow(session, nullptr, flow);
    if (!res.ok) fprintf(stderr, "error: %s\n", res.error.message.c_str());

    return 0;
}
```
