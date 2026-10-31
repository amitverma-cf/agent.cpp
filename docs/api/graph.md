# Graph runtime

`#include <agent-cpp/agent.hpp>`

The graph runtime runs workflows built from named nodes and edges, in the style of LangGraph 1.0 and Google ADK workflow agents. It adds scoped key/value state, a per-run event log, checkpoints keyed by `thread_id`, and interrupts for human approval.

It lives next to `Flow` / `AgentScheduler` ([fsm.md](fsm.md), [scheduler.md](scheduler.md)), which will later move onto it. Everything is plain structs and function pointers: no `std::function`, no virtual calls, no exceptions. A graph is built once at setup, so its nodes use owning strings. Inference inside Agent nodes goes through the usual arena-backed `run_turn`.

---

## Nodes and edges

```cpp
enum class GraphNodeKind { Agent, Function, Tool, Parallel, Loop };

struct GraphNode {
    std::string name;
    GraphNodeKind kind = GraphNodeKind::Function;

    std::string system_prompt;               // Agent
    std::vector<std::string> allowed_tools;  // Agent; empty = every tool
    std::string prompt;                      // Agent: used when input_key is unset
    std::string input_key;                   // Agent: user message; Tool: JSON arguments
    std::string output_key;                  // Agent: final reply; Tool: tool result
    bool interrupt_before_tools = false;     // Agent

    GraphFunctionFn function = nullptr;      // Function
    std::string tool_name;                   // Tool
    GraphToolArgsFn tool_args = nullptr;     // Tool: overrides input_key
    std::vector<std::string> branches;       // Parallel
    std::string loop_body;                   // Loop
    int max_iterations = 1;                  // Loop
    GraphUntilFn until = nullptr;            // Loop
    void *user_data = nullptr;               // passed to function / tool_args / until

    std::string exit_to;                     // where kGraphEnd leads (set by the builders)
};

struct GraphEdge { std::string from; std::string to; GraphRouterFn router = nullptr; void *user_data = nullptr; };

struct Graph {
    std::string name;
    std::string entry;
    std::vector<GraphNode> nodes;
    std::vector<GraphEdge> edges;
    int max_steps = 1000;
};
```

| Kind | What it does |
|---|---|
| `Agent` | Calls `run_turn` on the node's own `FlowMemory` (`state.histories[name]`). Like a `FlowState`, it installs `system_prompt` for the turn only and limits tools to `allowed_tools`. The user message is `state[input_key]` if that key is set, otherwise `prompt`. The final reply is written to `state[output_key]`. |
| `Function` | `Result<void> fn(Session &, GraphState &, void *user_data)`. Returning an error fails the node. |
| `Tool` | Calls `execute_tool(tool_name, args)`. `args` comes from `tool_args(session, state, user_data)`, else `state[input_key]`, else `"{}"`. The result goes to `state[output_key]`. A tool error fails the node. |
| `Parallel` | Runs every branch, joins them, then follows its own outgoing edge. Built by `make_parallel`. |
| `Loop` | Loop controller. Built by `make_loop`. |

**Edges.** A static edge is `from -> to`. A conditional edge has a router, `Result<std::string_view> router(Session &, const GraphState &, void *)`, which returns the next node name or `kGraphEnd`. Each node has at most one outgoing edge; to fan out, use a `Parallel` node. A node with no outgoing edge goes to `kGraphEnd`. When a transition reaches `kGraphEnd` and the node has an `exit_to`, the run goes there instead; this is how the builders compose sub-graphs.

`run_graph` checks the graph before it runs anything. It rejects a missing entry node, duplicate or empty names, edges or branches that point at unknown nodes, more than one outgoing edge per node, and nodes missing their function or tool name. All of these return `InvalidConfig`.

### Builders (ADK workflow agents)

```cpp
Graph make_graph(GraphNode node);                                   // one-node graph
Graph make_sequential(std::string name, std::vector<Graph> steps);  // SequentialAgent
Graph make_parallel(std::string name, std::vector<Graph> branches); // ParallelAgent
Graph make_loop(std::string name, Graph body, int max_iterations,
                GraphUntilFn until = nullptr, void *user_data = nullptr); // LoopAgent
```

The builders return a flat graph, and they nest freely. Node names must be unique across the whole composed graph.

- **Sequential:** when a step ends (reaches `kGraphEnd`), the run continues at the next step's entry node.
- **Parallel:** a branch ends when it reaches `kGraphEnd` or routes back to the `Parallel` node.
- **Loop:** each iteration of `body` returns to the controller. The controller leaves the loop once `max_iterations` iterations have run, or earlier when `until(session, state, user_data)` returns `true`. `until` is checked after each iteration. The iteration counter is a private per-thread key (`__loop:<name>`) that is cleared when the loop exits.

```cpp
agent::GraphNode researcher{.name = "research", .kind = agent::GraphNodeKind::Agent,
                            .system_prompt = "You research.", .input_key = "topic", .output_key = "notes"};
agent::GraphNode critic{.name = "critic", .kind = agent::GraphNodeKind::Agent,
                        .system_prompt = "Criticise the notes.", .input_key = "notes", .output_key = "review"};

agent::Graph g = agent::make_sequential("pipeline", {
    agent::make_graph(researcher),
    agent::make_loop("refine", agent::make_graph(critic), 3, review_is_good),
    agent::make_parallel("publish", {agent::make_graph(to_markdown), agent::make_graph(to_html)}),
});
```

---

## Scoped state

```cpp
struct GraphState {
    std::map<std::string, std::string, std::less<>> values;
    std::map<std::string, FlowMemory, std::less<>> histories;  // per Agent node
    std::vector<std::string> written;                          // runtime bookkeeping
    bool interrupt_requested = false;
    std::string interrupt_reason;
};

void set_state(GraphState &, std::string_view key, std::string value);
std::string_view get_state(const GraphState &, std::string_view key, std::string_view fallback = {});
bool has_state(const GraphState &, std::string_view key);
```

Values are strings, either JSON or plain text. The key prefix decides the scope, following ADK:

| Prefix | Scope |
|---|---|
| `temp:` | The current call only. Removed when `run_graph` / `resume_graph` returns, and never checkpointed. |
| `user:`, `app:` | Persistent. Every write goes to the Session's `StoreOps` as `graph_state:<key>`. The list of known keys is kept as a JSON array under `graph_state_keys`, because `StoreOps` cannot list keys. They are loaded at the start of every run and are visible to every thread. |
| *(none)* | Per thread (`GraphRun::thread_id`). Carried in checkpoints; a later `run_graph` on the same thread starts from them. |

Write state with `set_state`, which records the key so the runtime can log it, persist it and check it for parallel conflicts. Changing `values` directly works, but the write is not logged, and `user:`/`app:` keys written that way are not persisted. A view returned by `get_state` stays valid until that key is written again.

**Parallel branches** each work on their own copy of `values`. At the join, every key a branch wrote is merged back. If two branches of the same fan-out write the same key, the run fails with `InvalidConfig`, for example: `parallel 'fan': state key 'same' written by branches 'left' and 'right'; parallel branches must write unique keys`. The same applies when one Agent node's history is used by two branches.

---

## Running

```cpp
enum class GraphDurability { Exit, Async, Sync };
enum class GraphRunStatus { Pending, Running, Completed, Interrupted, Failed };

struct GraphRun {
    std::string thread_id;                    // "" = "default"
    GraphDurability durability = GraphDurability::Async;
    GraphState state;                         // seed before; result after
    GraphRunStatus status;
    std::string next_node;                    // interrupted / failed node while paused
    uint64_t step = 0;
    Error error;
    std::string interrupt_node, interrupt_reason;
    std::vector<ToolCall> pending_tool_calls;
    std::vector<GraphEvent> events;
};

Result<void> run_graph(Session &, const Graph &, GraphRun &);
Result<void> resume_graph(Session &, const Graph &, std::string_view thread_id, GraphRun &, std::string_view human_input = {});
Result<void> resume_graph(Session &, const Graph &, std::string_view thread_id, GraphRun &, PermissionDecision, std::string_view human_input = {});
```

`run_graph` starts at `graph.entry`. It builds the starting state in this order: the per-thread values and Agent histories from the thread's latest checkpoint (if there is one), then the `user:`/`app:` keys from the store, then whatever the caller seeded in `run.state`. The return value:

- `ok()` when the run completed, or when it was interrupted (check `run.status`);
- the node's error when a node failed. `run.status` is then `Failed` and `run.next_node` names the failed node. A checkpoint has already been written, so `resume_graph` can retry from that node.

`max_steps` caps how many nodes one call may run (and how many each parallel branch may run), so a routing cycle cannot run forever.

---

## Checkpoints and durability

A checkpoint is a JSON object that holds the step counter, the next node, the status, interrupt details and pending tool calls, all non-`temp:` state values, and every Agent node history (in the same JSON format as `save_flow_memory`). It is written through `store_memory` under:

- `checkpoint:<thread_id>:<step>`, one per completed node;
- `checkpoint:<thread_id>:latest`.

| Mode | When checkpoints are written |
|---|---|
| `Exit` | Only when `run_graph` / `resume_graph` returns: completed, interrupted or failed. |
| `Async` | After every node, and on return. In v1 the write is synchronous, so this behaves exactly like `Sync`; the name is kept for a later background writer. |
| `Sync` | After every node, before the next one starts, and on return. |

`resume_graph` loads `checkpoint:<thread_id>:latest` and continues at its next node. Nodes that already completed do not run again. If the thread has no checkpoint it returns `KeyNotFound`; if the thread already completed it returns `ok()` and does nothing.

Like LangGraph, the runtime re-executes on resume. A node that failed or was interrupted runs again from its start, so tools should be idempotent. State a failing node wrote before returning its error stays in memory and in the failure checkpoint, but it is not logged or persisted to `user:`/`app:` storage.

Agent node `FlowMemory` ids are `<thread_id>:<node>`, so `run_turn` also saves them under `history:<thread_id>:<node>`. Per-node `TurnStats` are not checkpointed.

---

## Interrupts (human in the loop)

**From a Function node.** Call `request_interrupt(state, reason)` and return `ok()`. The run checkpoints and returns with `status == Interrupted`, and sets `interrupt_node` and `interrupt_reason`. `resume_graph(..., human_input)` runs **the same node again** with the reply in `temp:human_input`, so the node should check for that key before it interrupts:

```cpp
agent::Result<void> ask_name(agent::Session &, agent::GraphState &state, void *) {
    if (!agent::has_state(state, "temp:human_input")) {
        agent::request_interrupt(state, "need a name");
        return agent::ok();
    }
    agent::set_state(state, "name", std::string(agent::get_state(state, "temp:human_input")));
    return agent::ok();
}
// run_graph(...)                  -> run.status == Interrupted
// resume_graph(..., "hitl", run, "Ava") -> continues; state["name"] == "Ava"
```

**From an Agent node: `interrupt_before_tools = true`.** When the model asks for tool calls, the node stops before running them. The calls are exposed in `run.pending_tool_calls`, and the assistant message that requested them stays in the node's history. This is the graph form of the permission `Ask` decision:

- If no `Config::permission_check` is set, the node pauses before every tool call.
- If a `permission_check` is set, the node pauses only when at least one call in the round is rated `Ask`. Rounds with only `Allow` or `Deny` calls run without stopping.

On resume, each pending call is answered:

| Call's policy | Approved | Denied |
|---|---|---|
| `Ask` (or no policy) | runs; the policy's `Ask` counts as approved, so `permission_prompt` is not consulted | tool message `Error: tool call denied by human reviewer[: <human_input>]` |
| `Allow` | runs | runs |
| `Deny` | denied by `execute_tool` as usual | denied by `execute_tool` as usual |

The turn then continues, and the model sees the tool results. If the model asks for more tools later, the node pauses again.

You can pass the decision explicitly with `resume_graph(..., PermissionDecision::Allow / Deny, human_input)`. With the string-only overload, a `human_input` of `approve`, `approved`, `yes`, `y`, `allow` or `ok` (case-insensitive) counts as approval, and anything else is a denial.

---

## Event log

```cpp
enum class GraphEventType { NodeStart, NodeEnd, StateWrite, ToolCall, ToolResult, Interrupt, Resume, Checkpoint, NodeFailed };

struct GraphEvent {
    GraphEventType type;
    std::string node;
    std::string key;           // StateWrite: state key; ToolCall/ToolResult: tool name; Checkpoint: store key
    int64_t timestamp_ms;      // wall clock, ms since the Unix epoch
    std::string payload;       // value / arguments / result / reason / error, cut to 200 chars
};
```

Every call appends to `run.events`. The log records node start and end (or failure), every `set_state` write, every tool call and result (both Tool nodes and tool calls made inside Agent turns), interrupts, resumes and checkpoints. Events from parallel branches are appended branch by branch after the join; use the timestamps to see how they interleaved. The log lives in memory and is not persisted.

The existing `EventHook`s also fire: `OnStateTransition` with the node name when a node starts, and `OnToolCall` / `OnToolResult` for Tool nodes and for approved pending calls (`run_turn` fires them itself for its own calls). `EventType` was not extended.

---

## Parallel execution

When `providers::supports_parallel(session)` is true (an engine with `AGENT_ENGINE_FLAG_CONCURRENT`) and the Session has worker arenas, a top-level fan-out runs its branches on threads, at most `Config::max_parallel_subagents` at a time. Each thread uses its own `Session::worker_arenas` slot through the same `ambient::worker_arena_index` mechanism `AgentScheduler` uses. Otherwise the branches run one after another on the calling thread. A `Parallel` node nested inside a branch always runs its branches sequentially.

---

## v1 limitations

- Interrupts are not supported inside parallel branches: `request_interrupt` and `interrupt_before_tools` there fail the node with `InvalidConfig`.
- A fan-out is checkpointed as one step. If a branch fails, the whole fan-out runs again on resume.
- Checkpoints go through `StoreOps`; there is no separate `CheckpointerOps` yet. There is no replay or time travel to an older step, although every `checkpoint:<thread_id>:<step>` is kept.
- `Async` durability writes synchronously.
- Graphs are defined in C++ only; loading them from `agent.json` is not implemented yet.
