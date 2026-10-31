# Graph runtime

`#include <agent-cpp/agent.hpp>`

The graph runtime runs workflows built from named nodes and edges, in the style of LangGraph 1.0 and Google ADK workflow agents. It adds scoped key/value state, a per-run event log, checkpoints keyed by `thread_id` through a pluggable checkpointer, and interrupts for human approval.

It is the library's only workflow API: it replaced the earlier `Flow` state machine and `AgentScheduler` (see [Migrating from Flow / AgentScheduler](#migrating-from-flow--agentscheduler)). Everything is plain structs and function pointers: no `std::function`, no virtual calls, no exceptions. A graph is built once at setup, so its nodes use owning strings. Inference inside Agent nodes goes through the usual arena-backed `run_turn`.

You can also write graphs declaratively in JSON and load them with `load_graph_json`; see [agent-json.md](agent-json.md).

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
    std::string history_key;                 // Agent: conversation key; empty = the node name
    GraphContextFn context_provider = nullptr; // Agent: per-run system-prompt prefix
    std::vector<ToolHook> hooks;             // Agent: Pre/Post tool hooks
    std::string model;                       // Agent: run on this GraphRun::models model (docs/api/models.md)

    GraphFunctionFn function = nullptr;      // Function
    std::string tool_name;                   // Tool
    GraphToolArgsFn tool_args = nullptr;     // Tool: overrides input_key
    std::vector<std::string> branches;       // Parallel
    std::string loop_body;                   // Loop
    int max_iterations = 1;                  // Loop
    GraphUntilFn until = nullptr;            // Loop
    void *user_data = nullptr;               // passed to function / tool_args / until / context_provider

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
| `Agent` | Calls `run_turn` on the node's `FlowMemory` (`state.histories[history_key]`, by default `state.histories[name]`). It installs `system_prompt` (prefixed by `context_provider`'s text, if any) for the node's turn only, limits tools to `allowed_tools`, and activates its `hooks`. The user message is `state[input_key]` if that key is set, otherwise `prompt`. The final reply is written to `state[output_key]`. See [Agent nodes](#agent-nodes). |
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

## Agent nodes

### System prompt and tool allowlist

`system_prompt` becomes the conversation's system message (`history[0]`) for the duration of the node's turn and is then removed again (or the previous system message restored), so it is never stored in the history. `allowed_tools` scopes both the tool definitions the engine sees and what `execute_tool` will dispatch while the node runs; a model-requested call to any other tool gets the tool message `Error: Tool not allowed for the current agent node: <name>`. `Session::config.tools` itself is never changed. Empty = every registered tool.

```cpp
agent::GraphNode reader{.name = "read", .kind = agent::GraphNodeKind::Agent,
                        .system_prompt = "Inspect the workspace.",
                        .allowed_tools = {"read_file", "list_dir", "file_info"},
                        .input_key = "task", .output_key = "findings"};
```

### Private or shared conversation (`history_key`)

By default every Agent node has its own conversation, `state.histories[<node name>]`: a node that runs again (in a loop, or on a later `run_graph` of the same thread) continues where it left off, but it never sees what other nodes said. Give several Agent nodes the same `history_key` to have them continue **one** conversation, each under its own system prompt and tool allowlist. This is how a plan -> act -> summarise agent keeps its whole exchange in one context:

```cpp
agent::GraphNode plan{.name = "plan", .kind = agent::GraphNodeKind::Agent,
                      .system_prompt = "You are a careful planner.", .input_key = "task",
                      .history_key = "react"};
agent::GraphNode act{.name = "act", .kind = agent::GraphNodeKind::Agent,
                     .system_prompt = "You are an executor. Use tools precisely.",
                     .allowed_tools = {"read_file", "write_file", "run_command"},
                     .prompt = "Execute your plan now.", .history_key = "react"};
agent::GraphNode report{.name = "report", .kind = agent::GraphNodeKind::Agent,
                        .system_prompt = "You are a concise reporter.",
                        .prompt = "Summarise what was done.", .output_key = "summary",
                        .history_key = "react"};
agent::Graph g = agent::make_sequential("react", {agent::make_graph(plan), agent::make_graph(act),
                                                   agent::make_graph(report)});
```

Shared histories are checkpointed under their key like any other. Two branches of one `Parallel` node may not use the same history (the join fails with `InvalidConfig`).

### Per-run context (`context_provider`)

```cpp
using GraphContextFn = Result<std::string> (*)(Session &session, const GraphState &state, void *user_data);
```

Called before each run of the node, with the node's `user_data`. A non-empty result is prepended to `system_prompt` (separated by a newline) for that run only; nothing is written to the history. An error fails the node. Typical use: fold retrieved context (a [data source](data-sources.md), [RAG](rag.md) hits, values from state) into the prompt without a tool round trip.

```cpp
agent::Result<std::string> docs_context(agent::Session &session, const agent::GraphState &state, void *) {
    auto res = agent::query_data_source(session, "product_docs", agent::get_state(state, "question"));
    if (!res.ok) return agent::ok(std::string());   // answer without context
    return agent::ok("Relevant docs:\n" + res.value);
}
// answer.context_provider = docs_context;
```

### Tool hooks

```cpp
struct ToolHook {
    enum class When { Pre, Post };
    When when = When::Pre;
    std::string_view tool_name;   // empty = every tool; the view must outlive the node
    using HookFn = void (*)(Session &, FlowMemory &memory, std::string_view tool_name,
                            std::string_view args, std::string_view result, void *user_data);
    HookFn callback = nullptr;
    void *user_data = nullptr;
};
```

`GraphNode::hooks` are active only while that Agent node runs. For each tool call: the matching `Pre` hooks fire in order (`result` empty), the tool runs, then the matching `Post` hooks fire with its result (including `Error: ...` results). `memory` is the node's conversation. Calls approved on `resume_graph` fire the hooks too; calls a reviewer denied never run, so they fire none. Hooks of nodes inside a `Parallel` branch may be called from worker threads.

```cpp
static void audit(agent::Session &, agent::FlowMemory &, std::string_view tool, std::string_view args,
                  std::string_view result, void *) {
    std::printf("%s %s -> %s\n", std::string(tool).c_str(), std::string(args).c_str(), std::string(result).c_str());
}
act.hooks = {{.when = agent::ToolHook::When::Post, .tool_name = "write_file", .callback = audit}};
```

---

## Scoped state

```cpp
struct GraphState {
    std::map<std::string, std::string, std::less<>> values;
    std::map<std::string, FlowMemory, std::less<>> histories;  // per Agent node (or history_key)
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

**Parallel branches** each work on their own copy of `values`. At the join, every key a branch wrote is merged back. If two branches of the same fan-out write the same key, the run fails with `InvalidConfig`, for example: `parallel 'fan': state key 'same' written by branches 'left' and 'right'; parallel branches must write unique keys`. The same applies when one Agent history (node name or `history_key`) is used by two branches.

---

## Running

```cpp
enum class GraphDurability { Exit, Async, Sync };
enum class GraphRunStatus { Pending, Running, Completed, Interrupted, Failed };

struct GraphRun {
    std::string thread_id;                    // "" = "default"
    GraphDurability durability = GraphDurability::Async;
    Checkpointer checkpointer;                // default: the Session's StoreOps
    GraphState state;                         // seed before; result after
    GraphRunStatus status;
    std::string next_node;                    // interrupted / failed node while paused
    uint64_t step = 0;
    Error error;
    std::string interrupt_node, interrupt_reason;
    std::vector<ToolCall> pending_tool_calls;
    std::vector<GraphEvent> events;
    ModelRegistry *models = nullptr;          // resolves GraphNode::model; see models.md
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

A checkpoint is a JSON object that holds the step counter, the next node, the status, interrupt details and pending tool calls, all non-`temp:` state values, and every Agent node history (in the same JSON format as `save_flow_memory`). One is written after every completed node (see the durability modes below), keyed by `(thread_id, step)`, together with a small metadata object (`{"graph", "step", "next", "status"}`).

### Checkpointers

```cpp
struct GraphCheckpointInfo { uint64_t step; int64_t timestamp_ms; std::string metadata; };
struct CheckpointerConfig { std::string path; };

struct CheckpointerOps {
    OpenFn open;                 // Result<std::shared_ptr<void>>(const CheckpointerConfig &)
    PutFn put;                   // (state, thread_id, step, blob, metadata)
    GetLatestFn get_latest;      // highest step; KeyNotFound when none
    GetFn get;                   // (state, thread_id, step); KeyNotFound when missing
    ListFn list;                 // every step of a thread, oldest first
    DeleteThreadFn delete_thread;
};
const CheckpointerOps *in_memory_checkpointer();
const CheckpointerOps *store::sqlite_checkpointer();   // <agent-cpp/store_sqlite.hpp>, agent_store_sqlite

struct Checkpointer { const CheckpointerOps *ops; std::shared_ptr<void> state; std::shared_ptr<void> writer; };
Result<Checkpointer> open_checkpointer(const CheckpointerOps *ops, const CheckpointerConfig &config = {},
                                       size_t async_queue_capacity = 64);

Result<std::vector<GraphCheckpointInfo>> list_graph_checkpoints(Session &, std::string_view thread_id,
                                                                const Checkpointer & = {});
Result<void> delete_graph_thread(Session &, std::string_view thread_id, const Checkpointer & = {});
```

`CheckpointerOps` is a function-pointer table like `StoreOps` and `VectorStoreOps`. Implementations must be safe to call from several threads on one state. Select one per run with `GraphRun::checkpointer` (set it on the `GraphRun` you pass to `resume_graph` as well). Copies of a `Checkpointer` share its state and its Async writer.

| Checkpointer | Where checkpoints go |
|---|---|
| `{}` (default, `ops == nullptr`) | The Session's `StoreOps`, as in v1: `checkpoint:<thread_id>:<step>` and `checkpoint:<thread_id>:latest`, plus `checkpoint_meta:<thread_id>:<step|latest>` for listing. `StoreOps` cannot delete keys, so `delete_graph_thread` overwrites them with empty values. |
| `in_memory_checkpointer()` | Process memory; gone with the last copy of the `Checkpointer`. |
| `store::sqlite_checkpointer()` | A SQLite file (`CheckpointerConfig::path`; empty or `":memory:"` = in memory). WAL journal, `synchronous=NORMAL` (a committed checkpoint survives a process crash; an OS crash or power loss can lose the last transactions), table `graph_checkpoints` keyed by `(thread_id, step)`. Several processes may open the same file. |

```cpp
#include <agent-cpp/store_sqlite.hpp>

auto cp = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = "runs.sqlite3"});
agent::GraphRun run;
run.thread_id = "ticket-42";
run.checkpointer = cp.value;
agent::run_graph(session, graph, run);

// Later, in another process:
agent::GraphRun resumed;
resumed.checkpointer = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = "runs.sqlite3"}).value;
agent::resume_graph(session, graph, "ticket-42", resumed);
```

### Durability modes

| Mode | When checkpoints are written |
|---|---|
| `Exit` | Only when `run_graph` / `resume_graph` returns: completed, interrupted or failed. |
| `Async` (default) | After every node, by a background writer thread, and on return. |
| `Sync` | After every node, before the next one starts, and on return. |

**Async writer.** Each checkpointer opened with `open_checkpointer` owns one writer thread (started on first use, joined when the last copy of the `Checkpointer` goes away); the default StoreOps checkpointer gets one per call. The run hands each checkpoint to a bounded FIFO (`async_queue_capacity`, default 64) and keeps going; when the queue is full the run blocks until the writer catches up. Before `run_graph` / `resume_graph` returns, whether the run completed, was interrupted or failed, the queue is flushed and the final checkpoint is written synchronously, so after the call returns every checkpoint is stored. A write error from the queue is reported by that call (`run.status == Failed`, `run.error`).

**Crash window.** If the process dies mid-run, `Async` can lose the checkpoints still in the queue (at most `async_queue_capacity`): `resume_graph` then continues from the newest checkpoint that reached storage and re-runs the nodes after it. `Sync` loses at most the node that was running. `Exit` loses everything since the call started. None of this affects a run that returns normally.

### Resume

`resume_graph` loads the thread's latest checkpoint and continues at its next node. Nodes that already completed do not run again. If the thread has no checkpoint it returns `KeyNotFound`; if the thread already completed it returns `ok()` and does nothing.

Like LangGraph, the runtime re-executes on resume. A node that failed or was interrupted runs again from its start, so tools should be idempotent. State a failing node wrote before returning its error stays in memory and in the failure checkpoint, but it is not logged or persisted to `user:`/`app:` storage.

Agent `FlowMemory` ids are `<thread_id>:<history>` (the node name or its `history_key`), so `run_turn` also saves them under `history:<thread_id>:<history>` in the Session's store, whichever checkpointer is used. Per-node `TurnStats` are not checkpointed.

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

**Starting over instead of resuming.** A fresh `run_graph` on a thread whose last run paused before tool calls abandons those calls: they never run. So that the conversation stays valid (every tool call answered before the next user message), `run_graph` appends a tool message `Error: tool call not executed: the run was restarted with run_graph before the call was approved (...)` for each unanswered call at the end of every loaded history, and logs a `ToolResult` event for it. The model sees the calls as failed. Use `resume_graph` to actually answer them.

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

The existing `EventHook`s also fire: `OnStateTransition` with the node name when a node starts; `OnSubAgentComplete` with the branch's entry node name for each successful branch of a `Parallel` node (fired on the calling thread, in branch order, after the branches have joined); and `OnToolCall` / `OnToolResult` for Tool nodes and for approved pending calls (`run_turn` fires them itself for its own calls).

---

## Parallel execution

When `providers::supports_parallel(session)` is true (an engine with `AGENT_ENGINE_FLAG_CONCURRENT`) and the Session has worker arenas, a top-level fan-out runs its branches on threads, at most `Config::max_parallel_subagents` at a time. Each thread uses its own `Session::worker_arenas` slot (`ambient::worker_arena_index`). More branches than slots run in batches of `max_parallel_subagents`. Otherwise the branches run one after another on the calling thread. A `Parallel` node nested inside a branch always runs its branches sequentially.

With the llama engine the parallelism comes from **continuous batching** inside the plugin, not from the threads themselves: one shared `llama_context` (created with `n_seq_max = max_parallel_subagents`) folds every active generation's next token into a single `llama_decode` per round, so the branch threads mostly wait on that shared batch. See [architecture.md](../architecture.md#continuous-batching).

---

## v1 limitations

- Interrupts are not supported inside parallel branches: `request_interrupt` and `interrupt_before_tools` there fail the node with `InvalidConfig`.
- A fan-out is checkpointed as one step. If a branch fails, the whole fan-out runs again on resume.
- There is no replay or time travel to an older step, although every checkpoint is kept.
- `hooks` and `context_provider` are C++ only; `agent.json` can set `history_key` but not these.

---

## Migrating from Flow / AgentScheduler

`Flow`, `FlowState`, `FlowHook`, `run_flow`, `step_flow`, `AgentScheduler` and `SubAgentResult` were removed; `FlowMemory`, `run_turn` and the rest of the conversation API are unchanged.

| Old | Graph runtime |
|---|---|
| `FlowState` with `on_transition` calling `run_turn` | An `Agent` node (`prompt` / `input_key`, `output_key`). Custom work goes in a `Function` node. |
| `FlowState::system_prompt`, `allowed_tools` | `GraphNode::system_prompt`, `allowed_tools` (same per-turn scoping) |
| `FlowState::hooks` / `FlowHook` | `GraphNode::hooks` / `ToolHook` (same fields and callback signature) |
| `Flow::memory` shared by all states | Same `history_key` on the Agent nodes |
| `FlowState::isolated_memory = true` | The default: each Agent node has its own history |
| `FlowState::context_provider` | `GraphNode::context_provider` (`GraphContextFn`, reads `GraphState`) |
| `on_transition` returning the next state id / `"exit"` | Static edges, or a router (`GraphRouterFn`) returning a node name / `kGraphEnd` |
| `void *context` passed to `run_flow` | `GraphNode::user_data` / `GraphEdge::user_data`, or values in `GraphState` |
| `run_flow` | `run_graph` |
| `step_flow` | Interrupts (`request_interrupt`, `interrupt_before_tools`) + `resume_graph`, or the per-node `GraphEvent` log |
| `AgentScheduler::spawn` + `run_until_done` | `make_parallel` with one branch per subagent (concurrent on concurrent engines, up to `max_parallel_subagents` at a time) |
| `SubAgentResult` | Branch outputs in `GraphState` (`output_key`), per-branch `state.histories[...].stats`; a failing branch fails the run with its error |
| `OnStateTransition` / `OnSubAgentComplete` events | Fired at node start / per completed parallel branch |

Differences: a graph checkpoints after every node and can be resumed; parallel branches must write distinct state keys; and `AgentScheduler::pump`'s one-state-per-tick round robin has no equivalent (a branch runs to its end).
