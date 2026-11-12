# Guide: writing a graph

A graph is a workflow of named nodes joined by edges: Agent nodes (a chat turn with their own system
prompt, tool allowlist and conversation), Function nodes (your C++), Tool nodes (one tool call), and the
workflow nodes Sequential, Parallel and Loop. The runtime adds scoped state, checkpoints after every
node, human-in-the-loop interrupts and time travel. You can build graphs in C++ or write them as
`agent.json`; both produce the same `agent::Graph`.

All code below is from [`examples/03-graph-workflow/graph.cpp`](../../examples/03-graph-workflow/graph.cpp),
which runs on the mock engine (every Agent node answers `echo: <input>`) as the `example.guide_graph`
test, including the two JSON graphs in [`examples/03-graph-workflow/graphs/`](../../examples/03-graph-workflow/graphs).

## 1. Nodes and builders (C++)

Small helpers keep node construction short:

```cpp
agent::GraphNode agent_node(std::string name, std::string system_prompt, std::string input_key, std::string output_key) {
    return agent::GraphNode{.name = std::move(name),
                            .kind = agent::GraphNodeKind::Agent,
                            .system_prompt = std::move(system_prompt),
                            .input_key = std::move(input_key),
                            .output_key = std::move(output_key)};
}

agent::GraphNode function_node(std::string name, agent::GraphFunctionFn fn) {
    return agent::GraphNode{.name = std::move(name), .kind = agent::GraphNodeKind::Function, .function = fn};
}
```

An Agent node reads its user message from `state[input_key]` (or `prompt`) and writes the reply to
`state[output_key]`. A Function node is `Result<void> fn(Session &, GraphState &, void *user_data)`.
The ADK-style builders compose sub-graphs; names must be unique across the whole graph:

```cpp
// Function node: a stand-in reviewer that approves on the second round.
agent::Result<void> review(agent::Session &, agent::GraphState &state, void *) {
    const int rounds = std::atoi(std::string(agent::get_state(state, "rounds", "0")).c_str()) + 1;
    agent::set_state(state, "rounds", std::to_string(rounds));
    agent::set_state(state, "verdict", rounds >= 2 ? "APPROVED" : "needs work");
    return agent::ok();
}

// Loop condition, checked after every iteration.
agent::Result<bool> approved(agent::Session &, const agent::GraphState &state, void *) {
    return agent::ok(agent::get_state(state, "verdict") == "APPROVED");
}
```

```cpp
    agent::Graph draft = agent::make_graph(agent_node("draft", "Write a short draft.", "topic", "draft"));
    agent::Graph refine = agent::make_loop("refine", agent::make_graph(function_node("review", review)), 5, approved);
    agent::Graph title = agent::make_graph(agent_node("title", "Write a title.", "draft", "title"));
    agent::Graph summary = agent::make_graph(agent_node("summary", "Summarise.", "draft", "summary"));
    agent::Graph g = agent::make_sequential("article", {draft, refine, agent::make_parallel("publish", {title, summary})});
    agent::GraphRun run = fresh_run(session, "builders");
    agent::set_state(run.state, "topic", "solid-state batteries");
    auto res = agent::run_graph(session, g, run);
```

- `make_sequential` runs steps in order, `make_loop` repeats its body until `until` returns true or
  `max_iterations` is reached, `make_parallel` fans out and joins.
- Parallel branches run on threads when the engine is concurrent (llama: one continuously batched decode
  for all branches), otherwise one after another. Each branch works on a copy of the state and must
  write **distinct keys**; two branches writing one key fail the join.
- `run_graph` returns `ok()` when the run completed **or** paused (check `run.status`), and the node's
  error when one failed (the failure is checkpointed, so `resume_graph` retries that node).

Agent nodes also take `allowed_tools`, `hooks` (see [writing-a-tool.md](writing-a-tool.md#3-observing-tool-calls)),
`history_key` (nodes with the same key continue one conversation; by default each node has its own),
`context_provider` (per-run system-prompt text such as retrieved documents) and `interrupt_before_tools`.

## 2. Conditional edges

A router returns the next node's name (or `kGraphEnd`) from the state. The returned view must outlive
the call: a literal, a node name or a state value.

```cpp
// Function node + router: classify, then pick the next node from state.
agent::Result<void> classify(agent::Session &, agent::GraphState &state, void *) {
    const std::string_view request = agent::get_state(state, "request");
    agent::set_state(state, "temp:category", request.find("crash") != std::string_view::npos ? "bug" : "question");
    return agent::ok();
}

agent::Result<std::string_view> route_by_category(agent::Session &, const agent::GraphState &state, void *) {
    const std::string_view category = agent::get_state(state, "temp:category");
    if (category == "bug") return agent::ok(std::string_view("triage_bug"));
    if (category == "question") return agent::ok(std::string_view("answer"));
    return agent::ok(agent::kGraphEnd);
}
```

```cpp
    agent::Graph g;
    g.name = "triage";
    g.entry = "classify";
    g.nodes = {function_node("classify", classify), agent_node("triage_bug", "Ask for a stack trace.", "request", "answer"),
               agent_node("answer", "Answer the question.", "request", "answer")};
    g.edges = {{.from = "classify", .router = route_by_category}};
```

A node has at most one outgoing edge (static `to` or a `router`); a node without one ends the graph (or
its step, branch or loop body). `run.events` logs every node start/end, state write, tool call and
checkpoint.

## 3. State scopes

All values are strings in `GraphState::values`, written with `set_state` (so the runtime can log,
persist and conflict-check them). The prefix decides the lifetime:

| Key | Lifetime |
|---|---|
| `temp:<k>` | this `run_graph` / `resume_graph` call only; never checkpointed |
| `user:<k>`, `app:<k>` | persisted through the Session's memory store (`Config::memory.store`); visible to every thread |
| anything else | per thread (`GraphRun::thread_id`), carried in checkpoints |

```cpp
agent::Result<void> publish(agent::Session &, agent::GraphState &state, void *) {
    agent::set_state(state, "published", agent::get_state(state, "approval") == "yes" ? "true" : "false");
    agent::set_state(state, "user:last_topic", std::string(agent::get_state(state, "topic"))); // persisted, every thread
    agent::set_state(state, "temp:scratch", "gone after the call");                            // never checkpointed
    return agent::ok();
}
```

The example gives its Session the SQLite store so `user:` keys survive the process:

```cpp
    cfg.memory.store = agent::store::sqlite_store(); // user:/app: state persists in <workspace>/memory/db.sqlite3
```

## 4. Checkpointers and durability

A checkpoint (state, Agent conversations, next node, interrupt details) is written after every node,
keyed by `(thread_id, step)`. Where it goes is `GraphRun::checkpointer`:

| Checkpointer | Storage |
|---|---|
| `{}` (default) | the Session's memory store |
| `open_checkpointer(agent::in_memory_checkpointer())` | process memory |
| `open_checkpointer(agent::store::sqlite_checkpointer(), {.path = ...})` | a SQLite file (WAL); several processes may share it |

`GraphRun::durability` picks when: `Async` (default; a background writer, flushed before the call
returns), `Sync` (before the next node starts) or `Exit` (only when the call returns). After
`run_graph` returns, every checkpoint is stored whichever mode you pick; the modes differ only in what
a crash mid-run can lose.

```cpp
    auto cp = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = workspace + "/runs.sqlite3"});
```

```cpp
    agent::GraphRun run;
    run.thread_id = "doc-1";
    run.checkpointer = cp.value;
    run.durability = agent::GraphDurability::Sync;
    agent::set_state(run.state, "topic", "local inference");
    auto res = agent::run_graph(session, g, run);
```

`list_graph_checkpoints`, `list_graph_events` and `delete_graph_thread` inspect and clean up a thread.

## 5. Interrupts (human in the loop)

A Function node pauses the run with `request_interrupt`. `resume_graph` runs the **same node again** with
the answer in `temp:human_input`, so the node checks for it first:

```cpp
// Human in the loop: pause until someone answers, then record the decision.
agent::Result<void> approve(agent::Session &, agent::GraphState &state, void *) {
    if (!agent::has_state(state, "temp:human_input")) {
        agent::request_interrupt(state, "publish this draft?");
        return agent::ok();
    }
    agent::set_state(state, "approval", std::string(agent::get_state(state, "temp:human_input")));
    return agent::ok();
}
```

```cpp
agent::Graph approval_graph() {
    return agent::make_sequential("doc", {agent::make_graph(agent_node("write", "Write a short paragraph.", "topic", "draft")),
                                          agent::make_graph(function_node("approve", approve)),
                                          agent::make_graph(function_node("publish_doc", publish))});
}
```

The first `run_graph` returns with `run.status == Interrupted` and `run.interrupt_reason`. Later, possibly
in another process with the same checkpointer file:

```cpp
    // Later (possibly another process): reopen the same file and answer.
    agent::GraphRun resumed;
    resumed.checkpointer = cp.value;
    res = agent::resume_graph(session, g, "doc-1", resumed, "yes");
```

An Agent node with `interrupt_before_tools = true` pauses before running the tools the model asked for
(with a `permission_check`, only before calls it rates `Ask`); the calls are in
`run.pending_tool_calls`, and `resume_graph(..., PermissionDecision::Allow / Deny)` answers them.
Branches of a parallel fan-out can pause independently (`run.interrupts`, one per branch) and be
resumed one by one with `GraphBranchInput`. See [api/graph.md](../api/graph.md#interrupts-human-in-the-loop).

## 6. Time travel: fork_graph_run

Every checkpoint is kept, so a thread can be continued from an earlier step as a new thread, for
example with a corrected input. The source thread is not changed:

```cpp
    // Time travel: continue step 1 (after "write"; "approve" runs next) as a new thread with another topic.
    agent::GraphRun alt;
    alt.checkpointer = cp.value;
    agent::set_state(alt.state, "topic", "cloud inference");
    res = agent::fork_graph_run(session, g, "doc-1", 1, "doc-1-alt", alt);
```

Nodes after the forked step run again, with their side effects (tool calls are not replayed from a
log). Make tools idempotent or put an approval in front of them.

## 7. A different model per node

Agent nodes can run on named models from a `ModelRegistry` instead of the run's Session. The registry
loads them lazily under a RAM budget and evicts idle ones (least recently used first):

```cpp
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 8ull << 30});
    for (const char *name : {"small", "large"}) {
        agent::ModelSpec spec;
        spec.name = name;
        spec.config.provider = agent::AiProvider::Engine;
        spec.config.engine = engine;
        spec.config.model = name; // a GGUF path with a real engine
        spec.config.workspace_dir = workspace;
        spec.estimated_bytes = 1ull << 30; // the model file does not exist here, so do not estimate from disk
        auto reg = agent::register_model(registry, spec);
        check(reg.ok, "register_model", reg.error.message);
    }
    agent::GraphNode quick = agent_node("quick", "Answer briefly.", "question", "short");
    quick.model = "small";
    agent::GraphNode deep = agent_node("deep", "Answer in depth.", "question", "long");
    deep.model = "large";
    agent::Graph g = agent::make_sequential("two_models", {agent::make_graph(quick), agent::make_graph(deep)});

    agent::GraphRun run = fresh_run(session, "models");
    run.models = &registry; // resolves GraphNode::model; must outlive the call
```

A typical split is a small, fast model for routing and classification and a larger one for the answer,
or a vision model for one step. Checkpoints and state still go through the run's Session. See
[api/models.md](../api/models.md).

## 8. The same in agent.json

`load_graph_json` turns a JSON document into the same `Graph` (`parse_graph_json` validates without a
Session). Function nodes refer to C++ functions registered by name; routes and loop conditions are
built in:

```cpp
constexpr const char *kInlineGraph = R"({
  "name": "inline_review",
  "nodes": [
    {"name": "draft", "kind": "agent", "system_prompt": "Write a short draft.", "input_key": "topic", "output_key": "draft"},
    {"name": "refine", "kind": "loop", "body": "review", "max_iterations": 5, "until": {"key": "verdict", "equals": "APPROVED"}},
    {"name": "review", "kind": "function", "function": "review"}
  ],
  "edges": [{"from": "draft", "to": "refine"}]
})";
```

```cpp
    agent::register_graph_function("review", review); // before parsing: function nodes are looked up by name
    auto loaded = agent::load_graph_json(session, kInlineGraph);
```

Keep the `LoadedGraph` alive while its `graph` runs (it owns the route tables). The two bundled
examples run unchanged with `agent run` (the CLI picks the engine and model from the `engine` /
`model` hints, or `--model`):

- [`examples/03-graph-workflow/graphs/router.json`](../../examples/03-graph-workflow/graphs/router.json): a classifier agent, then a `route` on
  its answer to a coder, a mathematician or a general assistant.
- [`examples/03-graph-workflow/graphs/review_loop.json`](../../examples/03-graph-workflow/graphs/review_loop.json): a writer drafts, then a
  `loop` runs a reviewer whose `route` either ends the iteration on `APPROVED` or sends the feedback to
  a reviser that shares the writer's conversation (`history_key`).

```sh
agent run examples/03-graph-workflow/graphs/review_loop.json --set task="Explain what a KV cache is."
agent run examples/03-graph-workflow/graphs/router.json --set request="What is 17 * 23?"
```

`hooks` and `context_provider` are C++-only; everything else maps one to one. Schema and validation
rules: [api/agent-json.md](../api/agent-json.md).

## Checklist

- [ ] Unique node names; at most one outgoing edge per node; routers return views that outlive the call.
- [ ] Parallel branches write distinct keys and use distinct conversations.
- [ ] Interrupting Function nodes check `temp:human_input` before interrupting again.
- [ ] Tools and Function nodes with side effects are idempotent (resume and fork re-run nodes).
- [ ] A checkpointer that matches the lifetime you need (memory for tests, SQLite for anything resumable).

Reference: [api/graph.md](../api/graph.md), [api/agent-json.md](../api/agent-json.md), [api/models.md](../api/models.md).
