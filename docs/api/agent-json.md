# agent.json: graphs from JSON

`#include <agent-cpp/agent.hpp>`

An `agent.json` file describes a [graph](graph.md) declaratively. It lists the nodes, the edges (static, or routed on a state value), the workflow nodes (sequential / parallel / loop), the MCP servers to connect, and hints for the host about which engine and model to use. The loader turns it into an ordinary `agent::Graph`, which you run with `run_graph` / `resume_graph`. Examples are in [`examples/graphs/`](../../examples/graphs).

```cpp
void register_graph_function(std::string_view name, GraphFunctionFn fn, void *user_data = nullptr);
void unregister_graph_function(std::string_view name);

struct LoadedGraph {
    Graph graph;
    std::string description;
    std::string engine;                       // hint for the host, not acted on
    std::string model;                        // hint for the host, not acted on
    std::string checkpointer;                 // "sqlite", "memory" or "": hint for the host
    std::vector<McpServerConfig> mcp_servers; // as declared
    std::vector<std::string> warnings;        // non-fatal findings
    std::shared_ptr<void> storage;            // what graph's user_data pointers refer to
};

Result<LoadedGraph> parse_graph_json(std::string_view path_or_json);                 // no Session needed
Result<LoadedGraph> load_graph_json(Session &session, std::string_view path_or_json);
```

`path_or_json` is treated as JSON text when it starts with `{` or `[` (leading whitespace is skipped); anything else is a file path.

- **`parse_graph_json`** parses and validates the document. It does not start MCP servers and does not check that tools exist. A host can call it before it creates the Session, for example to read the `engine` / `model` hints.
- **`load_graph_json`** calls `parse_graph_json`, then connects every entry in `mcp_servers` with [`connect_mcp_server`](mcp.md). A server that is already connected under the same name is reused, with a warning. It then checks that every Tool node names a registered tool. If any step fails, it disconnects the servers it connected itself. It also writes each warning to the Session logger.

```cpp
agent::register_graph_function("check_budget", check_budget, &budget);  // before loading
auto loaded = agent::load_graph_json(session, "agent.json");
if (!loaded.ok) { std::cerr << loaded.error.message << "\n"; return 1; }
for (const auto &w : loaded.value.warnings) std::cerr << "warning: " << w << "\n";

agent::GraphRun run;
agent::set_state(run.state, "topic", "solid-state batteries");
auto res = agent::run_graph(session, loaded.value.graph, run);
```

Keep the `LoadedGraph` alive while its `graph` is in use, because route tables, loop conditions and static tool arguments live in `storage`. Moving a `LoadedGraph` is fine.

---

## Document

```json
{
  "name": "pipeline",
  "description": "free text",
  "engine": "llama",
  "model": "qwen3-4b-instruct",
  "checkpointer": "sqlite",
  "entry": "research",
  "max_steps": 1000,
  "mcp_servers": [ { "name": "fs", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] } ],
  "nodes": [ { "name": "research", "kind": "agent", "input_key": "topic", "output_key": "notes" } ],
  "edges": [ { "from": "research", "to": "__end__" } ]
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | file stem, or `"graph"` for inline JSON | `Graph::name` |
| `description` | string | | Copied to `LoadedGraph::description` |
| `engine`, `model` | string | | Hints for the host (the CLI uses them to pick the engine plugin and model). The loader only returns them |
| `checkpointer` | `"sqlite"` or `"memory"` | host default | Hint for where the host keeps [checkpoints](graph.md#checkpointers), copied to `LoadedGraph::checkpointer`; any other value is an error. `agent run` uses SQLite unless this says `"memory"` (then `--resume` cannot find the run in a later invocation). The loader does not open anything |
| `entry` | string | first node | Entry node |
| `max_steps` | integer >= 1 | 1000 | `Graph::max_steps` |
| `mcp_servers` | array | | MCP servers to connect at load time; see below |
| `nodes` | array, required, non-empty | | Nodes; see below |
| `edges` | array | | Edges; see below |

`"//"`, `"description"` and `"$schema"` are allowed on every object and ignored, so you can use them for comments. Other unknown fields produce a **warning**, not an error: typos still get noticed, and hosts such as the CLI can add their own fields.

### mcp_servers

```json
{ "name": "fs", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
  "env": { "NODE_NO_WARNINGS": "1" }, "timeout_ms": 60000, "working_directory": "/srv/data" }
```

`name` and `command` are required. `args` is an array of strings and `env` is an object whose values are strings. `timeout_ms` defaults to 30000 and must be > 0. These map one to one onto `McpServerConfig`. Each server's tools are available as `<name>.<tool>`, for Tool nodes and in `allowed_tools`.

**`sandbox`** (optional) runs the server in the [OS sandbox](../security-sandbox.md) (`McpServerConfig::sandbox`). It is off when the field is missing, as in C++:

```json
{ "name": "notes", "command": "/opt/mcp/notes-server", "working_directory": "/srv/notes",
  "sandbox": { "read_only_paths": ["/srv/reference"], "read_write_paths": ["/srv/notes/cache"],
               "network": "deny", "memory_limit_mb": 1024 } }
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `true` | Writing the object turns the sandbox on; `"enabled": false` keeps the settings but runs the server unsandboxed |
| `read_only_paths` | array of strings | `[]` | `SandboxPolicy::read_only_paths`, added to the server's program directory |
| `read_write_paths` | array of strings | `[]` | `SandboxPolicy::read_write_paths`, added to `working_directory` |
| `network` | `"allow"` or `"deny"` | `"allow"` | `SandboxPolicy::network` |
| `memory_limit_mb` | integer, 0 to 16777216 | 0 (none) | `SandboxPolicy::memory_limit_bytes` (MiB; `RLIMIT_AS` on Linux, so leave headroom for runtimes that reserve address space, such as Node or the JVM) |

`"sandbox": true` is short for `{}` (sandbox on, nothing extra listed) and `"sandbox": false` for no sandbox. The rest of the policy keeps the `SandboxPolicy` defaults: system directories read-only, `$TMPDIR` and `/var/tmp` read-write, no GPU. Paths are passed to the kernel as written: a relative path is resolved against the host's current directory (not `working_directory`), and `~` is not expanded. A path that does not exist is skipped with a warning when the server starts. Wrong types, an empty path, an unknown `network` value or a negative or fractional `memory_limit_mb` are errors; unknown fields inside `sandbox` are warnings. Servers launched through `npx`/`uvx` need their runtime and package cache listed, see [mcp.md](mcp.md).

---

## Nodes

Every node has a `name` and a `kind`. Names must be unique, non-empty, not `"__end__"`, and must not contain `#` (the loader uses `#` for names it generates).

| `kind` | Fields | Runtime node |
|---|---|---|
| `agent` | `system_prompt`, `prompt`, `input_key`, `output_key`, `allowed_tools` (array), `interrupt_before_tools` (bool), `history_key` (agent nodes with the same key share one conversation), `model` (run on this [registry](models.md) model; needs `GraphRun::models`) | `GraphNodeKind::Agent`. Same semantics as in [graph.md](graph.md#agent-nodes). `hooks` and `context_provider` are C++ only. A warning is given when there is neither `prompt` nor `input_key` |
| `tool` | `tool` (required), `args` (object), `input_key`, `output_key` | `GraphNodeKind::Tool`. `args` are sent as static JSON arguments, otherwise `state[input_key]` is used, otherwise `{}`. When both are set, `args` wins (with a warning) |
| `function` | `function` (required) | `GraphNodeKind::Function`, with the function registered under that name |
| `parallel` | `branches` (required, non-empty array of node names) | `GraphNodeKind::Parallel` |
| `loop` | `body` (required node name), `max_iterations` (required, >= 1), `until` | `GraphNodeKind::Loop` |
| `sequential` | `steps` (required, non-empty array of node names, no duplicates) | See below |

**Function nodes.** JSON cannot hold function pointers. Register the C++ function first with `register_graph_function(name, fn, user_data)`; the registry is process-wide and protected by a mutex. The loader looks the name up when it parses, so registering or replacing a function later does not change graphs that are already loaded. An unregistered name is an error.

**Loop condition.** `"until": {"key": "verdict", "equals": "APPROVED"}` stops the loop once `state[key]`, with surrounding whitespace trimmed, equals `equals`. It is checked after each iteration, like `GraphUntilFn`. Without `until`, the body runs `max_iterations` times.

### What belongs to a body, branch or step

In JSON the body of a loop, a branch of a parallel node and a step of a sequential node are each given as the name of their **entry node**. The sub-graph consists of that node plus every node reachable from it through edges and routes. Reachability stops at the owning workflow node, and nested workflow nodes count as single nodes. When the sub-graph reaches `__end__`, it finishes the body, branch or step, exactly as with `make_loop` / `make_parallel` / `make_sequential`:

- a loop body returns to the loop node;
- a parallel branch ends and is joined;
- a sequential step continues with the next step; after the last step, the run continues along the sequential node's own outgoing edge.

The loader implements this by setting `GraphNode::exit_to`. A node may belong to only one body, branch or step, or to the top level (the nodes reachable from `entry`). Anything else is an error, for example `node 'x' is part of both loop 'l1' body and loop 'l2' body`. This also catches an edge that jumps out of a loop body back into the top-level flow. Two branches of the same parallel node may lead to a shared node; it then runs in both branches.

A `sequential` node becomes a no-op Function node with an edge to its first step. The last step exits into a generated no-op node `<name>#done`, which carries the sequential node's outgoing edge. You will see `<name>#done` in the event log.

---

## Edges

```json
{ "from": "draft", "to": "review" }
{ "from": "review", "to": "__end__" }
{ "from": "classify", "route": { "key": "temp:category", "cases": { "code": "coder", "math": "mathematician" }, "default": "chatter" } }
```

Every edge has `from` and **exactly one** of `to` and `route`. Each node has at most one outgoing edge; for conditional targets use a route, and to fan out use a `parallel` node. Targets must be existing node names or `"__end__"`. A node without an outgoing edge ends the run, or ends its body, branch or step.

**Routes** are built-in conditional edges. When the node finishes, the router reads `state[key]`, trims surrounding whitespace, and compares it with each key of `cases` (an exact string comparison). The first match picks the target. If nothing matches, or the key is unset, the target is `default`, which defaults to `"__end__"` when omitted. The keys of `cases` are strings; the values are node names or `"__end__"`.

---

## Validation

Errors (`InvalidConfig`, or `ParseError` for malformed JSON, or `FilesystemError` for an unreadable file) name the source and the place, for example:

```
graphs/agent.json: nodes[3] ("fetch"): missing required field "tool"
graphs/agent.json: edges[1] (from "review"): unknown node 'publsh' in "to"
graphs/agent.json: edges[2] (from "a"): node 'a' already has an outgoing edge (edges[0]); use a route for conditional targets or a parallel node to fan out
graphs/agent.json: nodes[4] ("f"): function 'score' is not registered (call agent::register_graph_function first)
graph 'pipeline': tool node 'save' uses unknown tool 'fs.write'          (load_graph_json only)
mcp_servers[0] ("fs"): mcp server 'fs': Program not found on PATH: npx    (load_graph_json only)
```

The checks cover: types of every field, required fields, unknown kinds, duplicate or invalid names, unknown nodes in `entry`, `edges`, `branches`, `body`, `steps` and route cases/defaults, a workflow node that references itself, duplicate steps, `max_iterations` < 1, `max_steps` < 1, duplicate MCP server names, more than one outgoing edge per node, `to` together with `route`, and nodes claimed by two bodies/branches/steps.

**Warnings** (in `LoadedGraph::warnings`; the graph still loads):

- unknown fields;
- nodes that cannot be reached from `entry`;
- an agent node with neither `prompt` nor `input_key`;
- an `allowed_tools` entry that names no registered tool (`load_graph_json` only);
- **cycles without a conditional edge or loop node**, for example `cycle b -> c -> b has no conditional edge or loop node; it only stops at max_steps or when a node interrupts`. Such cycles are allowed rather than rejected, because a Function node in the cycle may interrupt the run on every pass (a chat loop with a human turn), but they are usually a mistake. Cycles that pass through a route are the normal way to loop on a condition and produce no warning. Cycles that pass through a `loop` node are bounded by `max_iterations`.

---

## Schema (informal)

```text
document     := { name?, description?, engine?, model?, entry?, max_steps?, checkpointer?: "sqlite" | "memory", mcp_servers?: [mcp_server], nodes: [node, ...], edges?: [edge] }
mcp_server   := { name, command, args?: [string], env?: {string: string}, timeout_ms?: int>0, working_directory?, sandbox?: bool | object }
node         := { name, kind: "agent",      system_prompt?, prompt?, input_key?, output_key?, allowed_tools?: [string], interrupt_before_tools?: bool, history_key?, model? }
              | { name, kind: "tool",       tool, args?: object, input_key?, output_key? }
              | { name, kind: "function",   function }
              | { name, kind: "parallel",   branches: [node name, ...] }
              | { name, kind: "loop",       body: node name, max_iterations: int>=1, until?: { key, equals } }
              | { name, kind: "sequential", steps: [node name, ...] }
edge         := { from: node name, to: node name | "__end__" }
              | { from: node name, route: { key, cases: { value: node name | "__end__" }, default?: node name | "__end__" } }
```

## Schemas

[`schemas/agent.schema.json`](../../schemas/agent.schema.json) is a JSON Schema (draft 2020-12) for this format; point an editor's `$schema` at it. It checks field types, required fields and enums. It does not check references between nodes, unique names or registered functions; the loader does.

## Examples

- [`research_pipeline.json`](../../examples/graphs/research_pipeline.json): a sequential pipeline of research, then a review loop with `until`, then parallel publishing.
- [`router.json`](../../examples/graphs/router.json): a classifier agent followed by a route to specialist agents.
- [`mcp_filesystem.json`](../../examples/graphs/mcp_filesystem.json): an MCP filesystem server, a Tool node with static `args`, and an agent with `interrupt_before_tools`.
