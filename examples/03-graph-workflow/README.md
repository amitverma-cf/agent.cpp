# 03 - graph workflows

Agents as nodes in a graph: sequential pipelines, a router, a review loop, an MCP tool node. `graph.cpp` builds a graph in code
and runs it with checkpoints; `graphs/*.json` are the same kind of graph as data, run with `agent run`.

```sh
./build/agent run examples/03-graph-workflow/graphs/review_loop.json --set task="Explain what a KV cache is."
```

See [the graph guide](../../docs/guides/writing-a-graph.md).
