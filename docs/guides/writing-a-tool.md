# Guide: writing a tool

Tools are functions the model can call by name. When a reply contains tool calls, `run_turn` dispatches
each one through the permission gate, feeds the result back as a `tool` message and continues until the
model answers in text (at most `Config::max_tool_call_rounds` rounds). This guide writes a tool, gates
it, observes it with hooks and adds the tools of an MCP server.

The code is [`examples/02-tools/tool.cpp`](../../examples/02-tools/tool.cpp). It runs on
the mock engine plugin, where a user message `call:<tool>` makes the "model" request `<tool>` with `{}`,
so the whole loop runs without a real model (test `example.guide_tool`).

## 1. The tool

```cpp
struct Tool {
    std::string_view name;
    std::string_view description;
    std::string_view parameter_schema;
    using CallbackFn = Result<std::string> (*)(std::string_view arguments, void *user_data);
    CallbackFn callback = nullptr;
    void *user_data = nullptr;
};
```

A tool is plain data: views and a function pointer, no `std::function` and no virtual calls. The views
must outlive the Session (string literals are ideal); state goes behind `user_data`:

```cpp
// Tool state lives outside the Session and is reached through user_data; it must outlive the Session.
struct WordCounter {
    int calls = 0;
};

agent::Result<std::string> count_words(std::string_view arguments, void *user_data) {
    auto *counter = static_cast<WordCounter *>(user_data);
    ++counter->calls;
    const std::string text = json_string(arguments, "text");
    int words = 0;
    bool in_word = false;
    for (char c : text) {
        const bool space = c == ' ' || c == '\n' || c == '\t';
        if (!space && !in_word) ++words;
        in_word = !space;
    }
    // ok(...) becomes the "tool" message; fail(...) becomes "Error: <message>" and the turn goes on.
    return agent::ok("{\"words\":" + std::to_string(words) + "}");
}

agent::Tool make_word_count_tool(WordCounter &counter) {
    return agent::Tool{
        .name = "word_count",
        .description = "Count the words in a text.",
        .parameter_schema = R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})",
        .callback = count_words,
        .user_data = &counter,
    };
}
```

- `arguments` is the raw JSON object the model produced. `agent.hpp` has no JSON parser; use your own
  (the example has a tiny `json_string` helper). Treat it as untrusted input: models produce invalid or
  hostile arguments.
- Return `agent::ok(result)` or `agent::fail<std::string>(code, message)`. A failure does not end the
  turn: the model sees `Error: <message>` and can retry. Never throw.
- The callback may be called from a worker thread when the tool is used inside parallel graph branches.
- **Tools that return media.** The string callback above is the stable form. A tool that produces
  files (an image, a recording) or tensors sets `parts_callback` instead (exactly one of the two) and
  returns a `ToolResult` with `text`, `media`, `tensors` and `is_error`; the media are attached to the
  history and reach vision-capable engines. See [api/tools.md](../api/tools.md#tools-that-return-parts).

Register tools in `Config::tools` before `init()`, next to the default filesystem and terminal tools
(`auto_default_tools`), or later with `session.config.tools.push_back(tool)` followed by
`agent::rebuild_tool_index(session)`:

```cpp
    cfg.tools = {make_word_count_tool(counter)}; // added next to the default filesystem/terminal tools
```

Always dispatch through `agent::execute_tool`, never by calling `Tool::callback` yourself: it applies the
permission gate and the node allowlist, and some built-in tools need the Session pointer it refreshes.

```cpp
    auto direct = agent::execute_tool(session, "word_count", R"({"text":"the quick brown fox"})");
```

## 2. The permission gate

Every tool call (built-in, custom, data-source, MCP, `use_skill`) passes `Config::permission_check`
first. It returns `Allow`, `Deny` or `Ask`; `Ask` is resolved by `Config::permission_prompt` (a CLI
would ask the user; without a prompt function, `Ask` means `Deny`):

```cpp
agent::PermissionDecision policy(std::string_view tool, std::string_view arguments, void *) {
    if (tool == "delete_path") return agent::PermissionDecision::Deny;
    if (tool == "run_command" || arguments.find("secret") != std::string_view::npos) return agent::PermissionDecision::Ask;
    return agent::PermissionDecision::Allow;
}

// Resolves Ask (a CLI would prompt the user here). true = approve.
bool prompt(std::string_view tool, std::string_view, void *user_data) {
    ++static_cast<Approvals *>(user_data)->asked;
    return tool != "run_command"; // approve everything but shell commands
}
```

```cpp
    cfg.permission_check = policy;
    cfg.permission_prompt = prompt;
    cfg.permission_user_data = &approvals;
```

A denied call returns `ErrorCode::PermissionDenied` from `execute_tool`; inside a turn the model sees
`Error: Tool denied by permission policy: <name>`. `agent::tools::default_permission_policy` is a
ready-made policy (Ask for `run_command`, `delete_path`, `move_path`). In a graph,
`GraphNode::interrupt_before_tools` turns `Ask` into a pause for human approval instead of a prompt
callback ([writing-a-graph.md](writing-a-graph.md#5-interrupts-human-in-the-loop)).

The gate decides *whether* a call runs; the sandboxes decide what it can touch: file tools are confined
to the workspace (`sandbox_filesystem`), `run_command` runs under an OS sandbox (`tool_sandbox`). Your
own tool runs in your process with your privileges, so validate paths and commands yourself.

## 3. Observing tool calls

**Session-wide event hooks** fire for every tool call made by `run_turn` (and by graph Tool nodes): the
payload of `OnToolCall` is the tool name, that of `OnToolResult` the result text.

```cpp
void on_tool_event(const agent::Event &event, void *user_data) {
    // OnToolCall / OnToolResult payloads are the tool name / the result text.
    const std::string_view payload(reinterpret_cast<const char *>(event.payload.data()), event.payload.size());
    std::printf("   event %s: %.*s\n", event.type == agent::EventType::OnToolCall ? "OnToolCall" : "OnToolResult",
                static_cast<int>(payload.size()), payload.data());
    ++*static_cast<int *>(user_data);
}
```

```cpp
    agent::register_event_hook(session, agent::EventType::OnToolCall, on_tool_event, &events);
    agent::register_event_hook(session, agent::EventType::OnToolResult, on_tool_event, &events);
```

**Per-node tool hooks** (`ToolHook`) belong to a graph Agent node and are active only while it runs.
`Pre` hooks see the arguments, `Post` hooks also the result (including `Error: ...`); they also get the
node's conversation. Together with `allowed_tools` this scopes tools per step of a workflow:

```cpp
void audit(agent::Session &, agent::Conversation &, std::string_view tool, std::string_view args, std::string_view result, void *user_data) {
    std::printf("   post-hook %.*s(%.*s) -> %.*s\n", static_cast<int>(tool.size()), tool.data(), static_cast<int>(args.size()), args.data(),
                static_cast<int>(result.size()), result.data());
    ++*static_cast<int *>(user_data);
}
```

```cpp
    agent::GraphNode counter_node{.name = "counter",
                                  .kind = agent::GraphNodeKind::Agent,
                                  .allowed_tools = {"word_count"},
                                  .prompt = "call:word_count",
                                  .output_key = "answer"};
    counter_node.hooks = {{.when = agent::ToolHook::When::Post, .tool_name = "word_count", .callback = audit, .user_data = &hooked}};
    agent::GraphRun run;
    auto graph = agent::run_graph(session, agent::make_graph(counter_node), run);
```

## 4. Tools from an MCP server

`connect_mcp_server` starts a [Model Context Protocol](https://modelcontextprotocol.io) server as a child
process (stdio transport), lists its tools and registers each as an ordinary `Tool` named
`<server>.<tool>`, so the permission gate, `allowed_tools`, hooks and graph Tool nodes apply unchanged:

```cpp
        auto mcp = agent::connect_mcp_server(session, {.name = "demo", .command = argv[2]});
        check(mcp.ok, "connect_mcp_server", mcp.ok ? std::to_string(mcp.value.size()) + " tools" : mcp.error.message);
        if (mcp.ok) {
            for (const agent::Tool &t : mcp.value) std::printf("   mcp tool: %.*s\n", static_cast<int>(t.name.size()), t.name.data());
            auto echo = agent::execute_tool(session, "demo.echo", R"({"text":"hi"})");
```

For example `{.name = "fs", .command = "npx", .args = {"-y", "@modelcontextprotocol/server-filesystem", "."}}`.
The Session owns the server processes (stopped by `disconnect_mcp_server` or when the Session goes
away). MCP servers are not sandboxed unless you set `McpServerConfig::sandbox`
([security-sandbox.md](../security-sandbox.md)). In `agent.json`, list them under `mcp_servers`
([api/agent-json.md](../api/agent-json.md)). Details: [api/mcp.md](../api/mcp.md).

## Checklist

- [ ] Name, description and schema are clear enough for a small model; string views outlive the Session.
- [ ] The callback validates its arguments and returns `fail(...)` instead of crashing or throwing.
- [ ] Side effects are idempotent where possible (graph resumes and forks re-run tool calls).
- [ ] Dangerous tools return `Ask` (or `Deny`) from your permission policy.
- [ ] Thread-safe if the tool may run in parallel graph branches.

Reference: [api/tools.md](../api/tools.md) (built-in tools, factory functions, skills),
[api/events.md](../api/events.md), [api/graph.md](../api/graph.md#tool-hooks).
