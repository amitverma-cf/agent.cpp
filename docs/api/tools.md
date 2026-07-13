# Tools

`#include <agent-cpp/agent.hpp>`

Tools are callable functions the LLM can invoke by name. When the model emits a tool call in its response, `run_turn` dispatches it automatically, feeds the result back as a `"tool"` message, and continues the inference loop.

---

## Tool struct

```cpp
struct Tool {
    std::string_view name;               // unique identifier; LLM uses this to call the tool
    std::string_view description;        // shown to the LLM in the system context
    std::string_view parameter_schema;   // JSON Schema for the arguments object

    using CallbackFn = Result<std::string>(*)(std::string_view arguments, void *user_data);
    CallbackFn callback  = nullptr;
    void      *user_data = nullptr;
};
```

### CallbackFn

The callback receives raw JSON arguments and a `user_data` pointer. It returns either:
- `ok(std::string result)` — the result is inserted into the conversation as a `"tool"` message.
- `fail(code, message)` — the error string `"Error: <message>"` is inserted instead; the turn continues.

Tool callbacks **must not** call `session.arena.reset()` or modify `session.config.tools` without calling `rebuild_tool_index` afterward.

---

## Registration

Tools are registered through `Config::tools` at session creation:

```cpp
agent::Config cfg;
cfg.tools = { my_tool_a, my_tool_b };
auto sess = agent::init(cfg);
```

When `auto_default_tools = true` (the default), the built-in filesystem, terminal, and context compressor tools are also added by `init()`. If a user-provided tool has the same `name` as a default tool, the user tool takes priority.

To add tools after `init()`:

```cpp
session.config.tools.push_back(my_new_tool);
agent::rebuild_tool_index(session);   // required after mutating tools
```

### O(1) dispatch

Tools are dispatched via `session.tool_by_name`, an `unordered_map<string, size_t>` that maps tool name to index in `session.config.tools`. `rebuild_tool_index` (re)builds this index. It is called automatically by `init()` and by `step_flow` when filtering to `allowed_tools`.

---

## Custom tool example

```cpp
struct DbCtx {
    std::string connection_string;
};

DbCtx db_ctx{ .connection_string = "postgresql://localhost/mydb" };

agent::Tool query_db{
    .name        = "query_database",
    .description = "Run a read-only SQL SELECT query and return results as JSON.",
    .parameter_schema =
        R"({
          "type": "object",
          "properties": {
            "sql": { "type": "string", "description": "SQL SELECT statement" }
          },
          "required": ["sql"]
        })",
    .callback = [](std::string_view args, void *ud) -> agent::Result<std::string> {
        auto *ctx = static_cast<DbCtx*>(ud);

        // parse args with simdjson or manually
        std::string sql = extract_string(args, "sql");
        if (sql.empty())
            return agent::fail<std::string>(agent::ErrorCode::ParseError, "Missing 'sql'");

        // execute query against ctx->connection_string ...
        return agent::ok(std::string(R"([{"id":1,"name":"Alice"}])"));
    },
    .user_data = &db_ctx,    // must outlive the session
};

cfg.tools = { query_db };
```

---

## Default tools

When `Config::auto_default_tools = true`, these tools are registered automatically at `init()`.

### Filesystem tools

All paths resolve relative to `workspace_dir`. Absolute paths are accepted but checked against the sandbox if `Config::sandbox_filesystem = true`.

#### `read_file`

Read a file. Returns the file content as a string, truncated at `max_bytes` if the file is larger.

```json
{
  "path": "src/main.cpp",
  "max_bytes": 16384
}
```

Returns: raw file content, with `\n[truncated at N bytes]` appended if the file is larger.

#### `write_file`

Write (overwrite) a file. Creates parent directories by default.

```json
{
  "path": "output/report.md",
  "content": "# Report\n\nHello world.",
  "create_dirs": true
}
```

Returns: `"wrote N bytes to /path/to/file"`

#### `append_file`

Append text to a file. Creates the file and parent directories if they do not exist.

```json
{
  "path": "logs/debug.log",
  "content": "2025-01-01: agent started\n"
}
```

Returns: `"appended N bytes to /path/to/file"`

#### `list_dir`

List directory contents. Returns a JSON array of `{name, type, size?}` objects. Omit `path` to list the workspace root.

```json
{ "path": "src" }
```

Returns: `[{"name":"main.cpp","type":"file","size":1234},{"name":"utils","type":"dir"}]`

#### `create_dir`

Create a directory and all missing parents.

```json
{ "path": "output/reports/2025" }
```

Returns: `"created /workspace/output/reports/2025"`

#### `delete_path`

Delete a file or directory tree recursively. Cannot delete the workspace root.

```json
{ "path": "tmp/scratch" }
```

Returns: `"deleted N entries at /path/to/scratch"`

#### `move_path`

Move or rename a file or directory.

```json
{ "from": "draft.md", "to": "published/final.md" }
```

Returns: `"moved /a -> /b"`

#### `file_info`

Get metadata for a path.

```json
{ "path": "models/my.gguf" }
```

Returns: `{"exists":true,"type":"file","size":4294967296,"path":"/workspace/models/my.gguf"}` or `{"exists":false}`.

### Terminal tool

#### `run_command`

Run a shell command in `workspace_dir`. stdout and stderr are merged. Output is capped at 32 KiB.

```json
{
  "command": "git log --oneline -5",
  "timeout_seconds": 30
}
```

Returns: `{"exit_code":0,"output":"abc1234 fix bug\ndef5678 add feature\n..."}`

`timeout_seconds` defaults to 30. Set to 0 for no timeout. On POSIX, the system `timeout` utility enforces the limit. On Windows, no timeout is applied when `timeout_seconds = 0`.

### Context compressor

#### `compress_context`

Summarise and compress the active conversation's older history to free context window space. The LLM invokes this tool automatically when context pressure is high, or you can expose it for the model to call explicitly.

```json
{ "keep_recent": 6 }
```

Returns: `"Context compressed."` on success.

---

## Sandbox

When `Config::sandbox_filesystem = true` (the default), the filesystem tools enforce that all resolved paths stay inside `workspace_dir`. The check resolves through `hal::resolve_under_workspace` (`weakly_canonical` + prefix-path comparison), so symlink and `..` escapes are blocked, not just lexical joins.

Rejected paths return `ErrorCode::SandboxViolation` with message `"Path is outside workspace_dir (sandbox mode is enabled)."`.

To allow unrestricted filesystem access:

```cpp
cfg.sandbox_filesystem = false;
```

## Disabling default tools

```cpp
cfg.auto_default_tools = false;
cfg.tools = { my_custom_tool };   // only this tool is registered
```

## Tool factory functions

You can create individual default tools manually if needed (for custom `WorkspaceContext` or to embed them in a different session):

```cpp
namespace agent::tools {

struct WorkspaceContext {
    const std::string *workspace_dir = nullptr;
    bool               sandbox       = true;
};

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
Tool              get_terminal_tool   (const std::string *workspace_dir);

}
```

`session.tool_context` holds the `shared_ptr<WorkspaceContext>` that keeps the context alive while the session exists.
