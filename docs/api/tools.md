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

    using PartsCallbackFn = Result<ToolResult>(*)(std::string_view arguments, void *user_data);
    PartsCallbackFn parts_callback = nullptr;   // set exactly one of callback / parts_callback
};
```

### CallbackFn

The callback receives raw JSON arguments and a `user_data` pointer. It returns either:
- `ok(std::string result)` — the result is inserted into the conversation as a `"tool"` message.
- `fail(code, message)` — the error string `"Error: <message>"` is inserted instead; the turn continues.

Tool callbacks **must not** call `session.arena.reset()` or modify `session.config.tools` without calling `rebuild_tool_index` afterward.

### Tools that return parts

A tool that produces files (an image, a recording, a mesh) or tensors sets `parts_callback` instead
of `callback` and returns a `ToolResult`:

```cpp
struct ToolResult {
    std::string             text;
    std::vector<Media>      media;     // {path, mime}: images, audio, video, any file
    std::vector<TensorData> tensors;
    bool                    is_error = false;
};
```

Exactly one of `callback` and `parts_callback` must be set (`execute_tool` fails with
`InvalidConfig` otherwise). Both kinds go through the same `allowed_tools` check, permission gate
and tool hooks.

Inside `run_turn` (and graph Agent nodes, including tool calls approved on `resume_graph`):

- `media` and `tensors` go on the `"tool"` `Message` (`Message::media` / `Message::tensors`). The
  media persist with the history (`save_flow_memory`, graph checkpoints) and reach the engine as
  `FILE` parts with every later inference of that conversation, so a vision model can look at an
  image a tool produced. Relative paths are made absolute.
- The message text is `text` followed by one `[attached: <path> (<mime>)]` line per media item (and
  `[tensor: <name> [dims]]` per tensor), so models without that input modality still learn about the
  file. `ToolHook::Post`, `EventType::OnToolResult` and the graph `ToolResult` event see this text
  (the event payload keeps the attachment lines when it clips the text).
- Media are only sent to an engine that declares the matching input capability (`image/*` →
  `AGENT_CAP_VISION_IN`, `audio/*` → `AGENT_CAP_AUDIO_IN`, `video/*` → `AGENT_CAP_VIDEO_IN`); other
  files stay in the history but out of the request. If the engine fails a request that carries tool
  media (e.g. a llama model loaded without `mmproj`), the request is retried once without it and that
  Session stops sending tool media (a warning is logged).
- `is_error = true`: the tool ran but failed. The conversation gets `Error: <text>`; the media are
  still attached. A failed `Result` from `parts_callback` is reported like a failed `callback`.

Calling tools directly:

```cpp
// Text + "[attached: ...]" lines; is_error -> ErrorCode::ToolError.
Result<std::string> execute_tool(Session &, std::string_view name, std::string_view arguments);
// The whole ToolResult; is_error comes back as ok() with is_error set.
Result<ToolResult>  execute_tool_parts(Session &, std::string_view name, std::string_view arguments);
```

A graph `Tool` node uses the string form: its `output_key` gets the text with the attachment lines,
and an `is_error` result fails the node.

```cpp
agent::Tool snapshot{
    .name = "snapshot",
    .description = "Capture the current camera frame.",
    .parameter_schema = R"({"type":"object","properties":{}})",
    .user_data = &camera,
    .parts_callback = [](std::string_view, void *ud) -> agent::Result<agent::ToolResult> {
        std::string path = static_cast<Camera *>(ud)->capture_png();   // your code
        agent::ToolResult r;
        r.text = "Captured a frame.";
        r.media.push_back({.path = path, .mime = "image/png"});
        return agent::ok(std::move(r));
    },
};
```

MCP tools return `ToolResult`s: `image` / `audio` content items become media (see [mcp.md](mcp.md#tool-calls)).

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

Tools are dispatched via `session.tool_by_name`, an `unordered_map<string, size_t>` that maps tool name to index in `session.config.tools`. `rebuild_tool_index` (re)builds this index. It is called automatically by `init()`; call it again after changing `session.config.tools`. Per-node `allowed_tools` (graph Agent nodes) does not touch the index: it is a thread-local allowlist checked by `execute_tool`.

### MCP servers

`connect_mcp_server(session, McpServerConfig{...})` starts a Model Context Protocol server over stdio and registers each of its tools as `<server>.<tool>`. They go through the same permission gate as every other tool. See [mcp.md](mcp.md).

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

        // parse args with any JSON parser (agent.hpp does not expose one) or manually
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

Read a file (a `parts_callback` tool).

```json
{
  "path": "src/main.cpp",
  "max_bytes": 16384,
  "offset": 0
}
```

- **Text files**: the content from `offset` (default 0), at most `max_bytes` (default 16384, clamped
  to 4 MiB). The cut never splits a UTF-8 character; invalid UTF-8 bytes become U+FFFD. A longer file
  gets `\n[truncated at N bytes]\n[T bytes in total; read more with "offset": K]`.
- **Images, audio, video** (by extension: png, jpg/jpeg, gif, webp, bmp, tga, wav, mp3, flac, ogg,
  opus, m4a, mp4, webm, mov, mkv, avi): returned as `ToolResult::media` with the text
  `image file cat.png (N bytes), attached`, so a vision / audio model looks at or listens to the file
  on its next inference (the `view_image` use case). A file outside the workspace (only possible with
  `sandbox_filesystem = false`) is first copied to `<workspace>/artifacts/read-<ms>-<name>`, because a
  sandboxed out-of-process engine can only read the workspace.
- **Other binary files** (containing NUL bytes): `[binary file <name>, N bytes; not shown as text]`.
- A directory fails (use `list_dir`).

Call it through `execute_tool` / `execute_tool_parts`: `read_file` has no string `callback`.

#### `write_file`

Write (overwrite) a file. Creates parent directories by default.

```json
{
  "path": "output/report.md",
  "content": "# Report\n\nHello world.",
  "create_dirs": true
}
```

Returns: `"wrote N bytes to /path/to/file"`. A missing (or non-string) `content` fails instead of
truncating the file.

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

Returns: `[{"name":"main.cpp","type":"file","size":1234},{"name":"utils","type":"dir"}]`, sorted by
name. At most 1000 entries; a larger directory ends with `{"truncated":true,"shown":1000}`.

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

Move or rename a file or directory. Cannot move the workspace root.

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

`timeout_seconds` defaults to 30 (negative = 30, capped at one day). Set to 0 for no timeout. On POSIX, the system `timeout` utility enforces the limit. On Windows, no timeout is applied when `timeout_seconds = 0`. Invalid UTF-8 in the output (including a character cut by the 32 KiB cap) becomes U+FFFD. A running command is not stopped by `request_cancel()`; only its timeout ends it.

### Context compressor

#### `compress_context`

Summarise the active conversation's older history into one system message
(`[Compressed history]: <summary>`) to free context window space. The model can call it (it acts on
the conversation that issued the call: in a graph, the node's history, shared ones included), and
`run_turn` also compresses automatically at the start of a turn when the history exceeds 70% of
`context_window` (keeping the 6 most recent messages). `agent::compress_context(session, memory,
keep_recent)` does the same directly.

```json
{ "keep_recent": 6 }
```

Returns: `"Context compressed."` on success (also when there was nothing to compress).

- The summary is written by the Session's own engine as a plain chat request (`AGENT_CAP_TEXT_GEN`,
  whatever `Config::default_task` is, without tool definitions), in a separate arena, so it works
  from inside a tool call and with process-isolated engines. An engine that cannot chat, or a failed
  request, skips compression (the pruner still trims the history).
- An assistant tool call and its results are never separated, and the assistant message whose tool
  call is running stays in place.
- Media in the summarized part are named in the summary prompt (`[file: <path> (<mime>)]`); the
  files themselves leave the context.
- `run_turn` sends the leading system messages (system prompt, summary, injected tool list) to the
  engine as **one** system message, and the pruner never drops them. A graph node's `system_prompt`
  goes in front of a summary instead of replacing it. A later compression folds the previous summary
  into the new one.

### Skill loader

#### `use_skill`

Registered automatically when `Config::skills_dir` is set (see [Skills](../guide.md#skills)). Loads the full `SKILL.md` content for a named skill, previously scanned by `load_skills_dir`, or a file bundled with it.

```json
{ "name": "pdf-tools" }
{ "name": "pdf-tools", "file": "references/forms.md" }
```

Returns: `Skill directory: <absolute path> ...` followed by the skill's full `SKILL.md` content.
With `file`: that file's text (up to 256 KiB), or for png/jpg/gif/webp/wav/mp3 the file as media.
`file` must resolve inside the skill directory (`SandboxViolation` otherwise). Returns `KeyNotFound`
if the name doesn't match a loaded skill. In a graph Agent node the skills list is only advertised
when `use_skill` is in the node's `allowed_tools` (or the list is empty). `SKILL.md` frontmatter
values may be quoted.

### Graph Agent nodes

`allowed_tools` also scopes the plain-text tool list injected for engines without native tool
calling: the engine only sees the node's tools.

---

## Sandbox

When `Config::sandbox_filesystem = true` (the default), the filesystem tools enforce that all resolved paths stay inside `workspace_dir`. The check resolves through `hal::resolve_under_workspace` (`weakly_canonical` + prefix-path comparison), so symlink and `..` escapes are blocked, not just lexical joins.

Rejected paths return `ErrorCode::SandboxViolation` with message `"Path is outside workspace_dir (sandbox mode is enabled)."`.

To allow unrestricted filesystem access:

```cpp
cfg.sandbox_filesystem = false;
```

`run_command` is a shell, so path checks cannot contain it. It runs under an **OS sandbox**
(`Config::tool_sandbox`, on by default): on Linux, Landlock limits it to the workspace (read-write),
the system directories (read-only) and `/tmp`, absolute paths and symlinks included, while
`sandbox_filesystem` is true; seccomp can block the network (`tool_sandbox.network =
SandboxNetwork::Deny`; allowed by default). See [../security-sandbox.md](../security-sandbox.md).

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
    SandboxPolicy      command_sandbox;          // run_command's OS sandbox (get_terminal_tool(ctx))
};

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
Tool              get_terminal_tool   (std::shared_ptr<WorkspaceContext> ctx); // sandboxed per ctx
Tool              get_terminal_tool   (const std::string *workspace_dir);       // no OS sandbox (legacy)

}
```

`session.tool_context` holds the `shared_ptr<WorkspaceContext>` that keeps the context alive while the session exists.
