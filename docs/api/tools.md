# Tools

`#include <agent-cpp/agent.hpp>`

This page is the reference for tools: how to define one, how the session registers and dispatches it, how the
permission gate and path grants limit it, and what the default tool set does. Tools are callable functions the LLM can
invoke by name. When the model emits a tool call in its response, `run_turn` dispatches it automatically, feeds the result
back as a `"tool"` message, and continues the inference loop.

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

Set exactly one of `callback` (text result) and `parts_callback` (a `ToolResult`). `execute_tool` fails with `InvalidConfig`
when neither or both are set. The `string_view` fields must stay valid for as long as the session exists. String literals
and static data are safe.

### CallbackFn

The callback receives raw JSON arguments and a `user_data` pointer. It returns one of:

- `ok(std::string result)`: the result is inserted into the conversation as a `"tool"` message.
- `fail(code, message)`: the text `"Error: <message>"` is inserted instead, and the turn continues.

Tool callbacks **must not** call `session.arena.reset()` or modify `session.config.tools` without calling
`rebuild_tool_index` afterward.

Always invoke a tool through `agent::execute_tool` or `execute_tool_parts`, never through the function pointers directly.
Some built-in tools (`use_skill`, `compress_context`) use a `Session*` cell that `execute_tool` refreshes right before
dispatch. A direct call can run against a stale Session pointer. It also skips the `allowed_tools` check and the permission
gate.

### Tools that return parts

A tool that produces files (an image, a recording, a mesh) or tensors sets `parts_callback` instead of `callback`, and
returns a `ToolResult`:

```cpp
struct ToolResult {
    std::string             text;
    std::vector<Media>      media;     // {path, mime}: images, audio, video, any file
    std::vector<TensorData> tensors;
    bool                    is_error = false;
};
```

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

Inside `run_turn`, and in graph Agent nodes (including tool calls approved on `resume_graph`):

- `media` and `tensors` go on the `"tool"` `Message` (`Message::media` / `Message::tensors`). The media persist with the
  history (`save_conversation`, graph checkpoints). They reach the engine as `FILE` parts with every later inference of that
  conversation, so a vision model can look at an image a tool produced. Relative paths are made absolute.
- The message text is `text` followed by one `[attached: <path> (<mime>)]` line per media item, and
  `[tensor: <name> [dims]]` per tensor. Models without that input modality still learn about the file.
  `ToolHook::Post`, `EventType::OnToolResult` and the graph `ToolResult` event see this text. The event payload keeps the
  attachment lines when it clips the text.
- Media are sent only to an engine that declares the matching input capability: `image/*` needs `AGENT_CAP_VISION_IN`,
  `audio/*` needs `AGENT_CAP_AUDIO_IN`, and `video/*` needs `AGENT_CAP_VIDEO_IN`. Other files stay in the history but out of
  the request. If the engine fails a request that carries tool media (for example, a llama model loaded without `mmproj`),
  the request is retried once without it. The Session then stops sending tool media, and a warning is logged.
- `is_error = true`: the tool ran but failed. The conversation gets `Error: <text>`, and the media are still attached. A
  failed `Result` from `parts_callback` is reported like a failed `callback`.

### Calling tools directly

```cpp
// Text + "[attached: ...]" lines; is_error -> ErrorCode::ToolError.
Result<std::string> execute_tool(Session &, std::string_view name, std::string_view arguments);
// The whole ToolResult; is_error comes back as ok() with is_error set.
Result<ToolResult>  execute_tool_parts(Session &, std::string_view name, std::string_view arguments);
```

Both run the `allowed_tools` check (graph Agent nodes), the permission gate, and the tool hooks.
A graph `Tool` node uses the string form: its `output_key` gets the text with the attachment lines, and an `is_error` result
fails the node.

---

## Registration

Tools are registered through `Config::tools` at session creation:

```cpp
agent::Config cfg;
cfg.tools = { my_tool_a, my_tool_b };
auto sess = agent::init(cfg);
```

When `auto_default_tools = true` (the default), `init()` also adds the default tools listed below. If a user-provided
tool has the same `name` as a default tool, the user tool takes priority.

To add tools after `init()`:

```cpp
session.config.tools.push_back(my_new_tool);
agent::rebuild_tool_index(session);   // required after mutating tools
```

### O(1) dispatch

Tools are dispatched through `session.tool_by_name`, an `unordered_map<string, size_t>` that maps a tool name to its index
in `session.config.tools`. `rebuild_tool_index` (re)builds this index. `init()` calls it automatically. Call it again after
changing `session.config.tools`. Per-node `allowed_tools` (graph Agent nodes) does not touch the index. It is a
thread-local allowlist checked by `execute_tool`.

### MCP servers

`connect_mcp_server(session, McpServerConfig{...})` starts a Model Context Protocol server over stdio. It registers each of
its tools as `<server>.<tool>`. They go through the same permission gate as every other tool. See [mcp.md](mcp.md).

---

## Permissions

`Config::permission_check` decides, per call, whether a tool may run. It is called with the tool name and the raw JSON
arguments:

```cpp
enum class PermissionDecision { Allow, Ask, Deny };
using PermissionCheckFn  = PermissionDecision (*)(std::string_view tool_name, std::string_view arguments, void *user_data);
using PermissionPromptFn = bool (*)(std::string_view tool_name, std::string_view arguments, void *user_data);

cfg.permission_check       = my_policy;   // nullptr = Allow everything (the default)
cfg.permission_prompt      = my_prompt;   // resolves Ask: return true to approve
cfg.permission_user_data   = &state;
```

- `Allow` runs the tool. `Deny` fails the call with `PermissionDenied`.
- `Ask` calls `permission_prompt`. If `permission_prompt` is `nullptr`, `Ask` is treated as `Deny`.

`tools::default_permission_policy` is an opt-in example. It returns `Ask` for `run_command`, `delete_path` and `move_path`,
and `Allow` for everything else. It is not wired in automatically. Set it on `permission_check` if you want it.

In a graph, `interrupt_before_tools` turns an `Ask` decision into a pause for human approval. See
[graph.md](graph.md#interrupts-human-in-the-loop).

### Path grants

Sandboxed tools cannot touch paths outside the workspace and the system locations. A **path grant** lets a tool use a path
it would otherwise be refused. Without a grant, the host's gate (if it set one) is asked. The sandbox is never switched
off. The host decides, per path and per kind of access, and can remember the answer.

```cpp
struct PathGrant {
    std::string path;           // a file or a folder (covers everything below it)
    bool write = false;         // false = read only; true = read and write
    std::string consumer = "*"; // "*", "files" (the file tools), "run_command" or "mcp:<server name>"
};
struct PathAccessRequest {
    std::string_view consumer;
    std::string_view path;      // absolute
    bool write;
};
enum class PathGateDecision { Deny, AllowOnce, AllowAlways };
using PathGateFn = PathGateDecision (*)(const PathAccessRequest &request, void *user_data);

cfg.path_grants      = { {.path = "/srv/reference", .write = false, .consumer = "*"} };
cfg.path_gate        = my_gate;      // asked for any other path; nullptr = only the grants apply
cfg.path_gate_user_data = &ui;
```

- `AllowAlways` is added to the session's own copy of the grants. Persisting it is up to the host.
- `AllowOnce` lets this one access through.
- `Deny` refuses the access.

---

## Custom tool example

```cpp
struct DbCtx {
    std::string connection_string;
};

DbCtx db_ctx{ .connection_string = "postgresql://localhost/mydb" };

// Your JSON parser: returns the "sql" field of the arguments object, or "" when it is missing.
std::string parse_sql(std::string_view arguments);

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
        auto *ctx = static_cast<DbCtx *>(ud);
        std::string sql = parse_sql(args);
        if (sql.empty())
            return agent::fail<std::string>(agent::ErrorCode::ParseError, "Missing 'sql'");

        // run the query against ctx->connection_string ...
        return agent::ok(std::string(R"([{"id":1,"name":"Alice"}])"));
    },
    .user_data = &db_ctx,    // must outlive the session
};

cfg.tools = { query_db };
```

---

## Default tools

When `Config::auto_default_tools = true` (the default), `init()` registers these tools. Each one can be called by the
model, or from code through `execute_tool`.

| Tool | Registered when | Purpose |
|---|---|---|
| `read_file`, `write_file`, `append_file`, `list_dir`, `create_dir`, `delete_path`, `move_path`, `file_info` | always | File operations, relative to `workspace_dir` |
| `run_command` | always | Shell command in `workspace_dir` under an OS sandbox |
| `compress_context` | always | Summarises older history |
| `use_skill` | `Config::skills_dir` is set | Loads a skill's instructions (see [Skill loader](#skill-loader)) |
| `tool_search`, `load_tool` | `Config::lazy_tools` is true | Loads tools on demand (see [Lazy tools](#lazy-tools)) |

### Filesystem tools

All paths resolve relative to `workspace_dir`. Absolute paths are accepted, but they are checked against the sandbox when
`Config::sandbox_filesystem = true`.

#### `read_file`

Read a file. A `parts_callback` tool.

```json
{
  "path": "src/main.cpp",
  "max_bytes": 16384,
  "offset": 0
}
```

- **Text files**: the content from `offset` (default 0), at most `max_bytes` (default 16384, clamped to 4 MiB). The cut
  never splits a UTF-8 character. Invalid UTF-8 bytes become U+FFFD. A longer file gets
  `\n[truncated at N bytes]\n[T bytes in total; read more with "offset": K]`.
- **Images, audio, video** (by extension: png, jpg/jpeg, gif, webp, bmp, tga, wav, mp3, flac, ogg, opus, m4a, mp4, webm,
  mov, mkv, avi): returned as `ToolResult::media`, with the text `image file cat.png (N bytes), attached`. A vision or audio
  model then looks at or listens to the file on its next inference (the `view_image` use case). A file outside the
  workspace (possible only with `sandbox_filesystem = false`) is first copied to
  `<workspace>/artifacts/read-<ms>-<name>`, because a sandboxed out-of-process engine can only read the workspace.
- **Other binary files** (containing NUL bytes): `[binary file <name>, N bytes; not shown as text]`.
- A directory fails. Use `list_dir` instead.

Call it through `execute_tool` or `execute_tool_parts`. `read_file` has no string `callback`.

#### `write_file`

Write (overwrite) a file. Creates parent directories by default (`create_dirs`, default `true`).

```json
{
  "path": "output/report.md",
  "content": "# Report\n\nHello world.",
  "create_dirs": true
}
```

Returns `"wrote N bytes to /path/to/file"`. A missing (or non-string) `content` fails instead of truncating the file.

#### `append_file`

Append text to a file. Creates the file and its parent directories if they do not exist.

```json
{
  "path": "logs/debug.log",
  "content": "2025-01-01: agent started\n"
}
```

Returns `"appended N bytes to /path/to/file"`.

#### `list_dir`

List directory contents. Returns a JSON array of `{name, type, size?}` objects, sorted by name. Omit `path` to list the
workspace root.

```json
{ "path": "src" }
```

Returns `[{"name":"main.cpp","type":"file","size":1234},{"name":"utils","type":"dir"}]`. At most 1000 entries. A larger
directory ends with `{"truncated":true,"shown":1000}`.

#### `create_dir`

Create a directory and all missing parents.

```json
{ "path": "output/reports/2025" }
```

Returns `"created /workspace/output/reports/2025"`.

#### `delete_path`

Delete a file or directory tree recursively. It cannot delete the workspace root.

```json
{ "path": "tmp/scratch" }
```

Returns `"deleted N entries at /path/to/scratch"`.

#### `move_path`

Move or rename a file or directory. It cannot move the workspace root.

```json
{ "from": "draft.md", "to": "published/final.md" }
```

Returns `"moved /a -> /b"`.

#### `file_info`

Get metadata for a path.

```json
{ "path": "models/my.gguf" }
```

Returns `{"exists":true,"type":"file","size":4294967296,"path":"/workspace/models/my.gguf"}`, or `{"exists":false}`.

### Terminal tool

#### `run_command`

Run a shell command in `workspace_dir`. stdout and stderr are merged. Output is capped at 32 KiB.

```json
{
  "command": "git log --oneline -5",
  "timeout_seconds": 30
}
```

Returns a JSON object:

```json
{"exit_code":0,"timed_out":false,"cancelled":false,"output":"abc1234 fix bug\ndef5678 add feature\n..."}
```

- `timed_out` is `true` when the timeout stopped the command. On POSIX, `exit_code` is then 124, as `timeout(1)` reports.
- `cancelled` is `true` when `request_cancel()` stopped it. The command and its child processes are stopped, and the call
  returns with the flag set.
- `sandbox_warning` is present when the OS sandbox is weaker on this machine than configured.
- Invalid UTF-8 in the output, including a character cut by the 32 KiB cap, becomes U+FFFD.

`timeout_seconds` defaults to 30. A negative value means 30, and values are capped at one day. `0` means no timeout on every
platform. On POSIX the command runs in its own process group, so a timeout or cancel stops the whole tree.

### Context compressor

#### `compress_context`

Summarises the active conversation's older history into one system message (`[Compressed history]: <summary>`), to free
context window space. The model can call it. It acts on the conversation that issued the call. In a graph, that is the
node's history, shared ones included.

`run_turn` also compresses automatically at the start of a turn when the history exceeds 70% of `context_window`, keeping
the 6 most recent messages. `agent::compress_context(session, memory, keep_recent)` does the same directly.

```json
{ "keep_recent": 6 }
```

Returns `"Context compressed."` on success, including when there was nothing to compress.

- The summary is written by the Session's own engine as a plain chat request (`AGENT_CAP_TEXT_GEN`, whatever
  `Config::default_task` is, without tool definitions). It runs in a separate arena, so it works from inside a tool call and
  with process-isolated engines. An engine that cannot chat, or a failed request, skips compression. The pruner still trims
  the history.
- An assistant tool call and its results are never separated. The assistant message whose tool call is running stays in
  place.
- Media in the summarised part are named in the summary prompt (`[file: <path> (<mime>)]`). The files themselves leave the
  context.
- `run_turn` sends the leading system messages (system prompt, summary, injected tool list) to the engine as **one** system
  message, and the pruner never drops them. A graph node's `system_prompt` goes in front of a summary instead of replacing
  it. A later compression folds the previous summary into the new one.

### Lazy tools

With `Config::lazy_tools = true`, the model starts with only `tool_search`, `load_tool` and `use_skill`. Every other tool
stays out of the prompt until the model loads it. This saves the prompt tokens of tools that are never used.

- `tool_search` takes an optional `query`, a word to match against tool names and descriptions. It returns one line per tool:
  `- name: description`, with `(loaded)` after the tools already loaded. Tools that the node's `allowed_tools` omit are not
  listed.
- `load_tool` takes `name` (one tool) or `names` (an array). It adds the tool to the conversation's
  `Conversation::active_tools`. The tool is advertised from the next request on, and stays loaded for the rest of the
  conversation, across context compression.
- Calling a tool that is not loaded fails with `KeyNotFound`, and the message tells the model to use `tool_search` and then
  `load_tool`.

In a graph Agent node, lazy tools apply only when `allowed_tools` includes `tool_search`. A node whose `allowed_tools` omits
it keeps its fixed tool list.

### Skill loader

#### `use_skill`

Registered automatically when `Config::skills_dir` is set (see [Skills](../guide.md#skills)). It loads the full `SKILL.md`
content for a named skill, scanned earlier by `load_skills_dir`, or a file bundled with it.

```json
{ "name": "pdf-tools" }
{ "name": "pdf-tools", "file": "references/forms.md" }
```

Returns `Skill directory: <absolute path> ...`, followed by the skill's full `SKILL.md` content. With `file`, it returns that
file's text (up to 256 KiB). For png, jpg, gif, webp, wav and mp3 files, it returns the file as media.

`file` must resolve inside the skill directory, or the call fails with `SandboxViolation`. It returns `KeyNotFound` if the
name does not match a loaded skill. In a graph Agent node, the skills list is advertised only when `use_skill` is in the
node's `allowed_tools`, or when that list is empty. `SKILL.md` frontmatter values may be quoted.

---

## Sandbox

When `Config::sandbox_filesystem = true` (the default), the filesystem tools enforce that all resolved paths stay inside
`workspace_dir`. The check resolves through `hal::resolve_under_workspace` (`weakly_canonical` plus a prefix-path
comparison). So symlink and `..` escapes are blocked, not just lexical joins.

Rejected paths return `ErrorCode::SandboxViolation` with the message `"Path is outside workspace_dir (sandbox mode is enabled)."`.
A path grant or the permission gate can allow a specific path (see [Path grants](#path-grants)).

To allow unrestricted filesystem access:

```cpp
cfg.sandbox_filesystem = false;
```

`run_command` is a shell, so path checks cannot contain it. It runs under an **OS sandbox** (`Config::tool_sandbox`, on by
default). On Linux, Landlock limits it to the workspace (read-write), the system directories (read-only) and `/tmp`. Absolute
paths and symlinks are included, while `sandbox_filesystem` is true. seccomp can block the network
(`tool_sandbox.network = SandboxNetwork::Deny`; allowed by default). On Windows the sandbox is a Job Object, which is
untested and has no filesystem or network rules. See [../security-sandbox.md](../security-sandbox.md).

## Disabling default tools

```cpp
cfg.auto_default_tools = false;
cfg.tools = { my_custom_tool };   // only this tool is registered
```

## Tool factory functions

You can create individual default tools manually, for example with a custom workspace context or to add them to a
different session:

```cpp
namespace agent::tools {

struct WorkspaceContext {
    const std::string *workspace_dir = nullptr;
    bool sandbox = true;
    SandboxPolicy command_sandbox;                  // run_command's OS sandbox (get_terminal_tool(ctx))
    const volatile int32_t *cancel = nullptr;       // polled by run_command; non-zero stops it
    std::shared_ptr<std::vector<PathGrant>> grants; // the session's path grants
    PathGateFn gate = nullptr;
    void *gate_user_data = nullptr;
};

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
Tool              get_terminal_tool   (std::shared_ptr<WorkspaceContext> ctx); // sandboxed per ctx
Tool              get_terminal_tool   (const std::string *workspace_dir);       // no OS sandbox (legacy)

PermissionDecision default_permission_policy(std::string_view tool_name, std::string_view arguments, void *user_data);

}
```

`session.tool_context` holds the `shared_ptr<WorkspaceContext>` that keeps the context alive while the session exists. The
context must outlive any tool built from it.
