# MCP client

`#include <agent-cpp/agent.hpp>`

agent.cpp includes a small [Model Context Protocol](https://modelcontextprotocol.io) client for the **stdio** transport. The server runs as a child process, and the two sides exchange newline-delimited JSON-RPC 2.0 messages on its stdin and stdout. Each tool the server offers is registered on the Session as an ordinary `Tool` named `<server>.<tool>`. That means `allowed_tools`, the permission gate (`Config::permission_check` / `permission_prompt`), graph Tool nodes and `interrupt_before_tools` all work on MCP tools exactly as on built-in ones.

The client uses only the operating system's process and pipe APIs plus simdjson. It has no network code and no other dependencies.

---

## API

```cpp
struct McpServerConfig {
    std::string name;                        // tool prefix: letters, digits, '_' or '-'; unique per Session
    std::string command;                     // program, looked up on PATH; not run through a shell
    std::vector<std::string> args;
    std::map<std::string, std::string> env;  // added to / overriding the inherited environment
    int timeout_ms = 30000;                  // per request
    std::string working_directory;           // empty = inherit
    SandboxPolicy sandbox;                   // OS sandbox, off by default (opt-in)
};

Result<std::vector<Tool>> connect_mcp_server(Session &session, const McpServerConfig &config);
Result<void> disconnect_mcp_server(Session &session, std::string_view name);
void disconnect_mcp_servers(Session &session);
std::vector<std::string> list_mcp_servers(const Session &session);
```

```cpp
agent::McpServerConfig fs{.name = "fs", .command = "npx",
                          .args = {"-y", "@modelcontextprotocol/server-filesystem", "."}};
auto tools = agent::connect_mcp_server(session, fs);
if (!tools.ok) { /* tools.error.message says what went wrong, including the server's stderr */ }
// session now has "fs.read_text_file", "fs.list_directory", ...
auto listing = agent::execute_tool(session, "fs.list_directory", R"({"path": "."})");
```

`connect_mcp_server` does the following, in order:

1. Starts `command args...` with pipes for stdin, stdout and stderr. On POSIX it uses `fork` + `execve` and looks the program up on `PATH` itself. On Windows it uses `CreateProcessW`, with arguments quoted by the MSVC rules.
2. Sends `initialize` (protocol version `2025-06-18`, no client capabilities) and then the `notifications/initialized` notification. If the server answers `initialize` with a JSON-RPC error (servers that predate the handshake, or stateless ones that drop it), the client logs a warning and carries on without the handshake.
3. Calls `tools/list`, following `nextCursor` until the last page. A cursor that repeats is an error.
4. Registers one `Tool` per remote tool. The description comes from `description`, falling back to `title`; the schema is `inputSchema`. Then it calls `rebuild_tool_index`. The tool names must not already be registered.

It returns copies of the registered `Tool`s. Their string views point into storage owned by the connection, so they stay valid until the server is disconnected.

`disconnect_mcp_server` unregisters the server's tools and shuts it down. Destroying the Session does the same for every server.

---

## Tool calls

A tool callback sends `tools/call` with `{"name": <remote tool>, "arguments": <args>}` and waits for the matching response.

- **Arguments** must be a JSON object; empty means `{}`. They are parsed and re-serialised in minified form, because a newline inside a message would break the line-based framing. Invalid JSON, or a value that is not an object, fails with `ParseError` before anything is sent.
- **Result `content`** is turned into text, one line per item:
  - `text`: the text itself.
  - `image` / `audio`: the base64 `data` is decoded and saved as `<workspace>/artifacts/mcp-<server>-<tool>-<ms>-<n>.<ext>`, where the extension comes from `mimeType`. The item becomes a note: `[image saved to /path/file.png (image/png, 1234 bytes)]`.
  - `resource`: its `text` is used, or its `blob` is saved like an image; `resource_link` becomes `[resource link name: uri]`.
  - If there is no content, the result is `structuredContent` as JSON.
- **`isError: true`** fails the call with `ErrorCode::HttpError`, and the content text becomes the message. Inside `run_turn` the model sees it as `Error: <text>`.

While waiting for its response the client also handles other traffic. It answers server-to-client `ping` requests, and rejects other server requests (sampling, roots, elicitation) with `-32601`, because it advertises none of those capabilities. It writes `notifications/message` to the log and ignores other notifications. A response with an id that does not match is skipped (see timeouts below).

### Errors

| Situation | Result |
|---|---|
| Program not found, cannot be executed, bad `working_directory` | `connect_mcp_server` fails with `ProviderInitFailed` |
| Server exits during the handshake | `ProviderInitFailed`, the message includes the exit code and the tail of its stderr |
| Server exits during a call | `NetworkError`: `mcp server 'x' exited while waiting for tools/call (exit code 3); stderr: ...`. Later calls fail at once with `... is not running` |
| No response within `timeout_ms` | `NetworkError` (`... timed out after N ms`). The client sends `notifications/cancelled` for that request id. The connection stays usable, and the late response is discarded when it arrives |
| A stdout line that starts with `{` but is not valid JSON | `ParseError` for the pending request (`... sent invalid JSON: ...`) |
| A stdout line that does not start with `{` (for example a launcher banner) | Ignored with a warning in the log. The spec forbids such lines, but some wrappers print them |
| JSON-RPC error response | `HttpError`, with the server's message and code |
| One line longer than 64 MiB | `ParseError`; the connection is closed |

There is no dedicated `ErrorCode` for MCP failures. `NetworkError` means the transport failed, `HttpError` means the remote side reported an error, and `ParseError` means the server's output was malformed.

---

## Process, threads and logging

- **Ownership.** Each connection is an internal `McpServer` object held by a `shared_ptr`. It is stored in `Session::retained_data_sources` as a `DataSource` named `mcp:<server>`. This means the Session owns the process, and moving the Session moves only the pointer. Tool `user_data` points at per-tool records owned by that object. Do not copy MCP tools into another Session's `Config::tools`. (The `DataSource`'s `query` also works: `{"name": "<tool>", "arguments": {...}}` performs a `tools/call`.)
- **Thread safety.** Requests to one server are serialised by a mutex, so parallel graph branches or threads can call its tools concurrently. Different servers are independent. `disconnect_mcp_server` waits for an in-flight call to finish.
- **stderr** is drained by one background thread per server. Each line goes to `Config::logger` and to the session log file, at `Info` level with the prefix `[mcp:<server>]`. The last 2 KiB are kept for error messages.
- **Shutdown** follows the stdio transport spec. The client closes the server's stdin and waits up to `min(timeout_ms, 2000)` ms for it to exit. Then it sends `SIGTERM` (on Windows, `TerminateProcess`), waits again, and finally sends `SIGKILL`.
- **SIGPIPE.** Writing to a server that has died does not raise `SIGPIPE` in the host. The signal is blocked on the writing thread and the pending signal is consumed. The write then fails with `NetworkError`.
- **Sandbox.** `McpServerConfig::sandbox` is off by default: servers like the filesystem server or anything launched through `npx`/`uvx` need broad access (their runtime, package caches, the directories they serve). When enabled, the program's directory is read-only, `working_directory` read-write, and everything else must be listed in `sandbox.read_only_paths` / `read_write_paths`; `sandbox.network = SandboxNetwork::Deny` keeps a local server offline. Sandboxed servers run in their own process group, and shutdown signals the whole group. What the kernel could not enforce is logged as a warning. See [../security-sandbox.md](../security-sandbox.md). In `agent.json`, the same policy is the per-server `"sandbox"` object ([agent-json.md](agent-json.md#mcp_servers)).
- **Windows** support (`CreateProcessW`, an environment block merged with the parent's, polled anonymous pipes) is written but not yet tested in CI.

## Not supported yet

Resources, prompts, sampling, roots, elicitation, progress notifications, `notifications/tools/list_changed` (reconnect to refresh the tools), and the Streamable HTTP transport.
