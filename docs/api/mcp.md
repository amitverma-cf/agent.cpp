# MCP client

`#include <agent-cpp/agent.hpp>`

This page is the reference for connecting Model Context Protocol servers. agent.cpp includes a small
[Model Context Protocol](https://modelcontextprotocol.io) client for the **stdio** transport. The server runs as a child
process, and the two sides exchange newline-delimited JSON-RPC 2.0 messages on its stdin and stdout. Each tool the server
offers is registered on the Session as an ordinary `Tool` named `<server>.<tool>`. So `allowed_tools`, the permission gate
(`Config::permission_check` / `permission_prompt`), graph Tool nodes and `interrupt_before_tools` all work on MCP tools
exactly as on built-in ones.

The client is compiled when `AGENT_WITH_MCP` is on (the default). Build with `-DAGENT_WITH_MCP=OFF` to leave it out. It
uses only the operating system's process and pipe APIs plus simdjson. It has no network code and no other dependencies.

---

## Quick start

```cpp
agent::McpServerConfig fs{.name = "fs", .command = "npx",
                          .args = {"-y", "@modelcontextprotocol/server-filesystem", "."}};
auto tools = agent::connect_mcp_server(session, fs);
if (!tools.ok) { /* tools.error.message says what went wrong, including the server's stderr */ }
// session now has "fs.read_text_file", "fs.list_directory", ...
auto listing = agent::execute_tool(session, "fs.list_directory", R"({"path": "."})");
```

---

## API

```cpp
struct McpServerConfig {
    std::string name;                        // tool prefix: 1-64 letters, digits, '_' or '-'; unique per Session
    std::string command;                     // program, looked up on PATH; not run through a shell
    std::vector<std::string> args;
    std::map<std::string, std::string> env;  // added to / overriding the inherited environment
    int timeout_ms = 30000;                  // per request (initialize, each tools/list page, tools/call)
    std::string working_directory;           // empty = inherit the host's
    SandboxPolicy sandbox;                   // OS sandbox, off by default (opt-in)
};

Result<std::vector<Tool>> connect_mcp_server(Session &session, const McpServerConfig &config);
Result<void> disconnect_mcp_server(Session &session, std::string_view name);
void disconnect_mcp_servers(Session &session);
std::vector<std::string> list_mcp_servers(const Session &session);   // names, in connection order
```

`connect_mcp_server` does the following, in order:

1. Starts `command args...` with pipes for stdin, stdout and stderr. On POSIX it uses `fork` and `execve`, and looks the
   program up on `PATH` itself. On Windows it uses `CreateProcessW`, with arguments quoted by the MSVC rules.
2. Sends `initialize` (protocol version `2025-06-18`, no client capabilities), then the `notifications/initialized`
   notification. If the server answers `initialize` with a JSON-RPC error (servers that predate the handshake, or
   stateless ones that drop it), the client logs a warning and carries on without the handshake.
3. Calls `tools/list`, following `nextCursor` until the last page. A cursor that repeats is an error.
4. Registers one `Tool` per remote tool. The description comes from `description`, falling back to `title`. The schema is
   `inputSchema`. Then it calls `rebuild_tool_index`. The tool names must not already be registered.

It returns copies of the registered `Tool`s. Their string views point into storage owned by the connection, so they stay
valid until the server is disconnected.

`disconnect_mcp_server` unregisters the server's tools and shuts it down. Destroying the Session does the same for every
server.

---

## Tool calls

A tool callback sends `tools/call` with `{"name": <remote tool>, "arguments": <args>}` and waits for the matching response.

- **Arguments** must be a JSON object. Empty means `{}`. They are parsed and re-serialised in minified form, because a
  newline inside a message would break the line-based framing. Invalid JSON, or a value that is not an object, fails with
  `ParseError` before anything is sent.
- **Result `content`** becomes a `ToolResult`. MCP tools are `Tool::parts_callback` tools, see
  [tools.md](tools.md#tools-that-return-parts).
  - `text`: one line of `ToolResult::text` per item.
  - `image` / `audio`: the base64 `data` is decoded and saved as `<workspace>/artifacts/mcp-<server>-<tool>-<ms>-<n>.<ext>`
    (an absolute path). The extension comes from `mimeType`. The file is added to `ToolResult::media`. Inside `run_turn`
    the file goes on the tool message, so a vision or audio model sees it on the next inference. The text gets
    `[attached: /path/file.png (image/png)]`, which is also what the string `execute_tool` returns. Undecodable data
    becomes a text note instead.
  - `resource`: its `text` is used, or its `blob` is saved like an image. `resource_link` becomes
    `[resource link name: uri]`.
  - If there is no content, the text is `structuredContent` as JSON.
- **`isError: true`** sets `ToolResult::is_error`. The media are kept. The string `execute_tool` then fails with
  `ErrorCode::ToolError` and the content text as the message. Inside `run_turn`, the model sees `Error: <text>`.

While waiting for its response, the client also handles other traffic. It answers server-to-client `ping` requests. It
rejects other server requests (sampling, roots, elicitation) with `-32601`, because it advertises none of those
capabilities. It writes `notifications/message` to the log and ignores other notifications. A response with an id that
does not match is skipped (see timeouts below).

### Errors

| Situation | Result |
|---|---|
| Program not found, cannot be executed, bad `working_directory` | `connect_mcp_server` fails with `ProviderInitFailed` |
| Server exits during the handshake | `ProviderInitFailed`. The message includes the exit code and the tail of its stderr. |
| Server exits during a call | `NetworkError`: `mcp server 'x' exited while waiting for tools/call (exit code 3); stderr: ...`. Later calls fail at once with `... is not running`. |
| No response within `timeout_ms` | `NetworkError` (`... timed out after N ms`). The client sends `notifications/cancelled` for that request id. The connection stays usable, and the late response is discarded when it arrives. |
| A stdout line that starts with `{` but is not valid JSON | `ParseError` for the pending request (`... sent invalid JSON: ...`) |
| A stdout line that does not start with `{` (for example a launcher banner) | Ignored with a warning in the log. The spec forbids such lines, but some wrappers print them. |
| JSON-RPC error response | `HttpError`, with the server's message and code |
| One stdout line longer than 64 MiB | `NetworkError` (`... sent a line longer than ...`). The server is marked dead and its calls fail from then on. |

There is no dedicated `ErrorCode` for MCP failures. `NetworkError` means the transport failed. `HttpError` means the
remote side reported an error. `ParseError` means the server's output was malformed.

---

## Secrets

An MCP server often needs a credential in its environment. Write a placeholder, not the value, in `env`:

```cpp
agent::McpServerConfig gh{.name = "github", .command = "npx",
                          .args = {"-y", "@modelcontextprotocol/server-github"},
                          .env = {{"GITHUB_TOKEN", "${secret:github_token}"}}};
```

- `Config::secret_resolver(text, who, secret_values, user_data)` expands each `${secret:NAME}` when the server starts.
  Your resolver decides whether the secret exists and whether it is bound to that consumer (`"mcp:github"` here). The
  resolver appends the values it inserted to `secret_values`.
- Placeholders are read from `env` only. A `${secret:...}` in `args` is refused, because command lines are visible to
  every process of the user.
- Without a resolver, an `env` value that names a secret makes `connect_mcp_server` fail.
- The secret values are kept out of the server's stderr log and out of tool results (they are replaced with `***`).

`Config::mcp_launcher` is called on a copy of every MCP server's config just before it starts, after secrets are resolved.
The host can rewrite `command` there, for example to run `npx` or `uvx` from its own private runtime.

---

## Process, threads and logging

- **Ownership.** Each connection is an internal `McpServer` object held by a `shared_ptr`. It is stored in
  `Session::retained_data_sources` as a `DataSource` named `mcp:<server>`. The Session owns the process, and moving the
  Session moves only the pointer. Tool `user_data` points at per-tool records owned by that object. Do not copy MCP tools
  into another Session's `Config::tools`. The `DataSource`'s `query` also works: `{"name": "<tool>", "arguments": {...}}`
  performs a `tools/call`.
- **Thread safety.** Requests to one server are serialised by a mutex, so parallel graph branches or threads can call its
  tools concurrently. Different servers are independent. `disconnect_mcp_server` waits for an in-flight call to finish.
- **stderr** is drained by one background thread per server. Each line goes to `Config::logger` and to the session log
  file, at `Info` level with the prefix `[mcp:<server>]`. The last 2 KiB are kept for error messages.
- **Shutdown** follows the stdio transport spec. The client closes the server's stdin and waits up to `min(timeout_ms, 2000)`
  ms for it to exit. Then it sends `SIGTERM` (on Windows, `TerminateProcess`), waits again, and finally sends `SIGKILL`.
- **SIGPIPE.** Writing to a server that has died does not raise `SIGPIPE` in the host. The signal is blocked on the writing
  thread and the pending signal is consumed. The write then fails with `NetworkError`.
- **Sandbox.** `McpServerConfig::sandbox` is off by default. Servers like the filesystem server, or anything launched through
  `npx` or `uvx`, need broad access: their runtime, package caches, and the directories they serve. When enabled, the
  program's directory is read-only and `working_directory` is read-write. Everything else must be listed in
  `sandbox.read_only_paths` / `read_write_paths`. `sandbox.network = SandboxNetwork::Deny` keeps a local server offline.
  Sandboxed servers run in their own process group, and shutdown signals the whole group. What the kernel could not enforce
  is logged as a warning. See [../security-sandbox.md](../security-sandbox.md).
  In `agent.json`, the same policy is the per-server `"sandbox"` object ([agent-json.md](agent-json.md#mcp_servers)).
- **Path grants.** A sandboxed server's access to paths outside its own sandbox goes through the same grants and gate as the
  file tools. The consumer name is `mcp:<server name>`. See [tools.md](tools.md#path-grants).
- **Windows** support (`CreateProcessW`, an environment block merged with the parent's, polled anonymous pipes) is run by the
  Windows test suite on Windows 11 (the project has no Windows CI).

## Servers reached by URL (Streamable HTTP)

A server can be given as an address instead of a program: `McpServerConfig::url` (and `headers`, for example
`{"Authorization", "Bearer ${secret:github}"}`; the secret is resolved like an env value). Nothing is started: every JSON-RPC message is an
HTTP POST to the address with `Accept: application/json, text/event-stream`. The reply is either one JSON document or a stream of
events (`text/event-stream`), and the response with the request's id is read out of it. The `Mcp-Session-Id` the server hands out on
`initialize` and the negotiated `MCP-Protocol-Version` are sent with every later request; a 404 after a session started ends the session.

The core has no TLS, so the host provides the POST: `Config::http_post` (`HttpPostFn`, returning an `HttpReply` with the status, the
`Content-Type`, the `Mcp-Session-Id` and the body). The CLI uses `agent::hub::http_post`; a program that links `agent_hub` can do the same:

```cpp
cfg.http_post = [](std::string_view url, const std::vector<std::pair<std::string, std::string>> &headers, std::string_view body,
                   int timeout_ms, void *) -> agent::Result<agent::HttpReply> {
    auto r = agent::hub::http_post(url, headers, body, agent::hub::http_options_from_env(), timeout_ms / 1000);
    if (!r.ok) return agent::fail<agent::HttpReply>(r.error.code, r.error.message);
    return agent::ok(agent::HttpReply{r.value.status, r.value.content_type, r.value.session_id, r.value.body});
};
agent::McpServerConfig remote;
remote.name = "docs";
remote.url = "https://mcp.example.com/mcp";
remote.headers["Authorization"] = "Bearer ${secret:docs}";
agent::connect_mcp_server(session, remote);
```

In the CLI's `mcp.json` a remote server is `"docs": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ${secret:docs}"}}`.
The client speaks the stateless 2026-07-28 revision: it probes `server/discover` and, when the server lists that version, sends no
`initialize`; every request carries version and client info in `_meta`, and HTTP requests carry `Mcp-Method` / `Mcp-Name`. A server that
answers the probe with anything else is handled with the earlier handshake (2025-06-18, session id).

## Resources and prompts

A server that offers resources or prompts gets up to four more tools next to its own: `<server>.list_resources`,
`<server>.read_resource` (`{"uri": ...}`), `<server>.list_prompts` and `<server>.get_prompt` (`{"name": ..., "arguments": {...}}`). The
answer is the server's JSON as it came (at most 20,000 characters), so the model reads a document or fetches a prompt on demand.

## The agent as an MCP server

`agent mcp serve [--workspace DIR] [--read-only]` serves the built-in tools (files of the workspace, `run_command`) over stdin and stdout, so
an editor or another agent can use them (JSON-RPC, one message per line; `initialize`, `ping`, `tools/list`, `tools/call`). The client decides
what to approve; `--read-only` offers only the tools that change nothing. See [../cli.md](../cli.md#agent-mcp).

## Not supported yet

Sampling, roots, elicitation, progress notifications, `notifications/tools/list_changed` (reconnect to refresh the tools), paginated
resource and prompt lists (the first page is shown), `subscriptions/listen`, multi round-trip input requests (an `input_required` result is an error) and the tasks extension.
