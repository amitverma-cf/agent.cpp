# Remote models (the `api` engine)

The `api` engine runs a chat against a remote OpenAI-compatible service (`POST {base_url}/chat/completions`) instead of a model file.
It is optional: the plugin is `agent-engine-api`, it needs the bun runtime, and it is the only engine that sends the conversation off the
machine. No provider SDK and no HTTP client is linked into C++. The plugin starts `bun bridge.js` as a child process and exchanges one JSON
object per line over its stdin and stdout; the script (embedded in the plugin and written to the files folder on first use) makes the
requests with plain `fetch` and has no npm dependencies.

## Add a model

```sh
agent runtime install bun                 # once; `agent setup` offers it as well
export NUMES_API_KEY=...                  # the key stays in the environment, never in a file
agent model add-api numes-glm --base-url https://api.numes.ai/api/v1 --model dealignai/glm-5.3-flash-uncensored-fp8 --key-env NUMES_API_KEY
agent chat --model numes-glm
```

`add-api NAME` writes `<models>/api/NAME.api.json` (base URL, the provider's model id, the name of the key variable, optionally
`--max-tries N`) and a `models.json` entry that points at it. The model then appears in `agent model ls`, in `/model` and in the
Model setting of the chat like any other, and `agent model rm NAME` removes it. `--key-env` defaults to `NUMES_API_KEY`.

## Options

The descriptor is the engine's `engine_options`. Any key can also be given or overridden with `--engine-options`.

| Option | Default | Meaning |
|---|---|---|
| `base_url` | required | `https://host/path/v1`; `/chat/completions` is appended |
| `model` | required | the provider's model id |
| `api_key_env` | `NUMES_API_KEY` | name of the environment variable that holds the key |
| `timeout_ms` | `90000` | per attempt, how long the provider may stay silent (restarts whenever data arrives) |
| `max_tries` | `6` | attempts per request |
| `retry_base_ms`, `retry_max_ms` | `1000`, `20000` | first backoff, doubled per retry up to the cap, then jittered to 50 to 100 percent |
| `stream` | `true` | `false` asks for one non-streamed reply |
| `extra_body` | none | object merged into the request body (provider-specific fields) |
| `bun` | `bun` on PATH | the bun executable; the CLI passes its private install (`agent runtime path bun`) |
| `bridge_dir` | the scratch folder | where the bridge script is written; the CLI uses `<home>/runtimes/api-bridge` |

## What is sent and read

Messages, tool definitions (`tools`, as OpenAI `function` entries), `max_tokens` and `temperature` go in the request body. A streamed
reply (server-sent events) is forwarded token by token; a provider that ignores `stream` and returns one JSON message works too. Tool
calls are accumulated by index, missing ids are generated, and the arguments must be a JSON object. `usage` (including cached prompt
tokens) is read when the provider reports it. Fields the engine does not know, in requests and in replies, are ignored. The `reasoning` /
`reasoning_content` field is not shown: only the answer text reaches the conversation. Images and other media are not sent.

## Flaky providers

Free and busy endpoints fail often, so every request is retried up to `max_tries` times with exponential backoff and jitter on:
HTTP 408, 425, 429 and 5xx, a time-out, a connection error, a body that is not JSON or has no choices, an empty reply (no text and no
tool call), tool-call arguments that are not a JSON object, and a stream that ends before the finish marker. Other errors (400, 401, 403,
404, a reply cut off by the token limit before any text) are reported at once. A bridge that died or hung is started again.

The conversation is untouched by a failed attempt: the same request is sent again. When part of an answer was already streamed to the
screen, a retry sends the answer again from the start and only the text beyond what was shown is streamed, so nothing appears twice. The
per-turn footer shows the number of retries (`| 2 retries`) when there were any. The key is read by the bridge from its environment,
used only in the `Authorization` header and removed from every error text. A cancel aborts the HTTP request.

## Sandbox and Local only

The engine host sandbox (Landlock and seccomp on Linux, an AppContainer on Windows) denies network sockets and starting programs, which is
right for engines that only run a model file and is exactly what this engine needs. The `api` engine therefore does not use the host: the
CLI loads it in process (`isolation: inprocess`). Nothing is lost for it: it holds no model weights, reads no model files, and the code
in the process is a small JSON and process-control layer; the part that talks to the network is a separate bun process. The environment of
that child is the CLI's own, so it sees the key variable and nothing else it should not. The bridge script is plain text next to the other
runtime files and can be read before it is used.

With `local_only` on (Settings), the CLI refuses to load the engine with a message, and the plugin refuses on its own when it is told
`"local_only": true`. `agent model add-api` itself only writes two small files.

## Limits

Chat only (text generation with tool calls). No vision input, no audio, no embeddings. The bridge needs bun; a Node or uv setup does not
substitute. A key kept in the OS secret store is not read: the key has to be in the environment variable named by `--key-env`. Windows
support is written but has not been tested.
