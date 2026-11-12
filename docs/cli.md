# The `agent` CLI

`agent` is the command-line app built on top of `libagent`. It has a setup wizard, an engine and model
manager, a text chat with streaming and tool calls, and a voice loop. It uses only `libagent`, `agent_hub` (Hugging
Face downloads), `agent_store_sqlite` (conversation storage), simdjson and miniaudio (sound devices).

```sh
cmake -B build -G Ninja            # AGENT_BUILD_CLI=ON by default (needs AGENT_BUILD_HUB, i.e. OpenSSL)
cmake --build build
./build/agent setup --yes
./build/agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
./build/agent chat
```

The CMake target is `agent_cli_bin` (the library target is already called `agent`). It produces a
binary named `agent` at the top of the build tree. The binary looks for engine plugins in these places:
next to itself, in `engines/` and its subdirectories (so the build tree works as is), and in
`../lib/agent/engines/` (an install prefix).

**Global flags.** `--no-sandbox` works with any command: it runs engines, tools and commands without the OS
sandbox. The sandbox is on by default; where the OS cannot provide one, or with `--no-sandbox`, things run
unsandboxed and a warning says so. There are no weaker fallbacks. See [security-sandbox.md](security-sandbox.md).

## Files

Everything the CLI uses lives in the global home (see [Where files live](#where-files-live)). The home is chosen at
`agent setup` and remembered in a pointer file, `home.json`, in the platform config folder (`%APPDATA%\agent\` on
Windows, `$XDG_CONFIG_HOME/agent/` or `~/.config/agent/` elsewhere).

| File | Location |
|---|---|
| `home.json` (pointer to the home) | the platform config folder (above); `--home DIR` and `$AGENT_HOME` override it for one run |
| `config.json`, `engines.json` | the home folder |
| models and `models.json` | `<home>/models/` |
| workspace (file tools, `memory/db.sqlite3`, `logs/`, `artifacts/`) | `<home>/workspaces/default/` |

The file tools are sandboxed to the workspace. An explicit `--config-dir DIR` (or `$AGENT_CONFIG_DIR`) keeps the older
layout instead: `config.json` and `engines.json` in that folder, and models and the workspace in the platform data
folders (`~/.local/share/agent/` on Linux, `%LOCALAPPDATA%\agent\` on Windows).

`config.json`:

```json
{
  "version": 1,
  "workspace": "/home/me/.local/share/agent/workspace",
  "models_dir": "/home/me/.local/share/agent/models",
  "default_model": "qwen2.5-0.5b-instruct-q4_k_m",
  "capabilities": ["llama", "audiocpp", "sdcpp", "onnx"],
  "backend": "cpu",
  "hardware": {"cpu_cores": 4, "nvidia_smi": false, "nvidia_device": false, "dri_nodes": []},
  "context_window": 4096,
  "max_tokens": 512,
  "temperature": 0.7,
  "memory_budget_mb": 0
}
```

`engines.json` is the manifest that `agent::load_engine_manifest` reads. The CLI also records each engine's
name, version, capabilities and backend, and keeps an optional `"isolation": "process" | "inprocess"`:

```json
{
  "engines": [
    {"name": "llama", "path": "/opt/agent/lib/agent/engines/agent-engine-llama.so", "version": "b12020",
     "backend": "local", "caps": ["text_gen", "tool_call", "vision_in", "audio_in"]}
  ]
}
```

A hand-written entry that has only `"path"` also works. The CLI then takes the name from the file name
(`agent-engine-<name>.so`).

## Where files live

agent.cpp keeps everything shared in one **global home**, a folder called `.agent.cpp-workspace` that you
choose during `agent setup` (default: your user folder). Each project has a small `.agents/` folder of its own.
See [workspace.md](workspace.md) for the full layout.

```
.agent.cpp-workspace/          global home
  config.json  engines.json    settings and the engine list
  engines/  models/            plugins, and model files with models.json
  skills/  mcp/  plugins/      available in every project
  runtimes/                    private bun / uv, never installed globally
  cache/  logs/                downloads in progress, GPU kernel caches, logs
  workspaces/default/          the default workspace (file tools, memory, artifacts)
<project>/.agents/             one project
  skills/  mcp.json            committed with the project
  memory/  cache/  local/      private, git-ignored
```

The home is found from `--home DIR`, then `$AGENT_HOME`, then the folder `agent setup` recorded. An explicit
`--config-dir` / `$AGENT_CONFIG_DIR` keeps the older layout (config and models in the platform folders).

## agent setup

```sh
agent setup                         # asks questions on a terminal
agent setup --yes --home ~/.agent.cpp-workspace --model qwen2.5-0.5b --model bge-small
agent setup --no-models             # layout, engines and config only
agent setup --caps llama,sdcpp --engines-dir /opt/engines --backend cpu
agent setup --yes --install-runtimes   # the private bun / uv
```

1. **Asks where the global home goes** (`--home DIR` answers it), creates its folders and remembers the choice.
2. **Detects hardware**: CPU cores, installed RAM, whether `nvidia-smi` is on `PATH`, `/dev/nvidia0`, and the
   `/dev/dri/renderD*` nodes. From that it suggests a backend: `cuda`, `vulkan` or `cpu`.
3. **Picks engines.** The default is the starter pack `llama,audiocpp,sdcpp,onnx`.
4. **Looks for each engine's plugin** (`agent-engine-<name>.so` / `.dll` / `.dylib`) in `--engines-dir`
   and the default plugin directories, loads it to check it, and records it in `engines.json`. Engines already
   listed are kept. Engines that are still missing are reported. Build them from source and register them with `agent engine add`.
5. **Shows the models this machine can run** and downloads **nothing by default**. On a terminal you choose numbers
   from the list (registry models whose engine is installed and whose size fits in 70% of RAM with room for the KV
   cache), or type any `owner/repo[:quant]`. Without a terminal, `--model NAME|owner/repo[:quant]` (repeatable) picks them.
6. **Downloads and smoke-tests each model**: it runs the real `agent task` on it (a short chat reply, an embedding, a
   rerank) and reports the speeds, so you know the engine, the model and the sandbox work together. Models that need
   audio or image input are downloaded but not smoke-tested.

## agent init

```sh
agent init            # creates .agents/ in the current folder
agent init ~/code/app
```

Creates `.agents/` with `skills/`, `memory/`, `cache/`, `local/` and a `.gitignore` that keeps the last three out of
git. The nearest ancestor of the working directory that has a `.agents/` folder is the project.

## agent doctor

```sh
agent doctor
```

Prints what is installed: the engines with their status, the private runtimes, the models, and how much disk each
area of the home uses (`models/`, `cache/`, `runtimes/`, `logs/`), with what is safe to remove. It changes nothing.

## agent clean

```sh
agent clean --downloads          # partial downloads (.part files)
agent clean --cache --yes        # rebuildable caches: saved prefix KV states, GPU kernel caches, package caches
agent clean --logs --yes         # old log files
```

Removes one area per flag. Models, memory and conversations are never removed. `--yes` skips the confirmation.

## agent secret

```sh
agent secret set OPENAI_KEY                 # the value is typed without echo; --stdin reads it from a pipe
agent secret ls
agent secret bind OPENAI_KEY mcp:github     # let that MCP server read the secret
agent secret unbind OPENAI_KEY mcp:github
agent secret rm OPENAI_KEY
```

Values go to the OS credential store: Windows Credential Manager, or `secret-tool` (libsecret) on
Linux. Where none of those is available the CLI keeps them in an encrypted file in the home folder. A value is never
printed again. MCP configs refer to secrets as `${secret:NAME}`, so the value never appears in a config file or in the
model's context. `bind` allows one consumer to read a secret: `mcp:NAME` (an MCP server), `tool:NAME` or `group:NAME`.
For an MCP server the binding pins its command and arguments, so changing the command needs a new `bind`.

## agent sandbox

```sh
agent sandbox ls                              # folders outside the workspace that sandboxed tools may use
agent sandbox grant ~/notes --for files       # read access for the file tools
agent sandbox grant ~/data --write --for mcp:files
agent sandbox revoke ~/notes
```

`--for` limits a grant to one consumer: `run_command`, `files` (the file tools) or `mcp:NAME`. A tool that needs a path
outside the workspace asks first, with `[o] allow once  [a] allow always  [n] no`. Grants are saved in the project's
`.agents/` folder (run `agent init` there first) or in the global home. The operating-system sandbox that enforces them is
described in [security-sandbox.md](security-sandbox.md).

## agent engine

```sh
agent engine ls                     # name, version, backend, ok/MISSING, path, caps
agent engine add build/src/engines/sdcpp/agent-engine-sdcpp.so
agent engine rm sdcpp               # forgets the entry; the file stays
```

`add` loads the plugin with `agent::load_engine_plugin` and prints the engine's name, version and
capabilities before it records the plugin. A plugin that fails to load (wrong ABI version, missing
dependency libraries) is rejected and the load error is shown.

`agent engine pull NAME` downloads a prebuilt engine from a project GitHub release (`--tag vX.Y.Z` for a specific one, `--repo owner/name` for a fork), checks it against the release's `manifest.json` and stores it in `<home>/engines/NAME/<version>/`. No release is published yet (the version is `dev-v0.0.x`), so build the engines from source with the options in the [README](../README.md#cmake-options) and register them with `agent engine add`.

## agent runtime

`agent setup` asks which JavaScript tooling to use, and `agent runtime use bun|node|nub` changes it later:

| Choice | `npx` / `bunx` | `npm` | `node` |
|---|---|---|---|
| `bun` (default) | `bun x` | `bun install` / `bun run` | `bun` |
| `node` (a private Node.js with pnpm; npm is never used) | `pnpm dlx` | `pnpm install` / `pnpm run` | the private `node` |
| `nub` ([nub](https://github.com/nubjs/nub), a pnpm-compatible CLI over Node.js) | `nub dlx` | `nub install` / `nub run` | left alone |

The choice is stored as `js_runtime` in `config.json`. Every MCP server start goes through it (`Config::mcp_launcher`): servers
from `mcp.json`, from `agent.json` graphs and from library code alike. `uvx` always maps to `uv tool run`.

```sh
agent runtime ls                    # what is installed, and which choice is selected
agent runtime use node              # choose node + pnpm (downloads what is missing)
agent runtime install js            # install what the current choice needs
agent runtime install bun           # --version 1.2.3 for a specific release
agent runtime install uv
agent runtime rm bun
agent runtime path uv
```

Private copies of [bun](https://bun.sh) and [uv](https://docs.astral.sh/uv/) in `<home>/runtimes/`, so MCP servers and
skills written for `npx` / `uvx` run without Node, npm or Python on the machine. The archive is checked against the published SHA-256 before it is unpacked. Once installed, an MCP
server whose command is `npx` / `bunx` runs through `bun x`, and `uvx` through `uv tool run`; every cache, tool and
Python install is redirected into `<home>/cache` (`BUN_INSTALL_CACHE_DIR`, `UV_CACHE_DIR`, `UV_TOOL_DIR`,
`UV_PYTHON_INSTALL_DIR`), nothing is written to the user's own profile, and the runtimes folder is read-only inside
the server's sandbox. Without the runtime the command is left alone and `agent mcp` says what to install.

## agent skill

```sh
agent skill search pdf              # skills.sh
agent skill add anthropics/skills/pdf        # --project for <project>/.agents/skills, --yes to skip the question
agent skill ls
agent skill rm pdf
```

`add` finds the skill's folder in its GitHub repository (one API call), shows the source, the number of files and
size and the scripts it contains, and asks before it downloads anything. A skill is text the model follows plus
optional helper files; scripts in it can only run through `run_command` in the sandbox, after you approve the
command. `.source.json` in the skill's folder records the repository and a SHA-256 for each file. At most 200 files
and 20 MiB.

## agent mcp

```sh
agent mcp ls                        # configured servers, their sandbox and which keys they use
agent mcp search filesystem         # the official MCP registry
agent mcp add io.example/files      # exact registry name; --project, --yes
agent mcp rm files                  # only servers added with `mcp add`
```

`add` turns an npm or pypi package that speaks stdio into a config entry (`npx -y pkg@version`, `uvx pkg@version`) and
writes it to `<home>/mcp/mcp.d/<name>.json` (or `<project>/.agents/mcp.d/`), one file per server. Environment
variables the registry marks secret become `${secret:name}` references: it prints the `agent secret set` and `bind`
commands to run, and the value never passes through the config or the model. Required settings with no default are
listed for you to fill in. Hosted-only and container-only servers are listed as such and are not added.

## agent model

```sh
agent model search [TEXT] [--task asr]       # the curated registry: short names, sizes, [tested] marks
agent model info owner/repo                  # quants with sizes, projectors, licence of any Hugging Face repo
agent model pull qwen3-4b                    # a registry name (pulls every part, e.g. a talker and its codec)
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M [--name qwen] [--allow-restricted]
agent model pull owner/repo --quant Q4_K_M [--mmproj auto|none|QUANT] [--no-mmproj]
agent model pull owner/repo:* [--include int8,tokens]     # every file of a folder model (TRELLIS)
agent model fit owner/repo:Q4_K_M           # will it run on this machine, and about how fast
agent model recommend [TEXT] [--limit 8]     # search Hugging Face, keep what fits, fastest first
agent model ls
agent model rm qwen [--keep-file]
```

A vision, audio or OCR repo also gets its multimodal projector (`mmproj-*.gguf`) in the same pull: the one with the
model's quant, else Q8_0, else F16. It is recorded in `models.json` and used automatically by `chat`, `task` and `run`.

**Will it run here?** `agent model fit` reads the model's own header (the first few megabytes of the GGUF, fetched with an HTTP
range request, or from a file on disk), works out the memory it needs (weights, KV cache for `--ctx` tokens at `--kv f16|q8|q4`,
buffers) and compares it with the RAM that is free now, or the GPU memory when an NVIDIA GPU is found. It also measures the
machine's memory bandwidth for a third of a second and estimates generation speed from it: about 60 % of the bandwidth divided
by the bytes read per token (for a mixture-of-experts model, only the active experts). The estimate is "about": on the 4-core
test machine it said 57 to 61 tokens per second for Qwen2.5-0.5B Q4_K_M, which measured 50 to 62. A model that does not fit
says so and exits with 1. `agent model recommend` runs the same check over the most downloaded GGUF repos that match your text
(default "instruct") and lists those that fit, fastest first. Nothing is downloaded but the headers; there is no catalogue to
keep up to date.

`pull` uses the hub (see [api/hub.md](api/hub.md)) for spec resolution, resumable downloads, sha256
checks and the licence policy. Progress is drawn on stderr. Ctrl-C cancels the download and keeps the
`.part` file, so the same command resumes it. If a model's licence restricts use, the pull is refused
unless you pass `--allow-restricted`. On a terminal you are asked instead. The first chat model you pull
becomes `default_model`.

## agent task

```sh
agent task text_gen -e llama -m qwen3-4b --text "Say hi" --max-tokens 16
agent task embed -e llama -m bge-small --engine-options '{"mode":"embed"}' --text "hello"
agent task rerank -e llama -m jina-reranker-tiny --engine-options '{"mode":"rerank"}' --text "capital of France" --doc "Paris is..." --doc "Bananas are..."
agent task image_gen -e sdcpp -m sdxs-512 --text "a red lighthouse" --params '{"steps":1,"cfg_scale":1}'
agent task asr -e audiocpp -m parakeet-tdt --file speech.wav
agent task generic -e onnx -m minilm-onnx --tensor input_ids:i64:1,4=101,2023,2003,102 --tensor attention_mask:i64:1,4=1,1,1,1
```

Runs one engine capability once (`text_gen`, `ocr`, `asr`, `tts`, `vad`, `diarize`, `image_gen`, `image_edit`, `video_gen`,
`mesh_gen`, `embed`, `rerank`, `decide`, `classify`, `generic`) and prints every output part: text, files with their size,
tensors with dtype, shape and the first values, and a `stats:` line with the speeds. Tools are not added to the prompt.

## agent chat

```sh
agent chat                          # default_model, else the first .gguf in models.json
agent chat --model qwen             # a models.json name ...
agent chat --model /tmp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf    # ... or a path
agent chat --engine llama           # engine name from engines.json, or a plugin path
agent chat --resume chat-20261031-142501
agent chat --list                   # saved conversations in this workspace
agent chat --plain                  # line-based mode instead of the full-screen interface
```

**The chat screen.** In a terminal, `agent chat` opens a full-screen interface: a title bar with the model and engine,
the conversation (answers are rendered as markdown: headings, bold, inline code, lists, code blocks), an input line, and a
footer with the speed and token counts of the last answer. Tool calls and results appear as dim lines. When a tool needs
permission a prompt replaces the input line: `y` to allow once, `n` to deny, `a` to allow it for the rest of the session.

| Key | Action |
|---|---|
| Enter | send the message (or run the `/command`) |
| Esc, or Ctrl-C while an answer is running | stop the answer |
| Ctrl-C or Ctrl-D at the prompt | quit |
| PgUp / PgDn, mouse wheel | scroll the conversation |

The slash commands are the same as in line mode (`/help` lists them). With `--plain`, with `--verbose`, or when input or
output is not a terminal, `agent chat` uses the line-based mode instead.

Other options are `--workspace DIR`, `--ctx N`, `--max-tokens N`, `--engine-options JSON` (forwarded
to the engine, for example `{"mmproj": "mmproj.gguf"}` for image input with llama), `--yes` and
`--verbose`.

**Engine isolation.** The engine runs in a separate `agent-engine-host` process (installed next to the
`agent` binary), so an engine crash, `abort()` or `exit()` fails the turn with an error instead of
killing the chat; the next message restarts the engine (the conversation is kept, the KV cache is not).
`--inprocess` loads the engine into the CLI instead (slightly lower latency, no crash protection), as
does `"isolation": "inprocess"` on the engine's `engines.json` entry. See
[api/engines.md](api/engines.md#isolation).

**Engine choice.** Unless you pass `--engine`, the engine follows from the model file type: `.gguf`
uses `llama`, `.onnx` uses `onnx`, and `.safetensors` / `.ckpt` use `sdcpp`. The engine's path comes
from `engines.json`. If the engine is not listed there, the CLI looks in the default plugin directories.

**After every turn** a line on stderr shows the engine's own measurements of the last model call: prompt speed
(tokens that were not already in the KV cache, per second), generation speed and the KV cache,
for example `[prompt 412 tok (380 cached) 320 tok/s | gen 64 tok 90 tok/s | KV 476/2048 q4_0 3.1 MiB]`. Engines that report
nothing (speech, image, ...) print no line. `/stats` adds the prefix-cache hit rate and the last call.

**The chat screen.** Tokens stream as the model writes them. Tool-call markup in the Hermes / Qwen
format (`<tool_call>...</tool_call>`) is hidden from the stream. Instead, each tool call is shown as
`[tool] name {arguments}` and each tool result as `[result] ...`, cut to one line. Files that engines
generate are listed as `[artifact n] path (mime)`.

**Engine log.** Engine libraries write their own log lines straight to stderr. llama.cpp alone prints
hundreds of lines per model load. The chat therefore redirects stderr to `<workspace>/logs/engine.log`.
The CLI's own warnings and errors still reach the terminal. Pass `--verbose` to see the engine output
in the terminal instead.

**Permissions.** The CLI uses `agent::tools::default_permission_policy`: `run_command`, `delete_path`
and `move_path` ask first, and every other tool runs without asking. On a terminal you answer
`y` (yes), `n` (no, the default) or `a` (always allow this tool for the rest of the session). When
stdin is not a terminal, these calls are denied unless you pass `--yes`.

**Conversations.** A conversation is saved after every turn in `<workspace>/memory/db.sqlite3`
(`agent_store_sqlite`), under an id like `chat-20261031-142501`. When you exit, the CLI prints that
id. To continue the conversation later, run `agent chat --resume <id>`.

**Cancelling.** Ctrl-C while the model is answering cancels the answer. Ctrl-C or Ctrl-D at the
prompt quits.

| Command | |
|---|---|
| `/exit` | quit (`/quit` also works) |
| `/clear` | forget the conversation so far; the system prompt is kept |
| `/model [NAME\|PATH]` | show the current model and the catalogue, or switch models. The conversation is kept. Earlier attachments stay in the history as `[attached mime: path]` text, and `--engine-options` are not applied to the new model. If the new model fails to load, the old one is reloaded. With `memory_budget_mb` set, the previous model stays loaded (idle) while it fits, so switching back is instant; see [api/models.md](api/models.md#cli). |
| `/image <path> [prompt]` | send an image or audio file with the message (`run_turn_with_media`). Put paths that contain spaces in quotes. The engine has to accept media, for example llama with `--engine-options '{"mmproj": "..."}'`. |
| `/open [n]` | list the generated artifacts, or open the n-th one with the OS default app (`xdg-open` / `open` / `ShellExecute`). Media is never rendered in the terminal. |
| `/stats` | token and tool-call counters for the conversation and the session, plus the loaded models (estimated size, in use / idle, last use) and the CLI's RSS |
| `/help` | list the commands |
| `//text` | send a message that starts with `/` |

When stdin is piped, the CLI prints each input line after `> `, so a transcript reads like the
interactive session:

```sh
printf 'List the files in the workspace.\n/stats\n' | agent chat --model qwen
```

## agent schedule

Runs an agent command on a timer with the operating system's own scheduler: cron on Linux, Task Scheduler on Windows.
agent.cpp starts no background process of its own.

```sh
agent schedule add nightly --daily 02:30 -- run graph.json --yes     # every day at 02:30
agent schedule add poll --every 15m -- run graph.json --yes          # every 15 minutes (m: 1-30 dividing 60, h: dividing 24, 1d)
agent schedule add weekdays --cron "0 9 * * 1-5" -- chat --plain ...  # a cron expression (Linux only)
agent schedule ls
agent schedule rm nightly
```

Everything after `--` is the `agent` command to run; the schedule stores the full path of this `agent` binary and its `--home`. A run
nobody watches cannot answer a permission prompt, so put `--yes` in the command if its tools need approval. Adding a name that
exists replaces it. On Linux the entries are lines in your crontab tagged `# agent.cpp:NAME`; other lines are never touched. On
Windows they are tasks in the `agent.cpp` folder of Task Scheduler.

Together with the `wait` node of [agent.json](api/agent-json.md#nodes) this gives sleep and wake without a daemon: a graph pauses
at a wait node and exits; `agent schedule add resume --every 15m -- run graph.json --resume THREAD --yes` continues it
whenever it is due.

## agent run

`agent run <graph.json>` runs a declarative graph (see [api/agent-json.md](api/agent-json.md) and
`examples/03-graph-workflow/graphs/`). It uses the same model and engine resolution as `agent chat`, with the
graph's `"model"` / `"engine"` hints as defaults that `--model` / `--engine` override.

```bash
agent run examples/03-graph-workflow/graphs/research_pipeline.json --model qwen2.5-0.5b --set "topic=Why is the sky blue?"
```

| Flag | |
|---|---|
| `--set key=value` | seed graph state (repeatable) |
| `--thread ID` | checkpoint thread. Default: a new thread `<graph name>-<unix time>` for every run; pass an id to continue that thread's state and conversations |
| `--resume ID [--input TEXT]` | continue an interrupted or failed run; `TEXT` is the human reply, or `approve` for pending tool calls |
| `--swap-models` | do not keep the default model loaded for the whole run: its nodes load it through the registry like any other model, so it is unloaded while a node runs another one. Implied when `memory_budget_mb` is set and some nodes name their own `"model"` |
| `--yes` | approve tool calls the permission policy would ask about |
| `--json` | print node events as one JSON object per line instead of the status table |
| `--model`, `--engine`, `--engine-options`, `--workspace`, `--inprocess` | as for `agent chat` |

Agent nodes can name their own model (`"model": "<catalogue name or path>"`). Those models are
loaded through the [model registry](api/models.md) under `memory_budget_mb`, so a graph can use
more models than fit in memory at once. The default model is then only loaded if some agent node
has no `"model"`.

While the graph runs, the CLI shows the state of every node on stderr: one line each with the node
name, its kind, the state (`running`, `done`, `interrupted` or `error`), the elapsed time and the last
tool call. Branches of a `Parallel` node appear as they start and run side by side. When stdout and
stderr are both terminals, `TERM` is not `dumb` and `NO_COLOR` is not set, the table is redrawn in place
with ANSI escapes (on Windows 10 and later the CLI turns on virtual terminal processing for this).
Otherwise, for example in a pipe or a CI log, it prints one line per event as it happens:

```
[node] research (agent) running
[node] research (agent) tool: web_search
[node] research (agent) done in 4.2s
```

With `--json` each event line is a JSON object (`{"event":"node_start","node":"research","kind":"agent","ts":...}`;
the events are `node_start`, `tool_call`, `node_end`, `interrupt` and `node_failed`). Library users get the same
feed from `GraphRun::on_event` (see [api/graph.md](api/graph.md)).

When the graph finishes, the CLI prints the status and the final state. When it is interrupted,
it prints the reason, any paused parallel branches and pending tool calls, and the command to
resume (`--input` answers every paused branch). Checkpoints and the graph event log live in
`<workspace>/memory/checkpoints.sqlite3` (the SQLite checkpointer), unless the graph file says
`"checkpointer": "memory"`. Runs checkpointed by earlier CLI versions in the workspace's key-value
store cannot be resumed.

## agent voice

`agent voice` is a spoken conversation: microphone → speech detection → speech-to-text → one LLM turn →
text-to-speech → speaker, in a loop until Ctrl-C. Speech runs on the `audiocpp` engine (found like any other
engine: `engines.json`, else next to the binary); the LLM turn is the same as in `agent chat`. A guide with
model downloads is in [guides/voice-assistant.md](guides/voice-assistant.md).

```sh
agent voice --asr-model ~/models/parakeet-tdt.gguf --tts-model ~/models/pocket-tts.gguf \
            --vad-model ~/models/firered-vad.gguf --model qwen
# no sound card (or a test): read the microphone from a WAV file, write the replies as WAV files
agent voice --asr-model ... --tts-model ... --input-wav question.wav --output-dir replies/ --max-turns 1
```

| Flag | |
|---|---|
| `--asr-model FILE` | audio.cpp speech-recognition GGUF, for example `parakeet-tdt` or `moonshine-tiny`. Required |
| `--tts-model FILE` | audio.cpp text-to-speech GGUF, for example `pocket-tts` (English) or `supertonic-3` (30+ languages). Required |
| `--vad-model FILE` | audio.cpp VAD GGUF, for example `firered-vad`: utterances are cut by the engine's streaming VAD. Without it an energy gate is used, which works in a quiet room only |
| `--voice ID` | optional TTS voice preset, if the TTS model has one |
| `-m, --model`, `-e, --engine`, `--engine-options`, `--ctx`, `--max-tokens`, `--workspace`, `--inprocess`, `-y, --yes`, `--verbose` | as for `agent chat` |
| `--input-wav FILE` | file backend: the microphone is this WAV (any rate, resampled to 16 kHz); the loop ends at its end |
| `--output-dir DIR` | file backend for the speaker: each reply becomes `DIR/reply-NNN.wav` |
| `--max-turns N` | stop after N answered utterances |
| `--barge-in` | keep listening while a reply plays; speaking over it stops playback. Needs headphones or echo cancellation: the CLI has none, so speakers would interrupt themselves |
| `--resume ID` | continue a saved conversation (voice and chat conversations share the store) |

**How a turn goes.** Audio is captured as 16 kHz mono f32 and fed to the detector in 100 ms chunks. When a
speech segment ends (the VAD decides when a segment ends), the segment plus 200 ms before and
150 ms after goes to the ASR session as an in-memory tensor. Transcripts without words (`[BLANK_AUDIO]`,
`(music)`) and utterances shorter than 250 ms are skipped. The transcript is the user message of a normal
`run_turn` (streamed, tools and permission prompts as in `agent chat`, saved after each turn as
`voice-YYYYMMDD-HHMMSS`). The reply is cleaned for speech (markdown, code blocks and tool-call markup are
dropped; numbers and `+ - = % / x` are spelled out), synthesized
by the TTS session and played. Without `--barge-in` the loop is half-duplex: what the microphone hears
while a reply is produced or played is dropped, so the agent does not answer itself.

The screen shows `[listening]`, `you: <transcript>`, `agent: <reply>`, `[spoken] 1.84 s of speech at
24000 Hz`, `[barge-in]` and `[error] ...` lines (an engine error skips that utterance; the loop goes on).

**Audio backends.** Sound devices go through [miniaudio](https://github.com/mackron/miniaudio) (vendored
single header, 0.11.26, public domain / MIT-0; it loads ALSA / PulseAudio / JACK / WASAPI / Core Audio at
run time, so the binary has no link-time audio dependency; `-DAGENT_CLI_AUDIO_DEVICE=OFF` leaves it out).
Only the CLI links it, never `libagent`. When there is no microphone, `agent voice` stops with a hint to use
`--input-wav`; when there is no speaker, replies go to `<workspace>/voice/` as WAV files. The device backend
was tested on Linux against PulseAudio virtual devices (a null sink as the microphone, another as the speaker: the question
was transcribed, answered and played), not on a physical sound card; the file backend is what the unit
tests run.

**Graph form.** A graph file cannot express this loop today: a graph runs on one engine session, and
`agent run` registers no function nodes for ASR / TTS.

## Mirrors and tests: `AGENT_HTTP_REWRITE`

`AGENT_HTTP_REWRITE="https://github.com=https://mirror.example/gh;https://skills.sh=http://127.0.0.1:8080"` sends every download whose URL starts
with the text before an `=` to the text after it (first match wins, rules separated by `;`). It covers engine and runtime downloads, the
skills.sh and MCP registries and GitHub's API. `HF_ENDPOINT` does the same for Hugging Face. The tests use both to run the real `agent`
binary against a stand-in server on the loopback interface.

## Code layout

`apps/cli/src/`:

- `cli_core.{hpp,cpp}`: the testable logic, built as the `agent_cli_core` library and covered by
  `tests/unit/test_cli.cpp`. It handles argument parsing, `config.json` and `engines.json`, model and
  engine resolution, slash commands, the stream filter and the progress bar.
- `cli_platform.cpp`: config and data directories, the executable's location, hardware detection,
  TTY checks, line reads that Ctrl-C can interrupt, the stderr redirect, and opening files with the
  default app.
- `commands.cpp`: `setup`, `engine`, `model` and `run`.
- `chat.cpp`: the chat screen.
- `audio.{hpp,cpp}` (+ `miniaudio_impl.c`): WAV read/write, resampling, and the file and miniaudio
  device backends behind small op tables (in `agent_cli_core`).
- `voice.hpp`, `voice_core.cpp`: the energy gate, the voice loop, flag handling and the audiocpp options
  built from model files (in `agent_cli_core`, covered by `tests/unit/test_voice.cpp`).
- `voice.cpp`: `agent voice` (engine sessions and the hooks the loop calls).
- `main.cpp`: dispatches to the commands.
