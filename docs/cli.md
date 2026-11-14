# The `agent` command

`agent` is the command-line app built on `libagent`. Run `agent` for the overview and `agent <command> --help` for the options of
one command.

| I want to | Command |
|---|---|
| set everything up the first time | [`agent setup`](#agent-setup) |
| chat with a local model | [`agent`](#agent-chat) |
| get a model | [`agent model`](#agent-model) |
| run one thing once (transcribe, speak, draw, embed) | [`agent task`](#agent-task) |
| run a multi-step workflow | [`agent run`](#agent-run) |
| talk by voice | [`agent voice`](#agent-voice) |
| add or remove engines | [`agent engine`](#agent-engine) |
| add skills or MCP servers | [`agent skill`](#agent-skill), [`agent mcp`](#agent-mcp) |
| control what tools may touch | [`agent sandbox`](#agent-sandbox), [`agent secret`](#agent-secret) |
| see what is installed, free disk | [`agent doctor`](#agent-init-doctor-clean), [`agent clean`](#agent-init-doctor-clean) |
| pause a graph and resume it later | [`agent schedule`](#agent-schedule), [`agent wake`](#agent-wake) |

`--no-sandbox` works with every command and turns the OS sandbox off for engines, tools and commands. `--home DIR` picks the
global home for one run. Everything is documented below; the same text is in `--help`.

## agent setup, agent home, agent update

`agent setup` and `agent update` are not code in this program: each downloads a script from the repository
([`scripts/setup.sh`](../scripts/setup.sh) / [`setup.ps1`](../scripts/setup.ps1) and [`install.sh`](../scripts/install.sh) /
[`install.ps1`](../scripts/install.ps1)) and runs it, so they can be improved without a new release. `AGENT_SCRIPTS` points at another
base URL or a folder with copies of the scripts.

```sh
agent setup                  # the files folder, the optional bun and uv runtimes, a model link or file; then the chat opens
agent setup --yes --model https://huggingface.co/owner/repo     # no questions
agent home [DIR]             # show, or move, the folder agent.cpp keeps its files in
agent update                 # same version: nothing; a newer release: replaced in place (SHA-256 checked); --check only looks
```

Options of `setup`: `--yes`, `--model LINK|FILE|owner/repo[:quant]`, `--no-model`, `--runtimes` / `--no-runtimes`, `--home DIR`. The GPU
backend and the engines are not asked about: they come with the build that was installed (CPU or Vulkan). Engines are not touched by
`update`; `agent engine pull NAME` fetches newer ones.

## agent chat

```sh
agent                             # or: agent chat
agent chat --model qwen           # a name from models.json, or a path to a model file
agent chat --resume chat-20261031-142501
agent chat --list                 # saved conversations
agent chat --plain                # line mode, for pipes
```

In a terminal this opens a full-screen chat laid out like a chat website, built with [RTXUI](https://github.com/ArthurSonzogni/RTXUI)
(HTML and CSS for the terminal):

- **Sidebar:** your saved conversations, grouped by project (the folder the agent works in). Click a session, or press `Ctrl+K`, move
  with Up/Down and press Enter. Picking a session from another project switches the agent to that folder: its files, tools and memory
  change with it (the model is loaded again for the new folder). `Ctrl+N` starts a new chat, `Ctrl+B` hides the sidebar. The projects
  are remembered in `workspaces.txt` in the agent home.
- **Conversation:** your messages in a frame, answers rendered as markdown (headings, lists, quotes, tables, framed code), tool calls as
  cyan lines, and under every answer a line with the model, the speed in tokens per second, the token counts and the time.
- **Input:** at the bottom, with a command palette (type `/` or press `Ctrl+P`), history on Up/Down, and a footer with the speeds and how
  full the context window is.

**The layout follows the width of the terminal** and changes the moment you resize it (a split pane, a phone over SSH, a tmux window):

| Width | Layout |
|---|---|
| 84 columns or more | sidebar on the left, conversation and input on the right |
| 70 to 83 | the sidebar becomes a drawer: a `☰` button (or `Ctrl+K` / `Ctrl+B`) slides it over the conversation, picking a session closes it |
| 50 to 69 | tighter margins; the project path leaves the title bar |
| 36 to 49 | the tip, the model line and the footer text are dropped; the context gauge stays |
| under 36 | the big logo gives way to the name |

A tool that needs permission asks `y` once, `n`, or `a` for the rest of the session. `Esc` (or Ctrl-C while answering) stops an answer;
Ctrl-C or Ctrl-D at the prompt quits. The mouse wheel, PgUp/PgDn and Home scroll the conversation; `End` jumps back to the newest line
and follows it again. Select text with the mouse to copy it. The screen uses four colors (black, white, neon cyan, hot pink).

| Command | What it does |
|---|---|
| `/model [NAME\|PATH]` | show the loaded model, or switch (the conversation is kept; if the new one fails, the old one is reloaded) |
| `/image PATH [text]` | send an image or audio file with a message (the engine must accept media, e.g. llama with an `mmproj`) |
| `/open [n]` | list files the model generated, or open the n-th |
| `/stats` | token, tool-call and cache counters, loaded models |
| `/clear` | forget the conversation (the system prompt stays) |
| `/diff` | what the tools changed in files this session, as a unified diff |
| `/undo` | put back the files of the last tool call (a file it created is removed) |
| `/doctor`, `/setup` | the commands of the same name, run inside the chat |
| `/exit`, `//text` | quit; send a message that starts with `/` |

Options: `-m/--model`, `-e/--engine`, `--workspace DIR`, `--ctx N`, `--max-tokens N`, `--engine-options JSON`, `-y/--yes`
(approve tool calls without asking), `--verbose` (show engine logs), `--inprocess`.

The engine follows from the model file (`.gguf` uses `llama`, `.onnx` uses `onnx`, `.safetensors` and `.ckpt` use `sdcpp`) unless
you pass `--engine`. Engines run in a separate `agent-engine-host` process, so a crash fails one turn and the next message restarts
it ([isolation](api/engines.md#isolation)). Conversations are saved after every turn and the id is printed on exit. Tools that
change files or run commands (`run_command`, `delete_path`, `move_path`) ask first; without a terminal they are denied unless you
pass `--yes`.

### Rules for a project

The rules you write for a project become part of the system prompt of every new conversation: `AGENTS.md` in the workspace, then
`.agents/AGENTS.md`, then every `.agents/rules/*.md` in name order, then your own `AGENTS.md` in the agent home. At most 16 KiB are
used; a file that does not fit is cut and says so. Nothing is read from outside these places.

### Without a terminal: `-p` and `--jsonl`

```sh
agent chat -p "Summarise README.md in one line" -y      # answer once and exit
git diff | agent chat -p - -y                           # - reads the prompt from stdin
```

`-p` (`--prompt`) prints only the answer on stdout; tool lines, permission lines and notes go to stderr, so the output can be piped.
The exit code is 1 when the turn failed. Tool calls that need permission are denied unless you pass `-y`. The conversation is saved
like any other (`--resume ID` continues it).

`--jsonl` is the editor integration: one JSON object per line on stdin and on stdout, no terminal, no daemon (the program ends when
stdin closes). Send `{"type":"prompt","text":"..."}` (a `/command` works too) and `{"type":"exit"}`. The program writes:

| `type` | other fields | meaning |
|---|---|---|
| `ready` | `session`, `model`, `engine`, `workspace` | once, when the model is loaded |
| `token` | `text` | a piece of the answer |
| `tool`, `note`, `error` | `text` | a tool call or result, a notice, an error |
| `permission_request` | `tool`, `arguments` | answer on the next stdin line: `{"type":"permission","allow":true,"always":false}` (not asked with `-y`) |
| `done` | `session`, `metrics` | the turn is over; `metrics` is the model, speed and token line |

### In the message: `@file`, `!command`

`@path` in a message adds the content of that file (relative to the workspace or absolute; text files up to 32 KiB each, 96 KiB in all) as a
fenced block after your text, and the chat says which files it attached. A line that starts with `!` runs a shell command of your own in the
project folder (`!git status`); its output is shown and added to the conversation, so the model can see it. These are your commands, not the
agent's: they are not sandboxed and do not ask.

### Modes, thinking, permissions

- `/mode plan` lets the agent only read (no file changes, no commands); `/mode auto` never asks; `/mode ask` (default) asks before changes. The
  status line shows the mode when it is not `ask`.
- `/thinking [on|off]` shows or hides the `<think>...</think>` blocks of reasoning models (hidden by default).
- Answering `a` to a permission question allows that tool for good **in this project**: it is kept in `<workspace>/.agents/permissions.json`.
  `/permissions` lists them and `/permissions clear` forgets them. The question shows the call's arguments, one per line.
- `/compact` summarises the oldest messages to free context (the last six stay as they are).

### Sessions

`/rename TITLE` names the conversation in the sidebar. `/delete` asks, `/delete yes` removes the saved conversation for good and starts a
new one. In the session picker (`Ctrl+K`) you can type to filter the list, and `Del` removes the highlighted conversation.

### The web

The model has a `web_fetch` tool (an http or https address; HTML is reduced to readable text, at most 20,000 characters) and, when
`config.json` has a `search_url`, a `web_search` tool: `"search_url": "http://localhost:8080/search?format=json&q={q}"`, for example a SearXNG
instance. Both ask first (like `run_command`); `-y` or `/mode auto` approve them. Nothing else on the model's side touches the network.

### Constrained answers

`--json` makes every answer valid JSON; `--grammar FILE` samples under a GBNF grammar (for example `root ::= "yes" | "no"`). Both are for the
llama engine and switch the tools off for the session (a grammar cannot share a turn with the tool-call grammar). In `task text_gen` the same
is `--params '{"json": true}'` or `--params '{"grammar": "..."}'`.

### Look

The chat uses four colours: black, white, turquoise and pastel pink, with white at reduced opacity for secondary text and lines. The window
frame is turquoise; the input box turns turquoise while it takes input. Icons are Unicode symbols; `AGENT_ICONS=nerd` uses Nerd Font glyphs
(install a [Nerd Font](https://www.nerdfonts.com) in your terminal first) and `AGENT_ICONS=ascii` plain characters.

### In an editor: `--acp`

`agent chat --acp` speaks the [Agent Client Protocol](https://agentclientprotocol.com) on stdin and stdout (JSON-RPC 2.0, one message per
line), so an editor such as Zed can run it as its agent. Supported: `initialize`, `session/new`, `session/prompt` (text and embedded
resource blocks), the `session/update` notifications `agent_message_chunk`, `tool_call` and `tool_call_update`, and
`session/request_permission` (allow once, allow always, reject). Not supported yet: `session/load`, images and audio in prompts, and
`session/cancel` while a turn is running (it is read after the turn). The model, sandbox and project rules are the ones of `agent chat`;
`--jsonl` is the simpler, agent.cpp-specific line protocol.

### Seeing and undoing edits

Before a built-in tool changes a file (`write_file`, `append_file`, `delete_path`, `move_path`), the chat remembers the file's
content (files up to 1 MiB; folders are not remembered). `/diff` shows every change of the session against the files as they are
now, and `/undo` puts back the files of the last tool call, newest first.

## agent model

```sh
agent model info owner/repo                       # quants, sizes, projectors, licence of any Hugging Face repo
agent model pull https://huggingface.co/Qwen/Qwen3-4B-GGUF   # a repository page: the quant is chosen for you
agent model pull C:\Downloads\model.gguf                   # a file you already have
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M [--name qwen] [--allow-restricted]
agent model pull owner/repo:* [--include int8]    # every file of a folder model (TRELLIS)
agent model ls
agent model rm qwen [--keep-file]
```

`pull` takes a Hugging Face link (a repository page, or the page or download link of one file), `owner/repo[:quant-or-file]`, or a model
file on this disk. There is no list of suggested models: the file itself says which engine runs it (llama.cpp, stable-diffusion.cpp,
audio.cpp, ONNX Runtime or TRELLIS.2), and it is put in that engine's folder under `models/` and recorded in `models.json`. A file already
installed is not fetched again. An embedding or reranker model is detected too; the command prints the `--engine-options` it needs. Downloads resume after Ctrl-C
(the `.part` file is kept), are checked against the published hash, and refuse models with restrictive licences unless you pass
`--allow-restricted` (on a terminal you are asked). Vision, audio and OCR repos also fetch their multimodal projector, record it
and use it automatically. The first chat model you pull becomes the default.

Pulled files go to a folder of their engine, so the files of two engines are never mixed: `models/llama/owner/repo/file.gguf`,
`models/sdcpp/...`, `models/onnx/...`. A registry name knows its engine; for `owner/repo` the engine follows the quant or file
(`:Q4_K_M` and GGUF repos are `llama`, `.onnx` is `onnx`, `.safetensors` is `sdcpp`); `--engine NAME` overrides it. Models pulled before
this stay where they are (`models.json` records absolute paths).

## agent task

```sh
agent task text_gen -e llama -m qwen3-4b --text "Say hi" --max-tokens 16
agent task asr      -e audiocpp -m parakeet-tdt --file speech.wav
agent task tts      -e audiocpp -m pocket-tts --text "Hello there"
agent task image_gen -e sdcpp -m sdxs-512 --text "a red lighthouse" --params '{"steps":1,"cfg_scale":1}'
agent task embed    -e llama -m bge-small --engine-options '{"mode":"embed"}' --text "hello"
agent task rerank   -e llama -m jina-reranker-tiny --engine-options '{"mode":"rerank"}' --text "capital of France" --doc "Paris" --doc "Banana"
agent task generic  -e onnx -m minilm-onnx --tensor input_ids:i64:1,4=101,2023,2003,102
```

Runs one capability of one engine and prints every output: text, files (path and size), tensors, and a `stats:` line with speeds.
Capabilities: `text_gen ocr asr tts vad diarize image_gen image_edit video_gen mesh_gen embed rerank decide classify generic`.
Tools are not added to the prompt. For `generic`, `--tensor name:dtype:dims=values` is repeatable (`f32 i64 i32 u8`).

## agent run

```sh
agent run flow.json --model qwen2.5-0.5b --set "topic=Why is the sky blue?"
agent run flow.json --resume THREAD --input approve
```

Runs a declarative graph ([format](api/agent-json.md), [guide](guides/writing-a-graph.md), examples in
`examples/03-graph-workflow/graphs/`). Model and engine resolve as in `chat`; the graph's own `"model"` / `"engine"` are the
defaults that `--model` / `--engine` override.

| Flag | Meaning |
|---|---|
| `--set key=value` | seed graph state (repeatable) |
| `--thread ID` | checkpoint thread; default is a new one per run |
| `--resume ID [--input TEXT]` | continue an interrupted or failed run; the text is the human reply, or `approve` for pending tool calls |
| `--swap-models` | unload the default model while a node runs another (implied when `memory_budget_mb` is set) |
| `--yes`, `--json` | approve tool calls; print events as JSON lines instead of the status table |
| `--model --engine --engine-options --workspace --inprocess` | as for `chat` |

The CLI shows every node's state (`running`, `done`, `interrupted`, `error`), time and last tool call, redrawn in place on a
terminal and one line per event in a pipe. When a run is interrupted it prints why and the command that resumes it. Checkpoints
live in `<workspace>/memory/checkpoints.sqlite3`.

## agent voice

```sh
agent voice --asr-model parakeet-tdt.gguf --tts-model pocket-tts.gguf --vad-model firered-vad.gguf --model qwen
agent voice --asr-model ... --tts-model ... --input-wav question.wav --output-dir replies/ --max-turns 1   # no sound card
```

A spoken loop: microphone, voice-activity detection, speech recognition, one chat turn (with tools), speech synthesis,
speaker, until Ctrl-C. It chains separate models; there is no single speech-to-speech model. Speech runs on the `audiocpp`
engine, the chat turn on the model you pick. Step-by-step with downloads: [guides/voice-assistant.md](guides/voice-assistant.md).

| Flag | Meaning |
|---|---|
| `--asr-model FILE`, `--tts-model FILE` | required: e.g. `parakeet-tdt` / `moonshine-tiny`, and `pocket-tts` (English) / `supertonic-3` (30+ languages) |
| `--vad-model FILE` | e.g. `firered-vad`; without it an energy gate is used, which works only in a quiet room |
| `--voice ID` | a voice preset of the TTS model |
| `--input-wav FILE`, `--output-dir DIR` | read the "microphone" from a WAV and write replies as `reply-NNN.wav` |
| `--max-turns N`, `--barge-in`, `--resume ID` | stop after N utterances; keep listening during playback (needs headphones); continue a conversation |
| `-m -e --engine-options --ctx --max-tokens --workspace --inprocess -y --verbose` | as for `chat` |

Without `--barge-in` the loop is half-duplex: sound heard while the agent speaks is dropped, so it never answers itself.
Replies are cleaned for speech (markdown and code are dropped, symbols spelled out). Sound devices use miniaudio, loaded at run
time; with no microphone `voice` tells you to use `--input-wav`, with no speaker replies go to `<workspace>/voice/`.

## agent engine

```sh
agent engine ls                       # name, version, ok/MISSING, path, capabilities
agent engine add path/to/agent-engine-sdcpp.so
agent engine pull sdcpp               # a prebuilt engine from a GitHub release (--tag, --repo)
agent engine rm sdcpp                 # forgets the entry; the file stays
```

`add` loads the plugin, prints its name, version and capabilities, and records it in `engines.json`; a plugin with the wrong ABI or
missing libraries is rejected with the reason. No release is published yet, so build the engines from source (see the
[README](../README.md#install)).

## agent skill

```sh
agent skill search pdf
agent skill add anthropics/skills/pdf [--project] [--yes]
agent skill ls
agent skill rm pdf
```

A skill is text the model follows, with optional helper files. `add` shows the source, the file count, size and any scripts, and asks
before downloading (at most 200 files and 20 MiB). Scripts only run through `run_command` in the sandbox, after you approve.

## agent mcp

```sh
agent mcp search filesystem
agent mcp add io.example/files [--project] [--yes]
agent mcp ls
agent mcp rm files
```

```sh
agent mcp serve [--workspace DIR] [--read-only]     # the agent's tools as an MCP server on stdin/stdout, for an editor or another agent
```

An entry in `mcp.json` can be a server on the web instead of a program: `"docs": {"url": "https://mcp.example.com/mcp", "headers":
{"Authorization": "Bearer ${secret:docs}"}}` (MCP Streamable HTTP; see [api/mcp.md](api/mcp.md#servers-reached-by-url-streamable-http)).
A server that offers resources or prompts gets `list_resources`, `read_resource`, `list_prompts` and `get_prompt` tools next to its own.

`add` turns an npm or PyPI stdio server from the official registry into a config entry (`<home>/mcp/mcp.d/NAME.json`, or
`<project>/.agents/mcp.d/`). Secret settings become `${secret:NAME}` references and `add` prints the `agent secret` commands to
run. Hosted-only and container-only servers are listed but not added. Servers published as npm or Python packages run through a
private `bun` or `uv` ([`agent runtime`](#agent-runtime)).

## agent sandbox

```sh
agent sandbox ls
agent sandbox grant ~/notes --for files
agent sandbox grant ~/data --write --for mcp:files
agent sandbox revoke ~/notes
```

Tools can use the workspace. A grant lets them use one more folder; `--for` limits it to `run_command`, `files` or `mcp:NAME`. When a
tool needs a path outside, it asks `[o] once  [a] always  [n] no`. Grants are kept in the project's `.agents/` or the global home.
How the sandbox itself works: [security-sandbox.md](security-sandbox.md).

## agent secret

```sh
agent secret set OPENAI_KEY            # typed without echo; --stdin reads a pipe
agent secret bind OPENAI_KEY mcp:github
agent secret ls | unbind | rm
```

Values go to the OS credential store (Credential Manager, or libsecret on Linux), or an encrypted file in the home where neither
exists, and are never shown again. MCP configs refer to `${secret:NAME}`, so the value stays out of config files and the model's
context. `bind` allows one consumer (`mcp:NAME`, `tool:NAME` or `group:NAME`); for an MCP server it pins the command, so changing
the command needs a new `bind`.

## agent runtime

```sh
agent runtime ls
agent runtime use bun|node|nub        # which JavaScript tooling MCP servers use
agent runtime install js|bun|uv
agent runtime path uv | rm bun
```

Private copies of [bun](https://bun.sh), [uv](https://docs.astral.sh/uv/) (and optionally Node with pnpm, or nub) live in
`<home>/runtimes/`, so servers written for `npx` and `uvx` run without Node or Python on the machine. Archives are checked against
their published SHA-256; all caches stay in `<home>/cache`; nothing is written to your profile.

## agent init, doctor, clean

```sh
agent init [DIR]                     # create .agents/ (skills, mcp.d, memory, cache, local) in a project
agent doctor                         # engines, runtimes, models and disk use per area; changes nothing
agent clean --downloads|--cache|--logs [--yes]
```

`doctor` also prints the sandbox line: `on (Landlock ABI 6, seccomp)` or `AppContainer and Job Object`, `off (--no-sandbox)`, or
`not supported on this system (...): running unsandboxed`. The same warning is shown when a chat, task or run starts and the sandbox
cannot confine anything; nothing is refused and nothing is faked.

`clean` removes one area per flag (partial downloads, rebuildable caches, old logs). Models, memory and conversations are never removed.

## agent update, completion

```sh
agent update --check        # is there a newer release? changes nothing
agent update                # runs the installer script: a newer release replaces this one in place
agent completion bash       # also: zsh, fish, powershell
```

Load a completion script in your shell, for
example `source <(agent completion bash)`, `agent completion fish | source`, or `agent completion powershell | Out-String | Invoke-Expression`.

## agent schedule

```sh
agent schedule add nightly --daily 02:30 -- run flow.json --yes
agent schedule add poll --every 15m -- wake
agent schedule add weekdays --cron "0 9 * * 1-5" -- run flow.json --yes     # cron expressions: Linux only
agent schedule ls | rm nightly
```

Registers an `agent` command with the OS scheduler (cron on Linux, Task Scheduler on Windows); agent.cpp runs no daemon. What follows
`--` is the command, stored with the full path of this binary and its `--home`. Put `--yes` in it if its tools need approval, since
nobody is there to answer. On Linux the entries are crontab lines tagged `# agent.cpp:NAME` and other lines are never touched.

## agent wake

`agent wake [--workspace DIR] [--dry-run]` resumes graph runs that paused at a `wait` node ([agent.json](api/agent-json.md#nodes))
and are now due. `agent run` records such a run in `<workspace>/waiting.json` with its original flags; `wake` re-runs each due one
with `--resume`. Together with `schedule add wake --every 15m -- wake` this gives sleep and wake with no background process.

## Files and layout

One global home (default `.agent.cpp-workspace` in your user folder, chosen at setup and remembered in `home.json` in the platform
config folder; `--home` and `$AGENT_HOME` override it) holds `config.json`, `engines.json`, `models/`, `engines/`, `skills/`, `mcp/`,
`runtimes/`, `cache/`, `logs/` and `workspaces/default/` (file tools, `memory/db.sqlite3`, `artifacts/`). Each project can have
a `.agents/` folder. Full layout and the contents of `config.json` and `engines.json`: [workspace.md](workspace.md).

`AGENT_HTTP_REWRITE="https://github.com=https://mirror.example/gh"` redirects downloads by URL prefix (rules separated by `;`), and
`HF_ENDPOINT` does the same for Hugging Face. Source: the CLI lives in `apps/cli/src/`; `cli_core` holds the testable parts (argument
parsing, slash commands, config), `commands.cpp` the subcommands, `chat.cpp` and `tui.cpp` the chat.
