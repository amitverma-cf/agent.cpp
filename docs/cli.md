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

## agent setup

```sh
agent setup                                      # asks four questions
agent setup --yes --model qwen2.5-0.5b           # no questions
agent setup --no-models --caps llama,sdcpp --backend cpu
```

Setup chooses the global home (see [workspace.md](workspace.md)), detects your CPU, RAM and GPU, finds the engine plugins you
built or downloaded and records them in `engines.json`, then lists the models your machine can run. It downloads nothing unless
you pick a model, and each model it downloads is smoke-tested with a short real run.

Options: `--home DIR`, `--model NAME|owner/repo[:quant]` (repeatable), `--no-models`, `--yes`, `--caps llama,audiocpp,sdcpp,onnx`,
`--engines-dir DIR`, `--models-dir DIR`, `--workspace DIR`, `--backend cpu|cuda|vulkan`, `--js-runtime bun|node|nub`,
`--download-engines`, `--install-runtimes`. Inside the chat, `/setup` asks the same questions.

## agent chat

```sh
agent                             # or: agent chat
agent chat --model qwen           # a name from models.json, or a path to a model file
agent chat --resume chat-20261031-142501
agent chat --list                 # saved conversations
agent chat --plain                # line mode, for pipes
```

In a terminal this opens a full-screen chat, built with [RTXUI](https://github.com/ArthurSonzogni/RTXUI) (HTML and CSS for the
terminal): a title bar with the model and engine, the conversation, an input line and a footer with speed and token counts.
Answers are rendered as markdown: headings, lists, quotes, tables and fenced code in a frame. Tool calls appear as cyan lines, and a
tool that needs permission asks `y` once, `n`, or `a` for the rest of the session. `Esc` (or Ctrl-C while answering) stops an answer;
Ctrl-C or Ctrl-D at the prompt quits. The mouse wheel, PgUp/PgDn and Home scroll the conversation; `End` jumps back to the newest line
and follows it again. Select text with the mouse to copy it. The screen uses four colors (black, white, neon cyan, hot pink).

| Command | What it does |
|---|---|
| `/model [NAME\|PATH]` | show the loaded model, or switch (the conversation is kept; if the new one fails, the old one is reloaded) |
| `/image PATH [text]` | send an image or audio file with a message (the engine must accept media, e.g. llama with an `mmproj`) |
| `/open [n]` | list files the model generated, or open the n-th |
| `/stats` | token, tool-call and cache counters, loaded models |
| `/clear` | forget the conversation (the system prompt stays) |
| `/doctor`, `/setup` | the commands of the same name, run inside the chat |
| `/exit`, `//text` | quit; send a message that starts with `/` |

Options: `-m/--model`, `-e/--engine`, `--workspace DIR`, `--ctx N`, `--max-tokens N`, `--engine-options JSON`, `-y/--yes`
(approve tool calls without asking), `--verbose` (show engine logs), `--inprocess`.

The engine follows from the model file (`.gguf` uses `llama`, `.onnx` uses `onnx`, `.safetensors` and `.ckpt` use `sdcpp`) unless
you pass `--engine`. Engines run in a separate `agent-engine-host` process, so a crash fails one turn and the next message restarts
it ([isolation](api/engines.md#isolation)). Conversations are saved after every turn and the id is printed on exit. Tools that
change files or run commands (`run_command`, `delete_path`, `move_path`) ask first; without a terminal they are denied unless you
pass `--yes`.

## agent model

```sh
agent model search [TEXT] [--task asr]            # tested models with short names
agent model info owner/repo                       # quants, sizes, projectors, licence of any Hugging Face repo
agent model pull qwen3-4b                         # a short name
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M [--name qwen] [--allow-restricted]
agent model pull owner/repo:* [--include int8]    # every file of a folder model (TRELLIS)
agent model ls
agent model rm qwen [--keep-file]
```

`pull` accepts a short name from the registry, or `owner/repo[:quant-or-file]` from Hugging Face. Downloads resume after Ctrl-C
(the `.part` file is kept), are checked against the published hash, and refuse models with restrictive licences unless you pass
`--allow-restricted` (on a terminal you are asked). Vision, audio and OCR repos also fetch their multimodal projector, record it
and use it automatically. The first chat model you pull becomes the default.

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

`clean` removes one area per flag (partial downloads, rebuildable caches, old logs). Models, memory and conversations are never removed.

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
