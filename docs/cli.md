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

## agent setup, update, uninstall, home

`agent setup`, `update` and `uninstall` are not code in this program: they run the installer script that came with the installed release
([`scripts/install.sh`](../scripts/install.sh) / [`install.ps1`](../scripts/install.ps1), kept in `<home>/scripts/`). The full description is
under [agent setup, update, uninstall](#agent-setup-update-uninstall) below. `agent home` prints the folder everything is kept in.

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

- **Sidebar:** the saved conversations, grouped by project (the folder the agent works in). Click a session, or press `Ctrl+K`, move
  with Up/Down and press Enter. Picking a session from another project switches the agent to that folder: its files, tools and memory
  change with it (the model is loaded again for the new folder). `Ctrl+N` starts a new chat, `Ctrl+B` hides the sidebar. The projects
  are remembered in `workspaces.txt` in the agent home.
- **Conversation:** the messages in a frame, answers rendered as markdown (headings, lists, quotes, tables, framed code), tool calls as
  cyan lines, and under every answer a line with the model, the speed in tokens per second, the token counts and the time.
- **Input:** at the bottom, with a command palette (type `/` or press `Ctrl+P`), history on Up/Down, and a footer with the speeds and how
  full the context window is.

**The layout follows the width of the terminal** and changes the moment it is resized (a split pane, a phone over SSH, a tmux window):

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
| `/models` | models in memory with size and idle time, and what the memory manager loaded or unloaded |
| `/load MODEL`, `/unload MODEL` | load an installed model of any kind now (speech, image, embeddings), or free a loaded one |
| `/clear` | forget the conversation (the system prompt stays) |
| `/diff` | what the tools changed in files this session, as a unified diff |
| `/undo` | put back the files of the last tool call (a file it created is removed) |
| `/doctor`, `/setup` | the commands of the same name, run inside the chat |
| `/exit`, `//text` | quit; send a message that starts with `/` |

Options: `-m/--model`, `-e/--engine`, `--workspace DIR`, `--ctx N`, `--max-tokens N`, `--engine-options JSON`, `-y/--yes`
(approve tool calls without asking), `--verbose` (show engine logs), `--inprocess`.

The engine follows from the model file unless `--engine` is passed: the file's own header tells a language model (`llama`) from a
speech model (`audiocpp`) or an image model (`sdcpp`); `.onnx` uses `onnx`. Engines run inside the `agent` process, so models are shared,
loaded on first use and unloaded by the model registry when memory is needed (the budget is 85% of the installed memory unless
**Settings > Memory budget** sets one; a model nobody used for ten minutes is unloaded). An engine entry with `"isolation": "process"` in
`engines.json` runs in a separate sandboxed `agent-engine-host` instead, where a crash fails one turn and the next message restarts it
([isolation](api/engines.md#isolation)). Engine tasks asked for in the chat (speech, images, embeddings, reranking) run in the same
process through the same registry. Conversations are saved after every turn and the id is printed on exit. Tools that
change files or run commands (`run_command`, `delete_path`, `move_path`) ask first; without a terminal they are denied unless
`--yes` is passed.

### Rules for a project

The rules written for a project become part of the system prompt of every new conversation: `AGENTS.md` in the workspace, then
`.agents/AGENTS.md`, then every `.agents/rules/*.md` in name order, then the personal `AGENTS.md` in the agent home. At most 16 KiB are
used; a file that does not fit is cut and says so. Nothing is read from outside these places.

### Without a terminal: `-p` and `--jsonl`

```sh
agent chat -p "Summarise README.md in one line" -y      # answer once and exit
git diff | agent chat -p - -y                           # - reads the prompt from stdin
```

`-p` (`--prompt`) prints only the answer on stdout; tool lines, permission lines and notes go to stderr, so the output can be piped.
The exit code is 1 when the turn failed. Tool calls that need permission are denied unless `-y` is passed. The conversation is saved
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
fenced block after the text, and the chat says which files it attached. A line that starts with `!` runs a personal shell command in the
project folder (`!git status`); its output is shown and added to the conversation, so the model can see it. These are the commands, not the
agent's: they are not sandboxed and do not ask.

### Modes, thinking, permissions

- `/mode plan` lets the agent only read (no file changes, no commands); `/mode auto` never asks; `/mode ask` (default, also `/mode manual`) asks before `run_command`, `delete_path`, `move_path`,
  `agent_run`, `web_fetch` and `web_search`. `write_file` and `append_file` do **not** ask: they are shown in the transcript and put back with `/undo` (`/diff`
  lists them), and `/mode plan` refuses them. The status line shows the mode when it is not `ask`.
- **How far a change reaches.** Every conversation has a folder of its own, `sessions/<id>/` in the state folder of the project
  ([workspace.md](workspace.md#the-state-of-a-project)); the agent is told its path and uses it for drafts, downloads and anything that
  is only for that conversation. A change is at one of three levels: **session** (that folder; never asks), **project** (the folder
  the chat works in) and **workspace** (other projects the agent knows, and the `workspaces/` folder of the home). A chat opened on a
  project starts at project level; a chat with no project (the default workspace) starts at session level. A change at a wider level
  than the conversation has is asked about in `ask` mode, and a yes widens the conversation for the rest of it. In `auto` mode the
  agent widens by itself and says so in the transcript (`[access] this conversation now works at workspace level`), inside fixed
  bounds: it never writes into the program, engines, runtimes or settings of the home, and it uses a folder outside all three levels
  only when the message names that folder. `/mode` shows the mode and the level.
- `/thinking [on|off]` shows or hides the `<think>...</think>` blocks of reasoning models (hidden by default).
- Answering `a` to a permission question allows that tool for good **in this project**: it is kept in `.agents/permissions.json` in the project's state folder.
  `/permissions` lists them and `/permissions clear` forgets them. The question shows the call's arguments, one per line.
- `/compact` summarises the oldest messages to free context (the last six stay as they are).

### Sessions

`/rename TITLE` names the conversation in the sidebar. `/delete` asks, `/delete yes` removes the saved conversation for good and starts a
new one. In the session picker (`Ctrl+K`) typing filters the list, and `Del` removes the highlighted conversation.

### Doing more by prompt

The chat can drive everything the `agent` command does. The model has these tools, and the user is asked before the ones that reach
outside the conversation:

| Tool | What it does |
|---|---|
| `agent_run` | runs `agent task`, `agent run`, `agent model`, `agent engine`, `agent skill`, `agent mcp`, `agent workflow`, `agent runtime`, `agent doctor`, `agent schedule` or `agent wake`: images, video, 3D meshes, speech, transcripts, embeddings, ONNX models, graphs, downloads. Generated files are listed by `/open` |
| `settings` | reads and changes the chat's safe settings when you ask in plain words ("make it faster", "more creative", "longer answers"): temperature, longest answer, context window, KV cache, speculative decoding, draft model, MTP and mode. It asks first, applies the change when the answer is finished and says what changed; the model reloads for most of them. Local only, the web address and the other privacy switches are never changed by the agent |
| `remember`, `recall` | lasting notes about the user or the project. Notes and passages of earlier chats that fit a new message are put in front of it automatically (see Memory search) |
| `todo` | a checklist for a request with several steps, rewritten as the work moves on |

`/task CAPABILITY --model NAME --text TEXT` and `/run FILE.json` run the same thing directly. A tool result longer than 6000 bytes
reaches the model as its beginning and end; the whole text is saved under `.agents/results/` in the project and the note says which
file, so the model can read or search it.

How the agent presents itself comes from `SOUL.md`, `IDENTITY.md` and `ROLE.md`, plain Markdown files in the project folder or in the
agent home, added to the system prompt next to `AGENTS.md`.

### Memory, timers, learned skills, plugins

These work without a command: the agent uses them when a request calls for it.

| What | How it works |
|---|---|
| Memory search | Notes (`remember`) and every message of the saved chats of a project are indexed in `memory/index.sqlite3` under the project. `recall` searches the index by words (full text, BM25) and by meaning (vectors) together and lists the closest passages. With an embedding model installed (for example `bge-small`) the meaning search is real; without one the index goes by words. A second run embeds only what is new. Before each message the same index is searched once: the notes that fit come first, then passages of earlier chats of the project, each marked "said in an earlier chat". A passage is kept only when it shares two words with the message or is close to it in meaning, and the whole block stays under 1,200 characters, so a message that fits nothing gets no block. |
| Timers | "Remind me in 20 minutes", "every morning at 9 check the build": the `timer` tool keeps the request in `timers.json` in the home folder and puts it to the agent as a message when it is due and the chat is idle. A timer that came due while the program was closed fires once when it is opened again. Work that has to run with the program closed belongs to [`agent schedule`](#agent-schedule), which uses the system scheduler. |
| Learned skills | After a task that took several steps, the agent keeps how it was done with `skill_learn` as `skills/NAME/SKILL.md` in the home folder; the skill is offered from the next message on. When the steps of a skill fail, the agent repairs it: new steps replace the old ones and the cause is kept under `## Lessons`. |
| Plugins | A plugin is a folder under `plugins/` in the home folder (or `.agents/plugins/` in a project) with a `plugin.json` and a script. Each tool it declares is offered to the model as `PLUGIN_TOOL` and asks before it runs. |

A plugin manifest:

```json
{ "name": "weather", "description": "Weather lookups", "runtime": "bun", "entry": "main.js", "timeout_seconds": 60,
  "tools": [ { "name": "forecast", "description": "Forecast for a city",
               "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } } ] }
```

`runtime` is `bun`, `node`, `nub`, `uv` or `python` (the private runtimes of [`agent runtime`](#agent-runtime); `python` falls back to
`python3` on the PATH). The script is started once per call in the plugin folder; the call arrives on standard input as
`{"tool": "forecast", "arguments": {...}, "workspace": "..."}` and what the script prints is the result. A non-zero exit code is an error.
A plugin runs outside the sandbox with the user's rights, like an MCP server, so only plugins from a trusted source belong there.

### Memory and crashes

Engines run inside the program, and the models share one memory budget (85% of the installed memory unless **Settings > Memory budget**
sets one).

- A model nobody used for ten minutes is unloaded.
- When an engine task (speech, an image, ...) does not fit beside the chat model, the chat model is set aside for the time the task takes
  and loaded again afterwards; the conversation is kept. Asked for in the middle of an answer, the task runs right after the answer.
- An engine that crashes takes the program with it once. The crash is recorded against the model that was running
  (`cache/crashed/` in the home folder); from the next start that model runs in its own `agent-engine-host` process, where the same fault
  fails one answer and the chat goes on. Deleting the file in `cache/crashed/` puts the model back in the program.

### Speed

What makes answers come faster, in the order to try it (all of it is under Settings, Model, Speed, or just ask: "make it faster"):

- **A smaller model, a shorter answer limit and a smaller context window** are the surest levers. Every token costs a pass over the whole model,
  and a long conversation costs more per token. Lower the longest answer and the context window (Settings, Model) if you do not need them.
- **KV cache type** (`kv_cache` in `config.json`: `auto`, `f16`, `q8_0`, `q4_0`). A smaller cache type uses less memory and reads less per token,
  which helps long conversations. `auto` picks `q4_0` or `q8_0` by the model's shape (on models that run through llama.cpp's own engine, see MTP below,
  `auto` is llama.cpp's default, `f16`). The very small models are the ones that go wrong at `q4_0`, so leave `auto` unless you need the memory.
- **Speculative decoding** (`speculative`: `off`, `prompt_lookup`, `draft`). The model guesses the next few tokens and checks them in one pass; every guess that
  matches is a token it did not have to generate alone. `prompt_lookup` (the default) guesses by copying from text already in the conversation, so it pays when
  the answer repeats something: quoting, rewriting a file, editing code. It works at the default temperature and with tools; when nothing repeats it costs nothing.
  `draft` lets a small model of the same family guess (set `draft_model` to an installed model; a model with another vocabulary cannot draft). It pays when the small
  model is much faster than the big one and often right; on a CPU with few cores it often does not. Measure with your own models: the line under each answer
  shows what the guesses did (`lookup on, 31 of 40 drafted tokens kept`).
- **MTP** (`mtp`, on by default). A model whose file carries a multi-token-prediction head (Qwen3.5 builds made with it) drafts for itself. The line under
  the answer says `MTP on` and how many guesses were kept. On a few CPU cores the extra passes can cost more than they save for open-ended writing
  (measured numbers are in [benchmarks](benchmarks.md)); switch `mtp` off then.

Changing any of these reloads the model. The line under each answer shows tokens per second, and the bar at the bottom shows `context 951/8192 tokens`: the
prompt (the whole conversation as the model reads it) against the model's window. It drops when the conversation is shortened or a new one starts.

### The web

The model has a `web_fetch` tool (an http or https address; HTML is reduced to readable text, at most 20,000 characters) and, when
`config.json` has a `search_url`, a `web_search` tool: `"search_url": "http://localhost:8080/search?format=json&q={q}"`, for example a SearXNG
instance. Both ask first (like `run_command`); `-y` or `/mode auto` approve them. Nothing else on the model's side touches the network.

### Constrained answers

`--json` makes every answer valid JSON; `--grammar FILE` samples under a GBNF grammar (for example `root ::= "yes" | "no"`). Both are for the
llama engine and switch the tools off for the session (a grammar cannot share a turn with the tool-call grammar). In `task text_gen` the same
is `--params '{"json": true}'` or `--params '{"grammar": "..."}'`.

### The screen

- **Sidebar:** the logo, a New chat button, the sessions (scrollable) and a Settings button. On a terminal narrower than 84 columns it is a
  screen of its own, opened with the menu icon in the title bar (like a phone app); below 70 columns it fills the whole terminal.
- **Title bar:** the session name and a trash icon that deletes the session, after a confirmation, from the list, the saved history and the
  memory.
- **Messages:** every user message has a copy icon and a delete icon (the delete removes the message and the model's answers to it from
  what the model sees, after a confirmation). Every answer has a copy icon, how fast it was written (tok/s, when the engine measured it) and
  how long the whole answer took. Copy goes to the clipboard through the terminal (OSC 52) and the system's clipboard tool.
- **Input:** a `mic` box on the left of the input records until it is clicked again, then writes what was said into the input with the first speech
  recognition model that is installed (`agent model pull <a speech model link>`). `/` lists commands, `@file` attaches a file, `!command` runs
  a shell command. A line under the input suggests the next useful key or command while typing.
- **Bottom bar:** CPU use, RAM used and installed, the size of the files folder; and on the right the engine, the models held in RAM, the mode
  and the context used and available.
- **Shortcuts:** `?` on an empty input, or F1, lists every keyboard shortcut; the sidebar shows F1 under the Settings button. To change one, open the Settings, choose Shortcuts, press Enter on an action and then the wanted keys (F1 to F12, Ctrl+letter, Alt+letter or Ctrl+Alt+letter; a key that is already used is refused). The choices are kept in `config.json` under `shortcuts`, and "Put the defaults back" clears them. (A terminal cannot report the Alt key alone.) Ctrl+O opens the
  settings; there, the categories are on the left, and each setting changes with Left and Right or by typing a new value.

### Look

The chat uses four colours: black, white, turquoise and pastel pink, with white at reduced opacity for secondary text and lines. The window
frame is turquoise; the input box turns turquoise while it takes input. Icons are Unicode symbols; `AGENT_ICONS=nerd` uses Nerd Font glyphs
(install a [Nerd Font](https://www.nerdfonts.com) in the terminal first) and `AGENT_ICONS=ascii` plain characters.

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
agent model pull https://huggingface.co/Qwen/Qwen3-4B-GGUF   # a repository page: the quant is chosen automatically
agent model pull C:\Downloads\model.gguf                   # an existing file
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M [--name qwen] [--allow-restricted]
agent model pull owner/repo:* [--include int8]    # every file of a folder model (TRELLIS)
agent model add-api numes-glm --base-url https://api.numes.ai/api/v1 --model dealignai/glm-5.3-flash-uncensored-fp8 --key-env NUMES_API_KEY
agent model ls
agent model rm qwen [--keep-file]
```

`pull` takes a Hugging Face link (a repository page, or the page or download link of one file), `owner/repo[:quant-or-file]`, or a model
file on this disk. There is no list of suggested models: the file itself says which engine runs it (llama.cpp, stable-diffusion.cpp,
audio.cpp, ONNX Runtime or TRELLIS.2), and it is put in that engine's folder under `models/` and recorded in `models.json`. A file already
installed is not fetched again. An embedding or reranker model is detected too; the command prints the `--engine-options` it needs. Downloads resume after Ctrl-C
(the `.part` file is kept), are checked against the published hash, and refuse models with restrictive licences unless
`--allow-restricted` is passed (a terminal asks for confirmation). Vision, audio and OCR repos also fetch their multimodal projector, record it
and use it automatically. The first chat model pulled becomes the default.

`add-api NAME --base-url URL --model ID [--key-env VAR] [--max-tries N]` registers a remote OpenAI-compatible model instead of a file: the
requests are made by the optional `api` engine through a bun child process, the key is read from the environment variable `VAR`
(default `NUMES_API_KEY`), failed requests are retried with backoff and the chat footer shows the retries. It is refused to start when
Local only is on. See [remote models](guides/remote-models.md).

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
engine, the chat turn on the selected model. Step-by-step with downloads: [guides/voice-assistant.md](guides/voice-assistant.md).

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
time; with no microphone `voice` asks for `--input-wav`, with no speaker replies go to `<workspace>/voice/`.

## agent engine

```sh
agent engine ls                       # name, version, ok/MISSING, path, capabilities
agent engine add path/to/agent-engine-sdcpp.so
agent engine pull llama               # the installer script adds an engine of the installed release (--backend vulkan|cuda12|cuda13|rocm|cpu)
agent engine rm sdcpp                 # an engine the installer added is removed with its files; one added by path is only forgotten
```

`add` loads the plugin, prints its name, version and capabilities, and records it in `engines.json`; a plugin with the wrong ABI or
missing libraries is rejected with the reason. Engines also ship in the release archives; to build one from source, see the
[README](../README.md#build-from-source).

## agent skill

```sh
agent skill search pdf
agent skill add anthropics/skills/pdf [--project] [--yes]
agent skill ls
agent skill rm pdf
```

```sh
agent skill browse                  # a built-in list of well-known skills with one line each
agent skill on|off NAME             # switch an installed skill on or off (SKILL.md is renamed SKILL.md.off)
```

A skill is text the model follows, with optional helper files. Installed skills are offered to the agent in the chat from the next message. `add` shows the source, the file count, size and any scripts, and asks
before downloading (at most 200 files and 20 MiB). Scripts only run through `run_command` in the sandbox, after approval.

## agent workflow

```sh
agent workflow save daily-report graph.json   # check the graph and keep it as <home>/graphs/daily-report.json
agent workflow ls | show NAME | rm NAME
agent run daily-report                        # a bare name runs the saved workflow, from any folder
```

In the chat the agent saves a workflow it was asked for with the `workflow_save` tool and runs it with `agent_run ["run","NAME"]`; the names of
the saved workflows are in its system prompt. A run that ends `status: interrupted` is waiting (a wait node, a request for input, or an agent
node with `interrupt_before_tools` that reached a tool call the policy rates "ask": `run_command`, `delete_path`, `move_path`). The agent shows
the question with `ask_user` and continues with `agent run NAME --resume THREAD --input ANSWER`; `approve` lets the pending tool calls run.

## agent mcp

```sh
agent mcp search filesystem
agent mcp add io.example/files [--project] [--yes]
agent mcp ls
agent mcp rm files
```

```sh
agent mcp browse                              # well-known servers: filesystem, memory, fetch, git, time, playwright, github, ...
agent mcp add filesystem [--folder DIR]       # a server from that list
agent mcp add NAME --run "npx -y PACKAGE"     # any package or command (also uvx ...), or: --url https://host/mcp
agent mcp on|off NAME                         # a server added with `add` is switched with its file (NAME.json.disabled)
```

In the chat, Settings > Skills and Settings > MCP servers do the same from the keyboard (installed items with on/off, a well-known list, search,
adding by name, package or address), and so does the agent when asked in plain words (`agent_run`). Servers and skills added in a chat are
loaded before the next message; a server that cannot start is reported in the result of the command. A project's `.agents/mcp.json` is read
for the chat's workspace even when `.agents/` was created after the program started. Without the private bun / uv runtime
(`agent runtime install bun`), `npx` / `uvx` of the system keep their package cache in the server's own data folder.

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
agent runtime install js|bun|uv|node|pnpm|nub
agent runtime path uv | rm bun
```

Private copies of [bun](https://bun.sh), [uv](https://docs.astral.sh/uv/) (and optionally Node with pnpm, or nub) live in
`<home>/runtimes/`, so servers written for `npx` and `uvx` run without Node or Python on the machine. Archives are checked against
their published SHA-256; all caches stay in `<home>/cache`; nothing is written to the profile.

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

Load a completion script in the shell, for
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

## agent setup, update, uninstall

These three do not contain any installing code. They run the installer script that came with the installed release
(`<home>/scripts/install.sh`, or `install.ps1` on Windows), so what a release installs and removes always matches that release. The same
script is what `curl -fsSL https://agent.qozu.fyi/install.sh | sh` and `irm https://agent.qozu.fyi/install.ps1 | iex` run for a first install,
and it can be called directly: `install.sh add llama`, `install.sh ls`, `install.sh rm sdcpp`.

```sh
agent setup                    # engines, runtimes and a first model; asks, or take --engines llama,onnx --runtimes bun,uv --model LINK --yes
agent engine pull llama        # add an engine of the installed release (--backend vulkan|cuda12|cuda13|rocm|metal|cpu to choose)
agent runtime install bun      # add a runtime: bun, uv, node, pnpm, nub
agent engine rm sdcpp          # remove one (agent runtime rm NAME)
agent update [--check]         # move to the newest release; installed engines and runtimes follow the versions it pins
agent uninstall [--keep-data] [--yes]
```

**Pinned.** Each release lists, in `scripts/components.tsv`, the exact version and SHA-256 of every component it works with: its own
engines, the official onnxruntime and the runtimes (bun, uv, node, pnpm, nub). `setup`, `engine pull` and `runtime install` use
the list of the installed release and never "the latest"; only `agent update` reads a newer one. A download whose SHA-256 differs is
refused and nothing is installed.

**Backends.** The script looks at the machine (operating system, CPU architecture, NVIDIA driver and CUDA version, Vulkan) and picks the
best backend a component has; `--backend` or `AGENT_BACKEND` chooses another. `agent engine pull NAME` says when a component has no
build for this system. `uninstall --keep-data` removes the program, engines, runtimes and caches and keeps models, conversations and settings;
without it the whole folder goes, with `AGENT_HOME` and the `PATH` entry. It always asks first (`--yes` skips the question, which a
script without a terminal needs). On Windows `update` and `uninstall` continue in a window of their own, because a running program cannot
be replaced or deleted.

## Files and layout

One folder holds everything, the one chosen when agent.cpp was installed: it is the environment variable `AGENT_HOME` (`--home` overrides it
for one run). In it: `bin/` (the program), `engines/`, `runtimes/`, `scripts/`, `config.json`, `models/`, `skills/`, `mcp/`, `cache/`,
`logs/` and `workspaces/` (the default workspace, and one state folder for each project folder the agent has worked in: conversations,
memory, logs, generated files). A project folder itself holds only its own files; it can have a `.agents/` folder for skills, MCP servers
and rules. Full layout and the contents of `config.json`: [workspace.md](workspace.md).

`AGENT_HTTP_REWRITE="https://github.com=https://mirror.example/gh"` redirects downloads by URL prefix (rules separated by `;`), and
`HF_ENDPOINT` does the same for Hugging Face. Source: the CLI lives in `apps/cli/src/`; `cli_core` holds the testable parts (argument
parsing, slash commands, config), `commands.cpp` the subcommands, `chat.cpp` and `tui.cpp` the chat.
