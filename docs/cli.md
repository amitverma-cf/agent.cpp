# The `agent` CLI

`agent` is the command-line app built on top of `libagent`. It has a setup wizard, an engine and model
manager, and a text chat with streaming and tool calls. It uses only `libagent`, `agent_hub` (Hugging
Face downloads), `agent_store_sqlite` (conversation storage) and simdjson.

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

## Files

| File | Location |
|---|---|
| `config.json` | `$AGENT_CONFIG_DIR`, else `%APPDATA%\agent\` (Windows), else `$XDG_CONFIG_HOME/agent/`, else `~/.config/agent/` |
| `engines.json` | next to `config.json` |
| `models.json` + model files | `models_dir` from `config.json`, else the hub default (`$AGENT_MODELS_DIR`, `~/.local/share/agent/models`, `%LOCALAPPDATA%\agent\models`) |
| workspace | `workspace` from `config.json`, else `~/.local/share/agent/workspace` (`%LOCALAPPDATA%\agent\workspace`). The file tools are sandboxed to it. Conversations go in `memory/db.sqlite3`, logs in `logs/`, and generated media in `artifacts/`. |

`--config-dir DIR` works with any command and overrides the config location.

`config.json`:

```json
{
  "version": 1,
  "workspace": "/home/me/.local/share/agent/workspace",
  "models_dir": "/home/me/.local/share/agent/models",
  "default_model": "qwen2.5-0.5b-instruct-q4_k_m",
  "capabilities": ["llama", "sherpa", "sdcpp", "onnx"],
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

## agent setup

```sh
agent setup                         # asks questions on a terminal
agent setup --yes                   # accepts the defaults; also what happens when stdin is not a TTY
agent setup --caps llama,sdcpp --workspace ~/agent-ws --models-dir /data/models --engines-dir /opt/engines --backend cpu
```

1. **Detects hardware.** It reports the CPU core count, whether `nvidia-smi` is on `PATH`, whether
   `/dev/nvidia0` exists, and which `/dev/dri/renderD*` nodes are present. From that it suggests a
   backend: `cuda`, `vulkan` or `cpu`. The backend is only recorded for now.
2. **Picks engines.** The default is the starter pack from ROADMAP §6: `llama,sherpa,sdcpp,onnx`.
   `trellis` is not included yet because it has no engine.
3. **Looks for each engine's plugin** (`agent-engine-<name>.so` / `.dll` / `.dylib`) in `--engines-dir`
   and the default plugin directories. It loads each plugin it finds to check it and records the plugin
   in `engines.json`. Engines already listed in `engines.json` are kept. For an engine it cannot find, it
   prints a note that explains how to build the engine and register it with `agent engine add`.
   Downloading prebuilt engine plugins from GitHub releases is not implemented yet.
4. **Writes `config.json` and `engines.json`** and creates the workspace directory. If a `config.json`
   already exists, its values are the defaults.

## agent engine

```sh
agent engine ls                     # name, version, backend, ok/MISSING, path, caps
agent engine add build/engines/sdcpp/agent-engine-sdcpp.so
agent engine rm sdcpp               # forgets the entry; the file stays
```

`add` loads the plugin with `agent::load_engine_plugin` and prints the engine's name, version and
capabilities before it records the plugin. A plugin that fails to load (wrong ABI version, missing
dependency libraries) is rejected and the load error is shown.

## agent model

```sh
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M [--name qwen] [--allow-restricted]
agent model ls
agent model rm qwen [--keep-file]
```

`pull` uses the hub (see [api/hub.md](api/hub.md)) for spec resolution, resumable downloads, sha256
checks and the licence policy. Progress is drawn on stderr. Ctrl-C cancels the download and keeps the
`.part` file, so the same command resumes it. If a model's licence restricts use, the pull is refused
unless you pass `--allow-restricted`. On a terminal you are asked instead. The first chat model you pull
becomes `default_model`.

## agent chat

```sh
agent chat                          # default_model, else the first .gguf in models.json
agent chat --model qwen             # a models.json name ...
agent chat --model /tmp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf    # ... or a path
agent chat --engine llama           # engine name from engines.json, or a plugin path
agent chat --resume chat-20261031-142501
agent chat --list                   # saved conversations in this workspace
```

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

## agent run

`agent run <graph.json>` runs a declarative graph (see [api/agent-json.md](api/agent-json.md) and
`examples/graphs/`). It uses the same model and engine resolution as `agent chat`, with the
graph's `"model"` / `"engine"` hints as defaults that `--model` / `--engine` override.

```bash
agent run examples/graphs/research_pipeline.json --model qwen2.5-0.5b --set "topic=Why is the sky blue?"
```

| Flag | |
|---|---|
| `--set key=value` | seed graph state (repeatable) |
| `--thread ID` | checkpoint thread (default `run`) |
| `--resume ID [--input TEXT]` | continue an interrupted or failed run; `TEXT` is the human reply, or `approve` for pending tool calls |
| `--yes` | approve tool calls the permission policy would ask about |
| `--model`, `--engine`, `--engine-options`, `--workspace`, `--inprocess` | as for `agent chat` |

Agent nodes can name their own model (`"model": "<catalogue name or path>"`). Those models are
loaded through the [model registry](api/models.md) under `memory_budget_mb`, so a graph can use
more models than fit in memory at once. The default model is then only loaded if some agent node
has no `"model"`.

When the graph finishes, the CLI prints the status and the final state. When it is interrupted,
it prints the reason, any paused parallel branches and pending tool calls, and the command to
resume (`--input` answers every paused branch). Checkpoints and the graph event log live in
`<workspace>/memory/checkpoints.sqlite3` (the SQLite checkpointer), unless the graph file says
`"checkpointer": "memory"`. Runs checkpointed by earlier CLI versions in the workspace's key-value
store cannot be resumed.

## Code layout

`tools/cli/src/`:

- `cli_core.{hpp,cpp}`: the testable logic, built as the `agent_cli_core` library and covered by
  `tests/unit/test_cli.cpp`. It handles argument parsing, `config.json` and `engines.json`, model and
  engine resolution, slash commands, the stream filter and the progress bar.
- `cli_platform.cpp`: config and data directories, the executable's location, hardware detection,
  TTY checks, line reads that Ctrl-C can interrupt, the stderr redirect, and opening files with the
  default app.
- `commands.cpp`: `setup`, `engine`, `model` and `run`.
- `chat.cpp`: the chat screen.
- `main.cpp`: dispatches to the commands.
