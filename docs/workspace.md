# Workspace layout

agent.cpp keeps everything it writes in one place: the **global home**. A project can have an `.agents/` folder for the files its
owner writes (skills, MCP servers, rules); the agent's own data about a project is in the home, not in the project. Nothing is installed globally, and the agent never needs `npm` or `python` on the
machine (see "Runtimes" below).

## One folder, chosen when agent.cpp is installed

The installer asks where agent.cpp should keep everything, once, before anything is downloaded (default `~/.agent.cpp`, or
`%USERPROFILE%\.agent.cpp` on Windows; it must be a full path, and an empty or new folder). Everything is inside it: the program, the
engines, the runtimes, the models, the scripts, the data. The folder is remembered as the user environment variable **`AGENT_HOME`**
(a line in the shell profiles on Linux and macOS) and `<folder>/bin` is put on `PATH`; those two entries are the only thing outside the
folder. The program reads `AGENT_HOME`, or, started by its full path, the folder it sits in. `--home DIR` overrides it for one run.
Removing the folder removes agent.cpp; `agent uninstall` does that and the two entries.

There is no pointer file and no second folder. The program and the data are not moved apart by setup: to use another folder, uninstall
(`--keep-data` keeps the models and conversations) and install again with the new one.

| Folder / file | What it holds |
|---|---|
| `bin/` | the program: `agent`, `agent-engine-host` and the `api` engine. Replaced on update |
| `scripts/` | `install.sh` / `install.ps1` and `components.tsv` of the installed release: what `agent setup`, `update`, `uninstall`, `engine pull` and `runtime install` run. They stay as installed until `agent update` |
| `config.json`, `engines.json` | settings; the engine plugins in use |
| `engines/` | one folder for each engine the installer added: our plugin and, where one exists, the upstream library it loads (onnxruntime), with a `.pin` file that names the backend and versions |
| `models/` | `models.json` and the model files, one folder per engine (`llama/`, `sdcpp/`, `onnx/`, `audiocpp/`): `models/llama/owner/repo/file.gguf`; projectors sit beside their model. Models pulled earlier stay where they are |
| `skills/` | skills available in every project |
| `mcp/` | MCP server definitions. Keys are never stored here: they are referenced by name |
| `plugins/` | third-party plugins, each with its own data folder |
| `runtimes/` | private `bun`, `uv`, `node`, `pnpm` and `nub` installs, used only by agent.cpp, put there by the installer script |
| `cache/` | downloads in progress, saved prefix KV states, GPU kernel and shader caches |
| `logs/` | engine and session logs |
| `workspaces/default/` | the default workspace: what the file tools may touch when a chat has no project, plus its memory and artifacts |
| `workspaces/<name>-<hash>/` | the state of one project folder (see below) |

`agent doctor` reports how much each of `models/`, `cache/` and `runtimes/` uses and what is safe to delete.

## The state of a project

A chat, task or graph run works in a workspace: the default one in the home, or a project folder of the user (`--workspace DIR`, the
folder picked in the chat). What the agent writes about a project folder goes to `workspaces/<folder name>-<hash of its path>/` in the
home, never into the project:

| Path in the state folder | What it holds |
|---|---|
| `memory/` | conversations and notes (`db.sqlite3`), the search index, graph checkpoints |
| `sessions/<id>/` | the folder of one conversation: drafts, downloads, scratch files |
| `artifacts/` | generated and received files (images, audio, MCP content) |
| `logs/`, `tmp/`, `cache/` | session and engine logs, engine scratch space, GPU caches |
| `.agents/results/`, `.agents/permissions.json` | long tool results kept in full; tools allowed for good |
| `waiting.json`, `voice/` | paused graph runs; spoken replies when there is no speaker |
| `project.txt` | the path of the project the folder belongs to |

A workspace inside the home keeps its state in itself. What an earlier version wrote into a project folder (`memory/`,
`.agents/permissions.json`, `.agents/results/`, `waiting.json`, and `logs/` when it holds only agent logs) is moved to the state folder
the first time the project is opened; when the conversations cannot be moved (the database is open in another program) the project
folder is used as before for that run. With `--config-dir` (no home) the state stays in the workspace.

Outside the home there is nothing but the `AGENT_HOME` variable and the `PATH` entry.

## A project: `.agents/`

`agent init` creates `.agents/` in a project folder. agent.cpp uses it for everything project-scoped and does not
depend on any other tool's folders.

| Path | Committed? | What it holds |
|---|---|---|
| `.agents/skills/` | yes | skills for this project |
| `.agents/mcp.json` | yes | MCP servers for this project; keys are referenced by name |
| `.agents/local/` | no | personal overrides: path grants for this project, MCP data |

`agent init` writes a `.gitignore` inside `.agents/` for the private folders. The project is the nearest
ancestor of the working directory that contains a `.agents/` folder.

## Without a home

`--config-dir DIR` (or `$AGENT_CONFIG_DIR`) uses a plain folder instead: `config.json` and `engines.json` in it, models and the workspace in the
platform data folders.

## SQLite files

Every database (conversations in `memory/db.sqlite3`, graph checkpoints, RAG chunks) uses a WAL journal and `synchronous=NORMAL`: a
commit survives a crash of the program, and a power cut can lose the last few commits but never corrupts the file. Rows carry a time
(`created_at` / `updated_at` in unix ms, `created_ms` on checkpoints, `ts` on graph events).

## Runtimes

Many MCP servers are npm or Python packages. `agent runtime install bun` / `uv` puts `bun` (for `npx`) and `uv` (for Python tools, with their
own Python) into `runtimes/`, checksummed; their caches are in `cache/` and the servers run there, sandboxed. Nothing touches the global
`PATH`.

## config.json and engines.json

`config.json` is written by `agent setup`:

```json
{
  "version": 1,
  "workspace": "/home/me/.agent.cpp/workspaces/default",
  "models_dir": "/home/me/.agent.cpp/models",
  "default_model": "qwen2.5-0.5b-instruct-q4_k_m",
  "capabilities": ["llama", "audiocpp", "sdcpp", "onnx"],
  "backend": "cpu",
  "context_window": 4096,
  "max_tokens": 512,
  "temperature": 0.7,
  "memory_budget_mb": 0,
  "idle_unload_minutes": 10,
  "stt_model": "",
  "tts_model": "",
  "vad_model": "",
  "tts_voice": "",
  "image_model": ""
}
```

`engines.json` is the engine manifest that `agent::load_engine_manifest` reads. The CLI records each engine's name, version,
capabilities and backend, and an optional `"isolation": "process" | "inprocess"`:

```json
{ "engines": [ { "name": "llama", "path": "/opt/agent/lib/agent/engines/agent-engine-llama.so", "version": "b12020",
                 "backend": "local", "caps": ["text_gen", "tool_call", "vision_in", "audio_in"] } ] }
```

An entry with only `"path"` works too; the name is taken from the file name (`agent-engine-<name>.so`).
