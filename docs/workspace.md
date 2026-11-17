# Workspace layout

agent.cpp keeps everything it writes in one place: the **global home**. A project can have an `.agents/` folder for the files its
owner writes (skills, MCP servers, rules); the agent's own data about a project is in the home, not in the project. Nothing is installed globally, and the agent never needs `npm` or `python` on the
machine (see "Runtimes" below).

## One folder: `~/.agent.cpp`

The installer puts everything in one folder, `~/.agent.cpp` (`%USERPROFILE%\.agent.cpp` on Windows): the program in `app/`, and next to
it the files below. Nothing is stored anywhere else; the only change outside the folder is a PATH entry for `app/`. Removing the folder
removes agent.cpp. Set `AGENT_HOME` before installing to use another folder, and `agent home DIR` moves the data to a different place
later (a small `home.json` in the install folder remembers it). `--home DIR` and `$AGENT_HOME` override it for a single run. A program
that does not run from an install (a build tree, a package) falls back to a pointer file in the platform config folder.

Older versions used three folders (the program in `~/.local/share/agent-cpp`, the files in `~/.agent.cpp-workspace`, the pointer in
`~/.config/agent`). The installer removes the old program folder; the old files folder keeps working through its pointer, or can be moved
to `~/.agent.cpp/`.

| Folder / file | What it holds |
|---|---|
| `app/` | the program: `agent`, `agent-engine-host` and the bundled engines. Replaced on update |
| `config.json`, `engines.json` | settings; the engine plugins in use |
| `engines/` | engine plugins, one folder per engine and version |
| `models/` | `models.json` and the model files, one folder per engine (`llama/`, `sdcpp/`, `onnx/`, `audiocpp/`): `models/llama/owner/repo/file.gguf`; projectors sit beside their model. Models pulled earlier stay where they are |
| `skills/` | skills available in every project |
| `mcp/` | MCP server definitions. Keys are never stored here: they are referenced by name |
| `plugins/` | third-party plugins, each with its own data folder |
| `runtimes/` | private `bun` and `uv` installs, used only by agent.cpp |
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

The only file outside the home is `home.json`, which names the home: it is inside the install folder, or in the platform config
folder for a program that does not run from an install.

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

## Older layouts

An explicit `--config-dir` or `$AGENT_CONFIG_DIR` keeps the previous layout: `config.json` and `engines.json` in that
folder, models and the workspace in the platform data folders.

## SQLite files

Every database (conversations in `memory/db.sqlite3`, graph checkpoints, RAG chunks) is opened the same way: WAL journal (writes
append to `<db>-wal`, the `<db>-shm` index is memory-mapped, readers never block the writer), `synchronous=NORMAL` (a commit
survives a crash of the program; a power cut can lose the last few commits but never corrupts the file), a 2 MiB page cache per
connection, a 64 MiB read-only memory map (the operating system's reclaimable file cache, not heap), temporary tables on disk and
a WAL that is truncated at 16 MiB. Every row carries a time: `created_at` / `updated_at` (unix ms) on conversations and RAG
chunks, `created_ms` on checkpoints, `ts` on graph events (`datetime(updated_at/1000, 'unixepoch')` reads them).

## Runtimes

Many MCP servers are published as npm or Python packages. Instead of requiring Node or Python, agent.cpp
`agent runtime install bun` / `uv` downloads `bun` (a drop-in for `npx`) and `uv` (for Python tools, including their own Python) into `runtimes/`
with a checksum, points their caches at `cache/`, and runs those servers there, sandboxed. Nothing touches the global
PATH.

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
  "memory_budget_mb": 0
}
```

`engines.json` is the engine manifest that `agent::load_engine_manifest` reads. The CLI records each engine's name, version,
capabilities and backend, and an optional `"isolation": "process" | "inprocess"`:

```json
{ "engines": [ { "name": "llama", "path": "/opt/agent/lib/agent/engines/agent-engine-llama.so", "version": "b12020",
                 "backend": "local", "caps": ["text_gen", "tool_call", "vision_in", "audio_in"] } ] }
```

An entry with only `"path"` works too; the name is taken from the file name (`agent-engine-<name>.so`).
