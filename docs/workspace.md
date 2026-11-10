# Workspace layout

agent.cpp keeps its files in two places and nowhere else: one **global home** and, optionally, an `.agents/`
folder inside each project. Nothing is installed globally, and the agent never needs `npm` or `python` on your
machine (see "Runtimes" below).

## The global home: `.agent.cpp-workspace`

You choose where it goes during `agent setup`. The choice is remembered in a small pointer file, `home.json`, in the
platform's config folder (`%APPDATA%\agent` on Windows, `~/.config/agent` elsewhere). `--home DIR` and `$AGENT_HOME`
override it for a single run.

| Folder / file | What it holds |
|---|---|
| `config.json`, `engines.json` | settings; the engine plugins in use |
| `engines/` | engine plugins, one folder per engine and version |
| `models/` | model files, `models.json`, projectors |
| `skills/` | skills available in every project |
| `mcp/` | MCP server definitions. Keys are never stored here: they are referenced by name |
| `plugins/` | third-party plugins, each with its own data folder |
| `runtimes/` | private `bun` and `uv` installs, used only by agent.cpp |
| `cache/` | downloads in progress, saved prefix KV states, GPU kernel and shader caches |
| `logs/` | engine and session logs |
| `workspaces/default/` | the default workspace: what the file tools may touch, plus memory and artifacts |

`agent doctor` reports how much each of `models/`, `cache/` and `runtimes/` uses and what is safe to delete.

## A project: `.agents/`

`agent init` creates `.agents/` in a project folder. agent.cpp uses it for everything project-scoped and does not
depend on any other tool's folders.

| Path | Committed? | What it holds |
|---|---|---|
| `.agents/skills/` | yes | skills for this project |
| `.agents/mcp.json` | yes | MCP servers for this project; keys are referenced by name |
| `.agents/memory/` | no | conversations and the memory database |
| `.agents/cache/` | no | project caches |
| `.agents/local/` | no | personal notes and overrides |

`agent init` writes a `.gitignore` inside `.agents/` for the three private folders. The project is the nearest
ancestor of the working directory that contains a `.agents/` folder.

## Older layouts

An explicit `--config-dir` or `$AGENT_CONFIG_DIR` keeps the previous layout: `config.json` and `engines.json` in that
folder, models and the workspace in the platform data folders.

## Runtimes

Many MCP servers are published as npm or Python packages. Instead of asking you to install Node or Python, agent.cpp
`agent runtime install bun` / `uv` downloads `bun` (a drop-in for `npx`) and `uv` (for Python tools, including their own Python) into `runtimes/`
with a checksum, points their caches at `cache/`, and runs those servers there, sandboxed. Nothing touches your global
PATH.
