# Guide: embedding the library

This guide builds a small C++ program on top of agent.cpp: a local chat model answers a question, calls a tool of
yours through a permission gate, and runs the engine in its own sandboxed process. It walks through the three
entry points you will use:

- `agent::open_session(engine, model)`: one call for the common case, all defaults.
- `agent::engine_config(engine, models, options, base)`: builds a `Config` for an engine and its model files.
- `agent::init(config)`: starts the session from that `Config`.

The examples in [`examples/`](../../examples/README.md) take no arguments and name their model files at the top of
`main()`. The program below combines [`hello`](../../examples/01-hello-agent/hello.cpp) and
[`tools`](../../examples/02-tools/tools.cpp) and adds the isolation settings.

## 1. Build and install agent.cpp

Follow [getting-started.md](../getting-started.md#build) for the prerequisites, then install the library, the CLI, the
engine host and the engine plugins into a prefix:

```sh
git clone --recurse-submodules https://github.com/amitverma-cf/agent.cpp && cd agent.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix $HOME/agent-cpp
```

The prefix holds the library and headers, the engine host, and the plugins (layout in [packaging.md](../packaging.md)).

## 2. Your project

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_agent CXX)
include(GNUInstallDirs)

# Ask for the llama engine. CMake fails here, with a message, if the install does not have it.
find_package(agent-cpp 0.1 REQUIRED COMPONENTS llama)

add_executable(my_agent main.cpp)
target_link_libraries(my_agent PRIVATE agent-cpp::agent)

# Copies the llama plugin and agent-engine-host next to my_agent (in engines/ and in the build folder),
# and installs them with the program. Engines you do not name are not copied.
agent_deploy_engines(my_agent llama)

install(TARGETS my_agent RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

Configure with the prefix from step 1:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/agent-cpp
cmake --build build
```

`agent-cpp::store_sqlite` (SQLite memory store, checkpointer, sqlite-vec) and `agent-cpp::hub` (Hugging Face downloads)
are optional extra targets; link them only if you use them.

## 3. The program

Save this as `main.cpp`:

```cpp
#include <agent-cpp/agent.hpp>

#include <cstdio>
#include <string>

namespace {

// A tool is a function from a JSON argument string to a JSON result. The model decides when to call it.
agent::Result<std::string> get_weather(std::string_view arguments, void *) {
    std::printf("[tool] get_weather(%.*s)\n", static_cast<int>(arguments.size()), arguments.data());
    return agent::ok(std::string(R"({"city":"Paris","temperature_c":18,"sky":"light rain"})"));
}

// The permission gate runs before every tool call: Allow, Deny, or Ask (Ask needs Config::permission_prompt).
agent::PermissionDecision gate(std::string_view tool, std::string_view, void *) {
    return tool == "run_command" ? agent::PermissionDecision::Deny : agent::PermissionDecision::Allow;
}

void print_token(std::string_view token, void *) {
    std::fwrite(token.data(), 1, token.size(), stdout);
}

} // namespace

int main() {
    agent::AgentRuntime runtime; // starts the process-wide backends; frees them at exit

    // The settings every session needs. base is passed to engine_config, which fills in the engine and its model.
    agent::Config base;
    base.workspace_dir = ".agent"; // init() creates it; file tools stay inside it
    base.context_window = 4096;
    base.max_tokens = 256;
    base.permission_check = gate;
    base.tools.push_back(agent::Tool{
        .name = "get_weather",
        .description = "Current weather in a city.",
        .parameter_schema = R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
        .callback = get_weather,
    });

    // Run the engine in an agent-engine-host child. A crash there fails the call (EngineCrashed); this process survives.
    base.engine_isolation = agent::EngineIsolation::Process;
    base.engine_timeout_ms = 5 * 60 * 1000;               // watchdog per request
    base.engine_sandbox.memory_limit_bytes = 8ull << 30;  // address-space cap for the host
    base.tool_sandbox.network = agent::SandboxNetwork::Deny; // run_command gets no network

    // Name the engine and each model file by role. "model" is the chat model; other engines take other roles.
    const agent::ModelSlot models[] = {{"model", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"}};
    auto cfg = agent::engine_config(agent::engines::llama, models, /*extra_options_json=*/{}, base);
    if (!cfg.ok) return std::fprintf(stderr, "config: %s\n", cfg.error.message.c_str()), 1;

    auto session = agent::init(cfg.value);
    if (!session.ok) return std::fprintf(stderr, "init: %s\n", session.error.message.c_str()), 1;

    agent::Conversation chat;
    auto reply = agent::run_turn(session.value, chat,
                                 "What is the weather in Paris right now? Use the get_weather tool.",
                                 /*stream=*/true, print_token);
    std::printf("\n");
    if (!reply.ok) return std::fprintf(stderr, "turn: %s\n", reply.error.message.c_str()), 1;

    std::printf("tool calls: %d\n", agent::get_stats(session.value).total_tool_calls);
    return 0;
}
```

The rest of the API follows the same pattern:

- `engine_config` copies `base` and adds the engine name, the model files and (optionally) the engine's own options as a
  JSON object in `extra_options_json`. The engine options are engine-specific; see [api/engines.md](../api/engines.md).
- `init` returns a `Result<Session>`. `run_turn` takes the session and a `Conversation`; keep as many conversations as you
  like per session. Keep one `run_turn` in flight per session.
- A model that takes several files is opened the same way, with one `ModelSlot` per role, for example
  `{{"model", "sd.gguf"}, {"vae", "vae.gguf"}}` for the sdcpp engine.
- The shortest form is `agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")`. It
  uses `.agent` as the workspace and the library defaults for everything else. Pass a `Config` as the third argument to
  change any of them.

Things to know:

- **Errors are values.** Every fallible call returns `agent::Result<T>`: check `.ok`, then use `.value` or
  `.error.code` / `.error.message`. Nothing throws.
- **`AgentRuntime`** starts and frees the process-wide backends. Create one in `main`, before any session.
- **`Session`** owns the engine instance, the tools, the memory store and the turn arena. It is move-only.
- **`Conversation`** is one conversation: its history and stats. `save_conversation` and `load_conversation` store it
  through the session's memory store.
- **The workspace** (`workspace_dir`) must be set, because `init()` does not guess it (`open_session` does). It is created
  with `models/`, `memory/`, `logs/` and `tmp/` folders. Generated files go to `<workspace>/artifacts/`.

## 4. Run it

The model path and the workspace are relative to the working directory. Put the model at
`models/qwen2.5-0.5b-instruct-q4_k_m.gguf` in the folder you run the program from. The repository's download script
fetches this file (and the others the examples use) into `./models` when it is run from that folder:

```sh
cd ~/my_agent_run                                # any folder you like
~/agent.cpp/examples/download-models.sh          # writes ./models (about 2.7 GB for all examples)
~/my_agent/build/my_agent                        # the program; the engine is found in build/engines/
```

The first run creates `.agent/` in that folder. The output is the streamed answer, one `[tool] get_weather(...)` line for the tool call,
and the number of tool calls.

## 5. Isolation and sandbox settings

| Setting | Library default | This program | What it does |
|---|---|---|---|
| `engine_isolation` | `Default`, which means in process (the CLI uses process) | `Process` | Runs the engine in `agent-engine-host`; a crash fails the call instead of the program. |
| `engine_sandbox` | on, applies only under `Process` | on, `memory_limit_bytes` = 8 GiB | The host can read the plugin, the model and the workspace; it can write only `artifacts/`, the workspace `tmp/` and `/dev/shm`; no network, no `/tmp`. |
| `engine_timeout_ms` | `0` (none) | 5 minutes | The host is killed if one request takes longer. |
| `tool_sandbox` (for `run_command`) | on, network allowed | network `Deny` | `run_command` runs with the workspace writable and system folders read-only. |
| `sandbox_filesystem` (file tools) | on | default | File tools accept only paths that resolve inside the workspace, symlinks included. |
| `permission_check` | none (every call allowed) | `gate` | Decides each tool call before its callback runs. |

Process isolation costs about 40 microseconds per turn on the measurement machine (under 1 microsecond in process), which is
small next to a model call. Details: [benchmarks.md](../benchmarks.md), [security-sandbox.md](../security-sandbox.md),
[api/engines.md](../api/engines.md#isolation).

If the engine needs files outside the model's folder (a projector, voices, a LoRA), name them in `engine_options` (the folder
of each path-like string is added read-only), or list them in `engine_sandbox.read_only_paths`.

## Next

- [Writing a tool](writing-a-tool.md): custom tools, the permission gate, hooks, MCP servers.
- [Writing a graph](writing-a-graph.md): multi-step workflows in C++ or `agent.json`.
- [Writing an engine](writing-an-engine.md): wrap another inference library as a plugin.
- [Examples](../../examples/README.md): the ten example programs and the models they use.
- [API reference](../index.md#documentation-map).
