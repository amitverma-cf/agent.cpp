# Guide: embedding the library

This guide installs agent.cpp, links it into your own CMake project and runs a first chat turn on a
local GGUF model through the llama engine plugin, with the engine in its own sandboxed process.

The complete program is [`examples/guides/getting_started_library.cpp`](../../examples/guides/getting_started_library.cpp);
it is built with the examples (`AGENT_BUILD_EXAMPLES`, on by default) and runs as a test on the mock
engine (`ctest -L examples`).

## 1. Build and install agent.cpp

```sh
git clone https://github.com/amitverma-cf/agent.cpp && cd agent.cpp
git submodule update --init --depth 1 vendor/simdjson vendor/llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix $HOME/.local/agent
```

The prefix now holds the library and headers, the `agent` CLI, the out-of-process engine host and the
engine plugins (layout in [packaging.md](../packaging.md)):

```
bin/agent  bin/agent-engine-host
include/agent-cpp/agent.hpp  engine_abi.h  store_sqlite.hpp  hub.hpp
lib/libagent.a  libagent_store_sqlite.a  libagent_hub.a
lib/agent/engines/agent-engine-llama.so
lib/cmake/agent-cpp/
```

Get a model: any chat GGUF works, for example
[Qwen2.5-0.5B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (~400 MB), or
`agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M` with the CLI.

## 2. Your project

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_agent CXX)

find_package(agent-cpp 0.1 REQUIRED)

add_executable(my_agent main.cpp)
target_link_libraries(my_agent PRIVATE agent-cpp::agent)
# The engine host is found next to the executable, in ../bin, on PATH or through $AGENT_ENGINE_HOST.
# For a program that lives elsewhere, bake in the installed one:
target_compile_definitions(my_agent PRIVATE AGENT_HOST_PATH="${agent-cpp_ENGINE_HOST}")
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/.local/agent
cmake --build build
```

`agent-cpp::store_sqlite` (SQLite memory store, checkpointer, sqlite-vec) and `agent-cpp::hub`
(Hugging Face downloads) are optional extra targets. The package also sets `agent-cpp_ENGINE_DIR`
(`<prefix>/lib/agent/engines`) and `agent-cpp_ENGINE_HOST` (`<prefix>/bin/agent-engine-host`).

## 3. The first program

```cpp
#include <agent-cpp/agent.hpp>

#include <cstdio>
#include <string>

namespace {

void print_token(std::string_view token, void *) {
    std::fwrite(token.data(), 1, token.size(), stdout);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <engine plugin or name> <model> [prompt] [workspace]\n", argv[0]);
        return 2;
    }

    agent::AgentRuntime runtime; // process-wide backend init/teardown

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = argv[1]; // a plugin path (loaded on demand) or the name of an already loaded engine
    cfg.model = argv[2];
    cfg.workspace_dir = argc > 4 ? argv[4] : ".workspace";
    cfg.context_window = 4096;
    cfg.max_tokens = 256;

    // Run the engine in an agent-engine-host child: a crash or abort() in the engine fails the call with
    // ErrorCode::EngineCrashed and the next call restarts the host; this process survives.
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_timeout_ms = 5 * 60 * 1000; // watchdog per request

    // The host's OS sandbox (Config::engine_sandbox) is on by default: plugin dir + model read-only,
    // workspace read-only, workspace/artifacts read-write, no network, no /tmp. Add a memory cap:
    cfg.engine_sandbox.memory_limit_bytes = 8ull << 30; // RLIMIT_AS of the host

    // run_command (a default tool) runs in its own sandbox (Config::tool_sandbox, on by default): the
    // workspace read-write, system dirs read-only. The network is allowed by default; turn it off:
    cfg.tool_sandbox.network = agent::SandboxNetwork::Deny;

    auto sess = agent::init(cfg);
    if (!sess.ok) {
        std::fprintf(stderr, "init failed: %s\n", sess.error.message.c_str());
        return 1;
    }
    agent::Session session = std::move(sess.value);

    agent::Conversation conversation;
    conversation.id = "getting-started";
    const char *prompt = argc > 3 ? argv[3] : "Say hello in one short sentence.";

    auto reply = agent::run_turn(session, conversation, prompt, /*stream=*/true, print_token);
    std::printf("\n");
    if (!reply.ok) {
        std::fprintf(stderr, "turn failed (%d): %s\n", static_cast<int>(reply.error.code), reply.error.message.c_str());
        return 1;
    }

    const agent::SessionStats stats = agent::get_stats(session);
    std::printf("turns=%d prompt_tokens=%d completion_tokens=%d\n", stats.total_turns, stats.total_prompt_tokens,
                stats.total_completion_tokens);
    return 0;
}
```

(The example file also installs a `Config::logger` that prints warnings; set
`cfg.engine_host_path = AGENT_HOST_PATH;` if you used the compile definition above.)

```sh
./build/my_agent $HOME/.local/agent/lib/agent/engines/agent-engine-llama.so \
                 ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf "What is a KV cache?"
```

Things to know:

- **Errors are values.** Every fallible call returns `agent::Result<T>`: check `.ok`, then use
  `.value` or `.error.code` / `.error.message`. Nothing throws.
- **`AgentRuntime`** initialises and frees process-wide backends; create one in `main`.
- **`Session`** owns the engine instance, the tools, the memory store and the per-turn `Arena`.
  It is move-only. One `run_turn` at a time per Session.
- **`Conversation`** is one conversation (its history and stats). Keep as many as you like per Session;
  `save_conversation` / `load_conversation` persist them through the memory store.
- **The workspace** (`workspace_dir`) is created by `init()`. Filesystem tools are confined to it
  (`sandbox_filesystem`, default true) and generated media lands in `<workspace>/artifacts/`.

## 4. Isolation and sandbox defaults

| Setting | Default | In this program |
|---|---|---|
| `engine_isolation` | `Default` = in process (unless an engines.json entry says otherwise) | `Process` |
| `engine_sandbox` | on: plugin + model read-only, workspace read-only, `artifacts/` read-write, no network, no `/tmp` | + 8 GiB `RLIMIT_AS` |
| `tool_sandbox` (`run_command`) | on: workspace read-write, system dirs read-only, `/tmp` read-write, network allowed | network denied |
| `sandbox_filesystem` (file tools) | on: paths must resolve inside the workspace | default |
| `permission_check` | none: every tool call is allowed | default ([writing-a-tool.md](writing-a-tool.md) adds one) |

In process is the fastest (a turn costs about 1 µs of overhead instead of 10 to 90 µs through the host,
see [benchmarks](../benchmarks.md)), but a crash in the engine takes your process with it. Process
isolation is what the `agent` CLI uses. If the engine needs files outside the model's directory (a
projector, voices, a LoRA), name them in `engine_options`: the directory of every string value that looks
like a path is added to the sandbox read-only; anything else goes in `engine_sandbox.read_only_paths`.
Details: [security-sandbox.md](../security-sandbox.md), [api/engines.md](../api/engines.md#isolation).

## Next

- [Writing a tool](writing-a-tool.md): custom tools, the permission gate, hooks, MCP servers.
- [Writing a graph](writing-a-graph.md): multi-step workflows in C++ or `agent.json`.
- [Writing an engine](writing-an-engine.md): wrap another inference library as a plugin.
- [API reference](../index.md#documentation-map).
