# agent.cpp

**Run AI agents on your own machine, from one small C++ library or one command.**

agent.cpp is a local-first agent runtime. Chat with a model that can read your files and run commands, transcribe speech,
speak, draw, turn a picture into a 3D mesh, or wire all of that into a graph that runs on a schedule. Nothing leaves your
computer, and you do not need Python, Node or a cloud account.

```text
$ agent
╭─ qwen2.5-0.5b-instruct · llama · ~/work ─────────────────────────╮
│ > summarise the TODO comments in src/ and write them to todo.md  │
│ [tool] run_command grep -rn TODO src/        allow? y / n / a    │
╰───────────────────────────────────────────────────────────────────╯
```

> Status: pre-1.0. Flags, APIs and file formats can still change. Targets Windows 11 and Linux.

## What you can do with it

| I want to | Use | Engine |
|---|---|---|
| chat with tools, vision and long context | `agent` | llama.cpp |
| transcribe audio | `agent task asr --file talk.wav` | audio.cpp (Parakeet, Moonshine, CrisperWhisper) |
| turn text into speech | `agent task tts --text "Hello"` | audio.cpp (pocket-tts, supertonic-3) |
| find speech in a recording, tell speakers apart | `agent task vad` / `diarize` | audio.cpp |
| talk to the agent by voice | `agent voice` | audio.cpp + llama.cpp |
| generate or edit images, make short videos | `agent task image_gen` | stable-diffusion.cpp |
| make a 3D mesh from a picture | `agent task mesh_gen` | TRELLIS.2 |
| run any ONNX model | `agent task generic` | ONNX Runtime |
| embed, rerank, classify | `agent task embed` / `rerank` / `classify` | llama.cpp |
| run a repeatable multi-step workflow | `agent run flow.json` | any |

Voice is a pipeline (speech recognition, then the model, then speech synthesis). There is no single speech-to-speech model
engine yet.

## Install

Prebuilt archives are not published yet; build from source. You need CMake 3.21+, Ninja and a C++20 compiler
(MSVC 2022, GCC 12+ or Clang 15+). Windows details and OpenSSL are in [docs/getting-started.md](docs/getting-started.md).

```sh
git clone --recurse-submodules https://github.com/amitverma-cf/agent.cpp
cd agent.cpp
cmake -B build -G Ninja
cmake --build build
cmake --install build --prefix ~/agent-cpp
```

Every engine is built by default and installed as its own plugin. Switch one off with its option (`-DAGENT_ENGINE_SDCPP=OFF`):

| Engine | Option | Built by default when |
|---|---|---|
| llama.cpp | `AGENT_ENGINE_LLAMA` | always |
| ONNX Runtime | `AGENT_ENGINE_ONNX` | always (downloads ONNX Runtime at configure time) |
| audio.cpp | `AGENT_ENGINE_AUDIOCPP` | the `vendor/audio.cpp` submodule is checked out |
| stable-diffusion.cpp | `AGENT_ENGINE_SDCPP` | the `vendor/stable-diffusion.cpp` submodule is checked out |
| TRELLIS.2 | `AGENT_ENGINE_TRELLIS` | a CUDA toolkit is found and the `vendor/trellis2cpp` submodule is checked out |

Your own application ships only the engines it names (see below), so having them all built costs your users nothing.

## First run

```sh
./build/agent setup                                  # picks a home folder, finds your hardware and engines, suggests models
./build/agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M
./build/agent                                        # opens the chat
```

On Windows use `.\build\agent.exe`. `agent --help` lists every command, and `agent <command> --help` explains one.

## The command line

| Command | What it does |
|---|---|
| `agent` | the full-screen chat (same as `agent chat`; `--plain` for a pipe-friendly mode) |
| `agent setup` | first-time setup: home folder, hardware check, engines, first model |
| `agent model pull\|ls\|rm\|search\|info` | download from Hugging Face, list, remove, browse tested models |
| `agent task <capability>` | run one engine capability once (`asr`, `tts`, `image_gen`, ...) |
| `agent run <graph.json>` | run a declarative graph; resumable |
| `agent voice` | microphone, speech recognition, model, speech synthesis, speaker |
| `agent engine ls\|add\|pull\|rm` | manage engine plugins |
| `agent skill`, `agent mcp` | add skills (`SKILL.md`) and MCP servers |
| `agent sandbox`, `agent secret` | folders tools may use; API keys in the OS credential store |
| `agent init`, `agent doctor`, `agent clean` | project folder, what is installed and disk use, cleanup |

Inside the chat: `/model`, `/image`, `/open`, `/stats`, `/clear`, `/help`, `/exit`. Full reference: [docs/cli.md](docs/cli.md).

## Use it as a library

Install it and link it from CMake:

```sh
cmake --install build --prefix ~/agent-cpp
```

```cmake
find_package(agent-cpp REQUIRED)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
```

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    auto session = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if (!session.ok) return std::fprintf(stderr, "%s\n", session.error.message.c_str()), 1;

    agent::Conversation chat;
    auto reply = agent::run_turn(session.value, chat, "Say hi in five words.", true,
                                 [](std::string_view token, void *) { std::fwrite(token.data(), 1, token.size(), stdout); });
    return reply.ok ? 0 : 1;
}
```

`agent::open_session` is the short path: one engine, one model file, defaults for the rest. For several model files, engine options or
a full `Config`, use `agent::engine_config` and `agent::init` ([the guide](docs/guide.md)). `agent::engines::llama`, `audiocpp`, `sdcpp`, `trellis` and `onnx` name the engines that ship with agent.cpp. The plugin is found next to
your program (in `engines/`), which one CMake line fills with just the engines you use:

```cmake
find_package(agent-cpp REQUIRED COMPONENTS llama)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
agent_deploy_engines(my_app llama)        # copies engines/agent-engine-llama.* next to my_app after each build, and into install()
```
 The library has one header, no exceptions (`Result<T>` everywhere), no virtual
calls, and works from C through [`agent_c.h`](docs/api/c-api.md).

It covers chat turns with tools, graphs, MCP servers, skills, RAG, memory stores and the permission gate. Start with
[the library guide](docs/guides/getting-started-library.md).

## Safety

Tools and engines run in an OS sandbox: Landlock and seccomp on Linux, an AppContainer on Windows 11. Engines run in their
own process, so a crash fails one turn, not your program. Tool calls that touch files or run commands ask first, unless
you pass `--yes`. `--no-sandbox` turns the sandbox off. See [docs/security-sandbox.md](docs/security-sandbox.md).

## Documentation

[Documentation index](docs/index.md) · [Getting started](docs/getting-started.md) · [CLI reference](docs/cli.md) ·
[C++ API](docs/api.md) · [C API](docs/api/c-api.md) · [Architecture](docs/architecture.md) ·
[Writing an engine](docs/guides/writing-an-engine.md) · [Writing a tool](docs/guides/writing-a-tool.md) ·
[Writing a graph](docs/guides/writing-a-graph.md) · [A voice assistant](docs/guides/voice-assistant.md) · [Choosing the parts you use](docs/guides/choosing-parts.md) ·
[Benchmarks](docs/benchmarks.md) · [Examples](examples/README.md)

Build options, packaging and the source layout are in [docs/getting-started.md](docs/getting-started.md) and
[docs/packaging.md](docs/packaging.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
