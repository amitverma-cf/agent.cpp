# agent.cpp

<p align="center"><img src="docs/images/chat.svg" alt="The agent.cpp chat screen: logo, New chat and sessions grouped by project on the left, the message box and mic at the bottom, CPU, RAM, engine, model, mode and context in the bottom bar" width="1000"></p>

**AGENT.CPP** is a local-first agent runtime. Chat with a model that can read your files and run commands, transcribe speech, speak, draw, turn a picture into a 3D mesh, or wire all of that into a graph that runs on a schedule.

`agent` is a full-screen chat for local models: Qwen, Llama, Gemma and any other GGUF model. It works on Linux and Windows, on
Intel, AMD and Arm, with or without a GPU. Prompts, files and voice recordings never leave the computer.

## Install

**Linux** (x86_64, aarch64):

```sh
curl -fsSL https://agent.qozu.fyi/install.sh | sh
```

**Windows** (x64, arm64), in PowerShell:

```powershell
irm https://agent.qozu.fyi/install.ps1 | iex
```

The installer asks where to keep everything (one folder, default `~/.agent.cpp`; it becomes the environment variable `AGENT_HOME`), installs
the program, then lets you pick the engines (llama, audiocpp, sdcpp, onnx, trellis) and runtimes (bun, uv, node, pnpm, nub) and a first
model. Each component is downloaded for this machine (the GPU backend when a driver is present) at the exact version the release pins, and
its SHA-256 is checked. llama.cpp and audio.cpp come straight from their own releases; agent.cpp only ships the small plugin over them. Later: `agent engine pull NAME`, `agent runtime install NAME`, `agent update`, `agent uninstall`.

## Chat

```sh
agent
```

The first launch opens the chat. Sessions are grouped by project folder, the bottom bar shows CPU, RAM, the loaded model and context use,
and the mic box next to the input starts voice input.

| Key | Action |
|---|---|
| `/` or `Ctrl+P` | command palette: `/model`, `/image`, `/open`, `/stats`, `/doctor`, `/clear`, `/help` |
| `Ctrl+N`, `Ctrl+K` | new chat, switch session |
| `Ctrl+B` | hide or show the sidebar |
| `F1` | every shortcut; **Settings > Shortcuts** rebinds them |
| `Esc` | stop an answer in progress |

Running a command, or deleting or moving a file, always asks first. Tools run inside an OS sandbox (Landlock and seccomp on Linux, an
AppContainer on Windows 11).

## More than chat

Every capability is one command, and all of them run offline.

| Goal | Command |
|---|---|
| Talk to the agent by voice | `agent voice` |
| Transcribe a recording | `agent task asr --file talk.wav` |
| Read text aloud | `agent task tts --text "Hello"` |
| Generate or edit an image, make a short video | `agent task image_gen` |
| Turn a picture into a 3D mesh | `agent task mesh_gen` |
| Run a repeatable workflow, resumable | `agent run flow.json` |
| Add models, MCP servers and skills | `agent model pull NAME`, `agent mcp`, `agent skill` |
| Check the installation, free disk space | `agent doctor`, `agent clean` |
| Remove agent.cpp and its data folder | `agent uninstall` (`--keep-data` keeps the models and conversations) |

`agent --help` lists every command and `agent <command> --help` explains one. The full reference is [docs/cli.md](docs/cli.md).

## Supported platforms

| | Builds |
|---|---|
| Linux | x86_64 and aarch64, glibc 2.39 or newer |
| Windows 11 | x64 and arm64 |
| Acceleration | CPU and Vulkan (Intel, AMD, Arm GPUs) |

CUDA, ROCm and macOS builds are not published. All archives are on the [releases page](https://github.com/amitverma-cf/agent.cpp/releases).

## Build from source

Requires CMake 3.21+, Ninja and a C++20 compiler (MSVC 2022, GCC 12+ or Clang 15+).

```sh
git clone --recurse-submodules https://github.com/amitverma-cf/agent.cpp
cd agent.cpp
cmake -B build -G Ninja
cmake --build build
./build/agent setup
```

Windows prerequisites, build options and engine switches are in [docs/getting-started.md](docs/getting-started.md).

## Embed it in a C++ program

The same runtime is available as a C++20 library, `libagent`: one header, no exceptions, no virtual calls and no allocation in the
per-turn path. A first program is on the [library guide](docs/guides/getting-started-library.md); the API is documented under
[docs/api.md](docs/api.md), and runnable programs are in [examples/](examples/README.md).

```cpp
#include <agent-cpp/agent.hpp>

int main() {
    agent::AgentRuntime runtime;
    auto session = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if (!session.ok) return 1;
    agent::Conversation chat;
    return agent::run_turn(session.value, chat, "Say hi in five words.").ok ? 0 : 1;
}
```

## Documentation

| Topic | Page |
|---|---|
| Using the `agent` command | [CLI reference](docs/cli.md), [getting started](docs/getting-started.md), [workspace layout](docs/workspace.md), [voice assistant](docs/guides/voice-assistant.md), [GPU support](docs/gpu.md) |
| Embedding the library | [library guide](docs/guides/getting-started-library.md), [usage guide](docs/guide.md), [API reference](docs/api.md), [C API](docs/api/c-api.md), [choosing parts](docs/guides/choosing-parts.md) |
| Extending | [writing an engine](docs/guides/writing-an-engine.md), [a tool](docs/guides/writing-a-tool.md), [a graph](docs/guides/writing-a-graph.md) |
| Background | [architecture](docs/architecture.md), [sandbox](docs/security-sandbox.md), [benchmarks](docs/benchmarks.md), [comparison](docs/comparison.md) |

Everything is indexed in [docs/index.md](docs/index.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
