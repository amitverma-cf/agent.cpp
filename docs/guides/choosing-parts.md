# Choosing the parts you use

libagent is modular. The core is small: sessions, the turn loop, tools, the OS sandbox, memory, events, skills and the permission gates. The
larger parts are switches in CMake, and every engine is its own plugin you build, install or leave out.

| Part | CMake option (default) | What you lose without it |
|---|---|---|
| Graph runtime and `agent.json` | `AGENT_WITH_GRAPH` (ON) | `Graph`, `run_graph`, checkpoints, `load_graph_json` (JSON graphs also need MCP) |
| MCP client | `AGENT_WITH_MCP` (ON) | `connect_mcp_server`, MCP servers in graphs |
| RAG | `AGENT_WITH_RAG` (ON) | `embed`, `rerank`, chunking, vector stores, the C API's `agent_c_embed` |
| Model registry | `AGENT_WITH_MODELS` (ON) | `ModelRegistry`: several models under a memory budget |
| llama.cpp engine | `AGENT_ENGINE_LLAMA` (ON) | text, vision, embeddings, rerank, decisions |
| audio.cpp engine | `AGENT_ENGINE_AUDIOCPP` (ON) | speech recognition, synthesis, VAD, diarization |
| stable-diffusion.cpp engine | `AGENT_ENGINE_SDCPP` (ON) | images and video |
| TRELLIS.2 engine | `AGENT_ENGINE_TRELLIS` (ON when CUDA is found) | image to 3D |
| ONNX Runtime engine | `AGENT_ENGINE_ONNX` (ON) | generic ONNX models |
| `agent` command | `AGENT_BUILD_CLI` (ON) | the CLI (it needs every library part above) |

The graph runtime uses the model registry (agent nodes can name their own model), so `AGENT_WITH_GRAPH=ON` turns `AGENT_WITH_MODELS` on by itself.

A small embedded build, with a chat engine and nothing else:

```sh
cmake -B build -DAGENT_WITH_GRAPH=OFF -DAGENT_WITH_MCP=OFF -DAGENT_WITH_RAG=OFF -DAGENT_WITH_MODELS=OFF -DAGENT_BUILD_CLI=OFF
```

A program that links only the core still gets sessions, tools, skills, the sandbox and every engine you built; the declarations of a part you
left out are removed from `agent.hpp` (`#if AGENT_WITH_GRAPH` and so on), so using one is a compile error, not a link error.

## Asking for parts from your project

```cmake
find_package(agent-cpp REQUIRED COMPONENTS graph rag llama audiocpp)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
```

`find_package` fails, with the missing component named, when the installed build lacks a part (`graph`, `mcp`, `rag`, `models`) or an engine
(`llama`, `audiocpp`, `sdcpp`, `trellis`, `onnx`). Ask only for what you use. In C++ the engines are `agent::engines::llama` and the like; add
your own with `agent::engines::add`.

## Shipping only the engines you use

Every engine is built by default, but your application ships just the ones you name:

```cmake
find_package(agent-cpp REQUIRED COMPONENTS llama)
agent_deploy_engines(my_app llama)
```

After each build that copies `agent-engine-llama` (plus the libraries it loads) into `engines/` next to `my_app`, adds it to `install()` and
copies `agent-engine-host`. Engines you do not list are not copied or installed. The program finds `engines/` at run time, so the folder can
move with the app; `agent::engines::search_path(dir)` adds another place to look.

## Using less at run time

- **Tools:** `Config::auto_default_tools = false` and list only the tools you want; the file and terminal tools stay out.
- **Skills and MCP servers:** nothing loads unless `Config::skills_dir` or an MCP server is set.
- **Sandbox:** engines and `run_command` run sandboxed by default (`Config::engine_sandbox`, `Config::tool_sandbox`); see [security-sandbox.md](../security-sandbox.md).
- **Memory:** an in-memory store by default; link `agent-cpp::store_sqlite` for a persistent one.
