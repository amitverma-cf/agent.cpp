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
| audio.cpp engine | `AGENT_ENGINE_AUDIOCPP` (OFF) | speech recognition, synthesis, VAD, diarization |
| stable-diffusion.cpp engine | `AGENT_ENGINE_SDCPP` (OFF) | images and video |
| TRELLIS.2 engine | `AGENT_ENGINE_TRELLIS` (OFF) | image to 3D |
| ONNX Runtime engine | `AGENT_ENGINE_ONNX` (OFF) | generic ONNX models |
| `agent` command | `AGENT_BUILD_CLI` (ON) | the CLI (it needs every library part above) |

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

## Using less at run time

- **Tools:** `Config::auto_default_tools = false` and list only the tools you want; the file and terminal tools stay out.
- **Skills and MCP servers:** nothing loads unless `Config::skills_dir` or an MCP server is set.
- **Sandbox:** engines and `run_command` run sandboxed by default (`Config::engine_sandbox`, `Config::tool_sandbox`); see [security-sandbox.md](../security-sandbox.md).
- **Memory:** an in-memory store by default; link `agent-cpp::store_sqlite` for a persistent one.
