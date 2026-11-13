# How agent.cpp compares

This page says where agent.cpp fits next to other tools people use to run models locally or to build agents.
It only describes what the code in this repository does. It makes no speed claims: the numbers we have are in
[benchmarks.md](benchmarks.md), measured on one shared machine, and we have not measured any other tool.

## At a glance

| | agent.cpp | llama.cpp | Ollama / LM Studio | LangChain / LlamaIndex | Hosted APIs |
|---|---|---|---|---|---|
| What it is | C++ library and `agent` CLI | C/C++ inference library | Model servers with model management | Python frameworks for agents and RAG | Models you call over the network |
| Runs models on your machine | yes | yes | yes | yes, with a local backend | no |
| Needs Python or Node | no | no | no | yes (Python) | no |
| Opens a server for clients | no | optional | yes | no | the provider's servers |
| Use from a C++ program | yes, one header | yes | through its HTTP API | no | through its HTTP API |
| OS sandbox for tools | yes (Linux, Windows 11) | - | - | - | - |

A dash means this page does not compare that row for that tool. The rows below give the detail, and they describe
the other tools only in general terms.

## llama.cpp

agent.cpp does not replace llama.cpp. It builds on it: the `llama` engine plugin is llama.cpp, loaded through
agent.cpp's engine ABI. You get llama.cpp's model support and its GGUF files. agent.cpp adds the agent loop,
tools, graphs, the permission gate, the sandbox, and the other engines (speech, images, 3D, ONNX) behind one API.

Use llama.cpp directly when you only need text inference. Use agent.cpp when you also want tools, graphs, or more
than one kind of model in one program.

## Ollama and LM Studio

Ollama and LM Studio are servers. They manage models and answer requests over HTTP for other programs.

agent.cpp has no server. It does not open a port or expose an HTTP API. It is a library you link into your program
and a command-line app. Models load in the process that uses them, or, with process isolation, in a child process
that the library starts and talks to over pipes. The `agent model` commands pull, list, search, and remove model
files from Hugging Face, which covers part of what those servers do.

Use Ollama or LM Studio when you want one model server that many apps share, or a desktop app with a model
browser. Use agent.cpp when you want the agent inside your own program and do not want a server running beside it.

## LangChain and LlamaIndex

LangChain and LlamaIndex are Python frameworks. They have large ecosystems of integrations for models, vector
stores, and data loaders. We have not compared those ecosystems, and agent.cpp has far fewer integrations.

agent.cpp is a C++20 library. It has one public header, `include/agent-cpp/agent.hpp`, a CMake package, and a
C API. It has no Python dependency. Its own parts cover the same ground in a smaller form: tools, MCP servers
(as a client), skills (`SKILL.md`), graphs written in C++ or in `agent.json`, RAG over a folder with SQLite or an
in-memory store, and memory stores. Engines run in process or in a sandboxed child process.

Use LangChain or LlamaIndex when your team works in Python, or you need a specific integration that agent.cpp does
not have. Use agent.cpp when the program has to be C++, runs on the same machine as the model, or must keep tools
inside an OS sandbox.

## Hosted APIs (OpenAI-style and similar)

agent.cpp has no client for hosted model APIs. Its provider setting has two values: a mock provider and the engine
provider, which runs local engine plugins. Its network use comes from downloading models and engine files, from MCP
servers you add, and from tools you let run commands with network access.

A hosted model is usually larger and needs no local hardware. agent.cpp runs what fits on your machine. Choose a
hosted API when quality matters more than keeping data local, or when your machine cannot run the model you need.

## What agent.cpp is good at

- **An embeddable C++ library.** One header, `Result<T>` for errors instead of exceptions, no virtual calls in the
  dispatch path, and a per-turn arena instead of per-turn heap allocation.
- **One API for several kinds of model.** Text and vision chat, speech recognition and synthesis, voice detection,
  image generation, image to 3D, embeddings, reranking, OCR, and ONNX models, through the engine plugins listed in
  the [README](../README.md).
- **An OS sandbox.** On Linux, engines and `run_command` run under Landlock, seccomp, and resource limits. On Windows 11,
  `run_command` and MCP servers run in an AppContainer, and engine hosts get a Job Object. See
  [security-sandbox.md](security-sandbox.md) for the exact rules and what is not covered.
- **Tools, skills, MCP and graphs.** Permission-gated tools, `SKILL.md` skills, MCP servers as tools, and resumable
  graphs with checkpoints and interrupts.
- **No Python or Node for the core.** The library and the CLI need neither. Some MCP servers you add are written in
  Node or Python, and `agent runtime` can install the tools they need.
- **Out-of-process engines.** An engine can run in its own process, so a crash fails one turn instead of the program.

## What agent.cpp is not

- **Not a hosted API or a model server.** It does not serve HTTP, and it does not call hosted model APIs.
- **Not a GUI.** The `agent` CLI is a terminal app. There is no graphical application.
- **Pre-1.0.** Flags, APIs, and file formats can still change. The README says the same.
- **Windows 11 and Linux only.** macOS is not tested and not supported.
- **No prebuilt downloads yet.** You build from source. See [getting-started.md](getting-started.md).
- **Not measured against other tools.** [benchmarks.md](benchmarks.md) has numbers for agent.cpp on one shared VM,
  with notes on their limits. Run them on your own hardware before you decide.
