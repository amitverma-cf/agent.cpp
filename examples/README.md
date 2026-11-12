# Examples

One folder per use case, numbered in the order to read them. Each has its own README. Build with the project
(`AGENT_BUILD_EXAMPLES` is on by default); binaries land in `build/examples/`. The ones that need no model run as smoke tests on the
mock engine: `ctest --test-dir build -L examples`.

| Folder | What it shows | Needs |
|---|---|---|
| [`01-hello-agent`](01-hello-agent) | open a session on an engine, send a prompt, print the answer | nothing (mock engine) |
| [`02-tools`](02-tools) | give the model a C++ tool and an MCP tool | nothing (mock engine) |
| [`03-graph-workflow`](03-graph-workflow) | agents as graph nodes: pipeline, router, review loop; JSON graphs run with `agent run` | nothing (mock engine); a chat GGUF for real runs |
| [`04-custom-engine`](04-custom-engine) | write an engine plugin and drive it through the engine host | nothing |
| [`05-rag-over-docs`](05-rag-over-docs) | index a folder of text, retrieve, answer | an embedding GGUF and a chat GGUF |
| [`06-image-critique`](06-image-critique) | an LLM writes a prompt, an image engine draws it, a vision model critiques, repeat | llama and sdcpp engines, three models |
| [`07-model-pull`](07-model-pull) | download a Hugging Face model with the hub library | network |
| [`08-engine-plugin-template`](08-engine-plugin-template) | a standalone CMake project for an out-of-tree engine plugin | an installed agent.cpp |

The guides in [`docs/guides`](../docs/guides) use the code of folders 01 to 04, and the tests compile it, so the guides cannot drift from the API.
