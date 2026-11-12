# Examples

Built with the project when `AGENT_BUILD_EXAMPLES` is on (the default); binaries land in
`build/examples/`. The ones that need no model run as smoke tests on the mock engine:
`ctest --test-dir build -L examples`.

| Program | What it shows | Needs |
|---|---|---|
| [`basic/`](basic/basic.cpp) (`build/basic`) | `infer`, `run_turn`, memory, stats on the Mock provider | nothing |
| [`agent_pull/`](agent_pull/agent_pull.cpp) (`build/agent_pull`) | downloading a model with `agent_hub` | network |
| [`guides/`](guides) | the code of the [guides](../docs/guides): embedding the library, an engine plugin, a tool, a graph | mock engine (tests); llama + a GGUF for real runs |
| [`image_critique/`](#image-critique-loop) | LLM writes a prompt, stable-diffusion.cpp draws, a vision model critiques, repeat | llama + sdcpp engines, a chat, an SD and a vision model |
| [`rag_docs/`](#rag-over-a-folder-of-documents) | index a folder, retrieve, answer with citations | llama engine, an embedding and a chat model |
| [`graphs/`](#graph-json-examples) | `agent.json` graphs: router, review loop, research pipeline, MCP | `agent run` + a chat model |

## Image critique loop

[`image_critique/image_critique.cpp`](image_critique/image_critique.cpp): three Sessions on three
models. A chat model writes an image prompt for a goal, the sdcpp engine draws it (`AGENT_CAP_IMAGE_GEN`,
progress on stderr), a vision model (llama + libmtmd projector) critiques the picture and scores it out
of 10, and the critique goes back into the prompt writer's conversation for the next round. It stops at
a score of 8 or after `--rounds` (3).

```sh
cmake -B build -G Ninja -DAGENT_ENGINE_SDCPP=ON        # the sdcpp engine is opt-in (builds its own ggml)
cmake --build build
./build/examples/image_critique \
    --llama build/engines/llama/agent-engine-llama.so --llm models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --sdcpp build/engines/sdcpp/agent-engine-sdcpp.so --sd-model models/sdxs-512-tinySDdistilled_Q8_0.gguf \
    --vlm models/SmolVLM-256M-Instruct-Q8_0.gguf --mmproj models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
    --goal "a red lighthouse on a rocky coast at sunset"
```

Without every piece it prints which flags are missing (`not configured`) and exits with status 2; an
sdcpp plugin that does not load (not built, wrong ABI) is reported before any model loads.
`--sd-options` is forwarded to the sdcpp engine (default: 1 step, cfg 1, 512x512, seed 42, which suits
the distilled SDXS model; use e.g. `{"steps":20,"cfg_scale":7}` for SD 1.5). Images go to
`<workspace>/artifacts/`.

Tested end to end with the models above (Qwen2.5-0.5B, SDXS-512 Q8_0, SmolVLM-256M, 2 rounds, about
4 minutes per round on a busy 4-core VM): the loop runs, but these models are too small to make it
useful. SmolVLM-256M describes the picture without the requested `SCORE:` line (reported as score -1),
and the 0.5B prompt writer repeated its prompt, which with the fixed seed gives the same picture again.
Use a 3B+ chat model and a stronger VLM (and a different `seed` per run) for real feedback.

## RAG over a folder of documents

[`rag_docs/rag_docs.cpp`](rag_docs/rag_docs.cpp): every `.md` / `.txt` file under `--docs` is chunked
(800 bytes, 100 overlap) and embedded with `rag_index` (llama engine, `{"mode":"embed"}`), the question
is answered from the top-k chunks of `rag_query` (optionally reranked with `--rerank`), and the chat
model cites the chunk ids it used.

```sh
./build/examples/rag_docs --llama build/engines/llama/agent-engine-llama.so \
    --embed models/bge-small-en-v1.5-q8_0.gguf --chat models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --docs docs/ --question "How do I run an engine out of process?" \
    [--rerank models/jina-reranker-v1-tiny-en-Q8_0.gguf] [--k 4]
```

Tested with bge-small-en-v1.5 Q8_0, jina-reranker-v1-tiny-en and Qwen2.5-0.5B over this repository's
`docs/` (30 files): the default question retrieved a chunk of the engine guide and the answer named
`agent-engine-host`, `ErrorCode::EngineCrashed` and the automatic restart. (Timings on the shared,
heavily loaded test VM were not meaningful: see [docs/benchmarks.md](../docs/benchmarks.md) for clean numbers.)

The index is in memory and rebuilt on every run; `agent::store::sqlite_vector_store()` persists it
([docs/api/rag.md](../docs/api/rag.md)). To let an agent search on its own, add `make_rag_tool` to its
Session instead of injecting the context.

## Graph JSON examples

Declarative graphs for `agent run` or `agent::load_graph_json` ([docs/api/agent-json.md](../docs/api/agent-json.md)).
All of them parse in the test suite, and `router.json` / `review_loop.json` run on the mock engine in
the `example.guide_graph` test.

| File | Shape |
|---|---|
| [`router.json`](graphs/router.json) | classifier agent, then a `route` on its one-word answer to a coder / mathematician / chat agent |
| [`review_loop.json`](graphs/review_loop.json) | writer drafts; a `loop` runs a reviewer whose `route` ends on `APPROVED` or hands the feedback to a reviser sharing the writer's conversation (`history_key`) |
| [`research_pipeline.json`](graphs/research_pipeline.json) | sequential research, a critic loop with `until`, parallel publishing |
| [`mcp_filesystem.json`](graphs/mcp_filesystem.json) | an MCP filesystem server, a Tool node with static `args`, `interrupt_before_tools` |

```sh
./build/agent run examples/graphs/review_loop.json --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --set "task=Explain what a KV cache is in three sentences."
./build/agent run examples/graphs/router.json --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --set "request=What is 17 * 23?"
```

## Not written yet

The engine support for these exists, but the examples are not written:

- **OCR → extraction**: `ocr` (`AGENT_CAP_OCR`) returns plain text; an example that parses the text into fields is open.
- **Decision-model router**: `decide` / `decide_score` / `decide_yes_no` (`AGENT_CAP_DECIDE`) can route a request. `router.json` routes on a chat model's one-word answer for now.
- **Video storyboard**: sdcpp generates video (`AGENT_CAP_VIDEO_GEN`, Wan 2.1 tested); a storyboard example is open.
- **Voice assistant**: `agent voice` ([guide](../docs/guides/voice-assistant.md)) rather than a separate example.
