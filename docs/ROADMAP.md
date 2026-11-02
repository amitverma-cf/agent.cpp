# agent.cpp Roadmap (draft for agreement, rev 3)

Status: **in progress** (see the status board in §4). Drafted 2026-10-31, updated 2026-11-02.
Rev 3 folds in all answers so far (see §6).

## 0. What we are building

agent.cpp is **"llama.cpp for agents, across all modalities"**: a local, offline-first C++ runtime in three layers. **No inference engine is baked in.** Every engine is an add-on that the user picks at install time (CLI) or in a config file (library).

| Layer | Purpose | Ships as |
|---|---|---|
| **Engines** | Plug-and-play adapters around existing inference projects (llama.cpp, whisper.cpp, stable-diffusion.cpp, parakeet.cpp, sherpa-onnx, ...). Each one is built and loaded independently with its **own** copy of ggml/ORT. We wrap these projects and don't modify them. | `agent-engine-<name>` shared library + manifest |
| **Core** | Agent primitives inspired by LangGraph / Google ADK / pi-agent-core: agent loop, tools, graph workflows, scoped state, checkpoints, events, memory/RAG, artifacts. Knows nothing about any specific engine, and has **no storage or network dependencies**: persistence goes through pluggable op tables (in-memory defaults). | `libagent` + `agent.hpp` |
| **CLI** | The release product: install wizard, engine/model manager, a text TUI, plus the concrete infrastructure the core leaves pluggable: **SQLite** (checkpoints, state, history), **sqlite-vec** (RAG store), **httplib/OpenSSL** (downloads). | `agent` binary |

Non-goals:
- cloud or API providers, Python at runtime
- rendering images, audio or video inside the TUI. Generated media is written to the workspace and opened with the OS default app (`xdg-open` / `start` / `open`).
- patching upstream engines

Network is used only to download engines and models (Hugging Face, GitHub releases) and, optionally, by local MCP servers.

Kept constraints: no exceptions, no vtables (function-pointer op tables), arena-backed per-turn views, a single public header for the core, C++20.

Platforms: **Windows and Linux are first class** (CI + manual testing). macOS builds in CI only, best effort, untested by the maintainer.

---

## 1. Research findings

### 1.1 Engine landscape (candidates for add-ons)

| Capability | Engine (C/C++) | Notes |
|---|---|---|
| LLM text, tool calls | **llama.cpp** | Releases are `bNNNNN` (b12019 on 2026-10-31). KV API is now `llama_memory_*`; new `llama_batch_ext_*` + `llama_process()` for mixed token+embedding input; many old calls deprecated. |
| Vision / audio / video **input** to an LLM | llama.cpp **libmtmd** | Image, audio, video input: Gemma 4, Qwen2.5/3-VL, Qwen3-Omni, Voxtral, Qwen3-ASR, InternVL, SmolVLM. |
| **OCR** | llama.cpp libmtmd | PaddleOCR-VL, GLM-OCR, DeepSeek-OCR, Dots.OCR, HunyuanOCR (each needs a specific prompt format) |
| **Decision models** (choice / score / yes-no) | llama.cpp | New Oct 2026: `/v1/systemone` server endpoint ("System One" format). Models: Julia-1 144M, Laya 421M, Kev-4B, lev 4B, Clef 27B (all Apache-2.0). OpenJev 27B excluded (NC). We wrap the logic, not the HTTP endpoint. |
| **Embeddings / rerank** (RAG) | llama.cpp embedding + rerank modes | Feeds the core RAG pipeline (vector store is pluggable; sqlite-vec in the CLI) |
| **STT** | whisper.cpp v1.9.x | Mature, ggml |
| **STT** | **parakeet.cpp** (mudler, MIT, ggml) | NVIDIA Parakeet TDT/CTC/RNNT 110M–1.1B; GGUF on HF (`mudler/parakeet-cpp-gguf`, CC-BY-4.0); CPU/CUDA/HIP/Vulkan/Metal |
| **STT + TTS + VAD + diarization** | **sherpa-onnx** (k2-fsa, ONNX Runtime) | One engine covering **Kokoro**, Parakeet, Whisper, VITS/Piper and more; C/C++ API; **Windows x64/arm64**. Best breadth per integration effort. |
| **TTS** | Kokoro (via sherpa-onnx or TTS.cpp) | 82M, Apache-2.0 |
| ~~TTS~~ | ~~Fish Audio S2 Pro~~ (`s2.cpp` / `audio.cpp`) | **Excluded for now**: Fish Audio Research License (non-commercial) |
| ~~TTS~~ | ~~Phonon~~ (Gradium) | **Excluded for now**: CC BY-NC 4.0 weights, and no C++ runtime yet |
| **TTS** | Qwen3-TTS via `qwen3-tts.cpp` / `qwentts.cpp` (ggml) | Voice cloning |
| **Audio multi-model** | `audio.cpp` (ggml framework) | Several TTS/STT families behind one API: candidate "meta-engine" |
| **Image + video gen** | **stable-diffusion.cpp** | SD/SDXL, FLUX.2/3 Klein, Qwen-Image 2.1, Z-Image; video: Wan 2.1/2.2, LTX-2.3/2.5, HunyuanVideo 1.5, MiniMax-H3 |
| **3D** | TRELLIS.2 ggml ports: **`trellis2cpp`** (localai-org, MIT, flat C ABI; used by `engines/trellis`), `trellis.cpp` (pwilkin, MIT, C++ only, no library C API) | Image → GLB. TRELLIS.2 weights are MIT, but the required DINOv3 conditioning encoder is under the custom DINOv3 License: check it against the catalogue policy below before listing |
| **Long tail** | ONNX Runtime (generic graph runner) | Classifiers, detectors, custom models |

Licences vary per **model**, not just per engine. Policy for v1: the built-in catalogue lists **only free, open-source models with no usage restrictions** (Apache-2.0, MIT, BSD, CC-BY and similar). Paid/proprietary models and models whose licences restrict use (non-commercial, research-only, custom "open-weight" terms) are **not shipped or listed**. Users can still add any model they are entitled to use through their own `models.json` entry, at their own responsibility. The model manifest carries a licence field, and the CLI shows it before download.

### 1.2 Agent framework patterns to adopt
- **LangGraph 1.0**: state graph plus a checkpointer keyed by `thread_id`, giving durable execution, **interrupts** (human in the loop), resume and time travel. Durability modes `exit | async | sync`. Replay re-runs side effects, so tools need idempotency.
- **Google ADK**: `LlmAgent` plus workflow agents **Sequential / Parallel / Loop**. An **event log** per session is the source of truth. **Scoped state** via key prefixes (`temp:`, session, `user:`, `app:`). Parallel branches write unique keys.
- **pi-mono**: strict layers (`pi-ai` / `pi-agent-core` / `pi-tui` / `pi-coding-agent`), each usable alone, with a minimal core plus skills, extensions and prompt templates. This mirrors our Engines / Core / CLI split.
- **MCP**: the 2026-07-28 revision is reportedly stateless; stdio is still standard. There is no official C++ SDK, so we write a small stdio client ourselves.

### 1.3 Consequences for the design
1. Engines vendor their own ggml, and a shared ggml is not required, so **each engine must be its own shared library** loaded at runtime. Static-linking two ggml copies into one binary causes symbol clashes.
2. Plugins must cross the DLL boundary safely on **Windows (MSVC)**. The boundary between core and plugin therefore has to be a **plain C ABI** (C structs + function pointers, no STL types across it). The *public* `agent.hpp` stays idiomatic C++; only the internal plugin interface is C.
3. Config format: **JSON**. simdjson is already vendored (no new dependency), it parses without exceptions, and every tool can produce it. YAML would need a new parser (yaml-cpp throws exceptions; rapidyaml is a large addition) for a cosmetic gain. Comments go in a `"//"` or `"description"` field. *(Revisit only if users ask.)*

---

## 2. Audit of the current code: keep / rework / remove

| Area | Today | Decision |
|---|---|---|
| `Result<T>`, `Error`, `Arena` | Solid | **Keep** |
| `ProviderOps` closed enum (`Mock, LlamaCpp, OpenAICompatible, OnnxRuntime`) | Engines compiled into core, one per session | **Replace** with the runtime engine plugin registry |
| `OpenAICompatible` provider | Cloud path | **Remove** |
| ONNX provider | Single `Run()`, CPU, 1 thread | **Keep code**: move out of core into the `onnx` engine plugin and grow it (EP selection, threads, IoBinding) |
| llama.cpp provider + continuous batching | Older API, grammar removed | **Move** into the `llama` engine plugin and port to the current API |
| Mock provider | Tests | **Keep** as a built-in test engine (the only one compiled into core) |
| `Flow` FSM + `step_flow` + `void* context` | Ad hoc | **Done**: removed; the graph runtime replaces it (per-node prompts/allowlists/`ToolHook`s, `history_key`, `context_provider`, `user_data`) |
| `AgentScheduler` + cron | Threading tied to provider type | **Done**: removed, replaced by graph parallel branches; cron dropped from core |
| `FlowMemory` / `Message` / `TensorData` | Text-centric | **Replace** with `Part`-based messages + artifact refs |
| SQLite KV + write-back cache | Inside the core library | **Move to CLI**: becomes the CLI's implementation of the core `CheckpointerOps` / `StateStoreOps` |
| sqlite-vec DataSource | Inside the core library | **Move to CLI**: the CLI's implementation of the core `VectorStoreOps` |
| fs / terminal / compressor / skills / permissions tools | Useful | **Keep**, port to the new tool result type |
| cpp-httplib + OpenSSL | In core | **Move** to the CLI hub (downloads) only |
| `examples/agent_cli` | REPL | **Replace** with the `agent` CLI |
| `docs/api/*` | Old API | **Rewrite** after the API freeze |

---

## 3. Target architecture

```
                         agent (CLI)
   install wizard · engine manager · model hub (HF) · TUI (text only) · open-with-OS
   SQLite checkpointer/state · sqlite-vec vector store · httplib+OpenSSL downloads
                              │
                         libagent (core)
   Agent loop · Graph runtime (seq / parallel / loop / interrupt) · Event log
   Scoped state · Checkpointer/State/Vector store interfaces (in-memory defaults)
   Tools · Skills · Permissions · MCP client · Artifact store (artifact:// ids) · RAG pipeline
   Engine registry ── loads plugins listed in engines.json (C plugin ABI)
                              │ dlopen / LoadLibrary
   ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
   llama      whisper    parakeet   sherpa     sdcpp      trellis    onnx  ...
  (llm,vlm,  (stt)      (stt)      (stt,tts,  (image,    (3d)
   ocr,embed,                       vad,diar)  video)
   rerank,decision)        each with its own vendored ggml / ORT
```

### 3.1 Plugin ABI (`include/agent-cpp/engine_abi.h`, plain C)
- `agent_engine_get(uint32_t abi_version) -> const AgentEngineV1*`: the single exported symbol.
- `AgentEngineV1 { name, version, caps_bitmask, create, destroy, load_model, unload_model, run, cancel, estimate_memory, free_result }`.
- Data crosses as C views: `{const char*, size_t}`, typed buffers `{dtype, shape*, ndim, data*, bytes}`, or file paths for big media.
- Callbacks: `on_chunk` (tokens, audio PCM, frames), `on_progress(step, total)`, `cancel_flag`.
- ABI version check plus a capability query at load. A misbehaving plugin gets an error result, never a crash of the host.

### 3.2 Capability taxonomy
`TextGen · ToolCall · VisionIn · AudioIn · VideoIn · OCR · ASR · TTS · VAD · Diarize · ImageGen · ImageEdit · VideoGen · MeshGen · Embed · Rerank · Decide (choice/score/yes-no) · Classify · Generic (raw tensors)`

### 3.3 Configuration files (JSON)
- `engines.json`: which plugins are installed, their paths, and backend flavour (cpu / cuda / vulkan).
- `models.json`: logical model names → engine + files (HF repo/file/quant) + params + **licence**.
- `agent.json` / graph files: declarative agents and workflows for CLI users (nodes, edges, tools, which model per node).
- Library users either edit these files or build the same structs in C++.

### 3.4 Storage is pluggable, not built in
The core declares function-pointer tables, keeping the no-vtable rule:
- `CheckpointerOps { put, get, list, delete }`, keyed by `thread_id` + checkpoint id
- `StateStoreOps { get, set, erase, scan_prefix }` for `user:`/`app:`-scoped state and long-term memory
- `VectorStoreOps { upsert, query_topk, erase }` for RAG

The core ships only **in-memory** implementations (zero dependencies), and a **JSON-file** checkpointer is optional. The CLI registers **SQLite** and **sqlite-vec** implementations. Library users either use the in-memory ones, write their own (Postgres, LMDB, ...), or opt in to the CLI's SQLite module, which is published as an optional `agent-store-sqlite` library.

### 3.5 Media handling
Engines write outputs (wav, png, mp4, glb) into `workspace/artifacts/`. The core passes `artifact://<id>` handles between nodes and tools. The TUI prints the path, and the `open` command hands the file to the OS default app.

---

## 4. Phased plan with checklists

### Status board (updated 2026-11-02)

**In flight** (parallel worktrees, integrated to `main` one commit at a time after build + tests):
- [x] Out-of-process engine host (`agent-engine-host`, per-engine `isolation`, crash → `EngineCrashed` + auto-restart, hang watchdog, cross-process cancel). Fixes [LIMITATIONS](LIMITATIONS.md) §1
- [x] Remove `Flow`/`AgentScheduler`; the graph runtime becomes the only workflow API (tool hooks, shared/isolated history, context providers merged in)
- [x] `engines/trellis`: TRELLIS.2 image → GLB (coarse path verified on CPU)
- [x] `cmake --install` / `find_package(agent-cpp)` + Linux release tarball script ([docs/packaging.md](packaging.md))
- [x] OS sandbox on Linux: Landlock + seccomp + rlimits + new session for the engine host (default on, no network) and `run_command` (default on), opt-in for MCP servers; macOS profile and Windows Job Object written, untested ([docs/security-sandbox.md](security-sandbox.md)). Fixes [LIMITATIONS](LIMITATIONS.md) §2 on Linux
- [x] Explicit task on the public API (`InferRequest::task`, `run_task`, `Config::default_task`) instead of the lowest-capability-bit heuristic (still the automatic default)
- [x] ABI v2: `struct_size` on the engine table, request and response (`AGENT_HAS_FIELD`, `agent_response_reset`), `AGENT_REQUEST_FLAG_AUTO_TASK`; every plugin in `engines/` updated (ABI 1 plugins are refused)
- [x] Engine host v2: request ids + worker pool for concurrent requests (llama continuous batching works under process isolation), reused shared-memory segments for tensors/BYTES >= 64 KiB

**Next up** (in order):
- [x] Engine host v2: concurrent requests (llama continuous batching works under process isolation), shared memory for tensors/media
- [x] Model registry with RAM/VRAM budget and LRU unload (`ModelRegistry`, `acquire_model` leases, `GraphNode::model`, CLI `memory_budget_mb`; see [docs/api/models.md](api/models.md))
- [x] ABI v2: `struct_size` fields for compatible growth (`AGENT_ENGINE_ABI_VERSION` 2, `AGENT_HAS_FIELD`)
- [x] `CheckpointerOps` + persisted event log; per-branch checkpoints and interrupts in parallel fan-outs (SQLite checkpointer, background Async writer, `fork_graph_run`, nested fan-outs share the worker slots)
- [ ] OS sandboxing follow-ups: ~~test real engines under the default engine policy~~ (done: llama, onnx, sherpa, sdcpp, TRELLIS load paths; GPU device nodes now opt-in via `SandboxPolicy::gpu`), ~~`"sandbox"` for MCP servers in `agent.json`~~ (done); open: test the macOS profile (and apply it without allocating after `fork()`), Windows AppContainer (filesystem/network) and handle-list inheritance, GPU builds under the sandbox
- [x] Hub: proxy auth (`user:pass@` in `HTTPS_PROXY`/`HTTP_PROXY`, `NO_PROXY`), gated repos (`HF_TOKEN` or the HF token file, sent only to Hugging Face hosts); split GGUF downloads done. Unit-tested offline, not against a live gated repo
- [ ] Windows build + test pass (owner)
- [x] Voice loop: mic/speaker I/O + VAD (sherpa) → ASR → LLM → TTS (`agent voice`, [guide](guides/voice-assistant.md)); file backend tested end to end, real sound devices untested

**Recently done** (see the phase lists below): llama decide (choice / score / yes-no) and OCR; sdcpp video without ffmpeg (MJPEG AVI or PNG frames); whisper, parakeet and Qwen3-TTS engines; tools returning media parts; every engine tested under the engine sandbox, GPU device nodes opt-in, `sandbox` for MCP servers in agent.json; ggml engines built without Intel AMX by default (in-process ORT engines corrupted its per-thread tile state, see [LIMITATIONS](LIMITATIONS.md) §1); SDK docs and benchmark harness. Earlier: engine plugin ABI + loader + manifest; llama (b12020, Jinja templates, native tool calls, vision/audio input, embed/rerank), sdcpp, sherpa and onnx engines; media/progress/cancel in the turn loop; graph runtime + `agent.json` loader; MCP stdio client; RAG pipeline; Hugging Face hub; `agent` CLI (`setup`, `engine`, `model`, `chat`, `run`); SQLite moved out of the core.

### Phase 0: Decide and de-risk (1 to 2 weeks)
- [ ] Agree on this roadmap; record decisions as ADRs in `docs/adr/`
- [ ] Spike: C plugin ABI. Load two plugins with **different** ggml versions into one process on **Windows (MSVC)** and Linux; confirm no symbol clashes (use `-fvisibility=hidden` / `RTLD_LOCAL`)
- [ ] Spike: llama.cpp `llama_batch_ext`/`llama_process` + libmtmd image input from inside a plugin
- [ ] Spike: sherpa-onnx (Kokoro TTS + Parakeet STT) as a plugin on Windows
- [x] Draft `engine_abi.h` v1 and the capability taxonomy
- [x] Draft JSON schemas for `engines.json`, `models.json`, `agent.json` ([schemas/](../schemas))
- [ ] Pin submodules; CI matrix Windows (MSVC) + Linux (GCC/Clang) + macOS (build only)

### Phase 1: Clean slate (1 to 2 weeks)
- [x] Remove the OpenAI-compatible provider and cron
- [x] Remove `AgentScheduler` provider-specific threading (`Flow` / `AgentScheduler` removed; the graph runtime's parallel branches replace them)
- [x] Move the ONNX provider into `engines/onnx/` (`agent-engine-onnx` plugin, built against a prebuilt ORT release; `AiProvider::OnnxRuntime` and the vendored ORT headers removed)
- [x] Move SQLite, sqlite-vec and the memory write-back cache out of `libagent` into `tools/store-sqlite/` (used by the CLI)
- [x] Move llama.cpp out of core into `engines/llama/` (`agent-engine-llama` plugin; `AiProvider::LlamaCpp` removed)
- [x] Move httplib/OpenSSL into `tools/hub/` (CLI only); `libagent` depends only on simdjson
- [x] New tree: `include/agent-cpp/{agent.hpp,engine_abi.h}`, `src/core/`, `engines/<name>/`, `tools/cli/`, `tools/hub/`, `tools/store-sqlite/`
- [x] CMake `install()`/`export()` for `find_package(agent-cpp)`; each engine builds standalone against the installed ABI header
  - [x] `cmake --install` + `find_package(agent-cpp)` (`agent-cpp::agent`, `::store_sqlite`, `::hub`; static or `BUILD_SHARED_LIBS`), CLI in `bin/`, plugins + bundled runtime libraries in `lib/agent/engines/`; simdjson dropped from the public header; out-of-tree check in `tests/consumer/` (see [docs/packaging.md](packaging.md))
  - [x] engines build standalone against the installed `engine_abi.h`: the package exports `agent-cpp::engine_abi`, `agent_add_engine()` and `agent_install_engine()`; [`templates/engine-plugin/`](../templates/engine-plugin) builds out of tree against an installed prefix and loads with `agent engine add` (verified on Linux). The engines in `engines/` are still built in-tree
- [x] Tests stay green on the Mock engine

### Phase 2: Core v1 (3 to 4 weeks)
- [x] Engine registry: load plugins by path, ABI/version check, capability query (`AiProvider::Engine`, mock engine plugin)
- [x] Engine discovery from `engines.json` (`load_engine_manifest`)
- [x] Model registry from `models.json`: lazy load, LRU unload under a RAM/VRAM budget (see [docs/api/models.md](api/models.md))
  - [ ] VRAM accounting beyond `gpu_layers > 0` (GPU query, partial offload); per-model RSS for out-of-process engines
- [x] `StoreOps` key-value store + in-memory default
- [x] `CheckpointerOps` / `VectorStoreOps` + in-memory implementations
  - [x] `VectorStoreOps` + `in_memory_vector_store()` (brute-force cosine); sqlite-vec implementation in `agent-store-sqlite` (see [docs/api/rag.md](api/rag.md))
  - [x] `CheckpointerOps` + `in_memory_checkpointer()`; SQLite implementation in `agent-store-sqlite` (see [docs/api/graph.md](api/graph.md#checkpointers))
- [x] Media on messages (`Media`/`MediaView`, `run_turn_with_media`); engine file outputs land in `<workspace>/artifacts/` and persist with history
- [x] Run requests with cancel (`request_cancel`/`clear_cancel`), progress (`Config::on_progress`) and chunk streaming
- [x] Out-of-process engines: `agent-engine-host` serves one plugin over a binary pipe protocol; `Config::engine_isolation` / engines.json `"isolation"`; crashes become `ErrorCode::EngineCrashed` with automatic restart (rate-limited), per-request watchdog, cancel escalates to kill; CLI default (see [docs/api/engines.md](api/engines.md#isolation))
  - [x] OS sandboxing / rlimits for the host (Linux; `Config::engine_sandbox`)
  - [x] shared memory for large tensors, concurrent requests per host (engine host v2)
  - [ ] Windows build
- [ ] Agent loop (pi-agent-core style) with tool-call limits
- [x] Tools return `{parts}` (text and/or artifacts): `ToolResult` {text, media, tensors, is_error} via `Tool::parts_callback` (the string `callback` still works), `execute_tool_parts`; tool media go on the tool `Message`, persist with the history and reach engines that declare the input modality as FILE parts (see [docs/api/tools.md](api/tools.md#tools-that-return-parts)); MCP image/audio content becomes media
- [x] `llama` engine v1: llama.cpp b12020, Jinja chat templates + native tool-call parsing (common_chat), continuous batching
- [ ] Freeze public `agent.hpp` v1 and write `docs/api`

### Phase 3: Workflows, state, durability (3 weeks)
- [x] Graph runtime: agent/tool/function nodes, conditional edges, **Sequential / Parallel / Loop** (see [docs/api/graph.md](api/graph.md))
- [x] Event log as the source of truth; hooks subscribe to events
  - [x] v1: in-memory per-run `GraphRun::events` (node start/end, state writes, tool calls, interrupts, checkpoints) + existing `EventHook`s
  - [x] v2: persisted through the checkpointer (`list_graph_events`); not used to replay tool results
- [x] Scoped state (`temp:`/session/`user:`/`app:`)
- [x] Checkpointing by `thread_id` through `CheckpointerOps`; resume / replay; durability modes
  - [x] v1: checkpoints by `thread_id` through `StoreOps`, `resume_graph`, durability modes `Exit`/`Async`/`Sync`
  - [x] v2: `CheckpointerOps` (`GraphRun::checkpointer`; the StoreOps layout stays the default), real `Async` durability (background writer, bounded queue, flushed on return); time travel via `fork_graph_run` (continue any stored step as a new thread)
- [ ] `agent-store-sqlite`: SQLite checkpointer/state + sqlite-vec vector store (CLI and opt-in)
  - [x] sqlite-vec vector store (`store::sqlite_vector_store()`) and SQLite checkpointer (`store::sqlite_checkpointer()`, WAL); a SQLite `StateStoreOps` is still open (`user:`/`app:` state uses `StoreOps`)
- [x] **Interrupts** for human approval (reuse the permission `Ask`), including inside top-level parallel branches (`GraphRun::interrupts`, per-branch `resume_graph`)
- [x] Per-branch checkpoints: a resumed fan-out re-runs only unfinished branches
- [x] Load graphs from `agent.json` (`load_graph_json`, see [docs/api/agent-json.md](api/agent-json.md))
- [x] Port skills, permissions, context compression, fs/terminal tools: tested under graph Agent nodes with the mock engine (allowed_tools, shared `history_key` conversations, approval interrupts, process-isolated engines); `read_file` returns images/audio/video as media; `compress_context` uses the Session's engine (TEXT_GEN, own arena) and keeps tool-call groups intact; leading system messages are merged and never pruned; per-node tool/skill lists; `use_skill` loads bundled files (see [docs/api/tools.md](api/tools.md#default-tools))
  - [x] `run_command` stops on `request_cancel()`: the command and its child processes are killed (POSIX: SIGTERM to the process group, SIGKILL after 500 ms; Windows: Job Object) and the result reports `"cancelled":true` with the partial output; a timeout now kills the whole tree too (Windows path untested)
  - [ ] context compression is not tested with a real model
- [x] MCP stdio client (`connect_mcp_server`, see [docs/api/mcp.md](api/mcp.md))
- [x] RAG pipeline: chunk → embed → `VectorStoreOps` → rerank → inject (`rag_index` / `rag_query` / `search_knowledge` tool via `make_rag_tool`; see [docs/api/rag.md](api/rag.md)). A dedicated graph node kind is not added: an Agent node with the tool, or a Function node calling `rag_query`, covers it

### Phase 4: Engine add-ons (ongoing; each is an independent task)
Priority wave A (v1 **starter pack**, installed by default by `agent setup`):
- [x] `llama`: + libmtmd (vision/audio in), OCR models, embeddings, rerank, **decision models** (choice/score/yes-no); video input and native System One heads are the open sub-items below
  - [x] libmtmd image + audio input (`engine_options.mmproj`); video input still open (needs ffmpeg)
  - [x] embeddings + rerank (`engine_options.mode` `"embed"` / `"rerank"`, `AGENT_CAP_EMBED` / `AGENT_CAP_RERANK`; tested with bge-small-en-v1.5, jina-reranker-v1-tiny-en, bge-reranker-v2-m3)
  - [x] decision models, generic: `AGENT_CAP_DECIDE` choice / score / yes-no from option log-probabilities with any instruct GGUF (+ images via mmproj), `agent::decide` / `decide_score` / `decide_yes_no` ([api/decide.md](api/decide.md)); tested with Qwen2.5-0.5B-Instruct
  - [x] OCR: `AGENT_CAP_OCR` with model-family prompts (PaddleOCR-VL, GLM-OCR, DeepSeek-OCR, dots.ocr, HunyuanOCR, generic fallback), `agent::ocr`; tested with GLM-OCR (Q8_0, MIT) and SmolVLM-256M (generic prompt)
  - [ ] native System One decision models (Julia-1, Laya, Kev-4B, lev, Clef: `<arch>.decision.type`); detected and rejected for now, their per-type heads/prompts live in llama-server only
- [x] `sherpa`: Kokoro TTS, Parakeet / Whisper STT, VAD, diarization (Windows-friendly breadth); more TTS families and the Windows build are open below
  - [x] v1 (`engines/sherpa`, sherpa-onnx 2.0.1 prebuilt C API): Kokoro TTS (WAV artifact + streamed audio chunks, progress, cancel), Whisper / NeMo transducer (Parakeet) ASR; Linux x64 tested with kokoro-en-v0_19 + whisper-tiny.en
  - [x] Silero VAD (`AGENT_CAP_VAD`, offline + streaming mode) and pyannote + speaker-embedding diarization (`AGENT_CAP_DIARIZE`); audio input also as an F32 tensor; tested with silero_vad.onnx, pyannote segmentation-3.0 + NeMo TitaNet small
  - [ ] VITS/Matcha/Kitten TTS, Windows build
- [ ] `sdcpp`: stable-diffusion.cpp txt2img / img2img / video (Wan, LTX) with progress + cancel
  - [x] txt2img + img2img with progress + per-step cancel (`agent-engine-sdcpp`, stable-diffusion.cpp master-1028, opt-in `AGENT_ENGINE_SDCPP`)
  - [x] video (`AGENT_CAP_VIDEO_GEN`): MJPEG AVI or PNG frames, no ffmpeg; tested with Wan2.1-T2V-1.3B Q4_0 (~1 min on CPU for a short clip)
  - [ ] LTX video, image-to-video
- [ ] `trellis`: TRELLIS.2 image → GLB (3D gen)
  - [x] image → GLB through trellis2.cpp's C API (`agent-engine-trellis`, localai-org/trellis2cpp `2f3e6e2`, opt-in `AGENT_ENGINE_TRELLIS`): coarse / 512 / 1024 / PBR paths chosen by the GGUFs present, seed/steps/guidance overrides, per-step progress; real generation verified on CPU for the coarse path only (2 steps, ~13 min on 4 shared cores)
  - [ ] verify the 512 / 1024 / textured paths end to end (needs a GPU or much more CPU time), cancel inside a generation (no hook in the port), GPU backends (nested build is CPU-only)
- [ ] `onnx`: generic ONNX Runtime engine (classifiers, detectors, custom models), grown from the existing provider
  - [x] v1: named tensors in/out (`AGENT_CAP_GENERIC`), concurrent `run()`, threads, graph optimization level, CPU/CUDA EP selection
  - [ ] IoBinding / device-resident outputs, more EPs (DirectML, CoreML, TensorRT), cancel during `Run()`, task-specific pre/post-processing (classify, embed, detect)
Wave B (optional add-ons):
- [x] `whisper`: whisper.cpp STT (`agent-engine-whisper`, whisper.cpp v1.9.5 C API, opt-in `AGENT_ENGINE_WHISPER`): ASR (language / translate / beam size, streamed segments, progress, cancel) + whisper.cpp's Silero VAD (`AGENT_CAP_VAD`, also as a pre-filter); Linux x64 CPU tested with ggml-tiny.en + ggml-silero-v5.1.2 on samples/jfk.wav
- [x] `parakeet`: parakeet.cpp (ggml STT) (`agent-engine-parakeet`, parakeet.cpp v0.6.1 C API, opt-in `AGENT_ENGINE_PARAKEET`): offline ASR with decoder / language selection; Linux x64 CPU tested with tdt_ctc-110m-q8_0 on the repo's speech.wav. Not yet: parakeet.cpp's VAD / diarization / streaming APIs, cancel inside a transcription (no hook)
- [x] `qwen3tts`: Qwen3-TTS (voice cloning), after licence check (code MIT, weights Apache-2.0) (`agent-engine-qwen3tts`, qwentts.cpp `5916bf7` C API, opt-in `AGENT_ENGINE_QWEN3TTS`): TTS with streamed audio chunks, progress, per-frame cancel, voice cloning from a request audio part or `options.voice`, named speakers / voice design for the other talkers; Linux x64 CPU tested with qwen-talker-0.6b-base-Q8_0 ("hello world" and a jfk.wav-cloned "good morning" transcribed back by whisper tiny.en). Not yet tested: customvoice / voicedesign talkers, GPU, thread count control (qwentts picks it)
- [ ] more 3D engines as ggml/ONNX ports appear (e.g. Hunyuan3D if a commercially licensed port exists)
Watch list: audio.cpp as a meta-engine; llama.cpp native TTS / image gen when they land. Fish Audio S2 and Phonon are revisited only if their licences change.
- [ ] Each engine: manifest, tiny-model smoke test, example graph, licence notes

### Phase 5: CLI release (3 weeks, can start mid-Phase 3)
- [ ] `agent setup` wizard: detect GPU (CUDA/Vulkan/CPU), pick capabilities (starter pack preselected), download matching **prebuilt** engine plugins from GitHub releases (checksum-verified)
  - [x] v1 (`tools/cli`, see [docs/cli.md](cli.md)): reports CPU cores / `nvidia-smi` / `/dev/nvidia0` / `/dev/dri` render nodes and suggests a backend, prompts on a TTY (flags otherwise), picks the starter pack, finds locally built plugins next to the binary or in `<prefix>/lib/agent/engines`, validates them by loading, writes `config.json` + `engines.json`
  - [ ] download prebuilt plugins from GitHub releases (needs the release pipeline below)
- [ ] Release pipeline that builds every engine plugin × backend (cpu/cuda/vulkan) × OS (win/linux)
- [x] `agent engine add|rm|ls`, `agent model pull|rm|ls` (HF download with resume + checksum, licence prompt)
  - [x] model hub library `agent_hub` (`tools/hub/`, see [docs/api/hub.md](api/hub.md)): HF spec/quant resolution, download with redirects + Range resume + sha256, proxy/CA support, `models.json` catalogue, restricted-licence refusal; `agent_pull` example
  - [x] wire it into the `agent` binary as `agent model pull|rm|ls` with an interactive licence prompt (progress bar, Ctrl-C cancel + resume)
  - [x] `agent engine add|rm|ls` (plugins validated with `load_engine_plugin`)
- [ ] `agent chat` / `agent run <agent.json>`: text TUI with streaming, tool-call display, approval prompts, multi-agent status; `open <artifact>` launches the OS app
  - [x] `agent chat`: streaming, tool calls + results, y/n/always approval prompts (`default_permission_policy`), `/exit /clear /model /image /open /stats`, Ctrl-C cancels a turn, conversations persisted with `agent_store_sqlite` (`--resume ID`, `--list`), model by catalogue name or path, engine by file type or `--engine`
  - [x] `agent run <agent.json>`: `--set`, `--thread`, `--resume/--input`, final state + interrupt report
  - [x] multi-agent status view: `agent run` shows one line per graph node (state, elapsed time, last tool call), redrawn in place on a terminal, parallel branches included (`GraphRun::on_event`)
- [ ] Mic input / speaker output for voice agents (small C audio I/O library such as miniaudio, decided in Phase 0)
  - [x] miniaudio 0.11.26 vendored, CLI only (`tools/cli/src/audio.{hpp,cpp}`): 16 kHz mono capture, playback, plus a WAV file backend; `agent voice` with barge-in
  - [ ] verify capture / playback on real hardware (Linux ALSA/PulseAudio, Windows WASAPI); echo cancellation for barge-in on loudspeakers
- [x] User config at `%APPDATA%\agent\` (Windows) and `~/.config/agent/` (Linux) (`$AGENT_CONFIG_DIR` / `--config-dir` override; the Windows path is untested)
- [ ] Release artifacts: Windows x64 (CPU/CUDA/Vulkan), Linux x64 (CPU/CUDA/Vulkan), macOS (CI only)
  - [x] Linux tarball script `scripts/package.sh` (`agent-<version>-linux-x64[-<backend>].tar.gz`, version from `project(VERSION)`); CPU tested, CUDA/Vulkan untested; no CI, no uploads
  - [ ] Windows and macOS packaging (TODO in [docs/packaging.md](packaging.md))

### Phase 6: SDK and docs (finalize at release)
- [x] Guides: [embed the library](guides/getting-started-library.md), [write an engine plugin](guides/writing-an-engine.md), [write a tool](guides/writing-a-tool.md), [write a graph](guides/writing-a-graph.md) (JSON + C++); every C++ snippet is taken from `examples/guides/`, which builds with `AGENT_BUILD_EXAMPLES` and runs on the mock engine under ctest (`-L examples`)
- [ ] Examples: voice assistant (STT → LLM → TTS) (done: `agent voice` + [guide](guides/voice-assistant.md); no graph JSON form, a graph runs on one engine session), image agent (LLM → SD → VLM critique loop), video storyboard, RAG over docs, OCR → extraction, decision-model router
  - [x] image critique loop (`examples/image_critique`: LLM prompt → sdcpp → SmolVLM critique → repeat; run end to end with Qwen2.5-0.5B / SDXS / SmolVLM-256M, which are too small for useful critiques), RAG over a folder of docs (`examples/rag_docs`, run with bge-small + jina reranker + Qwen2.5-0.5B over `docs/`), graph JSON set (`examples/graphs/router.json`, `review_loop.json`, run on the mock engine in ctest and `review_loop.json` with `agent run` on Qwen2.5-0.5B); see [examples/README.md](../examples/README.md)
  - [ ] OCR → extraction and decision-model router (wait for OCR / decision-model engine support; TODO in examples/README.md), video storyboard (needs video generation)
- [x] Engine plugin template: [`templates/engine-plugin/`](../templates/engine-plugin) (standalone CMake project against the installed package, C echo engine + test harness); not a separate repository or cookiecutter
- [ ] Benchmarks: tokens/s, time to first audio, peak memory per pipeline (done: `agent_bench` and [docs/benchmarks.md](benchmarks.md) with tokens/s, time to first token, peak RSS, in-process vs process isolation, parallel branches and per-turn core overhead, measured on a noisy shared 4-core VM; open: time to first audio, no TTS model was available, and a quiet-machine re-run)

---

## 5. Remaining open questions
None blocking. New questions get added here as Phase 0 spikes report back.

## 6. Decisions log
| # | Question | Decision |
|---|---|---|
| 1 | TTS/STT engines | Support many current engines as add-ons (Kokoro, Parakeet, Whisper, ...); see #11 for licence exclusions |
| 2 | Shared ggml | Not required; engines are plug-and-play plugins, each with its own ggml/ORT, upstream unmodified |
| 3 | Config format | **JSON** (reasons in §1.3) |
| 4 | Platforms | Windows + Linux first class; macOS CI-only, best effort |
| 5 | Scope | 3D, video, OCR, decision models, RAG and its models are all in scope; nothing hard-baked, everything on demand |
| 6 | Media in TUI | None; files open in the OS default apps |
| 7 | Language | C++ throughout (C only for the plugin ABI header) |
| 8 | Plugin distribution | **Prebuilt** binaries from GitHub releases |
| 9 | Starter pack | llama + sherpa (Kokoro/Parakeet) + sdcpp + **trellis (3D)** + **onnx** |
| 10 | ONNX | Keep: existing provider moves into the `onnx` engine plugin |
| 11 | Licences | Catalogue = open-source, unrestricted licences only. Restricted (NC/research/custom) and proprietary models are not shipped (Fish S2, Phonon, OpenJev excluded); users may add them manually via `models.json` |
| 12 | Storage/network deps | Not in `libagent`. SQLite, sqlite-vec and httplib/OpenSSL live in the CLI (SQLite store published as opt-in `agent-store-sqlite`); core has pluggable op tables with in-memory defaults |

## Sources
- llama.cpp multimodal: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md · llama.h: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h · releases: https://github.com/ggml-org/llama.cpp/releases
- Decision models in llama.cpp: https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp
- FOSDEM 2026 multimodal talk: https://fosdem.org/2026/schedule/event/LRZJEH-llama-cpp-multimodal/
- stable-diffusion.cpp: https://github.com/leejet/stable-diffusion.cpp
- parakeet.cpp: https://github.com/mudler/parakeet.cpp · GGUF: https://huggingface.co/mudler/parakeet-cpp-gguf
- sherpa-onnx: https://p.codekk.com/detail/c++/k2-fsa/sherpa-onnx
- Fish Audio S2 (s2.cpp): https://github.com/rodrigomatta/s2.cpp · audio.cpp: https://github.com/0xShug0/audio.cpp
- Phonon: https://gradium.ai/blog/release-phonon-open-weight · https://huggingface.co/gradium/phonon-40m-en
- qwen3-tts.cpp: https://github.com/predict-woo/qwen3-tts.cpp · TTS.cpp: https://github.com/lookbe/TTS.cpp
- TRELLIS.2 ggml: https://github.com/localai-org/trellis2cpp · https://github.com/pwilkin/trellis.cpp/wiki
- LangGraph checkpointers: https://docs.langchain.com/oss/langgraph/checkpointers · ADK overview: https://futureagi.com/blog/what-is-google-adk-2026 · pi-mono: https://pyshine.com/Pi-Mono-Full-Stack-AI-Agent-Toolkit/
- MCP transports (draft): https://modelcontextprotocol.net/specification/draft/basic/transports
