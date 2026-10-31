# agent.cpp Roadmap (draft for agreement, rev 3)

Status: **proposal**, nothing implemented yet. Date: 2026-10-31.
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
| **3D** | TRELLIS.2 ggml ports (`trellis2cpp` with C ABI, `trellis.cpp`) | Image → GLB |
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
| `Flow` FSM + `step_flow` + `void* context` | Ad hoc | **Replace** with the graph runtime |
| `AgentScheduler` + cron | Threading tied to provider type | **Replace** with graph parallel branches; **drop cron** from core |
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

### Phase 0: Decide and de-risk (1 to 2 weeks)
- [ ] Agree on this roadmap; record decisions as ADRs in `docs/adr/`
- [ ] Spike: C plugin ABI. Load two plugins with **different** ggml versions into one process on **Windows (MSVC)** and Linux; confirm no symbol clashes (use `-fvisibility=hidden` / `RTLD_LOCAL`)
- [ ] Spike: llama.cpp `llama_batch_ext`/`llama_process` + libmtmd image input from inside a plugin
- [ ] Spike: sherpa-onnx (Kokoro TTS + Parakeet STT) as a plugin on Windows
- [x] Draft `engine_abi.h` v1 and the capability taxonomy
- [ ] Draft JSON schemas for `engines.json`, `models.json`, `agent.json`
- [ ] Pin submodules; CI matrix Windows (MSVC) + Linux (GCC/Clang) + macOS (build only)

### Phase 1: Clean slate (1 to 2 weeks)
- [x] Remove the OpenAI-compatible provider and cron
- [ ] Remove `AgentScheduler` provider-specific threading (replaced by the graph runtime in Phase 3)
- [ ] Move the ONNX provider and its vendored ORT headers into `engines/onnx/` (kept, not deleted)
- [x] Move SQLite, sqlite-vec and the memory write-back cache out of `libagent` into `tools/store-sqlite/` (used by the CLI)
- [x] Move llama.cpp out of core into `engines/llama/` (`agent-engine-llama` plugin; `AiProvider::LlamaCpp` removed)
- [ ] Move httplib/OpenSSL into `tools/hub/` (CLI only); `libagent` depends only on simdjson
- [ ] New tree: `include/agent-cpp/{agent.hpp,engine_abi.h}`, `src/core/`, `engines/<name>/`, `tools/cli/`, `tools/hub/`, `tools/store-sqlite/`
- [ ] CMake `install()`/`export()` for `find_package(agent-cpp)`; each engine builds standalone against the installed ABI header
- [ ] Tests stay green on the Mock engine

### Phase 2: Core v1 (3 to 4 weeks)
- [x] Engine registry: load plugins by path, ABI/version check, capability query (`AiProvider::Engine`, mock engine plugin)
- [ ] Engine discovery from `engines.json`
- [ ] Model registry from `models.json`: lazy load, LRU unload under a RAM/VRAM budget
- [x] `StoreOps` key-value store + in-memory default
- [ ] `CheckpointerOps` / `VectorStoreOps` + in-memory implementations
- [x] Media on messages (`Media`/`MediaView`, `run_turn_with_media`); engine file outputs land in `<workspace>/artifacts/` and persist with history
- [x] Run requests with cancel (`request_cancel`/`clear_cancel`), progress (`Config::on_progress`) and chunk streaming
- [ ] Agent loop (pi-agent-core style) with tool-call limits
- [ ] Tools return `{parts}` (text and/or artifacts)
- [x] `llama` engine v1: llama.cpp b12020, Jinja chat templates + native tool-call parsing (common_chat), continuous batching
- [ ] Freeze public `agent.hpp` v1 and write `docs/api`

### Phase 3: Workflows, state, durability (3 weeks)
- [ ] Graph runtime: agent/tool/function nodes, conditional edges, **Sequential / Parallel / Loop**
- [ ] Event log as the source of truth; hooks subscribe to events
- [ ] Scoped state (`temp:`/session/`user:`/`app:`)
- [ ] Checkpointing by `thread_id` through `CheckpointerOps`; resume / replay; durability modes
- [ ] `agent-store-sqlite`: SQLite checkpointer/state + sqlite-vec vector store (CLI and opt-in)
- [ ] **Interrupts** for human approval (reuse the permission `Ask`)
- [ ] Load graphs from `agent.json`
- [ ] Port skills, permissions, context compression, fs/terminal tools
- [ ] MCP stdio client
- [ ] RAG pipeline node: chunk → embed → `VectorStoreOps` → rerank → inject

### Phase 4: Engine add-ons (ongoing; each is an independent task)
Priority wave A (v1 **starter pack**, installed by default by `agent setup`):
- [ ] `llama`: + libmtmd (vision/audio/video in), OCR models, embeddings, rerank, **decision models** (choice/score/yes-no)
- [ ] `sherpa`: Kokoro TTS, Parakeet / Whisper STT, VAD, diarization (Windows-friendly breadth)
- [ ] `sdcpp`: stable-diffusion.cpp txt2img / img2img / video (Wan, LTX) with progress + cancel
- [ ] `trellis`: TRELLIS.2 image → GLB (3D gen)
- [ ] `onnx`: generic ONNX Runtime engine (classifiers, detectors, custom models), grown from the existing provider
Wave B (optional add-ons):
- [ ] `whisper`: whisper.cpp STT
- [ ] `parakeet`: parakeet.cpp (ggml STT)
- [ ] `qwen3tts`: Qwen3-TTS (voice cloning), after licence check
- [ ] more 3D engines as ggml/ONNX ports appear (e.g. Hunyuan3D if a commercially licensed port exists)
Watch list: audio.cpp as a meta-engine; llama.cpp native TTS / image gen when they land. Fish Audio S2 and Phonon are revisited only if their licences change.
- [ ] Each engine: manifest, tiny-model smoke test, example graph, licence notes

### Phase 5: CLI release (3 weeks, can start mid-Phase 3)
- [ ] `agent setup` wizard: detect GPU (CUDA/Vulkan/CPU), pick capabilities (starter pack preselected), download matching **prebuilt** engine plugins from GitHub releases (checksum-verified)
- [ ] Release pipeline that builds every engine plugin × backend (cpu/cuda/vulkan) × OS (win/linux)
- [ ] `agent engine add|rm|ls`, `agent model pull|rm|ls` (HF download with resume + checksum, licence prompt)
- [ ] `agent chat` / `agent run <agent.json>`: text TUI with streaming, tool-call display, approval prompts, multi-agent status; `open <artifact>` launches the OS app
- [ ] Mic input / speaker output for voice agents (small C audio I/O library such as miniaudio, decided in Phase 0)
- [ ] User config at `%APPDATA%\agent\` (Windows) and `~/.config/agent/` (Linux)
- [ ] Release artifacts: Windows x64 (CPU/CUDA/Vulkan), Linux x64 (CPU/CUDA/Vulkan), macOS (CI only)

### Phase 6: SDK and docs (finalize at release)
- [ ] Guides: embed the library, write an engine plugin, write a tool, write a graph (JSON + C++)
- [ ] Examples: voice assistant (STT → LLM → TTS), image agent (LLM → SD → VLM critique loop), video storyboard, RAG over docs, OCR → extraction, decision-model router
- [ ] Engine plugin template repo / cookiecutter
- [ ] Benchmarks: tokens/s, time to first audio, peak memory per pipeline

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
