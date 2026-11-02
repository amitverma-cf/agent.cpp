# Known limitations and how to fix them

Status as of 2026-10-31. Each entry says what breaks, why, and the planned fix. Items tracked in
[ROADMAP.md](ROADMAP.md) link there.

## 1. Robustness and isolation

| Limitation | Why | Fix |
|---|---|---|
| ~~An engine crash kills the whole process.~~ **Resolved with `EngineIsolation::Process`** (default in the `agent` CLI): the plugin runs in an `agent-engine-host` child process; a segfault, `abort()`, `exit()`, escaped exception or OOM kill fails the in-flight call with `ErrorCode::EngineCrashed` and the next call restarts the host (limit: `Config::engine_max_restarts` per minute). See [api/engines.md](api/engines.md#isolation). | Library default is still `InProcess` (unchanged behaviour, lowest latency); an in-process engine can still take the process down. | Opt in per session (`Config::engine_isolation`) or per engine (`"isolation": "process"` in engines.json). |
| ~~Process isolation v1 copies tensors / BYTES parts through the pipe and serializes requests per host.~~ **Resolved in engine host v2**: request ids multiplex concurrent `run()`s of `AGENT_ENGINE_FLAG_CONCURRENT` engines through one host (parallel graph branches and llama continuous batching work out of process), and payloads >= 64 KiB go through reused shared-memory segments. | The Windows path (pipes, file mappings) is written but not compiled. | Build and test on Windows. |
| ~~sherpa-onnx calls `exit()` on malformed model files~~ (e.g. a corrupt tokens file). | Upstream behaviour in its loaders. | **Contained** under process isolation: `init()` (or the restart) fails with `EngineCrashed` ("exited with status ..."). In-process it still exits. |
| Only the llama plugin has an exception firewall (around `common_chat`). sd.cpp / TRELLIS ports may throw across the C ABI. | Upstream C++ code throws; the core ABI forbids unwinding across it. | Contained under process isolation (the host dies with `SIGABRT`, the call fails with `EngineCrashed`); narrow catch sites in plugins that call throwing upstream APIs are still wanted for in-process use. |
| ~~No memory budget~~: **resolved with `ModelRegistry`** ([api/models.md](api/models.md)): models load lazily under a RAM (and best-effort VRAM) budget, idle unpinned models are unloaded LRU-first, and an acquire that cannot fit fails with `ErrorCode::OutOfMemory` (`agent chat` / `agent run` use it with `memory_budget_mb` from config.json). Only Sessions created through the registry are counted. | Memory is *estimated* from file sizes × per-engine factors (+ a llama KV-cache estimate), not measured; VRAM counts only with `gpu_layers > 0`; `process_rss_bytes()` does not see `agent-engine-host` children. | Set `ModelSpec::estimated_bytes` for measured models; per-host `rlimit` / Windows Job Objects for the engine host (an OOM kill of a host is already reported as `EngineCrashed`); GPU memory queries. |
| ggml's Intel AMX kernels are compiled out of every engine by default (`AGENT_GGML_AMX=OFF`). | ggml loads the AMX tile configuration once per thread and assumes nobody changes it; ONNX Runtime's MLAS (sherpa, onnx) and other plugins' ggml copies do, so an in-process mix on an AMX Xeon made llama return garbage. Only `GGML_NATIVE` builds on AMX CPUs are affected (packaged builds never enable AMX). See `cmake/ggml_no_amx.cmake`. | `-DAGENT_GGML_AMX=ON` when every engine runs with process isolation; a proper fix is upstream (reload the tile config per call, as MLAS does). |
| ~~No hang watchdog~~: **resolved for process isolation**: `Config::engine_timeout_ms` kills a host whose request takes longer (`EngineCrashed`), and a cancel the engine ignores escalates to killing the host after `Config::engine_cancel_grace_ms`. Model loading (`create()`) has no deadline. In-process engines can still block a turn forever. | `run()` is synchronous; cancel is cooperative. | Use process isolation for engines that may wedge. |

## 2. Security

| Limitation | Why | Fix |
|---|---|---|
| ~~Tools, MCP servers and engines run with the user's full privileges.~~ **Linux: resolved** for the engine host (`Config::engine_sandbox`, on: plugin/model read-only, artifacts read-write, no network) and `run_command` (`Config::tool_sandbox`, on: workspace read-write, system read-only) with Landlock + seccomp + rlimits + a new session; see [security-sandbox.md](security-sandbox.md). | MCP servers stay unsandboxed by default (`McpServerConfig::sandbox` is opt-in) because `npx`/filesystem-style servers need broad access. In-process engines and user tools are never sandboxed. Kernels without Landlock (< 5.13) only get a warning. | Opt MCP servers in per server (`McpServerConfig::sandbox`, or `"sandbox"` per server in `agent.json`). |
| macOS and Windows sandboxing is partial and untested: macOS writes a `sandbox_init()` profile (applied after `fork()`, where compiling it is not async-signal-safe); Windows uses a Job Object (kill-on-close, memory and process limits) but enforces **no** filesystem or network rules, and the child inherits every inheritable handle. | AppContainer needs persistent ACL changes on every allowed directory. | Test the macOS profile (apply it through `sandbox-exec` or a parent-compiled profile); AppContainer with per-directory ACL grants plus `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`. See [security-sandbox.md](security-sandbox.md#windows-written-untested). |
| ~~The engine sandbox can break a real engine.~~ **Verified on Linux** (CPU builds): llama (chat, vision, embed, rerank), onnx, sherpa (TTS, ASR, VAD, diarization), sdcpp and TRELLIS (load and error paths; no weights here) run under the default policy with no extra paths ([details](security-sandbox.md#real-engines-under-the-default-engine-policy)). | Model files outside the model's directory and not named in `engine_options` are still invisible to the host. GPU builds are untested (no GPU here). | List such paths in `engine_sandbox.read_only_paths`; set `engine_sandbox.gpu = true` for CUDA/ROCm/Vulkan (the CLI does it for a non-`cpu` backend). |
| `run_command` is a shell, and the network is allowed by default inside its sandbox. Home-directory config (`~/.gitconfig`, `~/.npm`) is not readable. | By design (package managers, git, curl). | Keep it behind `Ask` (default policy); set `tool_sandbox.network = SandboxNetwork::Deny` when tools should be offline; list config dirs in `tool_sandbox.read_only_paths`. |

## 3. Platforms

| Limitation | Fix |
|---|---|
| Windows and macOS code paths (plugin loader, CLI platform layer, MCP process spawning, packaging) have never been compiled. | Build and test on Windows (owner), then add CI once wanted. |
| The TRELLIS nested build may still assume a single-config generator (`engines/sdcpp` handles Visual Studio/Xcode/Ninja Multi-Config; verified with Ninja Multi-Config on Linux, not with Visual Studio). | Same pattern as sdcpp: `--config $<CONFIG>` plus per-config archive dirs. |
| No GPU backend has been tested (CUDA/Vulkan/Metal). | Test the llama/sd.cpp/ORT GPU builds; add a backend field to `engines.json`. |

## 4. Engine ABI and core API

| Limitation | Fix |
|---|---|
| ~~The task for non-chat engines is inferred as "TEXT_GEN, else the lowest capability bit".~~ **Resolved**: `InferRequest::task`, `agent::run_task(session, task, request)` and `Config::default_task` (for `run_turn`) pick the task explicitly; 0 keeps the automatic rule. | sherpa still routes by content when no task is set. |
| ~~`agent_engine_v1` has no struct-size field, so it can't grow compatibly.~~ **Resolved in ABI 2**: `agent_engine_v1`, `agent_request` and `agent_response` carry `struct_size`; readers check it (`AGENT_HAS_FIELD`) before touching fields past the v2 baseline. | `agent_part` / `agent_message` arrays still change only with a new ABI version. |
| Cancel granularity differs per engine: ONNX only before `Run()`, llama per token, sd.cpp per step, sherpa per sentence. | Document per engine. Under process isolation a cancel the engine doesn't honour within `engine_cancel_grace_ms` kills the host (hard cancel). |

## 4b. Tools

| Limitation | Fix |
|---|---|
| Tool media is sent to an engine whenever it declares the input capability, but the llama plugin declares VISION_IN/AUDIO_IN even without an `mmproj`. The first request carrying tool media then fails and is retried without it (one wasted call; the Session stops sending tool media). | A per-instance capability query in the engine ABI. |
| Tool tensors (`ToolResult::tensors`) reach the engine and the event log as a note, but are not persisted with the history (`save_flow_memory` / checkpoints store media, not tensors). | Persist tensors as artifacts if needed. |
| ~~`run_command` is not interrupted by `request_cancel()`~~: **resolved**: it polls the session's cancel flag and kills the command and its child processes (SIGTERM to the process group, SIGKILL after 500 ms; a Job Object on Windows, untested) and returns `"cancelled":true` with the partial output. A process that left the command's session / Job Object on its own can outlive the cancel. | None needed. |
| Context compression is only tested with the mock engine (summary quality with small local models is unknown); the automatic trigger runs at most once per turn, at its start. | Test with a real model; trigger inside long tool loops. |

## 5. Graph runtime

| Limitation | Fix |
|---|---|
| ~~Interrupts inside parallel branches are not supported.~~ **Resolved** for fan-outs on the top-level chain: a branch may pause, the run reports every paused branch (`GraphRun::interrupts`) and `resume_graph(..., std::span<const GraphBranchInput>)` answers them one by one. | Interrupts inside nested fan-outs still fail with `InvalidConfig`. |
| ~~A parallel fan-out is checkpointed as one step, so a resume re-runs the whole fan-out.~~ **Resolved**: per-branch progress is checkpointed after every branch node; a resume re-runs only unfinished branches, from the node they stopped at. | A nested fan-out is still one node of its branch (re-run whole). |
| ~~`Async` durability writes synchronously (it behaves like `Sync`).~~ **Resolved**: a background writer thread per checkpointer with a bounded queue, flushed before `run_graph` / `resume_graph` return. A crash can lose the checkpoints still queued (see [api/graph.md](api/graph.md#durability-modes)). | Use `Sync` when re-running a few nodes after a crash is not acceptable. |
| ~~Checkpoints go through `StoreOps`; there is no `CheckpointerOps`~~ **Resolved**: `CheckpointerOps` with in-memory and SQLite (`store::sqlite_checkpointer()`) implementations; the StoreOps layout stays the default. ~~There is no time-travel replay, and the event log is in memory only.~~ **Resolved**: the event log is persisted through the checkpointer (`list_graph_events`) and `fork_graph_run` continues any stored step as a new thread. | Replay re-executes tool side effects (see [api/graph.md](api/graph.md#time-travel-fork_graph_run)); tools must be idempotent. |

## 6. Models and modalities

| Limitation | Fix |
|---|---|
| Small models drift from their tool-call format; the llama engine falls back to a loose JSON scan. | Grammar-constrained tool calls (lazy grammars) once llama.cpp's grammar path is stable for us; prefer ≥1.5B instruct models. |
| Audio input (libmtmd) is wired but untested; video input is off (libmtmd shells out to ffmpeg). | Test with an audio projector (e.g. Qwen2.5-Omni/Voxtral); video via our own frame extraction. |
| TTS covers Kokoro only (sherpa); English Kokoro on sherpa 2.x needs a lexicon, and words missing from it are skipped. | Add more TTS families (VITS/Piper, Qwen3-TTS via its ggml port). |
| Parakeet ASR (sherpa transducer) is untested; there is no VAD or diarization yet. | Test the transducer path; wire sherpa VAD/diarization. |
| sd.cpp is tested only with a single-file SD1.x-class model; FLUX/SD3/Qwen-Image and video (Wan/LTX) are untested. | Test split-model configs and add a video task. |
| Decisions (`AGENT_CAP_DECIDE`) use generic option log-probability scoring; native System One decision models (Julia-1, Laya, Kev, lev, Clef) are rejected. Image decisions are wired through mmproj but untested. | Port the per-type decision heads from llama-server's `server-decision.cpp`; test with a VLM. |
| OCR prompts for PaddleOCR-VL, DeepSeek-OCR, dots.ocr and HunyuanOCR come from upstream docs and are untested here (GLM-OCR and the generic prompt are tested); OCR returns plain text only, no layout boxes. | Test each family; parse grounding / layout output where models emit it. |
| Rerank mode can't fully detect non-reranker GGUFs (llama.cpp exposes no classifier-head query). | Upstream API, or a model-card allow-list. |
| No document loaders for RAG (PDF, HTML, ...). | Optional loaders in the CLI layer. |

## 7. CLI, hub and MCP

| Limitation | Fix |
|---|---|
| `agent setup` cannot download prebuilt engines yet. | Release pipeline publishing per-engine/per-backend archives. |
| The CLI hides only the Hermes/Qwen `<tool_call>` markup from the token stream. | Use the engine's parsed message instead of filtering raw tokens. |
| Hub: proxy auth and gated-repo tokens are unit-tested offline only; split GGUF download is unit-tested only. | Test with a real authenticating proxy, `HF_TOKEN` and a real split repo. |
| MCP: stdio only (no HTTP transport), no resources/prompts/sampling, tested against a fake server only. | Implement Streamable HTTP and the remaining features; test with real servers. |
| No multi-agent status view in the TUI. | Render graph events live in `agent run` / `agent chat`. |

## 8. Process

| Limitation | Fix |
|---|---|
| `AGENTS.md` says the project does not accept predominantly AI-generated PRs, but this codebase is now largely AI-written (with the owner's approval, trunk-based). | The owner should decide whether to reword `AGENTS.md`. |
