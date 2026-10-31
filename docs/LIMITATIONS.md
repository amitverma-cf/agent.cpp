# Known limitations and how to fix them

Status as of 2026-10-31. Each entry says what breaks, why, and the planned fix. Items tracked in
[ROADMAP.md](ROADMAP.md) link there.

## 1. Robustness and isolation

| Limitation | Why | Fix |
|---|---|---|
| **An engine crash kills the whole process.** Plugins are `dlopen`ed into the host and run on its threads. A segfault, `abort()` (e.g. llama.cpp `GGML_ASSERT`), `exit()`, an escaped C++ exception or OOM in any engine takes down the CLI or the embedding app. | Threads share one address space. The `Arena` is a per-turn allocator for zero hot-path allocations; it does not isolate failures. | **Out-of-process engine host** (in progress): an `agent-engine-host` executable loads one plugin and serves the same C ABI over IPC (pipes for control, shared memory for tensors/media). The core gets an `isolation: "process"` mode per engine (default for third-party plugins). Host death becomes an `ErrorCode` plus an automatic restart; in-process stays available for trusted engines and lowest latency. |
| **sherpa-onnx calls `exit()` on malformed model files** (e.g. a corrupt tokens file). | Upstream behaviour in its loaders; existence checks can't catch corrupt contents. | Run sherpa in the engine host (above). |
| Only the llama plugin has an exception firewall (around `common_chat`). sd.cpp / TRELLIS ports may throw across the C ABI. | Upstream C++ code throws; the core ABI forbids unwinding across it. | Engine host; plus narrow catch sites in plugins that call throwing upstream APIs. |
| No memory budget: loading several large models can OOM the machine. | The model registry with LRU unload is not implemented yet. | Model registry with a RAM/VRAM budget (ROADMAP Phase 2); per-host `rlimit` / Windows Job Objects once engines run out of process. |
| No hang watchdog: a wedged engine blocks the turn forever. | `run()` is synchronous; cancel is cooperative. | Engine host + per-request deadline; kill and restart the host on timeout. |

## 2. Security

| Limitation | Why | Fix |
|---|---|---|
| Tools, MCP servers and engines run with the user's full privileges. | The filesystem sandbox is path-based only; the permission gate decides *whether* a call runs, not *what* it can touch. | OS sandboxing for the engine host, tool runner and MCP children: Landlock + seccomp on Linux, AppContainer / Job Objects on Windows, sandbox-exec on macOS. |
| `run_command` is a shell. | By design. | Keep it behind `Ask` (default policy); run it inside the sandbox above. |

## 3. Platforms

| Limitation | Fix |
|---|---|
| Windows and macOS code paths (plugin loader, CLI platform layer, MCP process spawning, packaging) have never been compiled. | Build and test on Windows (owner), then add CI once wanted. |
| The TRELLIS nested build may still assume a single-config generator (`engines/sdcpp` handles Visual Studio/Xcode/Ninja Multi-Config; verified with Ninja Multi-Config on Linux, not with Visual Studio). | Same pattern as sdcpp: `--config $<CONFIG>` plus per-config archive dirs. |
| No GPU backend has been tested (CUDA/Vulkan/Metal). | Test the llama/sd.cpp/ORT GPU builds; add a backend field to `engines.json`. |

## 4. Engine ABI and core API

| Limitation | Fix |
|---|---|
| The task for non-chat engines is inferred as "TEXT_GEN, else the lowest capability bit". Multi-capability engines (sherpa: ASR+TTS) route by content as a workaround. | Explicit task on the public API (e.g. `InferRequest::task`, `run_task(...)`), which already exists internally as `providers::infer_task`. |
| `agent_engine_v1` has no struct-size field, so it can't grow compatibly. | Add `struct_size` to the request/response/engine structs in ABI v2. |
| Cancel granularity differs per engine: ONNX only before `Run()`, llama per token, sd.cpp per step, sherpa per sentence. | Document per engine; the engine host makes hard cancel (kill) possible. |

## 5. Graph runtime

| Limitation | Fix |
|---|---|
| Interrupts inside parallel branches are not supported. | Per-branch checkpoints. |
| A parallel fan-out is checkpointed as one step, so a resume re-runs the whole fan-out. | Same. |
| `Async` durability writes synchronously (it behaves like `Sync`). | Background checkpoint writer. |
| Checkpoints go through `StoreOps`; there is no `CheckpointerOps`, no time-travel replay, and the event log is in memory only. | `CheckpointerOps` + a persisted event log (ROADMAP Phase 3). |

## 6. Models and modalities

| Limitation | Fix |
|---|---|
| Small models drift from their tool-call format; the llama engine falls back to a loose JSON scan. | Grammar-constrained tool calls (lazy grammars) once llama.cpp's grammar path is stable for us; prefer ≥1.5B instruct models. |
| Audio input (libmtmd) is wired but untested; video input is off (libmtmd shells out to ffmpeg). | Test with an audio projector (e.g. Qwen2.5-Omni/Voxtral); video via our own frame extraction. |
| TTS covers Kokoro only (sherpa); English Kokoro on sherpa 2.x needs a lexicon, and words missing from it are skipped. | Add more TTS families (VITS/Piper, Qwen3-TTS via its ggml port). |
| Parakeet ASR (sherpa transducer) is untested; there is no VAD or diarization yet. | Test the transducer path; wire sherpa VAD/diarization. |
| sd.cpp is tested only with a single-file SD1.x-class model; FLUX/SD3/Qwen-Image and video (Wan/LTX) are untested. | Test split-model configs and add a video task. |
| Rerank mode can't fully detect non-reranker GGUFs (llama.cpp exposes no classifier-head query). | Upstream API, or a model-card allow-list. |
| No document loaders for RAG (PDF, HTML, ...). | Optional loaders in the CLI layer. |

## 7. CLI, hub and MCP

| Limitation | Fix |
|---|---|
| `agent setup` cannot download prebuilt engines yet. | Release pipeline publishing per-engine/per-backend archives. |
| The CLI hides only the Hermes/Qwen `<tool_call>` markup from the token stream. | Use the engine's parsed message instead of filtering raw tokens. |
| Hub: proxy credentials are ignored; gated repos are untested; split GGUF download is unit-tested only. | Proxy auth; test with `HF_TOKEN` and a real split repo. |
| MCP: stdio only (no HTTP transport), no resources/prompts/sampling, tested against a fake server only. | Implement Streamable HTTP and the remaining features; test with real servers. |
| No multi-agent status view in the TUI. | Render graph events live in `agent run` / `agent chat`. |

## 8. Process

| Limitation | Fix |
|---|---|
| `AGENTS.md` says the project does not accept predominantly AI-generated PRs, but this codebase is now largely AI-written (with the owner's approval, trunk-based). | The owner should decide whether to reword `AGENTS.md`. |
