# Benchmarks

Measured with `agent_bench` (`bench/`, option `AGENT_BUILD_BENCH`). **Read the caveat first: these
numbers come from a shared 4-core VM that other jobs were using at the same time (load average about
11 during the runs), so absolute figures are noisy and only meant to show orders of magnitude and
shape.** Re-run on your own hardware before drawing conclusions.

## Machine

| | |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz (Sapphire Rapids class: AVX2, AVX-512, AMX), 4 vCPUs, shared VM |
| RAM | 16 GiB |
| OS / kernel | Ubuntu 24.04, Linux 6.18 |
| Compiler | GCC 13.3, Release, `GGML_NATIVE=ON`, no GPU, no sanitizers |
| Commit | origin/main `d1127eb` plus the SDK branch |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M (`qwen2.5-0.5b-instruct-q4_k_m.gguf`), CPU, llama engine |

## Reproduce

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAGENT_BUILD_BENCH=ON
cmake --build build
build/agent_bench --llama build/src/engines/llama/agent-engine-llama.so \
                  --model qwen2.5-0.5b-instruct-q4_k_m.gguf \
                  --mock build/src/engines/mock/agent-engine-mock.so \
                  --reps 3 --gen-tokens 128 --prompt-words 200 --branches 4 --turns 2000 \
                  --json results.json --md results.md
```

The engine options are paths to plugin files. Each plugin is in the build tree under `src/engines/<name>/`, named
`agent-engine-<name>` with the platform's extension (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows); the paths above
are for Linux. Running `agent_bench` with no arguments prints the list of options. Sections run when their inputs are
given: `llm` and `parallel` need `--llama` and `--model`, `overhead` needs `--mock`, `tts` needs `--audiocpp` and
`--tts-model`.

## Core overhead per turn (mock engine)

`run_turn` on the mock engine plugin, which replies instantly, 2000 turns after 50 warm-up turns.
This is the cost of the core itself (Arena, message views, dispatch) and, for process isolation,
the pipe round trip to `agent-engine-host`.

| isolation | turns | mean us | p50 us | p99 us |
|---|---|---|---|---|
| in-process | 2000 | 0.7 | 0.7 | 0.8 |
| process | 2000 | 43.3 | 41.5 | 87.2 |

The core adds under a microsecond per turn; process isolation adds roughly 40 us, which is
negligible next to any real inference.

## Large payloads through the engine host (mock engine, process isolation)

One f32 tensor of the given size goes to `agent-engine-host` and the mock engine echoes it back, so the same number of bytes travels in both
directions. Median of 9 calls after 2 warm-up calls, 4 vCPUs. `pipe` is `AGENT_ENGINE_SHM=0` (everything through the pipe); `shared memory`
is the default for payloads of 64 KiB and more ([engines.md](api/engines.md#isolation)).

| payload | pipe ms | shared memory ms | speed-up |
|---|---|---|---|
| 1 MiB | 0.7 | 0.3 | 2.4x |
| 8 MiB | 11.6 | 6.6 | 1.8x |
| 32 MiB | 63.7 | 44.3 | 1.4x |
| 128 MiB | 256.1 | 167.5 | 1.5x |

Shared memory removes the copies through the pipe, not the copies the engine itself makes, so the gain levels off at about 1.5x for large
tensors. Images and audio for the llama and audio.cpp engines are passed as file paths and are not copied at all. Reproduce with
`agent_bench --mock <plugin> --host <agent-engine-host> --payload 128`.

## LLM: prefill, generation, time to first token

`infer()` with streaming, temperature 0, KV cache cleared before every repetition, medians of 3
runs, 279 prompt tokens, 128 generated tokens, default thread count.

| isolation | prompt tok/s | gen tok/s | TTFT ms | total ms | load ms | peak RSS MiB (this process / host) |
|---|---|---|---|---|---|---|
| in-process | 292.5 | 10.1 | 953.8 | 13601 | 1368 | 827 / - |
| process | 200.0 | 19.1 | 1395.1 | 8049 | 330 | 56 / 826 |

Across three complete runs on this VM, in-process generation ranged 5.2 to 10.1 tok/s and process
isolation 12.7 to 19.1 tok/s. The two modes run the same engine code, so the gap between them is
contention from neighbouring jobs on the shared VM, not a real advantage of process isolation. The
reliable observations are: memory is the model plus KV cache plus compute buffers (about 830 MiB
here) and lives in the host process under isolation, leaving the application at about 56 MiB; and
the first-token latency is dominated by prefill.

## Graph: one branch vs parallel branches

A `make_parallel` node with N Agent branches on one llama instance (`max_parallel_subagents` = 4),
temperature 0. The model ended its reply early (12 tokens per branch with this prompt), so each run
is short; treat it as a smoke-level scaling check.

| isolation | branches | gen tok (all) | wall s | aggregate tok/s | per-branch tok/s |
|---|---|---|---|---|---|
| in-process | 1 | 12 | 2.61 | 4.6 | 4.6 |
| in-process | 4 | 48 | 2.54 | 18.9 | 4.7 |
| process | 1 | 12 | 0.94 | 12.7 | 12.7 |
| process | 4 | 48 | 2.80 | 17.2 | 4.3 |

Four branches finish in about the wall time of one in-process (continuous batching on one model),
so aggregate throughput scales close to linearly in that run. In the first run, the one-branch
in-process case took 95 s, an outlier from machine contention, which is why single numbers here
should not be trusted.

## Not measured

- **Time to first audio chunk (TTS):** the harness implements it (`--audiocpp`, `--tts-model`), but
  no TTS model was available on the machine used, so it has not been run.
- **Speech recognition, VAD and diarization** with the audio.cpp engine: not benchmarked yet.
- GPU backends and larger models.

## Graph evaluation, speculation and memory pressure (4 vCPU, 8 GB, CPU only)

Measured with small driver scripts that are not part of the repository. Each check runs one graph or request through the `agent` CLI and records pass or fail and the time; temperature 0. The memory-pressure runs start a ballast process that takes RAM away while the model runs.

**Graph suite** (chat, router, tool call through a grammar-constrained `read_file`, approval pause, resume, review loop, research
pipeline with a parallel step). The checks are lenient: they show that a model can drive the graph machinery, not how good its
answers are.

| Model | Passed | Fastest check | Pipeline |
|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M | 7/7 | 1 s | 13 s |
| LFM3-1.7B Q4_K_M | 7/7 | 1 s | 26 s |
| TwIL-8B Q3_K_M | 7/7 | 3 s | 186 s |

LFM3 is the model for sub-agents: it passes everything at roughly seven times the speed of TwIL. TwIL is the main model when
answer quality matters. Mixed graphs (TwIL planner and merger, LFM3 workers on a 6.5 GB budget) swap the models in and out of
memory: 3 loads for one run.

**Prompt-lookup speculation** on a request whose answer repeats its input: TwIL-8B 18.9 s to 10.3 s (113 drafted tokens accepted
in 15 rounds), identical text. Qwen2.5-0.5B 1.8 s to 0.8 s, identical for the first four repetitions and then a different (earlier)
stop, a near-tie in a very small model. LFM3 does not use it (hybrid).

**Memory pressure** (TwIL-8B, 6 turns, peak RSS 6.4 GB; a ballast process takes RAM away, weights are re-read from the disk and
the rest goes to the swap file): full RAM 10.4 s, 2 GB taken 11.4 s (+10%), 3.5 GB taken 13.4 s (+29%, 1.3 GB swapped). Memory
is what limits how many models run together; the disk costs less than a third of the speed when it has to stand in for RAM.

**No effect:** `batch` / `ubatch` (LFM3, peak RSS 1987 MiB at 4096 context for 512/256/128/64; the footprint is the mapped weights
plus the KV cache) and a q8 instead of the automatic KV type.

## Qwen3.5-4B (MTP build), speculation, MTP, tool calling, LoRA, sandbox (4 vCPU, 8 GB, CPU only)

**Graph suite:** Qwen3.5-4B Q5_K_M passes all 7 checks (pipeline 100 s, loop 23 s). TwIL-8B Q3 needed 186 s for the pipeline.

**Speculation in our engine.** Hybrid models (Qwen3.5, LFM) cannot cut their recurrent state back, so the engine snapshots it before
verifying a draft and restores it (then decodes the accepted prefix again) when part of the draft is rejected. Qwen3.5-4B on a
request that repeats its input: 10.6 s to 5.5 s (15 rounds, 113 drafted tokens accepted), identical text.

**MTP, measured with upstream `llama-cli --spec-type draft-mtp` (our engine does not drive the MTP head).** Generation speed in
tokens/s on the same model: output that repeats the input 12.3 plain, 15.6 n-gram, 18.6 MTP (n=3, +51%); an open question 15.3 plain,
14.4 n-gram, 12.3 MTP (-20%). On four CPU cores the draft passes cost about as much as they save, so MTP pays only for predictable
output. It is not a default here; prompt-lookup speculation costs nothing when there is nothing to look up.

**Function calling, BFCL v3 (25 cases per category, tool-call grammar on; a driver outside the repository sends each case to the engine and checks the returned call):** simple 80%, multiple 92%,
parallel 84%, irrelevance (call nothing) 96%. The first run scored 0/25 on parallel calls: the engine had not enabled
`parallel_tool_calls`, so the grammar allowed one call per reply. Fixed.

**LoRA** (`ggml-org/LoRA-Qwen2.5-3B-Instruct-abliterated` on Qwen2.5-3B-Instruct Q4_K_M, greedy): our output at scale 1.0 is
identical to upstream `llama-completion` ("mutex works in two sentences, the first sentence is the first, ..."): the degraded text is
that adapter on that base, not the engine. At scale 0.5 both give a normal answer.

**Sandbox red team on Linux, 14 of 14 as expected** (a driver outside the repository runs forbidden reads, writes, deletes and a symlink escape from inside the sandbox): reading `/etc/shadow`, another folder, the models
folder; writing `/tmp`, next to the workspace, the home folder, `../`; deleting outside; a symlink out of the workspace are all
refused. Writing inside the workspace and the internet work. A write grant opens exactly that folder, and a read-only grant does not
allow writing.

**Every engine, one real model each:** image generation (sdxs, 4 s), OCR (GLM-OCR reads "Lighthouse" from the generated image), image
description (SmolVLM and Qwen3.5-4B both describe the lighthouse), text embedding (bge, Qwen3-VL-Embedding-2B), reranking, ONNX
embedding (MiniLM), 3D (TRELLIS: 22 423 vertices, 369 s, 3.5 GB), audio into
an LLM (Voxtral answers "a fox" about the clip; asked to "transcribe" it refuses, a property of that model's instruction following).

**Not supported:** video input (the llama engine has no video capability; frames can be sent as images), and images or audio into an
embedding model (embed mode rejects `mmproj`). Julia (a native decision model) is answered through llama.cpp's `/v1/systemone` handler, called in process.
