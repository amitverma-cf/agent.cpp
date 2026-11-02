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
build/agent_bench --llama build/engines/llama/agent-engine-llama.so \
                  --model qwen2.5-0.5b-instruct-q4_k_m.gguf \
                  --mock build/engines/mock/agent-engine-mock.so \
                  --reps 3 --gen-tokens 128 --prompt-words 200 --branches 4 --turns 2000 \
                  --json results.json --md results.md
```

`agent_bench --help` lists every option. Sections run when their inputs are given: `llm` and
`parallel` need `--llama` and `--model`, `overhead` needs `--mock`, `tts` needs `--sherpa` and
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

- **Time to first audio chunk (TTS):** the harness implements it (`--sherpa`, `--tts-model`), but
  no TTS model was available on the machine used, so it has not been run. [ROADMAP](ROADMAP.md)
  keeps this open.
- GPU backends, larger models, and Windows or macOS.
