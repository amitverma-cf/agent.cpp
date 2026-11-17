# GPU backends

GPU support comes from the upstream releases. The installer detects the machine and downloads the matching llama.cpp (or audio.cpp)
build at the version `scripts/components.tsv` pins; agent.cpp compiles nothing for a GPU. The project's test machines have no GPU, so
these builds are used as upstream ships them and have not been run on a card here.

| Backend | Detected by | llama.cpp | audio.cpp |
|---|---|---|---|
| `cuda12`, `cuda13` | `nvidia-smi` (the driver's CUDA version) | Linux x86_64 (12, 13), Linux aarch64 (13), Windows x64 (12, 13), Windows arm64 (13) | Linux x86_64 and Windows x64 (12, 13) |
| `rocm` | `/dev/kfd` or `/opt/rocm` | Linux x86_64, Windows x64 | not offered |
| `vulkan` | Vulkan loader and a render device | all Linux and Windows platforms | Linux x86_64, Windows x64 |
| `metal` | macOS on Apple silicon | macOS arm64 | macOS arm64 |
| `cpu` | always | every platform | Linux x86_64, Windows x64, macOS x86_64 |

CUDA builds download llama.cpp's `cudart` archive with them (several hundred MB); audio.cpp's CUDA builds use the same one.

`--backend NAME` or `AGENT_BACKEND` overrides the detection on `agent setup`, `agent engine pull` and the installer. A build that needs
a newer C library than the machine has (audio.cpp on Linux needs glibc 2.43) is skipped, and the installer says why. The choice is
kept in `<home>/engines/<name>/.pin` and `engines.json`; `agent doctor` shows it.

sdcpp and trellis run on the CPU only. onnx uses the official onnxruntime build, without its GPU providers.

## Sandbox and caches

These apply only when an engine runs in its own process (`"isolation": "process"` in `engines.json`, or
`Config::engine_isolation = Process`) with the engine sandbox on. The `agent` CLI sets `engine_sandbox.gpu` whenever the engine's
backend is not `cpu`; a library program sets `cfg.engine_sandbox.gpu = true`.

- The sandbox opens `/dev/dri`, `/dev/kfd` and `/dev/nvidia*` read-write (`src/hal/sandbox.cpp`).
- Drivers keep compiled kernels and shaders under `$HOME`, which the sandbox hides, so the core creates `Config::gpu_cache_dir`
  (default `<state folder>/cache/gpu` in the home) and points the drivers at it: `CUDA_CACHE_PATH` (limit 1 GiB),
  `MESA_SHADER_CACHE_DIR` (1G), `__GL_SHADER_DISK_CACHE_PATH`. Without it every start recompiles them.
- Not redirected: the HIP and Intel caches.
- The Windows sandbox is partial ([security-sandbox.md](security-sandbox.md)).

## Checking a GPU

1. `agent engine pull llama --backend cuda12` (or `vulkan`, `rocm`), then `agent doctor` to see the recorded backend.
2. Run a model twice with the engine isolated: `agent task text_gen -e llama -m MODEL --text hi --max-tokens 8`. The second start
   should be clearly faster, and `cache/gpu/` under the project's state folder should hold files.
3. Compare with `--inprocess`, which runs the engine without the host and its sandbox.
4. If the device is not found, read the sandbox warnings in `logs/engine.log` (`--verbose` prints them).
5. Report the driver and OS version, both start times and the engine log in an issue.
