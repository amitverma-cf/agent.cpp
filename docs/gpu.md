# GPU engines: what is installed, what is tested

> **Status: untested on hardware.** The project's test machines have no GPU. The installer picks a GPU build from llama.cpp's and
> audio.cpp's own releases; that these builds start and use the card has not been observed here. Treat every GPU path as
> experimental until someone has run the checklist and reported back.

## Where a GPU build comes from

Nothing is compiled for a GPU by agent.cpp. The installer (`install.sh` / `install.ps1`) detects the machine and downloads the upstream
release of llama.cpp (or audio.cpp) for it, at the version `scripts/components.tsv` pins:

| Backend | How it is detected | llama.cpp builds | audio.cpp builds |
|---|---|---|---|
| `cuda12`, `cuda13` | `nvidia-smi` reports the driver's CUDA version | Linux x86_64 (12, 13), Linux aarch64 (13), Windows x64 (12, 13), Windows arm64 (13) | not offered |
| `rocm` | a ROCm install is found | Linux x86_64, Windows x64 | not offered |
| `vulkan` | a Vulkan loader and device are found | all four platforms | Linux x86_64, Windows x64 |
| `metal` | macOS | not published by agent.cpp | not offered |
| `cpu` | always | all four platforms | Linux x86_64, Windows x64 |

`--backend NAME` (or `AGENT_BACKEND`) overrides the detection, on `agent setup`, `agent engine pull` and the installer itself. When a build
needs a newer C library than the machine has (audio.cpp on Linux needs glibc 2.43), the installer says so and goes on without it.
The choice is recorded per engine in `<home>/engines/<name>/.pin` and `engines.json`.

## What the code does

Three things change when a GPU backend is in use. All three apply only when the engine runs in its own process
(`EngineIsolation::Process`, the CLI's default) with the engine sandbox on and `engine_sandbox.gpu = true`. CPU builds are
unaffected.

- **Device nodes.** The sandboxed host may open `/dev/dri` (DRM render and card nodes: Vulkan, Intel, ROCm), `/dev/kfd`
  (ROCm compute) and `/dev/nvidia*` read-write (`src/hal/sandbox.cpp`). Without `gpu`, these are not reachable.
- **Cache folder.** GPU drivers compile kernels and shaders on first use and keep them in a cache folder under `$HOME`,
  which the engine sandbox hides. The core creates `Config::gpu_cache_dir` (default `<state folder>/cache/gpu`, in the home) with `cuda/`,
  `mesa/` and `nv/` inside, makes it writable for the host, and points the drivers at it (`src/agent/providers/engine_provider.cpp`):
  - `CUDA_CACHE_PATH` = `cuda/`, and `CUDA_CACHE_MAXSIZE` = 1 GiB (the driver default is 256 MiB)
  - `MESA_SHADER_CACHE_DIR` = `mesa/`, and `MESA_SHADER_CACHE_MAX_SIZE` = `1G`
  - `__GL_SHADER_DISK_CACHE_PATH` = `nv/`, and `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP` = `1`

  Without this, a sandboxed start recompiles the kernels and shaders again. The cost depends on the driver and the kernels: seconds to minutes.
- **Who turns it on.** The `agent` CLI sets `engine_sandbox.gpu` whenever the engine's `backend` is not `cpu`. A library
  program sets `cfg.engine_sandbox.gpu = true` itself. See [security-sandbox.md](security-sandbox.md).

Other engines:

- **TRELLIS** runs on the CPU and needs no CUDA ([api/engines.md](api/engines.md#the-trellis-engine)). **sdcpp** is a CPU build.
- **onnx** uses the official onnxruntime release; its CUDA provider would need its own runtime and is not set up.

## What is not covered

- **HIP caches.** The sandbox opens `/dev/kfd` and `/dev/dri` for ROCm, but the HIP cache folder is not redirected. Its compiled code goes
  to the default folder under `$HOME`, which the sandbox hides.
- **Metal (macOS).** No macOS build is published.
- **Intel oneAPI / Level Zero / SYCL.** Not offered.
- **Windows.** The sandbox is partial there ([security-sandbox.md](security-sandbox.md)). The cache redirect is written for Linux first.

## Why it is untested

The environment variables above are the documented knobs of the CUDA driver, Mesa and the NVIDIA GL / Vulkan driver. Whether a
given driver version honours them under Landlock has not been observed. Nothing in this repository has opened a GPU device
node from the sandboxed host.

## What to test (checklist for someone with a GPU)

1. Install, or add the engine with the backend: `agent engine pull llama --backend cuda12` (or `vulkan`, `rocm`); `agent doctor` shows
   which backend is recorded.
2. Run a model twice, through the CLI so the engine runs in its sandboxed host:
   `agent task text_gen -e llama -m <model> --text hi --max-tokens 8`.
   - The **second** run should start clearly faster than the first. If both are slow, the cache is not being written. Look for
     files under `<home>/workspaces/.../cache/gpu/`; there should be some after the first run.
3. Compare with the sandbox off. `--inprocess` runs the engine in the CLI's own process, which turns off the host and its
   sandbox, so it is the comparison point. The start time of the sandboxed second run should match the in-process one.
4. If the engine cannot find the device, look at the sandbox warnings in the engine log. `agent chat` and `agent voice` write it to
   `<workspace>/logs/engine.log` (`--verbose` prints it). The device nodes must be in the list of read-write paths.
5. CUDA: pick the build that matches the driver (`cuda12` or `cuda13`); the upstream kernels are precompiled for common cards, and the cache covers the rest.
6. ROCm / HIP and any other backend: note where it writes its cache, and report it.
7. Report the results in an issue: driver and OS version, first and second start time, and the engine log.

## Known gaps

- The HIP and Intel caches are not redirected (see above).
- No GPU backend has been run by the project.
- The GPU paths are not covered by the test suite.
