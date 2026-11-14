# GPU engines: what is built, what is tested

> **Status: untested.** The project has no GPU hardware to test on, and the cloud environment used for testing has none.
> Everything below describes what the code does and what a person with a GPU should check. Treat every GPU path as
> experimental until someone has run the checklist and reported back.

## What the code does

Three things change when a GPU backend is in use. All three apply only when the engine runs in its own process
(`EngineIsolation::Process`, the CLI's default) with the engine sandbox on and `engine_sandbox.gpu = true`. CPU builds are
unaffected.

- **Device nodes.** The sandboxed host may open `/dev/dri` (DRM render and card nodes: Vulkan, Intel, ROCm), `/dev/kfd`
  (ROCm compute) and `/dev/nvidia*` read-write (`src/hal/sandbox.cpp`). Without `gpu`, these are not reachable.
- **Cache folder.** GPU drivers compile kernels and shaders on first use and keep them in a cache folder under `$HOME`,
  which the engine sandbox hides. The core creates `Config::gpu_cache_dir` (default `<workspace>/cache/gpu`) with `cuda/`,
  `mesa/` and `nv/` inside, makes it writable for the host, and points the drivers at it (`src/agent/providers/engine_provider.cpp`):
  - `CUDA_CACHE_PATH` = `cuda/`, and `CUDA_CACHE_MAXSIZE` = 1 GiB (the driver default is 256 MiB)
  - `MESA_SHADER_CACHE_DIR` = `mesa/`, and `MESA_SHADER_CACHE_MAX_SIZE` = `1G`
  - `__GL_SHADER_DISK_CACHE_PATH` = `nv/`, and `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP` = `1`

  Without this, a sandboxed start recompiles the kernels and shaders again. The cost depends on the driver and the kernels: seconds to minutes.
- **Who turns it on.** The `agent` CLI sets `engine_sandbox.gpu` whenever the engine's `backend` is not `cpu`. The backend is
  recorded per engine in `engines.json` (or `config.json`) and is set with `agent setup --backend cpu|cuda|vulkan`. A library
  program sets `cfg.engine_sandbox.gpu = true` itself. See [security-sandbox.md](security-sandbox.md).

## How the engines get a GPU build

The engine CMake files do not choose a GPU backend. The llama engine builds llama.cpp as a subproject, so its options are
passed through the CMake cache:

```sh
cmake -B build -G Ninja -DAGENT_ENGINE_LLAMA=ON -DGGML_CUDA=ON     # or -DGGML_VULKAN=ON
```

`scripts/package.sh --backend cuda|vulkan` sets the same option for the packaged llama build. Neither path has been run on a GPU.

Other engines:

- **TRELLIS** (`AGENT_ENGINE_TRELLIS`) is on by default when its submodule is checked out. It runs on the CPU and needs no CUDA
  ([api/engines.md](api/engines.md#the-trellis-engine)); a GPU path is not part of this tree.
- **sdcpp** and **onnx** take no GPU option from this tree. ONNX Runtime's CUDA provider would need its own runtime and is not set up.

## What is not covered

- **ROCm / HIP.** The sandbox opens `/dev/kfd` and `/dev/dri` for it, but the CLI accepts only `cpu`, `cuda` and `vulkan` as
  backends, and the HIP cache folder is not redirected. Its compiled code goes to the default folder under `$HOME`, which the
  sandbox hides.
- **Metal (macOS).** No device-file path is involved, and no cache is redirected.
- **Intel oneAPI / Level Zero / SYCL.** Not built or tested.
- **Windows.** The sandbox is partial there ([security-sandbox.md](security-sandbox.md)). The cache redirect is written for Linux first.

## Why it is untested

The environment variables above are the documented knobs of the CUDA driver, Mesa and the NVIDIA GL / Vulkan driver. Whether a
given driver version honours them under Landlock has not been observed. Nothing in this repository has opened a GPU device
node from the sandboxed host.

## What to test (checklist for someone with a GPU)

1. Build the llama engine with a GPU backend (see above), install it, and record the backend:
   `agent setup --backend cuda` (or `vulkan`), or set `"backend": "cuda"` for the engine in `engines.json`.
2. Run a model twice, through the CLI so the engine runs in its sandboxed host:
   `agent task text_gen -e llama -m <model> --text hi --max-tokens 8`.
   - The **second** run should start clearly faster than the first. If both are slow, the cache is not being written. Look for
     files under `<workspace>/cache/gpu/`; there should be some after the first run.
3. Compare with the sandbox off. `--inprocess` runs the engine in the CLI's own process, which turns off the host and its
   sandbox, so it is the comparison point. The start time of the sandboxed second run should match the in-process one.
4. If the engine cannot find the device, look at the sandbox warnings in the engine log. `agent chat` and `agent voice` write it to
   `<workspace>/logs/engine.log` (`--verbose` prints it). The device nodes must be in the list of read-write paths.
5. CUDA only: build llama.cpp with the architectures of the target card (`-DCMAKE_CUDA_ARCHITECTURES=<sm>`) so the kernels
   are precompiled. A PTX-only build has to compile at every start, and that is what the cache is for.
6. ROCm / HIP, Metal and any other backend: note where it writes its cache, and report it.
7. Report the results in an issue: driver and OS version, first and second start time, and the engine log.

## Known gaps

- The HIP, Metal and Intel caches are not redirected (see above).
- No GPU backend has been built or run by the project.
- The GPU paths are not covered by the test suite.
