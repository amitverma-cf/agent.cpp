# GPU engines: what is built, what is tested

> **Status: untested.** The project has no GPU hardware to test on. Everything below describes what the
> code does and exactly what a person with a GPU should check. Treat every GPU path as experimental
> until someone has run the checklist and reported back.

## What exists

- `engine_sandbox.gpu = true` makes the sandboxed engine host able to open the GPU device nodes
  (`/dev/dri`, `/dev/kfd`, `/dev/nvidia*`). The `agent` CLI sets it whenever the engine's `backend` in
  `engines.json` / `config.json` is not `cpu`. See [security-sandbox.md](security-sandbox.md).
- **Kernel and shader caches.** GPU drivers compile kernels and shaders on first use and keep the result in
  a cache folder under `$HOME`. The engine sandbox hides `$HOME`, so before this change the cache could not
  be written and everything was recompiled at every start (seconds to minutes on CUDA). With
  `engine_sandbox.gpu` set, the core now:
  - creates `Config::gpu_cache_dir` (default `<workspace>/cache/gpu`) with `cuda/`, `mesa/` and `nv/` inside,
  - allows the sandbox to write there, and
  - points the drivers at it: `CUDA_CACHE_PATH` (and `CUDA_CACHE_MAXSIZE` = 1 GiB), `MESA_SHADER_CACHE_DIR`
    (and `MESA_SHADER_CACHE_MAX_SIZE`), `__GL_SHADER_DISK_CACHE_PATH` (and `..._SKIP_CLEANUP`).
- CPU builds are unaffected.

## Why it is untested

No NVIDIA, AMD or Intel GPU is available to the project, and the cloud sandbox used for testing has none.
The environment variables above are the documented knobs of the CUDA driver, Mesa and the NVIDIA GL / Vulkan
driver, but whether a given driver version honours them under Landlock has not been observed.

## What to test (checklist for someone with a GPU)

1. Build a GPU engine (for example llama with CUDA or Vulkan) and register it with `"backend": "cuda"` (or
   `"vulkan"`) in `engines.json`.
2. Run any model twice: `agent task text_gen -e llama -m <model> --text hi --max-tokens 8`.
   - The **second** run should start clearly faster than the first. If both are slow, the cache is not
     being written: look at `<workspace>/cache/gpu/*` (it should hold files after the first run).
3. Run with the sandbox disabled (`engine_sandbox.enabled = false`, or `--inprocess`) for comparison. The
   start time of the sandboxed second run should match the unsandboxed one.
4. If the engine fails to find the device, check the sandbox warnings in the engine log: the device nodes
   must be listed as read-write.
5. CUDA only: build the engine with the SASS architectures of the target card (`CMAKE_CUDA_ARCHITECTURES`)
   so nothing needs JIT compilation at all; PTX-only builds depend on this cache.
6. ROCm / HIP and Metal have their own caches that are **not** redirected yet (HIP keeps compiled code
   under `~/.cache`, Metal under the system cache). Report where they write if you test them.
7. Report results (driver and OS version, first and second start time) in an issue.

## Known gaps

- ROCm / HIP, Intel Level Zero / SYCL and Metal caches are not redirected.
- The sandbox on macOS and Windows is partial ([security-sandbox.md](security-sandbox.md)); the cache
  folders matter on Linux first.
