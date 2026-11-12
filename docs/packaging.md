# Installing and packaging

## Version

The version is set once, in `project("agent.cpp" VERSION x.y.z)` in the top-level `CMakeLists.txt`.
`agent version`, the CMake package version (`agent-cppConfigVersion.cmake`) and the Linux tarball
names all come from it.

## `cmake --install`

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /opt/agent
```

Installed layout:

| Path | What |
|---|---|
| `bin/agent` | the CLI |
| `bin/agent-engine-host` | runs engines out of process (found next to `agent`) |
| `lib/agent/engines/agent-engine-*.so` | engine plugins, plus the runtime libraries they load through their `$ORIGIN` RPATH (`libonnxruntime.so.1`, ...) |
| `lib/libagent.a` (or `.so`) | the library |
| `lib/libagent_store_sqlite.a`, `lib/libagent_hub.a` | the optional SQLite store and Hugging Face hub libraries |
| `include/agent-cpp/` | `agent.hpp`, `engine_abi.h`, `store_sqlite.hpp`, `hub.hpp` |
| `lib/cmake/agent-cpp/` | `agent-cppConfig.cmake`, `agent-cppConfigVersion.cmake`, exported targets, `agent-cppEngine.cmake` (engine plugin helpers) |
| `share/doc/agent/` | `README.md`, `LICENSE`, `cli.md` |

The library directory is `CMAKE_INSTALL_LIBDIR` (`lib`, or `lib64` on some distributions). The engine
directory is always `lib/agent/engines`, because that is where the CLI looks (`<exe>/../lib/agent/engines`,
see [cli.md](cli.md)). An installed `agent setup` finds the installed plugins there with no flags.

The mock engine plugin (`agent-engine-mock`, used by the tests) is not installed. Set
`-DAGENT_INSTALL=OFF` to generate no install rules (the default when agent.cpp is added with
`add_subdirectory`).

### Static or shared

`BUILD_SHARED_LIBS` (default `OFF`) switches `libagent`, `agent_store_sqlite` and `agent_hub` between
static and shared. Every symbol in the libraries is exported for now. Engine plugins are always shared
modules, and the CLI's internal `agent_cli_core` is always static.

RPATHs:

- `agent` with static libraries: no RPATH. With shared libraries: `$ORIGIN/../lib` (the `CMAKE_INSTALL_LIBDIR`).
- shared `libagent_store_sqlite.so` / `libagent_hub.so`: `$ORIGIN`, so they find `libagent.so` next to them.
- `agent-engine-onnx`: `$ORIGIN`, so it finds its bundled runtime libraries.
- `agent-engine-llama` and `agent-engine-sdcpp` link llama.cpp / stable-diffusion.cpp statically and need
  only system libraries (`libstdc++`, `libgomp` for OpenMP, `libc`).

## Using the library: `find_package(agent-cpp)`

```cmake
find_package(agent-cpp 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
# optional:
target_link_libraries(my_app PRIVATE agent-cpp::store_sqlite)  # SQLite KV store + sqlite-vec
target_link_libraries(my_app PRIVATE agent-cpp::hub)           # Hugging Face downloads (if built)
```

Configure with `-DCMAKE_PREFIX_PATH=<prefix>`. Pre-1.0, a request for version `0.1` accepts any `0.1.x`
(`SameMinorVersion`). The package also sets `agent-cpp_ENGINE_DIR` to `<prefix>/lib/agent/engines`.

The public headers do not include simdjson: it is compiled into `libagent` and is an implementation
detail. Static packages pull in `Threads` (and `OpenSSL` for `agent-cpp::hub`) with
`find_dependency`.

`tests/consumer/` is a minimal out-of-tree project that does this (a Mock-provider turn plus the SQLite
store):

```sh
cmake --install build --prefix /tmp/agent-prefix
cmake -S tests/consumer -B /tmp/agent-consumer -DCMAKE_PREFIX_PATH=/tmp/agent-prefix
cmake --build /tmp/agent-consumer && /tmp/agent-consumer/consumer
```

## Engine plugins outside this tree

The package also serves engine authors. `find_package(agent-cpp)` provides the header-only target
`agent-cpp::engine_abi` (`<agent-cpp/engine_abi.h>`), `agent_add_engine(<target> <sources>...)` (a shared
library without the `lib` prefix and with hidden visibility), `agent_install_engine(<target> [libs...])`
(installs into `<prefix>/lib/agent/engines/`, like the in-tree function of the same name) and
`agent-cpp_ENGINE_HOST` (the installed `agent-engine-host`, for process-isolation tests).
[`templates/engine-plugin/`](../templates/engine-plugin) is a standalone project using them: a C echo
engine plus a test harness that loads it in process and through the host. Checked by installing to a
temporary prefix, building a copy of the template outside the source tree, running its test and
registering the result with `agent engine add`. See [guides/writing-an-engine.md](guides/writing-an-engine.md).

## Linux tarball: `scripts/package.sh`

This script makes a Linux-only tarball from a local build.

```sh
scripts/package.sh                                   # agent-0.1.2-linux-x64.tar.gz, llama engine, CPU
scripts/package.sh --engines llama,sdcpp
scripts/package.sh --engines llama,onnx --onnxruntime-root ~/onnxruntime-linux-x64-1.23.2
scripts/package.sh --engines llama,audiocpp
scripts/package.sh --backend vulkan                  # agent-0.1.2-linux-x64-vulkan.tar.gz
```

The script configures a Release build (`build-package/` by default, `--build-dir` to change it), builds
only the installed targets, installs with `--strip` into a staging directory and packs:

```
agent-<version>-linux-<arch>[-<backend>]/
  bin/agent
  lib/agent/engines/agent-engine-*.so  (+ bundled runtime libraries)
  share/doc/agent/README.md, LICENSE, cli.md, licenses/<third party>.LICENSE
```

The tarball also holds `bin/agent-engine-host`: the CLI runs engines in that process by default
(`EngineIsolation::Process`), so the packaged `agent` needs it next to the binary.

next to a `.sha256` file in `dist/` (`--out-dir`). Headers, static libraries and the CMake package are
left out: the tarball is the end-user CLI. Before packing it runs `bin/agent version` and fails if
`ldd` reports an unresolved library for the binary or any plugin.

By default ggml is built with `GGML_NATIVE=OFF`, which enables AVX, AVX2, FMA, F16C and BMI2 (roughly
x86-64-v3) instead of `-march=native`; `--native` builds for the packaging machine only. Native builds still leave out Intel AMX unless
`-DAGENT_GGML_AMX=ON`.

The binary links the system OpenSSL 3 (`libssl.so.3`, `libcrypto.so.3`), `libstdc++` and glibc of the
build machine, so build on the oldest distribution you want to support.

Not done yet:

- The `cuda` and `vulkan` backends only pass `GGML_CUDA` / `GGML_VULKAN` to the llama build. They have
  not been built or tested, and the CUDA runtime libraries are not bundled.
- Licence files for cpp-httplib, sqlite-vec and SQLite (public domain) are not in `vendor/`, so they are
  not copied into `licenses/`.

## Release archives

No release archives are published yet (the version is `dev-v0.0.x`, and there are no GitHub Releases).
Build from source as described in [getting-started.md](getting-started.md#track-a-the-agent-cli), or use the Linux
tarball script above. The script is run by hand. No GitHub Action runs automatically: the CI and release
workflows are manual-dispatch only.

## Windows notes

- Plugin DLLs are loaded with `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`
  (`src/hal/dylib.cpp`), so a plugin finds the ONNX Runtime DLLs next to it.
- `agent_hub` needs OpenSSL 3 at build time.
- Build from source with the steps in [getting-started.md](getting-started.md#track-a-the-agent-cli).

