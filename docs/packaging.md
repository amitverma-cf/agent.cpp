# Installing and packaging

## Version

The version is set once, in `project("agent.cpp" VERSION x.y.z)` in the top-level `CMakeLists.txt` (currently 1.2.2).
`agent version`, the CMake package version (`agent-cppConfigVersion.cmake`) and the Linux tarball names all come from it.

## Which engines are built

Every engine is a plugin file, and each one has its own switch (`AGENT_ENGINE_<NAME>`):

| Engine | Default | Condition |
|---|---|---|
| `llama` | `ON` | Needs `vendor/llama.cpp`. |
| `onnx` | `ON` | CMake downloads the pinned ONNX Runtime release at configure time. Use `-DAGENT_ENGINE_ONNX=OFF` to avoid that. |
| `sdcpp` | `ON` when `vendor/stable-diffusion.cpp` is checked out | Its header is used; with `AGENT_SDCPP_PREBUILT` the release library is linked, otherwise sd.cpp is compiled (its nested ggml submodule is needed too). |
| `audiocpp` | `ON` when `vendor/audio.cpp` is checked out | |
| `trellis` | `ON` | A small plugin that runs the trellis.cpp release program `trellis-cli` ([api/engines.md](api/engines.md#the-trellis-engine)); nothing of trellis.cpp is compiled. |

### Plugins over an upstream release (what a release does)

llama.cpp and audio.cpp publish their own builds for every backend (CPU, Vulkan, CUDA, ROCm, ...). A release of agent.cpp does not compile
them: the `llama` and `audiocpp` plugins are compiled against the release that is pinned in `scripts/components.tsv` and link its shared
libraries, so a plugin is a few hundred kilobytes and compiles in seconds; the installer downloads the upstream part at the pinned version
for the backend the machine can run.

```sh
scripts/install.sh fetch llama build/upstream/llama           # the pinned upstream release, unpacked (install.ps1 fetch on Windows)
cmake -S . -B build -DAGENT_LLAMA_PREBUILT=$PWD/build/upstream/llama -DAGENT_AUDIOCPP_PREBUILT=$PWD/build/upstream/audiocpp
```

- `llama`: the plugin uses llama.cpp's own libraries `llama`, `llama-common` and `mtmd`, whose interfaces change with every commit, and
  compiles llama.cpp's scheduler sources (`tools/server/server-context` and its neighbours: the code `llama-server` and `llama-cli` run on,
  with multi-token prediction, draft models and native decision models), so `vendor/llama.cpp` must be at the pinned tag (`b12020`; checked:
  the commit is the tag's). Those sources are compiled into the plugin the same way on every system, because llama.cpp's Windows
  `llama-server-impl.dll` does not export them. Only they are compiled, not llama.cpp.
- `audiocpp`: only the C header `audiocpp.h` is used, which is the release's.
- `onnx`: always linked against the official onnxruntime release (`AGENT_ORT_VERSION` in `cmake/onnxruntime_release.cmake`).
- `sdcpp`: the plugin uses stable-diffusion.cpp's C header, which must be at the pinned release tag (`master-1106-532d2c1`), and links the
  release's `stable-diffusion` library.
- `trellis`: trellis.cpp publishes programs and no library, so the plugin runs the release's `trellis-cli` (found next to the plugin) and
  has nothing to link; the installer downloads the program for the machine's backend (Vulkan, CUDA, ROCm).
- Where the upstream build needs a newer C library than the machine has (audio.cpp's Linux builds need glibc 2.43), the installer says so
  and installs nothing, instead of installing a plugin that cannot load (`needs` column of `components.tsv`).

Clone with `git clone --recurse-submodules` so the optional engines are included. A plain clone still builds and says which
engines it skipped. Build options are listed in [getting-started.md](getting-started.md#build-options).

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
| `bin/agent-engine-host` | runs engines out of process (found next to `agent`, and next to a program that uses `agent_deploy_engines`) |
| `lib/agent/engines/agent-engine-<name>.so` | engine plugins, with the runtime libraries they load through their `$ORIGIN` RPATH (`libonnxruntime.so.1`, `libaudiocpp*`, ...). On Windows the DLLs are installed here too. |
| `lib/libagent.a` (or `.so` / `.dll`) | the library |
| `lib/libagent_store_sqlite.a`, `lib/libagent_hub.a` | the optional SQLite store and Hugging Face hub libraries |
| `include/agent-cpp/` | `agent.hpp`, `engine_abi.h`, `agent_c.h`, `store_sqlite.hpp`, `hub.hpp` |
| `lib/cmake/agent-cpp/` | `agent-cppConfig.cmake`, `agent-cppConfigVersion.cmake`, the exported targets, `agent-cppEngine.cmake` (the engine helper functions) |
| `share/doc/agent/` | `README.md`, `LICENSE`, `cli.md` |

The library directory is `CMAKE_INSTALL_LIBDIR` (`lib`, or `lib64` on some distributions). The engine directory is always
`lib/agent/engines`, because that is where the CLI looks (`<exe>/../lib/agent/engines`, see [cli.md](cli.md)). An installed
`agent` finds the installed plugins there with no flags.

The mock engine plugins (`agent-engine-mock*`, used by the tests) are not installed by default, so CLI archives never carry
them. Library users who want model-free tests build with `-DAGENT_INSTALL_MOCK_ENGINE=ON`; the plugins then go to
`lib/agent/testing` (not `lib/agent/engines`, so the CLI does not list them) and the `testing` component finds them:

```cmake
find_package(agent-cpp REQUIRED COMPONENTS testing)
add_executable(my_tests my_tests.cpp)
target_link_libraries(my_tests PRIVATE agent-cpp::agent agent-cpp::testing)
```

`agent-cpp::testing` defines `AGENT_CPP_MOCK_ENGINE` (answers chat with canned text and calls tools on request) and
`AGENT_CPP_MOCK_ALL_ENGINE` (every capability with deterministic stand-in output) as the plugin paths; set one as
`Config::engine` with `provider = AiProvider::Engine` (`Config::engine` takes a path to a plugin).

`-DAGENT_INSTALL=OFF` generates no install rules; this is the default when agent.cpp is added with `add_subdirectory`.

### Static or shared

`BUILD_SHARED_LIBS` (default `OFF`) switches `libagent`, `agent_store_sqlite` and `agent_hub` between static and shared.
Every symbol in the libraries is exported. Engine plugins are always shared modules, and the CLI's internal
`agent_cli_core` is always static.

RPATHs:

- `agent` with static libraries: no RPATH. With shared libraries: `$ORIGIN/../lib`.
- shared `libagent_store_sqlite` and `libagent_hub`: `$ORIGIN`, so they find `libagent` next to them.
- `agent-engine-host` with shared libraries: `$ORIGIN/../lib`, so it finds `libagent` in the prefix's `lib`.
- `agent-engine-onnx` and `agent-engine-audiocpp`: `$ORIGIN`, so they find their bundled runtime libraries.
- `agent-engine-llama` (prebuilt mode, what a release is) and `agent-engine-audiocpp` find the upstream shared libraries next to them (`$ORIGIN`). A source build of `agent-engine-llama` and `agent-engine-sdcpp` links llama.cpp and stable-diffusion.cpp statically and needs only system
  libraries (`libstdc++`, `libgomp` for OpenMP, `libc`).

## Using the library: `find_package(agent-cpp)`

```cmake
find_package(agent-cpp 1.2 REQUIRED)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
# optional:
target_link_libraries(my_app PRIVATE agent-cpp::store_sqlite)  # SQLite KV store + sqlite-vec
target_link_libraries(my_app PRIVATE agent-cpp::hub)           # Hugging Face downloads (if built)
```

Configure with `-DCMAKE_PREFIX_PATH=<prefix>`. A request for version `1.2` accepts any `1.2.x`
(`SameMinorVersion`). The package also sets:

- `agent-cpp_ENGINE_DIR`: `<prefix>/lib/agent/engines`
- `agent-cpp_ENGINE_HOST`: `<prefix>/bin/agent-engine-host` (the host used with `EngineIsolation::Process`)

### Components

Components tell CMake which parts of this install the program needs. A required component that is missing stops
configuration with a message that says how to build it:

```cmake
find_package(agent-cpp 1.2 REQUIRED COMPONENTS graph rag llama)
```

| Component | Found when |
|---|---|
| `graph`, `mcp`, `rag`, `models` | this install was built with `AGENT_WITH_<NAME>=ON` (the library part is compiled in) |
| `llama`, `audiocpp`, `sdcpp`, `trellis`, `onnx` | the plugin `agent-engine-<name>.*` is in `lib/agent/engines` |

Each component sets `agent-cpp_<name>_FOUND`. Only list the components the program uses. `store_sqlite` and `hub` are
targets, not components: link them as shown above.

The public headers do not include simdjson: it is compiled into `libagent` and is an implementation detail. Static packages
pull in `Threads` (and `OpenSSL` for `agent-cpp::hub`) with `find_dependency`.

`tests/consumer/` is a minimal out-of-tree project that does this (a Mock-provider turn plus the SQLite store):

```sh
cmake --install build --prefix /tmp/agent-prefix
cmake -S tests/consumer -B /tmp/agent-consumer -DCMAKE_PREFIX_PATH=/tmp/agent-prefix
cmake --build /tmp/agent-consumer && /tmp/agent-consumer/consumer
```

## Without installing: `FetchContent` or `add_subdirectory`

When agent.cpp is part of the build instead of an installed package, only the library is built unless more is requested: the tests, the
examples, the CLI and the install rules are off (they default to on only when agent.cpp is the top-level project).

```cmake
include(FetchContent)
FetchContent_Declare(agent-cpp
    GIT_REPOSITORY https://github.com/amitverma-cf/agent.cpp.git
    GIT_TAG        main                  # pin a commit for reproducible builds
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES vendor/simdjson vendor/llama.cpp)   # only what is built; add vendor/audio.cpp, ... for those engines
set(AGENT_BUILD_HUB       OFF CACHE BOOL "" FORCE)  # ON needs OpenSSL
set(AGENT_ENGINE_ONNX     OFF CACHE BOOL "" FORCE)  # downloads ONNX Runtime at configure time
set(AGENT_ENGINE_SDCPP    OFF CACHE BOOL "" FORCE)
set(AGENT_ENGINE_AUDIOCPP OFF CACHE BOOL "" FORCE)
set(AGENT_ENGINE_TRELLIS  OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(agent-cpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
```

The llama.cpp engine plugin is built in the build tree (`AGENT_ENGINE_LLAMA`, default on; set it `OFF` for a library without engines). To
run a program from the build tree, `agent_deploy_engines(my_app)` (from `cmake/agent-cppEngine.cmake`) puts the plugins and
`agent-engine-host` next to it. The submodules that are not listed are not cloned.

## vcpkg

`ports/agent-cpp` is an overlay port of the library (core, SQLite store, optionally the downloader): engines are not part of it, they are
plugins built or downloaded separately. It builds the commit named in its portfile; `scripts/update-vcpkg-port.sh [COMMIT]` pins it to
another one (it computes both SHA512 values).

```sh
vcpkg install agent-cpp --overlay-ports=path/to/agent.cpp/ports            # core + store
vcpkg install "agent-cpp[hub]" --overlay-ports=path/to/agent.cpp/ports     # + agent::hub (needs openssl)
```

```cmake
find_package(agent-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE agent-cpp::agent agent-cpp::store_sqlite)
```

## Shipping an application with engines

`agent_deploy_engines(<target> <engine>...)` (defined in `cmake/agent-cppEngine.cmake`, available after
`find_package(agent-cpp)`) packages the engines an application uses. A complete `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_app CXX)
include(GNUInstallDirs)

find_package(agent-cpp 1.2 REQUIRED COMPONENTS llama)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
agent_deploy_engines(my_app llama)

install(TARGETS my_app RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

`agent_deploy_engines` does three things:

1. After every build of `my_app`, it copies the named plugins and the runtime libraries they load into
   `<my_app folder>/engines/`. For `audiocpp`, the `libaudiocpp*` files come along; for `onnx`, `libonnxruntime*`.
   Engines that are not named are not copied, so they never reach the release.
2. `cmake --install` installs the same files to `<bindir>/engines`.
3. It copies `agent-engine-host` next to the program (and installs it to `<bindir>`), for `EngineIsolation::Process`.

The program finds the engines at run time. The name `"llama"` resolves to `engines/agent-engine-llama.so` (or `.dll`, `.dylib`)
next to the executable, so the folder can be moved with the program. Nothing in the program is compiled against an absolute path.
The host is found next to the executable as well, so no `AGENT_ENGINE_HOST` setting is needed.

Things to know:

- Call `find_package` with the engine as a component, or the deploy function stops with an error. The engine plugin must be installed
  in `agent-cpp_ENGINE_DIR`.
- The engine files are found at configure time. After installing a new engine, run CMake again.
- The deploy function does not install the program or custom libraries. It also does not copy `libagent` when built
  with `BUILD_SHARED_LIBS=ON`; in that case the folder is only complete if `libagent` is installed next to the program or
  found through its RPATH. The default static build needs nothing else.
- To ship an engine that is not in this tree, install its plugin into `lib/agent/engines` (see
  [writing an engine](guides/writing-an-engine.md)) and name it in `agent_deploy_engines`, the same way.

## Engine plugins outside this tree

The package also serves engine authors. `find_package(agent-cpp)` provides the header-only target
`agent-cpp::engine_abi` (`<agent-cpp/engine_abi.h>`), `agent_add_engine(<target> <sources>...)` (a shared library without the
`lib` prefix and with hidden visibility), `agent_install_engine(<target> [libs...])` (installs into `<prefix>/lib/agent/engines/`,
like the in-tree function of the same name) and `agent-cpp_ENGINE_HOST` (the installed `agent-engine-host`, for
process-isolation tests).
[`examples/08-engine-plugin-template/`](../examples/08-engine-plugin-template) is a standalone project using them: a C echo
engine plus a test harness that loads it in process and through the host. See [guides/writing-an-engine.md](guides/writing-an-engine.md).

## Linux tarball: `scripts/package.sh`

This script makes a Linux-only tarball from a local build with llama.cpp compiled from source. Releases do not use it (see "Releases" below); it is run by hand.

```sh
scripts/package.sh                                   # agent-1.2.1-linux-x64.tar.gz, llama engine, CPU
scripts/package.sh --engines llama,sdcpp
scripts/package.sh --engines llama,onnx --onnxruntime-root ~/onnxruntime-linux-x64-1.23.2
scripts/package.sh --engines llama,audiocpp
scripts/package.sh --backend vulkan                  # agent-1.2.1-linux-x64-vulkan.tar.gz
```

The script configures a Release build (`build-package/` by default, `--build-dir` to change it), builds only the installed
targets, installs with `--strip` into a staging directory and packs:

```
agent-<version>-linux-<arch>[-<backend>]/
  bin/agent
  bin/agent-engine-host
  lib/agent/engines/agent-engine-*.so  (+ bundled runtime libraries)
  share/doc/agent/README.md, LICENSE, cli.md, licenses/<third party>.LICENSE
```

The CLI runs engines in the host by default (`EngineIsolation::Process`), so the packaged `agent` needs `agent-engine-host`
next to it; the script includes it.

The tarball is written to `dist/` (`--out-dir`) with a `.sha256` file next to it. Headers, static libraries and the CMake
package are left out: the tarball is the end-user CLI. Before packing it runs `bin/agent version` and fails if `ldd` reports
an unresolved library for the binary or any plugin.

By default ggml is built with `GGML_NATIVE=OFF`, which enables AVX, AVX2, FMA, F16C and BMI2 (roughly x86-64-v3) instead of
`-march=native`; `--native` builds for the packaging machine only. Native builds still leave out Intel AMX unless
`-DAGENT_GGML_AMX=ON`.

The binary links the system OpenSSL 3 (`libssl.so.3`, `libcrypto.so.3`), `libstdc++` and glibc of the build machine, so build
on the oldest supported distribution.

Not done yet:

- The `cuda` and `vulkan` backends only pass `GGML_CUDA` / `GGML_VULKAN` to the llama build. They have not been built or
  tested, and the CUDA runtime libraries are not bundled. The installer gets CUDA, ROCm and Vulkan builds from llama.cpp's releases instead ([gpu.md](gpu.md)).
- Licence files for cpp-httplib, sqlite-vec and SQLite (public domain) are not in `vendor/`, so they are not copied into
  `licenses/`.

## Releases

A release is a set of small files, not one archive: `agent-cpp-core-<tag>-<os>-<arch>` (the program) and one
`agent-engine-<name>-<tag>-<os>-<arch>` for each of the thin plugins (llama, audiocpp, onnx, sdcpp, trellis), made by `scripts/package-release.sh`, plus `SHA256SUMS`. The upstream
parts (llama.cpp, audio.cpp, stable-diffusion.cpp, trellis.cpp, onnxruntime, bun, uv, node, pnpm, nub) are not rebuilt or copied: the installer script downloads them from their own projects.
`scripts/components.tsv` lists every component with its pinned version, per operating system, architecture and backend, and the SHA-256 of
each file. It is written by

```sh
python3 scripts/pin-components.py --tag v1.2.2 > scripts/components.tsv     # before the tag is made; the release workflow checks it
```

Raising the pin of an upstream project means changing `PINS` in that script, running it, and testing `tests/modular` (the installer tests) on Linux
and Windows. The workflow's `publish` job refuses a tag whose `components.tsv` was written for another tag.

The git tag `v1.2.2` matches the version in `CMakeLists.txt`. The CI and release workflows in `.github/workflows/` are
manual-dispatch only; nothing runs on a push. Build from source as described in [getting-started.md](getting-started.md#build),
or use the Linux tarball script above, which is run by hand.

## Windows notes

- Plugin DLLs are loaded with `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`
  (`src/hal/dylib.cpp`), so a plugin finds the ONNX Runtime DLLs next to it.
- `agent-engine-host.exe` is installed and deployed like the Linux host.
- `agent_hub` needs OpenSSL 3 at build time.
- Build from source with the steps in [getting-started.md](getting-started.md#build).
