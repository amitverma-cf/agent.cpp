# Installing and packaging

## Version

The version is set once, in `project("agent.cpp" VERSION x.y.z)` in the top-level `CMakeLists.txt` (currently 1.0.0).
`agent version`, the CMake package version (`agent-cppConfigVersion.cmake`) and the Linux tarball names all come from it.

## Which engines are built

Every engine is a plugin file, and each one has its own switch (`AGENT_ENGINE_<NAME>`):

| Engine | Default | Condition |
|---|---|---|
| `llama` | `ON` | Needs `vendor/llama.cpp`. |
| `onnx` | `ON` | CMake downloads the pinned ONNX Runtime release at configure time. Use `-DAGENT_ENGINE_ONNX=OFF` to avoid that. |
| `sdcpp` | `ON` when `vendor/stable-diffusion.cpp` is checked out | Its own nested ggml submodule is needed too. |
| `audiocpp` | `ON` when `vendor/audio.cpp` is checked out | |
| `trellis` | `ON` when a CUDA compiler is found and `vendor/trellis2cpp` is checked out | The option text says it needs CUDA; [api/engines.md](api/engines.md#the-trellis-engine) describes its nested build as CPU-only. Check before relying on it. |

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
`agent setup` finds the installed plugins there with no flags.

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
- `agent-engine-llama` and `agent-engine-sdcpp` link llama.cpp and stable-diffusion.cpp statically and need only system
  libraries (`libstdc++`, `libgomp` for OpenMP, `libc`).

## Using the library: `find_package(agent-cpp)`

```cmake
find_package(agent-cpp 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
# optional:
target_link_libraries(my_app PRIVATE agent-cpp::store_sqlite)  # SQLite KV store + sqlite-vec
target_link_libraries(my_app PRIVATE agent-cpp::hub)           # Hugging Face downloads (if built)
```

Configure with `-DCMAKE_PREFIX_PATH=<prefix>`. Pre-1.0, a request for version `0.1` accepts any `0.1.x`
(`SameMinorVersion`). The package also sets:

- `agent-cpp_ENGINE_DIR`: `<prefix>/lib/agent/engines`
- `agent-cpp_ENGINE_HOST`: `<prefix>/bin/agent-engine-host` (the host used with `EngineIsolation::Process`)

### Components

Components tell CMake which parts of this install the program needs. A required component that is missing stops
configuration with a message that says how to build it:

```cmake
find_package(agent-cpp 0.1 REQUIRED COMPONENTS graph rag llama)
```

| Component | Found when |
|---|---|
| `graph`, `mcp`, `rag`, `models` | this install was built with `AGENT_WITH_<NAME>=ON` (the library part is compiled in) |
| `llama`, `audiocpp`, `sdcpp`, `trellis`, `onnx` | the plugin `agent-engine-<name>.*` is in `lib/agent/engines` |

Each component sets `agent-cpp_<name>_FOUND`. Only list the components your program uses. `store_sqlite` and `hub` are
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

When agent.cpp is part of your build instead of an installed package, only the library is built unless you ask for more: the tests, the
examples, the CLI and the install rules are off (they default to on only when agent.cpp is the top-level project).

```cmake
include(FetchContent)
FetchContent_Declare(agent-cpp
    GIT_REPOSITORY https://github.com/amitverma-cf/agent.cpp.git
    GIT_TAG        main                  # pin a commit for reproducible builds
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES vendor/simdjson vendor/llama.cpp)   # only what you build; add vendor/audio.cpp, ... for those engines
set(AGENT_BUILD_HUB       OFF CACHE BOOL "" FORCE)  # ON needs OpenSSL
set(AGENT_ENGINE_ONNX     OFF CACHE BOOL "" FORCE)  # downloads ONNX Runtime at configure time
set(AGENT_ENGINE_SDCPP    OFF CACHE BOOL "" FORCE)
set(AGENT_ENGINE_AUDIOCPP OFF CACHE BOOL "" FORCE)
set(AGENT_ENGINE_TRELLIS  OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(agent-cpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
```

The llama.cpp engine plugin is built in your build tree (`AGENT_ENGINE_LLAMA`, default on; set it `OFF` for a library without engines). To
run a program from the build tree, `agent_deploy_engines(my_app)` (from `cmake/agent-cppEngine.cmake`) puts the plugins and
`agent-engine-host` next to it. The submodules you do not list are not cloned.

## vcpkg

`ports/agent-cpp` is an overlay port of the library (core, SQLite store, optionally the downloader): engines are not part of it, they are
plugins you build or download separately. It builds the commit named in its portfile; `scripts/update-vcpkg-port.sh [COMMIT]` pins it to
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

find_package(agent-cpp 0.1 REQUIRED COMPONENTS llama)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE agent-cpp::agent)
agent_deploy_engines(my_app llama)

install(TARGETS my_app RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

`agent_deploy_engines` does three things:

1. After every build of `my_app`, it copies the named plugins and the runtime libraries they load into
   `<my_app folder>/engines/`. For `audiocpp`, the `libaudiocpp*` files come along; for `onnx`, `libonnxruntime*`.
   Engines you do not name are not copied, so they never reach your release.
2. `cmake --install` installs the same files to `<bindir>/engines`.
3. It copies `agent-engine-host` next to the program (and installs it to `<bindir>`), for `EngineIsolation::Process`.

The program finds the engines at run time. The name `"llama"` resolves to `engines/agent-engine-llama.so` (or `.dll`, `.dylib`)
next to the executable, so the folder can be moved with the program. Nothing in the program is compiled against an absolute path.
The host is found next to the executable as well, so no `AGENT_ENGINE_HOST` setting is needed.

Things to know:

- Call `find_package` with the engine as a component, or the deploy function stops with an error. The engine plugin must be installed
  in `agent-cpp_ENGINE_DIR`.
- The engine files are found at configure time. After installing a new engine, run CMake again.
- The deploy function does not install your program or your own libraries. It also does not copy `libagent` when you built
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

This script makes a Linux-only tarball from a local build.

```sh
scripts/package.sh                                   # agent-0.1.2-linux-x64.tar.gz, llama engine, CPU
scripts/package.sh --engines llama,sdcpp
scripts/package.sh --engines llama,onnx --onnxruntime-root ~/onnxruntime-linux-x64-1.23.2
scripts/package.sh --engines llama,audiocpp
scripts/package.sh --backend vulkan                  # agent-0.1.2-linux-x64-vulkan.tar.gz
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
on the oldest distribution you want to support.

Not done yet:

- The `cuda` and `vulkan` backends only pass `GGML_CUDA` / `GGML_VULKAN` to the llama build. They have not been built or
  tested, and the CUDA runtime libraries are not bundled.
- Licence files for cpp-httplib, sqlite-vec and SQLite (public domain) are not in `vendor/`, so they are not copied into
  `licenses/`.

## Releases

The git tag `v0.1.2` matches the version in `CMakeLists.txt`. The CI and release workflows in `.github/workflows/` are
manual-dispatch only; nothing runs on a push. Build from source as described in [getting-started.md](getting-started.md#build),
or use the Linux tarball script above, which is run by hand.

## Windows notes

- Plugin DLLs are loaded with `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`
  (`src/hal/dylib.cpp`), so a plugin finds the ONNX Runtime DLLs next to it.
- `agent-engine-host.exe` is installed and deployed like the Linux host.
- `agent_hub` needs OpenSSL 3 at build time.
- Build from source with the steps in [getting-started.md](getting-started.md#build).
