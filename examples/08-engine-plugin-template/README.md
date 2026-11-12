# agent.cpp engine plugin template

A standalone CMake project that builds an agent.cpp engine plugin (`agent-engine-echo`) against an
**installed** agent.cpp, plus a test harness. Copy this directory out of the agent.cpp tree to start a
new engine; nothing here refers back to the source tree.

```
CMakeLists.txt        find_package(agent-cpp), agent_add_engine(), tests, agent_install_engine()
src/echo_engine.c     the engine: plain C against <agent-cpp/engine_abi.h>
test/engine_test.cpp  loads the plugin through libagent, in process and under process isolation
```

## Build and test

```sh
# 1. agent.cpp itself, installed somewhere (headers, libagent, agent-engine-host, the agent CLI)
cmake -S agent.cpp -B agent.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build agent.cpp/build
cmake --install agent.cpp/build --prefix /opt/agent

# 2. this project
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/agent
cmake --build build
ctest --test-dir build --output-on-failure

# 3. install next to the CLI and register it
cmake --install build --prefix /opt/agent
/opt/agent/bin/agent engine add /opt/agent/lib/agent/engines/agent-engine-echo.so
/opt/agent/bin/agent engine ls
```

`find_package(agent-cpp)` provides:

| | |
|---|---|
| `agent-cpp::engine_abi` | header-only target for `<agent-cpp/engine_abi.h>`; a plugin needs nothing else |
| `agent_add_engine(<target> <sources>...)` | shared library without the `lib` prefix, hidden visibility, `--exclude-libs,ALL` on Linux |
| `agent_install_engine(<target> [libs...])` | installs into `<prefix>/lib/agent/engines/` (plus bundled runtime libraries) |
| `agent-cpp::agent` | libagent, used here only by the test harness |
| `agent-cpp_ENGINE_HOST` | the installed `agent-engine-host`, for process-isolation tests |
| `agent-cpp_ENGINE_DIR` | `<prefix>/lib/agent/engines` |

## Making it yours

1. Rename the target to `agent-engine-<name>` and set the engine's `name`, `version` and `caps` in the
   `agent_engine_v1` table.
2. Load the model in `create()` (`config_json` carries `model`, `context_window`, `max_tokens`,
   `temperature`, `parallel` and `options` = `Config::engine_options`), run it in `run()`.
3. Link your inference library with `target_link_libraries(agent-engine-<name> PRIVATE ...)`. Prefer
   static linking; shared runtime libraries go next to the plugin (RPATH `$ORIGIN`) and are listed in
   `agent_install_engine`.
4. Update the expected replies in `test/engine_test.cpp`.

The rules (struct sizes, ownership, streaming, progress, cancel, sandbox) are in
[docs/guides/writing-an-engine.md](../../docs/guides/writing-an-engine.md) and
[docs/api/engines.md](../../docs/api/engines.md).
