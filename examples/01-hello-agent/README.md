# 01 - hello agent

The smallest programs: open a session on an engine, send one prompt, print the answer.

- `getting_started.cpp`: `engine_config()` + `init()` + `run_turn()`; the code of [the library guide](../../docs/guides/getting-started-library.md).
- `basic.cpp`: the same on the mock engine, no model needed.

```sh
cmake -B build -DAGENT_BUILD_EXAMPLES=ON && cmake --build build
./build/examples/guide_getting_started build/src/engines/mock/agent-engine-mock.so mock-model "hello" /tmp/ws
```
