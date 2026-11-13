# Working in this repository (for coding agents and people)

agent.cpp is a high-performance C++20 framework with strict architectural and memory rules. These notes say what they are and how to build
and test a change.

## Rules of the codebase

1.  **No allocations on the hot path.** Do not use `std::string`, `std::vector` or `new` in the per-turn dispatch path: building
    `MessageView` / `ToolCallView` / `TensorView` for a provider call, or the token-generation loop. Use `std::string_view` and
    `std::span` pointing into the `Arena` there. This does not extend to a `Flow`'s persistent state (`Conversation::history` and
    friends): the `Arena` resets every turn, so anything that must survive across turns (conversation history, key-value memory)
    owns its storage and lives outside it.
2.  **No C++ exceptions.** Error handling is monadic: fallible functions return `Result<T>`. Do not use `try`, `catch` or `throw`.
3.  **No vtables.** Providers, memory and tools are selected through data-oriented dispatch: `constexpr` arrays of function pointers
    (`ProviderOps`, `StoreOps`, `ToolsOps`).
4.  **Public headers.** The C++ API is the single header `include/agent-cpp/agent.hpp` (`Result<T>`, `std::string_view` / `std::span`
    views into the `Arena`, PascalCase types, snake_case free functions). Two C interfaces sit beside it: `agent_c.h` (the library from
    C) and `engine_abi.h` (engine plugins). Avoid breaking existing struct fields or function signatures.
5.  **No new dependencies** and no code that bypasses the `Arena` allocator.

## Layout

`src/agent/` holds four subsystems you will touch most: `providers/`, `memory/`, `tools/` and `executor/`. Engines are plugins under
`src/engines/`; the CLI is `apps/cli/`. [docs/architecture.md](docs/architecture.md) explains how they fit together.

## Build and test

```sh
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure        # or ./build/test_agent "[tag]" for one subsystem
scripts/format.sh                                 # clang-format; run it when you are about to push, not while editing
```

Keep changes small and focused, add a test for new behavior, and update the docs under `docs/` and the comments in `agent.hpp` when the
public API changes. [CONTRIBUTING.md](CONTRIBUTING.md) has the full checklist (sanitizers, valgrind for concurrency, coverage).
