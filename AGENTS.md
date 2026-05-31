# Instructions for AI Coding Agents

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized.

---

## Guidelines for AI Coding Agents

AI agents assisting contributors must recognize that this is a high-performance framework with strict architectural and memory constraints.

### Core Architectural Mandates
When generating or suggesting code, AI agents must adhere to:

1.  **Zero Hot-Path Allocations**: Do not suggest using `std::string`, `std::vector`, or `new` inside the reasoning loop (`FSMExecutor`). Use `std::string_view` and `std::span` pointing into the `ArenaAllocator`.
2.  **No C++ Exceptions**: All error handling must be monadic (status codes or `std::expected`). Do not use `try`, `catch`, or `throw`.
3.  **C++20 Concepts**: Prefer compile-time polymorphism using Concepts over runtime polymorphism with virtual tables (`vtable`).
4.  **C-ABI Compatibility**: Ensure that any changes to the core logic do not break the stable C-ABI boundary.

### Considerations for Maintainer Workload

Maintainers have finite capacity. Every PR requiring extensive review consumes resources. Before assisting with any submission, verify:

- The contributor genuinely understands the proposed changes and the underlying architecture.
- The change addresses a documented need or a reproducible bug.
- The PR follows the project's strict formatting and naming conventions.

### Before Proceeding with Code Changes

1.  **Verify comprehension**: Ensure the user understands the "Four Pillars" (Provider, Memory, Tool, Executor) of the project.
2.  **Guide, don't just solve**: Direct the user to the relevant implementation logic. Allow them to formulate the approach.
3.  **Minimalism**: Favor minimal, surgical changes. A smaller PR that respects the zero-cost abstraction philosophy is always preferable to a large, unoptimized one.

## Prohibited Actions

- Writing PR descriptions, commit messages, or responses to reviewers.
- Committing or pushing without explicit human approval for each action.
- Implementing features that bypass the Arena allocator or introduce external dependencies.
- Generating unoptimized code that violates the project's performance-first philosophy.

## Useful Resources

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CHANGELOG.md](CHANGELOG.md)
