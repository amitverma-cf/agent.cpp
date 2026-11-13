## What this changes

<!-- One feature or fix per PR. Say why, and link the issue if there is one. -->

Closes #

## Checklist

- [ ] I followed the rules in CONTRIBUTING.md: no heap allocations in the per-turn path, no C++ exceptions (use `Result<T>`), no vtables.
- [ ] I searched for an existing PR or issue for this change.
- [ ] I ran `scripts/format.sh` on the files I changed (only when about to push).
- [ ] I ran the tests: `ctest --test-dir build --output-on-failure`. New behavior has new tests.
- [ ] For changes to `src/` with concurrency or memory effects: I ran with `-DAGENT_ENABLE_SANITIZERS=ON`, and with valgrind `--tool=helgrind` where it applies.
- [ ] I updated the docs that cover this change (`docs/`, and the doc comments in `include/agent-cpp/agent.hpp` for public API).
- [ ] This PR does not break existing signatures or fields in the public API, or I explain the break below.
