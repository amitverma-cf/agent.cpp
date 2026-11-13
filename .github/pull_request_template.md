## What this changes

<!-- One feature or fix per PR. Say why, and link the issue if there is one. -->

Closes #

## AI use

<!-- Required. See the AI Usage Policy in CONTRIBUTING.md and AGENTS.md. -->

- [ ] Most of this code was written by me. AI tools (if any) only helped with corrections or with expanding code I had already designed.
- [ ] I say below how AI was used, and I reviewed every changed line myself.
- [ ] I can explain every line in this PR if a maintainer asks.

How AI was used (write "none" if it was not used):

## Checklist

- [ ] I followed the rules in CONTRIBUTING.md and AGENTS.md: no heap allocations in the per-turn path, no C++ exceptions (use `Result<T>`), no vtables.
- [ ] I searched for an existing PR or issue for this change.
- [ ] I ran `scripts/format.sh` on the files I changed (only when about to push).
- [ ] I ran the tests: `ctest --test-dir build --output-on-failure`. New behavior has new tests.
- [ ] For changes to `src/` with concurrency or memory effects: I ran with `-DAGENT_ENABLE_SANITIZERS=ON`, and with valgrind `--tool=helgrind` where it applies.
- [ ] I updated the docs that cover this change (`docs/`, and the doc comments in `include/agent-cpp/agent.hpp` for public API).
- [ ] This PR does not break existing signatures or fields in the public API, or I explain the break below.
