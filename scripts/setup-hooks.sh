#!/bin/sh
# Points git at the repository's hooks (.githooks/): format check on commit, build + tests on push.
cd "$(dirname "$0")/.." && git config core.hooksPath .githooks && echo "git hooks enabled (core.hooksPath = .githooks)"
