#!/bin/sh
# Points git at the repository hooks (.githooks/): format check, build and tests before a push.
cd "$(dirname "$0")/.." && git config core.hooksPath .githooks && echo "git hooks enabled (core.hooksPath = .githooks)"
