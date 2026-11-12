#!/bin/sh
# Formats the C++ files with clang-format and the repository .clang-format. Run it only when about to push: clang-format reorders includes
# (guard windows.h blocks with clang-format off), so never run it on files you are editing.
#   scripts/format.sh          every tracked .cpp/.hpp/.h outside vendor/
#   scripts/format.sh FILE...  just those files
cd "$(dirname "$0")/.." || exit 1
if [ $# -gt 0 ]; then files="$*"; else files=$(git ls-files '*.cpp' '*.hpp' '*.h' | grep -v '^vendor/'); fi
echo "$files" | xargs clang-format -i
