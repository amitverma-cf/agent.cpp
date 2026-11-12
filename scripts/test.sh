#!/bin/sh
# The one test script, run by hand: configure, build and run the tests that need no model (they use the mock engines).
#
#   scripts/test.sh [--build-dir DIR] [--with-engines] [--jobs N] [--all]
#
#   --with-engines  also build the real engines (llama, audiocpp, sdcpp, trellis, onnx) so their model tests can run
#                   when the AGENT_TEST_* environment variables point at models (see CONTRIBUTING.md)
#   --all           run every test, including the ones that need a model, network or the OS keychain (they skip themselves
#                   when their inputs are missing)
# Packaging a release is a separate step: scripts/package.sh.
set -eu
cd "$(dirname "$0")/.."
build=build
engines=OFF
jobs=$(nproc 2>/dev/null || echo 4)
all=0
while [ $# -gt 0 ]; do
    case "$1" in
    --build-dir) build=$2; shift 2 ;;
    --with-engines) engines=ON; shift ;;
    --jobs) jobs=$2; shift 2 ;;
    --all) all=1; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DAGENT_ENGINE_LLAMA=$engines -DAGENT_ENGINE_AUDIOCPP=$engines \
    -DAGENT_ENGINE_SDCPP=$engines -DAGENT_ENGINE_TRELLIS=$engines -DAGENT_ENGINE_ONNX=$engines
cmake --build "$build" --config Release -j "$jobs"
if [ "$all" = 1 ]; then
    ctest --test-dir "$build" -C Release --output-on-failure -j 2
else
    ctest --test-dir "$build" -C Release --output-on-failure -j 2 -LE 'model|network|keychain' || ctest --test-dir "$build" -C Release --rerun-failed --output-on-failure -j 1
fi
