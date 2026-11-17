#!/bin/bash
# Tests the modular install on this machine (Linux; the box). It uses the files of a release build (a folder with SHA256SUMS and the
# agent-cpp-core-* and agent-engine-* archives, e.g. from `gh run download` of the release workflow) and the real upstream downloads,
# in a folder of its own: nothing outside it is touched (no shell profile, no environment).
#
#   run.sh RELEASE_DIR [TAG]            full run: install, every component, update path, uninstall
set -u
REL="${1:?release folder}"; TAG="${2:-local}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="${AGENT_SCRIPTS_DIR:-$HERE/../../scripts}"
T=${MODTEST_DIR:-/tmp/agent-modtest}; rm -rf "$T"; mkdir -p "$T"
export AGENT_HOME="$T/root" AGENT_RELEASE="$REL" AGENT_SCRIPTS="$SCRIPTS" AGENT_VERSION="$TAG" AGENT_NO_ENV=1 AGENT_YES=1 AGENT_NO_START=1 NO_COLOR=1
pass=0; fail=0
check() { # name, condition result
    if [ "$2" = ok ]; then pass=$((pass + 1)); printf 'PASS %s\n' "$1"; else fail=$((fail + 1)); printf 'FAIL %s  ::  %s\n' "$1" "$2"; fi
}
ok_if() { if eval "$2" >/dev/null 2>&1; then check "$1" ok; else check "$1" "$2"; fi; }

echo "== install"
sh "$SCRIPTS/install.sh" install --yes 2>&1 | tail -8
ok_if "install: the program is in bin/" "[ -x '$AGENT_HOME/bin/agent' ]"
ok_if "install: the marker and the pinned scripts are there" "[ -f '$AGENT_HOME/.agent-root' ] && [ -f '$AGENT_HOME/scripts/components.tsv' ] && [ -f '$AGENT_HOME/scripts/install.sh' ]"
ok_if "install: the program reports the folder" "[ \"\$('$AGENT_HOME/bin/agent' home)\" = '$AGENT_HOME' ]"
ok_if "install: nothing outside the folder (home)" "[ -z \"\$(find '$T' -mindepth 1 -maxdepth 1 ! -name root)\" ]"

echo "== ls"
sh "$AGENT_HOME/scripts/install.sh" ls 2>&1 | tail -14

echo "== every engine and runtime"
ENGINES="${ENGINES:-llama audiocpp sdcpp onnx trellis}"
RUNTIMES="${RUNTIMES:-bun uv node pnpm nub}"
for n in $ENGINES $RUNTIMES; do
    sh "$AGENT_HOME/scripts/install.sh" add "$n" 2>&1 | tail -3
done
for n in $ENGINES; do
    ok_if "engine $n: installed with its pin" "[ -f '$AGENT_HOME/engines/$n/.pin' ]"
    ok_if "engine $n: the program lists it as ok" "'$AGENT_HOME/bin/agent' doctor 2>&1 | grep -Eq '^ +$n +ok'"
done
for n in $RUNTIMES; do ok_if "runtime $n: installed" "[ -f '$AGENT_HOME/runtimes/$n/.pin' ]"; done
ok_if "runtime bun runs" "'$AGENT_HOME/runtimes/bun/bun' --version"
ok_if "runtime uv runs" "'$AGENT_HOME/runtimes/uv/uv' --version"
ok_if "runtime node runs" "'$AGENT_HOME/runtimes/node/node' --version"
ok_if "runtime pnpm is there" "[ -f '$AGENT_HOME/runtimes/pnpm/bin/pnpm.cjs' ]"
ok_if "runtime nub runs" "'$AGENT_HOME/runtimes/nub/nub' --version"
ok_if "agent runtime path finds bun" "[ \"\$('$AGENT_HOME/bin/agent' runtime path bun)\" = '$AGENT_HOME/runtimes/bun/bun' ]"

echo "== rm and add again"
sh "$AGENT_HOME/scripts/install.sh" rm bun 2>&1 | tail -1
ok_if "rm: bun is gone" "[ ! -d '$AGENT_HOME/runtimes/bun' ]"
ok_if "rm: the other runtimes stay" "[ -d '$AGENT_HOME/runtimes/uv' ]"
sh "$AGENT_HOME/scripts/install.sh" add bun 2>&1 | tail -1
ok_if "add: bun is back" "[ -x '$AGENT_HOME/runtimes/bun/bun' ]"

echo "== a download that does not match its sum is refused"
cp "$AGENT_HOME/scripts/components.tsv" "$T/good.tsv"
awk -F'\t' 'BEGIN{OFS="\t"} $2=="uv" && $7!="@sums" && !d {$7="0000000000000000000000000000000000000000000000000000000000000000"; d=1} {print}' "$T/good.tsv" >"$AGENT_HOME/scripts/components.tsv"
sh "$AGENT_HOME/scripts/install.sh" rm uv >/dev/null 2>&1
out="$(sh "$AGENT_HOME/scripts/install.sh" add uv 2>&1 || true)"
ok_if "a wrong SHA-256 is refused" "printf '%s' '$out' | grep -qi 'mismatch'"
ok_if "a wrong SHA-256 installs nothing" "[ ! -d '$AGENT_HOME/runtimes/uv' ]"
cp "$T/good.tsv" "$AGENT_HOME/scripts/components.tsv"

echo "== uninstall --keep-data, then uninstall"
mkdir -p "$AGENT_HOME/models/llama/keep" && echo x >"$AGENT_HOME/models/llama/keep/m.gguf"
sh "$AGENT_HOME/scripts/install.sh" uninstall --keep-data --yes 2>&1 | tail -2
ok_if "keep-data: the program is gone" "[ ! -d '$AGENT_HOME/bin' ] && [ ! -d '$AGENT_HOME/engines' ]"
ok_if "keep-data: the models stay" "[ -f '$AGENT_HOME/models/llama/keep/m.gguf' ]"
sh "$SCRIPTS/install.sh" install --yes >/dev/null 2>&1
sh "$AGENT_HOME/scripts/install.sh" uninstall --yes 2>&1 | tail -1
ok_if "uninstall: the folder is gone" "[ ! -e '$AGENT_HOME' ]"

echo "SUMMARY $pass passed, $fail failed"
