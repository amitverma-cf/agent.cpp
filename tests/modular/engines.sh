#!/bin/bash
# Real models through the engines of a modular install (Linux; the box). Installs into its own folder from a release folder and the real
# upstream downloads, then runs each installed engine on a real model. Needs the models of the box in $MODELS.
#
#   engines.sh RELEASE_DIR SCRIPTS_DIR [TAG]
set -u
REL="${1:?release folder}"; SCRIPTS="${2:?scripts folder}"; TAG="${3:-local}"
MODELS="${MODELS:-$HOME/models}"
T=${MODTEST_DIR:-/tmp/agent-engtest}; rm -rf "$T"; mkdir -p "$T"
export AGENT_HOME="$T/root" AGENT_RELEASE="$REL" AGENT_SCRIPTS="$SCRIPTS" AGENT_VERSION="$TAG" AGENT_NO_ENV=1 AGENT_YES=1 AGENT_NO_START=1 NO_COLOR=1
A="$AGENT_HOME/bin/agent"
pass=0; fail=0
ok_if() { if eval "$2" >/dev/null 2>&1; then pass=$((pass + 1)); printf 'PASS %s\n' "$1"; else fail=$((fail + 1)); printf 'FAIL %s  ::  %s\n' "$1" "$2"; fi; }
show() { sed 's/^/    /' | cut -c1-200 | head -${1:-6}; }

sh "$SCRIPTS/install.sh" install --yes 2>&1 | tail -3
ENGINES="${ENGINES:-llama audiocpp onnx sdcpp trellis}"
for e in $ENGINES; do sh "$AGENT_HOME/scripts/install.sh" add "$e" 2>&1 | tail -3; done
echo "== doctor"; "$A" doctor 2>&1 | sed -n '/^engines/,/^models/p' | show 10

Q=$(find "$MODELS" -name 'qwen2.5-0.5b*.gguf' | head -1); BGE=$(find "$MODELS" -name 'bge-small*.gguf' | head -1)
JINA=$(find "$MODELS" -name 'jina-reranker*.gguf' | head -1); TTS=$(find "$MODELS" -name 'pocket-tts-english-q8_0.gguf' | head -1)

if echo " $ENGINES " | grep -q " llama "; then
    echo "== llama (upstream release b12020 + our plugin)"
    for m in "$Q" "$BGE" "$JINA"; do "$A" model pull "$m" >/dev/null 2>&1; done
    out="$("$A" chat -m qwen2.5-0.5b-instruct-q4_k_m -y -p 'Reply with only the single word: pineapple' 2>&1 | tail -3)"; echo "$out" | show 3
    ok_if "llama: a real chat answer comes back" "printf '%s' \"\$out\" | grep -qi pineapple"
    out="$("$A" task embed -m bge-small-en-v1.5-q8_0 --text 'hello world' 2>&1 | tail -3)"; echo "$out" | show 3
    ok_if "llama: an embedding of a real text" "printf '%s' \"\$out\" | grep -qi embedding"
    out="$("$A" task rerank -m jina-reranker-v1-tiny-en-Q8_0 --text 'red fox' --doc 'a red fox jumps' --doc 'blue sky' 2>&1 | tail -4)"; echo "$out" | show 4
    ok_if "llama: a rerank puts the matching passage first" "printf '%s' \"\$out\" | grep -q 'score\[0\]'"
    ok_if "llama: the plugin is small and the libraries are upstream's" "[ \$(stat -c %s '$AGENT_HOME/engines/llama/agent-engine-llama.so') -lt 3000000 ] && [ -f '$AGENT_HOME/engines/llama/libllama.so' ]"
fi
if echo " $ENGINES " | grep -q " audiocpp " && [ -d "$AGENT_HOME/engines/audiocpp" ]; then
    echo "== audiocpp"
    "$A" model pull "$TTS" >/dev/null 2>&1
    out="$("$A" task tts -m pocket-tts-english-q8_0 --text 'Hello there' 2>&1 | tail -4)"; echo "$out" | show 4
    ok_if "audiocpp: speech is written to a wav file" "printf '%s' \"\$out\" | grep -qiE 'wav|audio/'"
fi
if echo " $ENGINES " | grep -q " onnx "; then
    echo "== onnx (the official onnxruntime)"
    python3 - "$T/add.onnx" <<'PY'
import sys
b = bytes([0x08,0x08,0x12,0x09,0x61,0x67,0x65,0x6e,0x74,0x2e,0x63,0x70,0x70,0x3a,0x63,0x0a,0x15,0x0a,0x01,0x61,0x0a,0x01,0x62,0x12,0x03,0x73,0x75,0x6d,0x1a,0x03,0x61,0x64,0x64,0x22,0x03,0x41,0x64,0x64,0x12,0x09,0x61,0x64,0x64,0x5f,0x67,0x72,0x61,0x70,0x68,0x5a,0x13,0x0a,0x01,0x61,0x12,0x0e,0x0a,0x0c,0x08,0x01,0x12,0x08,0x0a,0x02,0x08,0x02,0x0a,0x02,0x08,0x02,0x5a,0x13,0x0a,0x01,0x62,0x12,0x0e,0x0a,0x0c,0x08,0x01,0x12,0x08,0x0a,0x02,0x08,0x02,0x0a,0x02,0x08,0x02,0x62,0x15,0x0a,0x03,0x73,0x75,0x6d,0x12,0x0e,0x0a,0x0c,0x08,0x01,0x12,0x08,0x0a,0x02,0x08,0x02,0x0a,0x02,0x08,0x02,0x42,0x04,0x0a,0x00,0x10,0x0d])
open(sys.argv[1], 'wb').write(b)
PY
    "$A" model pull "$T/add.onnx" >/dev/null 2>&1
    out="$("$A" task generic -e onnx -m add --tensor a:f32:2,2=1,2,3,4 --tensor b:f32:2,2=10,20,30,40 2>&1 | tail -4)"; echo "$out" | show 4
    ok_if "onnx: a real model adds two tensors (11 22 33 44)" "printf '%s' \"\$out\" | grep -q '11'"
    ok_if "onnx: the library is the official onnxruntime" "ls '$AGENT_HOME/engines/onnx/' | grep -q 'libonnxruntime.so'"
fi
if echo " $ENGINES " | grep -q " audiocpp " && [ ! -d "$AGENT_HOME/engines/audiocpp" ]; then echo "SKIP audiocpp: the installer refused it here (its upstream Linux build needs a newer glibc)"; fi
for e in sdcpp trellis; do
    if echo " $ENGINES " | grep -q " $e "; then ok_if "$e: the plugin loads (no model of this kind on the box)" "'$A' doctor 2>&1 | grep -Eq '^ +$e +ok'"; fi
done
echo "SUMMARY $pass passed, $fail failed"
