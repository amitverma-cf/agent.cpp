#!/bin/sh
# The one update script, run by hand (nothing runs it for you, and it never commits):
#
#   scripts/update-deps.sh            list every vendored engine and the ONNX Runtime pin against the newest upstream
#   scripts/update-deps.sh bump NAME  move submodule NAME (llama.cpp, audio.cpp, stable-diffusion.cpp, ...) to upstream's latest
#   scripts/update-deps.sh ort        print the newest ONNX Runtime release with the sha256 values for cmake/onnxruntime_release.cmake
#
# After a bump, run scripts/test.sh; when it passes, run scripts/package.sh; commit the new pointer only then.
cd "$(dirname "$0")/.." || exit 1

ort_latest() {
    curl -fsSL https://api.github.com/repos/microsoft/onnxruntime/releases/latest | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("version:", d["tag_name"].lstrip("v"))
want = ("linux-x64-", "linux-aarch64-", "win-x64-", "win-arm64-", "osx-arm64-")
for a in d["assets"]:
    n = a["name"]
    if n.startswith("onnxruntime-") and any(w in n for w in want) and not any(x in n for x in ("gpu", "cuda", "training", "tensorrt", "rocm")) and n.endswith((".tgz", ".zip")):
        print(a.get("digest", "?").replace("sha256:", ""), n)
'
}

list() {
    printf '%-24s %-10s %-10s %s\n' submodule pinned upstream state
    git submodule status 2>/dev/null | while read -r line; do
        sha=$(echo "$line" | sed 's/^[ +-]//' | cut -d' ' -f1)
        path=$(echo "$line" | sed 's/^[ +-]//' | cut -d' ' -f2)
        url=$(git config -f .gitmodules --get "submodule.$path.url")
        head=$(git ls-remote "$url" HEAD 2>/dev/null | cut -f1)
        [ -z "$head" ] && head="(unreachable)"
        state="up to date"
        [ "$sha" != "$head" ] && state="differs"
        printf '%-24s %-10s %-10s %s\n' "${path#vendor/}" "$(echo "$sha" | cut -c1-9)" "$(echo "$head" | cut -c1-9)" "$state"
    done
    echo
    pinned=$(sed -n 's/^set(AGENT_ORT_VERSION "\(.*\)")/\1/p' cmake/onnxruntime_release.cmake)
    latest=$(ort_latest 2>/dev/null | sed -n 's/^version: //p')
    printf '%-24s %-10s %-10s %s\n' onnxruntime "$pinned" "${latest:-?}" "$([ "$pinned" = "$latest" ] && echo 'up to date' || echo differs)"
}

case "${1:-list}" in
list) list ;;
bump)
    name=${2:?usage: scripts/update-deps.sh bump NAME}
    path="vendor/$name"
    [ -d "$path" ] || { echo "no submodule $path" >&2; exit 1; }
    git submodule update --init --depth 1 "$path" >/dev/null 2>&1
    git -C "$path" fetch --depth 1 origin HEAD && git -C "$path" checkout FETCH_HEAD
    echo "$path is now at $(git -C "$path" rev-parse --short HEAD) (not committed)."
    echo "Next: scripts/test.sh, then scripts/package.sh; commit the pointer only if they pass."
    ;;
ort) ort_latest ;;
*) echo "usage: scripts/update-deps.sh [list | bump NAME | ort]" >&2; exit 2 ;;
esac
