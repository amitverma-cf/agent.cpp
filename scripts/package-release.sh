#!/usr/bin/env bash
# Packs a build tree into the files of an agent.cpp release: the program (core) and one archive for each engine plugin. The installer scripts
# (scripts/install.sh, install.ps1) put them together on the user's machine, so a user adds only the engines they want and one engine can be
# replaced without a new release of the rest.
#
#   scripts/package-release.sh --build-dir build --out-dir dist --tag v1.2.0 --os linux --arch x86_64 [--variant vulkan] [--ext-lib so]
#
#   agent-cpp-core-<tag>-<os>-<arch>.<tar.gz|zip>               agent, agent-engine-host, the api engine, licences (and the OpenSSL DLLs on Windows)
#   agent-engine-<name>-<tag>-<os>-<arch>[-vulkan].<tar.gz|zip> the plugin and the libraries it loads from its own folder
#   SHA256SUMS                                                  one line for each file (added to when it exists)
#
# The ONNX Runtime library is not packed with the onnx plugin: the installer downloads the official release (see scripts/components.tsv).
set -euo pipefail

build_dir=build; out_dir=dist; tag=""; os_name=""; arch=""; variant=""; no_core=""; lib_ext=""; exe_ext=""; ssl_dir=""; ssl_suffix=""
while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) build_dir=$2; shift 2 ;;
        --out-dir) out_dir=$2; shift 2 ;;
        --tag) tag=$2; shift 2 ;;
        --os) os_name=$2; shift 2 ;;
        --arch) arch=$2; shift 2 ;;
        --variant) variant=$2; shift 2 ;;
        --no-core) no_core=1; shift ;;
        --lib-ext) lib_ext=$2; shift 2 ;;
        --exe-ext) exe_ext=$2; shift 2 ;;
        --openssl-dir) ssl_dir=$2; shift 2 ;;
        --openssl-suffix) ssl_suffix=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$tag" ] && [ -n "$os_name" ] && [ -n "$arch" ] || { echo "--tag, --os and --arch are required" >&2; exit 2; }
if [ -z "$lib_ext" ]; then case "$os_name" in windows) lib_ext=dll ;; macos) lib_ext=dylib ;; *) lib_ext=so ;; esac; fi
if [ -z "$exe_ext" ] && [ "$os_name" = windows ]; then exe_ext=.exe; fi
src_dir=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out_dir"
out_dir=$(cd "$out_dir" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

archive() { # folder-with-the-files archive-name (without extension)
    if [ "$os_name" = windows ]; then
        if command -v 7z >/dev/null 2>&1; then (cd "$1" && 7z a -tzip -bso0 "$out_dir/$2.zip" . >/dev/null)
        elif command -v zip >/dev/null 2>&1; then (cd "$1" && zip -qr "$out_dir/$2.zip" .)
        else (cd "$1" && python3 -c "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', '.')" "$out_dir/$2"); fi
        echo "$2.zip"
    else
        tar -C "$1" -czf "$out_dir/$2.tar.gz" .
        echo "$2.tar.gz"
    fi
}
sums() { # file names
    for f in "$@"; do
        if command -v sha256sum >/dev/null 2>&1; then (cd "$out_dir" && sha256sum "$f"); else (cd "$out_dir" && shasum -a 256 "$f"); fi
    done
}

files=()

# ---- core (the GPU build adds engines only: the program is the same)
if [ -z "$no_core" ]; then
core="$work/core"; mkdir -p "$core/licenses"
cp "$build_dir/agent$exe_ext" "$core/"
[ -f "$build_dir/agent-engine-host$exe_ext" ] && cp "$build_dir/agent-engine-host$exe_ext" "$core/"
for f in "$build_dir"/agent-engine-api.$lib_ext "$build_dir"/src/engines/api/agent-engine-api.$lib_ext; do [ -f "$f" ] && cp "$f" "$core/" && break; done
if [ "$os_name" = windows ] && [ -n "$ssl_dir" ]; then cp "$ssl_dir"/libssl-3$ssl_suffix.dll "$ssl_dir"/libcrypto-3$ssl_suffix.dll "$core/"; fi
cp "$src_dir/LICENSE" "$src_dir/README.md" "$core/"
for l in simdjson cpp-httplib miniaudio llama.cpp sqlite-vec audio.cpp stable-diffusion.cpp trellis2cpp RTXUI; do
    for f in "$src_dir/vendor/$l/LICENSE" "$src_dir/vendor/$l/LICENSE.md" "$src_dir/vendor/$l/LICENSE.txt"; do
        [ -f "$f" ] && cp "$f" "$core/licenses/$l.LICENSE" && break
    done
done
files+=("$(archive "$core" "agent-cpp-core-$tag-$os_name-$arch")")
fi

# ---- one archive for each engine plugin
suffix=""; [ "$variant" = vulkan ] && suffix="-vulkan"
for plugin in "$build_dir"/src/engines/*/agent-engine-*.$lib_ext; do
    [ -f "$plugin" ] || continue
    base=$(basename "$plugin" ".$lib_ext")
    name=${base#agent-engine-}
    case "$name" in mock* | api) continue ;; esac # the test engines are not released; the api engine is part of the core
    stage="$work/engine-$name"; rm -rf "$stage"; mkdir -p "$stage"
    cp "$plugin" "$stage/"
    # Libraries the plugin loads from its own folder (libaudiocpp, ...). ONNX Runtime comes from upstream, not from here.
    for l in "$(dirname "$plugin")"/*.$lib_ext*; do
        [ -f "$l" ] || continue
        case "$(basename "$l")" in agent-engine-* | *onnxruntime*) continue ;; esac
        cp "$l" "$stage/"
    done
    files+=("$(archive "$stage" "agent-engine-$name-$tag-$os_name-$arch$suffix")")
done

: >>"$out_dir/SHA256SUMS"
for f in "${files[@]}"; do grep -v " \*\?$f\$" "$out_dir/SHA256SUMS" >"$work/s" || true; cat "$work/s" >"$out_dir/SHA256SUMS"; done
sums "${files[@]}" >>"$out_dir/SHA256SUMS"
printf '%s\n' "${files[@]}"
