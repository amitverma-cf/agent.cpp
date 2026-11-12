#!/usr/bin/env bash
# Builds a Linux release tarball of the agent CLI and the chosen engine plugins:
#   agent-<version>-linux-<arch>[-<backend>].tar.gz
#     bin/agent
#     lib/agent/engines/agent-engine-*.so (+ bundled runtime libraries)
#     share/doc/agent/{README.md,LICENSE,cli.md,licenses/}
# The version comes from project(VERSION) in CMakeLists.txt. See docs/packaging.md.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: scripts/package.sh [options] [-- extra cmake configure args]

  --engines LIST         comma-separated engine plugins: llama,onnx,sdcpp,audiocpp,trellis (default: llama)
  --backend NAME         cpu | cuda | vulkan (default: cpu); selects the ggml backend of llama
  --build-dir DIR        CMake build directory (default: build-package)
  --out-dir DIR          where the tarball goes (default: dist)
  --jobs N               parallel build jobs (default: 2)
  --onnxruntime-root DIR use this ONNX Runtime instead of downloading the pinned upstream release
  --native               optimise ggml for this machine (-march=native; not portable).
                         Default: GGML_NATIVE=OFF, i.e. AVX2/FMA/F16C/BMI2 (x86-64-v3).
  -h, --help             this help
EOF
}

src_dir=$(cd "$(dirname "$0")/.." && pwd)
engines=llama
backend=cpu
build_dir=build-package
out_dir=dist
jobs=2
ort_root=""
native=OFF
extra_args=()

while [ $# -gt 0 ]; do
    case "$1" in
    --engines) engines=$2; shift 2 ;;
    --backend) backend=$2; shift 2 ;;
    --build-dir) build_dir=$2; shift 2 ;;
    --out-dir) out_dir=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --onnxruntime-root) ort_root=$2; shift 2 ;;
    --native) native=ON; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) echo "package.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "package.sh: $*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "only Linux packaging is implemented (Windows/macOS: see docs/packaging.md)"
case "$(uname -m)" in
x86_64 | amd64) arch=x64 ;;
aarch64 | arm64) arch=arm64 ;;
*) arch=$(uname -m) ;;
esac

version=$(sed -n 's/^project("agent.cpp" VERSION \([0-9][0-9.]*\).*/\1/p' "$src_dir/CMakeLists.txt")
[ -n "$version" ] || die "could not read project(VERSION) from CMakeLists.txt"

# Engine selection -> CMake options and build targets.
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DAGENT_BUILD_CLI=ON -DAGENT_BUILD_HUB=ON -DBUILD_SHARED_LIBS=OFF
    -DAGENT_ENGINE_LLAMA=OFF -DAGENT_ENGINE_ONNX=OFF -DAGENT_ENGINE_SDCPP=OFF -DAGENT_ENGINE_AUDIOCPP=OFF)
targets=(agent agent_store_sqlite agent_hub agent_cli_bin agent-engine-host)
IFS=, read -r -a engine_list <<<"$engines"
for e in "${engine_list[@]}"; do
    case "$e" in
    llama) cmake_args+=(-DAGENT_ENGINE_LLAMA=ON -DGGML_NATIVE="$native") ;;
    sdcpp) cmake_args+=(-DAGENT_ENGINE_SDCPP=ON -DAGENT_SDCPP_BUILD_JOBS="$jobs") ;;
    onnx)
        # without --onnxruntime-root the pinned upstream release is downloaded by CMake (cmake/onnxruntime_release.cmake)
        cmake_args+=(-DAGENT_ENGINE_ONNX=ON)
        [ -n "$ort_root" ] && cmake_args+=(-DONNXRUNTIME_ROOT="$ort_root")
        ;;
    audiocpp) cmake_args+=(-DAGENT_ENGINE_AUDIOCPP=ON -DAGENT_AUDIOCPP_BUILD_JOBS="$jobs") ;;
    "") ;;
    *) die "unknown engine '$e' (llama, onnx, sdcpp, audiocpp)" ;;
    esac
    [ -n "$e" ] && targets+=("agent-engine-$e")
done

suffix=""
case "$backend" in
cpu) ;;
cuda | vulkan)
    [[ ",$engines," == *,llama,* ]] || die "--backend $backend only applies to the llama engine"
    upper=$(echo "$backend" | tr '[:lower:]' '[:upper:]')
    cmake_args+=(-DGGML_"$upper"=ON)
    suffix=-$backend
    ;;
*) die "unknown backend '$backend' (cpu, cuda, vulkan)" ;;
esac

name="agent-$version-linux-$arch$suffix"
case "$build_dir" in /*) ;; *) build_dir="$src_dir/$build_dir" ;; esac
case "$out_dir" in /*) ;; *) out_dir="$src_dir/$out_dir" ;; esac
stage="$build_dir/stage"
pkg="$build_dir/pkg/$name"

echo "== $name (engines: $engines, backend: $backend)"
generator=()
if [ ! -f "$build_dir/CMakeCache.txt" ] && command -v ninja >/dev/null 2>&1; then generator=(-G Ninja); fi
cmake -S "$src_dir" -B "$build_dir" "${generator[@]}" "${cmake_args[@]}" "${extra_args[@]}"
cmake --build "$build_dir" --parallel "$jobs" --target "${targets[@]}"

rm -rf "$stage" "$build_dir/pkg"
cmake --install "$build_dir" --prefix "$stage" --strip

# The tarball is the end-user CLI: no headers, static libraries or CMake package files.
mkdir -p "$pkg/bin" "$pkg/lib/agent" "$pkg/share/doc"
cp -a "$stage/bin/agent" "$pkg/bin/"
# Engines run out of process by default, so the host program must ship next to the CLI.
cp -a "$stage/bin/agent-engine-host" "$pkg/bin/"
if [ -d "$stage/lib/agent/engines" ]; then cp -a "$stage/lib/agent/engines" "$pkg/lib/agent/"; fi
cp -a "$stage/share/doc/agent" "$pkg/share/doc/"
lic="$pkg/share/doc/agent/licenses"
mkdir -p "$lic"
copy_license() { if [ -f "$1" ]; then cp "$1" "$lic/$2.LICENSE"; fi; }
copy_license "$src_dir/vendor/simdjson/LICENSE" simdjson # compiled into the CLI and every plugin
copy_license "$src_dir/vendor/cpp-httplib/LICENSE" cpp-httplib # the CLI's HTTP client
copy_license "$src_dir/vendor/miniaudio/LICENSE" miniaudio     # sound devices for `agent voice`
[[ ",$engines," == *,llama,* ]] && copy_license "$src_dir/vendor/llama.cpp/LICENSE" llama.cpp
[[ ",$engines," == *,sdcpp,* ]] && copy_license "$src_dir/vendor/stable-diffusion.cpp/LICENSE" stable-diffusion.cpp
[[ ",$engines," == *,onnx,* ]] && copy_license "${ort_root:-$build_dir/_deps/agent_onnxruntime_release-src}/LICENSE" onnxruntime
[[ ",$engines," == *,audiocpp,* ]] && copy_license "$src_dir/vendor/audio.cpp/LICENSE" audio.cpp

# Sanity checks: the binary runs and every plugin resolves its libraries.
"$pkg/bin/agent" version
status=0
for so in "$pkg/bin/agent" "$pkg/bin/agent-engine-host" "$pkg"/lib/agent/engines/agent-engine-*; do
    [ -e "$so" ] || continue
    if ldd "$so" | grep -q "not found"; then
        echo "package.sh: unresolved libraries in $so:" >&2
        ldd "$so" | grep "not found" >&2
        status=1
    fi
done
[ $status -eq 0 ] || die "missing runtime libraries"

mkdir -p "$out_dir"
tarball="$out_dir/$name.tar.gz"
tar -C "$build_dir/pkg" --owner=0 --group=0 --numeric-owner -czf "$tarball" "$name"
(cd "$out_dir" && sha256sum "$name.tar.gz" >"$name.tar.gz.sha256")
echo "== wrote $tarball ($(du -h "$tarball" | cut -f1))"
tar -tzvf "$tarball"
