#!/bin/sh
# Installs the agent CLI on Linux:  curl -fsSL https://agent.qozu.fyi | sh
#
# Picks the release archive that fits this machine (x86_64 or aarch64; the Vulkan build when a Vulkan driver is present), checks its
# SHA-256 against the release's SHA256SUMS, unpacks it to ~/.local/share/agent-cpp and links `agent` into ~/.local/bin.
# Environment: AGENT_VERSION (a tag such as v0.2.0; default: the latest release), AGENT_VARIANT (cpu or vulkan; default: detected),
#              AGENT_PREFIX (default ~/.local), AGENT_REPO (default amitverma-cf/agent.cpp).
set -eu

REPO="${AGENT_REPO:-amitverma-cf/agent.cpp}"
PREFIX="${AGENT_PREFIX:-$HOME/.local}"
DEST="$PREFIX/share/agent-cpp"
BIN="$PREFIX/bin"

say() { printf '%s\n' "$*"; }
die() { printf 'install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "this installer is for Linux (for Windows: irm https://agent.qozu.fyi | iex)"
case "$(uname -m)" in
    x86_64 | amd64) arch=x86_64 ;;
    aarch64 | arm64) arch=aarch64 ;;
    *) die "unsupported CPU: $(uname -m) (x86_64 and aarch64 are released)" ;;
esac

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q "$1" -O "$2"; }
else
    die "curl or wget is needed"
fi
if command -v sha256sum >/dev/null 2>&1; then
    sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    die "sha256sum or shasum is needed"
fi

variant="${AGENT_VARIANT:-}"
if [ -z "$variant" ]; then
    variant=cpu
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so'; then variant=vulkan; fi
fi
case "$variant" in cpu) suffix="" ;; vulkan) suffix="-vulkan" ;; *) die "AGENT_VARIANT must be cpu or vulkan" ;; esac

if [ -n "${AGENT_VERSION:-}" ]; then base="https://github.com/$REPO/releases/download/$AGENT_VERSION"; else base="https://github.com/$REPO/releases/latest/download"; fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM
fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" || die "cannot read $base/SHA256SUMS (is there a release?)"

pick() { grep -E " \*?agent-cpp-[^ ]*-linux-$arch$1\.tar\.gz\$" "$tmp/SHA256SUMS" | head -n 1; }
line="$(pick "$suffix")"
if [ -z "$line" ] && [ -n "$suffix" ]; then
    say "no Vulkan build for $arch in this release; using the CPU build"
    suffix=""
    line="$(pick "")"
fi
[ -n "$line" ] || die "this release has no Linux $arch archive"
want="$(printf '%s' "$line" | cut -d' ' -f1)"
name="$(printf '%s' "$line" | sed 's/^[^ ]* *\*\{0,1\}//')"

say "downloading $name"
fetch "$base/$name" "$tmp/$name" || die "download failed"
[ "$(sha "$tmp/$name")" = "$want" ] || die "SHA-256 mismatch for $name; nothing was installed"

rm -rf "$DEST.new"
mkdir -p "$DEST.new" "$BIN"
tar -xzf "$tmp/$name" -C "$DEST.new" --strip-components=1
rm -rf "$DEST"
mv "$DEST.new" "$DEST"
ln -sf "$DEST/agent" "$BIN/agent"

say "installed $name to $DEST"
case ":$PATH:" in *":$BIN:"*) ;; *) say "add $BIN to your PATH:  export PATH=\"$BIN:\$PATH\"" ;; esac
say "next: agent setup"
