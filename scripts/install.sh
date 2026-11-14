#!/bin/sh
# Installs the agent CLI on Linux:  curl -fsSL https://agent.qozu.fyi/install.sh | sh
#
# Picks the release archive that fits this machine (x86_64 or aarch64; the Vulkan build when a Vulkan driver is present), checks its
# SHA-256 against the release's SHA256SUMS, unpacks it to ~/.local/share/agent-cpp and links `agent` into ~/.local/bin.
# It then runs `agent setup` and opens the chat (AGENT_NO_START=1 only installs).
# Environment: AGENT_VERSION (a tag such as v1.0.0; default: the latest release), AGENT_VARIANT (cpu, vulkan or cuda; default: detected),
#              AGENT_PREFIX (default ~/.local), AGENT_REPO (default amitverma-cf/agent.cpp), NO_COLOR.
set -eu

REPO="${AGENT_REPO:-amitverma-cf/agent.cpp}"
PREFIX="${AGENT_PREFIX:-$HOME/.local}"
DEST="$PREFIX/share/agent-cpp"
BIN="$PREFIX/bin"

# ---- output: turquoise accents, grey detail, pastel pink for problems; plain text when not a terminal or NO_COLOR is set
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    T="$(printf '\033[38;2;64;224;208m')"; P="$(printf '\033[38;2;255;179;209m')"; G="$(printf '\033[2m')"
    B="$(printf '\033[1m')"; R="$(printf '\033[0m')"
else
    T=""; P=""; G=""; B=""; R=""
fi
step() { printf '  %s✓%s %s\n' "$T" "$R" "$*"; }
info() { printf '  %s·%s %s%s%s\n' "$G" "$R" "$G" "$*" "$R"; }
die() { printf '\n  %s✗ %s%s\n\n' "$P" "$*" "$R" >&2; exit 1; }

printf '\n  %s%sagent.cpp%s  %sinstaller%s\n  %s────────────────────────────%s\n\n' "$B" "$T" "$R" "$G" "$R" "$G" "$R"

[ "$(uname -s)" = "Linux" ] || die "This installer is for Linux. On Windows use: irm https://agent.qozu.fyi/install.ps1 | iex"
case "$(uname -m)" in
    x86_64 | amd64) arch=x86_64 ;;
    aarch64 | arm64) arch=aarch64 ;;
    *) die "Unsupported CPU: $(uname -m). x86_64 and aarch64 are released." ;;
esac

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL "$1" -o "$2"; }
    fetch_progress() { if [ -t 1 ]; then curl -fL --progress-bar "$1" -o "$2"; else curl -fsSL "$1" -o "$2"; fi; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q "$1" -O "$2"; }
    fetch_progress() { if [ -t 1 ]; then wget -q --show-progress "$1" -O "$2"; else wget -q "$1" -O "$2"; fi; }
else
    die "curl or wget is needed."
fi
if command -v sha256sum >/dev/null 2>&1; then
    sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    die "sha256sum or shasum is needed."
fi

variant="${AGENT_VARIANT:-}"
detected=""
if [ -z "$variant" ]; then
    variant=cpu
    if [ "$arch" = x86_64 ] && { command -v nvidia-smi >/dev/null 2>&1 || [ -e /proc/driver/nvidia/version ]; }; then variant=cuda; detected=" (NVIDIA driver found)"
    elif command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so'; then variant=vulkan; detected=" (Vulkan driver found)"; fi
fi
case "$variant" in cpu) suffix="" ;; vulkan) suffix="-vulkan" ;; cuda) suffix="-cuda" ;; *) die "AGENT_VARIANT must be cpu, vulkan or cuda." ;; esac

if [ -n "${AGENT_VERSION:-}" ]; then base="https://github.com/$REPO/releases/download/$AGENT_VERSION"; else base="https://github.com/$REPO/releases/latest/download"; fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM
fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" || die "Cannot read the release ($base). Is there a release yet?"

pick() { grep -E " \*?agent-cpp-[^ ]*-linux-$arch$1\.tar\.gz\$" "$tmp/SHA256SUMS" | head -n 1; }
line="$(pick "$suffix")"
if [ -z "$line" ] && [ "$suffix" = "-cuda" ]; then
    info "No CUDA build for $arch in this release; trying Vulkan"
    variant=vulkan; suffix="-vulkan"; line="$(pick "-vulkan")"
fi
if [ -z "$line" ] && [ -n "$suffix" ]; then
    info "No GPU build for $arch in this release; using the CPU build"
    variant=cpu; suffix=""; detected=""
    line="$(pick "")"
fi
[ -n "$line" ] || die "This release has no Linux $arch archive."
want="$(printf '%s' "$line" | cut -d' ' -f1)"
name="$(printf '%s' "$line" | sed 's/^[^ ]* *\*\{0,1\}//')"
version="$(printf '%s' "$name" | sed 's/^agent-cpp-\(v\{0,1\}[0-9][^-]*\)-.*/\1/')"

step "Linux $arch  ·  $variant build$detected"
step "Release ${version}"

printf '  %s↓%s Downloading %s\n' "$T" "$R" "$name"
fetch_progress "$base/$name" "$tmp/$name" || die "The download failed."
[ "$(sha "$tmp/$name")" = "$want" ] || die "SHA-256 mismatch for $name. Nothing was installed."
step "Checksum verified (SHA-256)"

rm -rf "$DEST.new"
mkdir -p "$DEST.new" "$BIN"
tar -xzf "$tmp/$name" -C "$DEST.new" --strip-components=1
rm -rf "$DEST"
mv "$DEST.new" "$DEST"
ln -sf "$DEST/agent" "$BIN/agent"
step "Installed to $DEST"
step "Command: $BIN/agent"
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) info "Add $BIN to your PATH to type just 'agent' in new shells:" ; info "  export PATH=\"$BIN:\$PATH\"" ;;
esac

# Set up and start. The script comes through a pipe, so the terminal is read from /dev/tty.
if [ -z "${AGENT_NO_START:-}" ] && [ -t 1 ] && [ -r /dev/tty ]; then
    printf '\n  %sSetting up agent.cpp ...%s\n\n' "$B" "$R"
    "$BIN/agent" setup </dev/tty || exit $?
    exec "$BIN/agent" </dev/tty
fi
printf '\n  Next: %sagent setup%s\n\n' "$T" "$R"
