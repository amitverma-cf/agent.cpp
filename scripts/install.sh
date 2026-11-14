#!/bin/sh
# Installs the agent CLI on Linux:  curl -fsSL https://agent.qozu.fyi/install.sh | sh
#
# Picks the release archive that fits this machine (x86_64 or aarch64; the Vulkan build when a Vulkan driver is present), checks its
# SHA-256 against the release's SHA256SUMS, unpacks it to ~/.local/share/agent-cpp and links `agent` into ~/.local/bin.
# It then runs the setup script (scripts/setup.sh) and opens the chat (AGENT_NO_START=1 only installs). Run again, it updates: the same
# version is left alone, a newer one replaces it in place (`agent update` does exactly this).
# Environment: AGENT_VERSION (a tag such as v1.0.0; default: the latest release), AGENT_VARIANT (cpu or vulkan; default: detected),
#              AGENT_PREFIX (default ~/.local), AGENT_REPO (default amitverma-cf/agent.cpp), NO_COLOR.
set -eu

# `agent update` starts this script and exits; on a system that locks running programs the old process must be gone first.
if [ -n "${AGENT_WAIT_PID:-}" ]; then
    n=0
    while kill -0 "$AGENT_WAIT_PID" 2>/dev/null && [ "$n" -lt 100 ]; do sleep 0.1 2>/dev/null || sleep 1; n=$((n + 1)); done
fi

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
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so' && ls /dev/dri/renderD* >/dev/null 2>&1; then variant=vulkan; detected=" (Vulkan driver found)"; fi
fi
case "$variant" in cpu) suffix="" ;; vulkan) suffix="-vulkan" ;; *) die "AGENT_VARIANT must be cpu or vulkan." ;; esac

if [ -n "${AGENT_VERSION:-}" ]; then base="https://github.com/$REPO/releases/download/$AGENT_VERSION"; else base="https://github.com/$REPO/releases/latest/download"; fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM
fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" || die "Cannot read the release ($base). Is there a release yet?"

pick() { grep -E " \*?agent-cpp-[^ ]*-linux-$arch$1\.tar\.gz\$" "$tmp/SHA256SUMS" | head -n 1; }
line="$(pick "$suffix")"
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

if [ -n "${AGENT_CHECK:-}" ]; then # agent update --check: only look
    if [ -r "$DEST/.version" ] && [ "$(cat "$DEST/.version" 2>/dev/null)" = "$name" ]; then step "Up to date ($version)"; else step "An update is available: $version"; fi
    exit 0
fi

if [ -z "${AGENT_FORCE:-}" ] && [ -r "$DEST/.version" ] && [ "$(cat "$DEST/.version" 2>/dev/null)" = "$name" ] && [ -x "$BIN/agent" ]; then
    step "Already installed and up to date"
    installed_now=""
else
    installed_now=1
fi

if [ -n "$installed_now" ]; then
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
printf '%s' "$name" > "$DEST/.version"
step "Command: $BIN/agent"
fi
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) info "Add $BIN to your PATH to type just 'agent' in new shells:" ; info "  export PATH=\"$BIN:\$PATH\"" ;;
esac

# Setup is its own script (scripts/setup.sh), fetched from the repository so it can be improved without a new release. It runs here, through
# the terminal (the installer itself comes through a pipe). `agent update` sets AGENT_NO_START: an update does not set anything up again.
SCRIPTS="${AGENT_SCRIPTS:-https://raw.githubusercontent.com/amitverma-cf/agent.cpp/main/scripts}"
if [ -z "${AGENT_NO_START:-}" ] && [ -t 1 ] && [ -r /dev/tty ]; then
    if fetch "$SCRIPTS/setup.sh" "$tmp/setup.sh"; then
        AGENT_BIN="$BIN/agent" sh "$tmp/setup.sh" </dev/tty
        exit $?
    fi
    info "Could not fetch the setup script; run agent setup later"
fi
[ -n "$installed_now" ] && printf '\n  Next: %sagent setup%s\n\n' "$T" "$R"
exit 0
