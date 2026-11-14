#!/bin/sh
# First-time setup of agent.cpp on Linux. `agent setup` downloads and runs this file from the repository (scripts/setup.sh), and the
# installer runs it at the end, so setup can be improved without a new release. It only uses commands the installed `agent` has.
#
#   agent setup [--yes] [--runtimes | --no-runtimes] [--model LINK | --no-model] [--home DIR]
#
# What it does: asks where agent.cpp keeps its files (when this agent can move it), offers the optional private runtimes (JavaScript tooling of
# your choice: bun, node with pnpm or nub, and uv, for MCP servers and skills; installed ones are checked and updated), and downloads a model from a Hugging Face link or file. The GPU backend and the engines are not asked: they come with
# the build that was installed. Without a terminal, or with --yes, nothing is asked.
set -eu

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    T="$(printf '\033[38;2;64;224;208m')"; P="$(printf '\033[38;2;255;179;209m')"; G="$(printf '\033[2m')"
    B="$(printf '\033[1m')"; R="$(printf '\033[0m')"
else
    T=""; P=""; G=""; B=""; R=""
fi
step() { printf '  %s✓%s %s\n' "$T" "$R" "$*"; }
info() { printf '  %s·%s %s%s%s\n' "$G" "$R" "$G" "$*" "$R"; }
die() { printf '\n  %s✗ %s%s\n\n' "$P" "$*" "$R" >&2; exit 1; }

yes=""; runtimes=""; model=""; no_model=""; home=""
while [ $# -gt 0 ]; do
    case "$1" in
        --yes | -y) yes=1 ;;
        --runtimes | --install-runtimes) runtimes=y ;;
        --no-runtimes) runtimes=n ;;
        --model) shift; model="${1:-}" ;;
        --no-model | --no-models) no_model=1 ;;
        --home) shift; home="${1:-}" ;;
        --reconfigure | --backend | --caps | --js-runtime) [ "$1" = --reconfigure ] || shift ;; # accepted for the old command line, ignored
        *) die "unknown option $1" ;;
    esac
    shift
done

agent="${AGENT_BIN:-}"
[ -n "$agent" ] || agent="$(command -v agent 2>/dev/null || true)"
[ -n "$agent" ] || { [ -x "$HOME/.local/bin/agent" ] && agent="$HOME/.local/bin/agent"; }
[ -n "$agent" ] || die "agent is not installed. Install it first: curl -fsSL https://agent.qozu.fyi/install.sh | sh"

ask=""
if [ -z "$yes" ] && [ -r /dev/tty ] && [ -t 1 ]; then ask=1; fi
prompt() { # question, default -> answer in $answer
    printf '  %s ' "$1" >/dev/tty
    read -r answer </dev/tty || answer=""
    [ -n "$answer" ] || answer="${2:-}"
}

printf '\n  %s%sagent.cpp%s  %ssetup%s\n  %s────────────────────────────%s\n\n' "$B" "$T" "$R" "$G" "$R" "$G" "$R"
step "$("$agent" version 2>/dev/null | head -n 1)"

# 1. Where the files live. The folder is created and remembered even when the default is kept: the runtimes and models need it.
if "$agent" home >/dev/null 2>&1; then
    current="$("$agent" home 2>/dev/null | head -n 1)"
    chosen="$current"
    if [ -n "$home" ]; then chosen="$home"
    elif [ -n "$ask" ]; then
        prompt "Where should agent.cpp keep its files (models, engines, skills, memory)? [$current]" "$current"
        chosen="$answer"
    fi
    "$agent" home "$chosen" >/dev/null 2>&1 && step "Files go to $chosen" || die "Cannot use $chosen as the files folder."
fi

# 2. The optional private runtimes: JavaScript tooling for MCP servers and skills (bun, node with pnpm, or nub) and uv for Python ones.
# What is installed is checked first; an older one is updated, a current one is left alone.
latest_of() { # the newest published version of a runtime, or nothing
    case "$1" in
        bun) curl -fsSL https://api.github.com/repos/oven-sh/bun/releases/latest 2>/dev/null | sed -n 's/.*"tag_name": *"bun-v\{0,1\}\([^"]*\)".*/\1/p' | head -n 1 ;;
        uv) curl -fsSL https://api.github.com/repos/astral-sh/uv/releases/latest 2>/dev/null | sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' | head -n 1 ;;
        nub) curl -fsSL https://api.github.com/repos/nubjs/nub/releases/latest 2>/dev/null | sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' | head -n 1 ;;
        pnpm) curl -fsSL https://registry.npmjs.org/pnpm/latest-10 2>/dev/null | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' | head -n 1 ;; # 11 and newer have another layout
        node) curl -fsSL https://nodejs.org/dist/latest-v24.x/SHASUMS256.txt 2>/dev/null | sed -n 's/.* node-v\([0-9.]*\)-.*/\1/p' | head -n 1 ;;
    esac
}
installed_of() { # the version of an installed runtime, or nothing
    rpath="$("$agent" runtime path "$1" 2>/dev/null)" || return 0
    if [ "$1" = pnpm ]; then # a script run by node: its version is in the package.json two folders up
        sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' "$(dirname "$(dirname "$rpath")")/package.json" 2>/dev/null | head -n 1
        return 0
    fi
    "$rpath" --version 2>/dev/null | head -n 1 | sed -n 's/^[^0-9]*\([0-9][0-9.]*\).*/\1/p'
}
ensure_runtime() {
    name="$1"
    have="$(installed_of "$name")"
    newest="$(latest_of "$name")"
    pin=""; [ "$name" = pnpm ] && [ -n "$newest" ] && pin="--version $newest"
    if [ -z "$have" ]; then
        printf '  %s↓%s Downloading %s\n' "$T" "$R" "$name"
        # shellcheck disable=SC2086
        if err="$("$agent" runtime install "$name" $pin 2>&1)"; then step "$name installed"; else info "$name could not be installed: $(printf '%s' "$err" | tail -n 1)"; fi
        return
    fi
    if [ -n "$newest" ] && [ "$newest" != "$have" ]; then
        printf '  %s↓%s Updating %s %s to %s\n' "$T" "$R" "$name" "$have" "$newest"
        # shellcheck disable=SC2086
        if err="$("$agent" runtime install "$name" $pin 2>&1)"; then step "$name updated to $newest"; else info "$name could not be updated: $(printf '%s' "$err" | tail -n 1)"; fi
    else
        step "$name $have is installed and up to date"
    fi
}
if [ -z "$runtimes" ] && [ -n "$ask" ]; then
    prompt "Install the optional runtimes for MCP servers and skills? [Y/n]" "y"
    case "$answer" in n | N | no | NO) runtimes=n ;; *) runtimes=y ;; esac
fi
[ -n "$runtimes" ] || runtimes=y # --yes and no terminal: the same default
if [ "$runtimes" = y ]; then
    js=bun
    if [ -n "$ask" ]; then
        printf '  JavaScript tooling:  1) bun   2) node with pnpm   3) nub\n' >/dev/tty
        prompt "Choose [1]:" "1"
        case "$answer" in 2 | node) js=node ;; 3 | nub) js=nub ;; *) js=bun ;; esac
    fi
    "$agent" runtime use "$js" >/dev/null 2>&1 || true
    case "$js" in node) ensure_runtime node; ensure_runtime pnpm ;; *) ensure_runtime "$js" ;; esac
    ensure_runtime uv
else
    info "Runtimes skipped: agent runtime use bun | node | nub and agent runtime install uv add them any time"
fi

# 3. A model: a link or a file, not a list.
if [ -z "$model" ] && [ -z "$no_model" ] && [ -n "$ask" ]; then
    prompt "Model to download now: a Hugging Face link, owner/repo, or a file on this computer (Enter to skip):" ""
    model="$answer"
fi
# A path dragged into the terminal comes with quotes: they are not part of the name.
model="$(printf '%s' "$model" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^[\"']//" -e "s/[\"']\$//")"
if [ -n "$model" ]; then
    printf '  %s↓%s Adding %s\n' "$T" "$R" "$model"
    "$agent" model pull "$model" || die "The model could not be added."
fi

have_model=""
listed="$("$agent" model ls 2>/dev/null || true)"
case "$listed" in "" | "no models"*) ;; *) have_model=1 ;; esac
if [ -n "$have_model" ] && [ -n "$ask" ]; then
    printf '\n  Done. Starting the chat ...\n\n'
    exec "$agent" </dev/tty
fi
if [ -n "$have_model" ]; then
    printf '\n  Done. Run %sagent%s to start the chat.\n\n' "$T" "$R"
else
    printf '\n  Done. Add a model with %sagent model pull <Hugging Face link or file>%s, then run %sagent%s.\n\n' "$T" "$R" "$T" "$R"
fi
