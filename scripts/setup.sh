#!/bin/sh
# First-time setup of agent.cpp on Linux. `agent setup` downloads and runs this file from the repository (scripts/setup.sh), and the
# installer runs it at the end, so setup can be improved without a new release. It only uses commands the installed `agent` has.
#
#   agent setup [--yes] [--runtimes | --no-runtimes] [--model LINK | --no-model] [--home DIR]
#
# What it does: asks where agent.cpp keeps its files (when this agent can move it), offers the optional private runtimes (bun and uv, for MCP
# servers and skills), and downloads a model from a Hugging Face link or file. The GPU backend and the engines are not asked: they come with
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

# 1. Where the files live. Only an agent that knows `agent home` can move it.
if "$agent" home >/dev/null 2>&1; then
    current="$("$agent" home 2>/dev/null | head -n 1)"
    if [ -n "$home" ]; then
        "$agent" home "$home" >/dev/null && step "Files go to $home"
    elif [ -n "$ask" ]; then
        prompt "Where should agent.cpp keep its files (models, engines, skills, memory)? [$current]" "$current"
        [ "$answer" = "$current" ] || { "$agent" home "$answer" >/dev/null && step "Files go to $answer"; }
        [ "$answer" != "$current" ] || step "Files go to $current"
    else
        step "Files go to $current"
    fi
fi

# 2. The optional private runtimes.
if [ -z "$runtimes" ] && [ -n "$ask" ]; then
    prompt "Download the optional bun and uv runtimes, for MCP servers and skills? [y/N]" "n"
    case "$answer" in y | Y | yes | YES) runtimes=y ;; *) runtimes=n ;; esac
fi
if [ "$runtimes" = y ]; then
    for r in bun uv; do
        printf '  %s↓%s Downloading %s\n' "$T" "$R" "$r"
        "$agent" runtime install "$r" >/dev/null 2>&1 && step "$r installed" || info "$r could not be installed (agent runtime install $r)"
    done
else
    info "Runtimes skipped: agent runtime install bun | uv adds them any time"
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
