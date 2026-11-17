#!/bin/sh
# Removes agent.cpp from Linux and macOS. `agent uninstall` runs this file after it has asked; it is given what to remove through the
# environment:
#   AGENT_UNINSTALL_HOME      the folder of the data (models, engines, conversations, ...); empty = leave the data
#   AGENT_UNINSTALL_PROGRAM   the folder of the program (the one on PATH)
#   AGENT_UNINSTALL_POINTER   the file that names the home (home.json), when it is outside both folders
# The program folder, the data folder, the pointer file, and the PATH line the installer added to the shell profiles are removed.
set -u

step() { printf '  \033[36m✓\033[0m %s\n' "$1"; }
info() { printf '  \033[90m· %s\033[0m\n' "$1"; }

remove_folder() {
    path="$1"
    [ -n "$path" ] && [ -e "$path" ] || return 0
    # A path with fewer than two parts is the root or a top-level folder; the user's home folder is never removed either.
    case "$path" in "/" | "$HOME" | "$HOME/" | /*/) info "Not removing $path"; return 0 ;; esac
    [ "$(printf '%s' "$path" | tr -cd '/' | wc -c)" -ge 2 ] || { info "Not removing $path"; return 0; }
    rm -rf "$path" 2>/dev/null
    if [ -e "$path" ]; then info "Some files in $path could not be removed"; else step "Removed $path"; fi
}

printf '\n  \033[36magent.cpp  uninstall\033[0m\n\n'
program="${AGENT_UNINSTALL_PROGRAM:-}"
data="${AGENT_UNINSTALL_HOME:-}"
remove_folder "$data"
case "$program" in "$data"/*) ;; *) remove_folder "$program" ;; esac
if [ -n "${AGENT_UNINSTALL_POINTER:-}" ] && [ -f "$AGENT_UNINSTALL_POINTER" ]; then
    rm -f "$AGENT_UNINSTALL_POINTER" && step "Removed $AGENT_UNINSTALL_POINTER"
fi
# The installer added one line to the shell profiles; the line is recognised by its comment.
for profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$profile" ] && grep -q '# agent.cpp$' "$profile" 2>/dev/null; then
        tmp="$profile.agent-uninstall.$$"
        grep -v '# agent.cpp$' "$profile" > "$tmp" && cat "$tmp" > "$profile" && step "Removed the PATH line from $profile"
        rm -f "$tmp"
    fi
done
printf '\n  \033[36magent.cpp is removed.\033[0m\n\n'
