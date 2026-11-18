#!/bin/sh
# agent.cpp on Linux and macOS: installs, sets up, updates, extends and removes it. This one script does all of it; the program only
# hands over to it (agent setup, agent update, agent uninstall, agent engine add ...).
#
#   curl -fsSL https://agent.qozu.fyi/install.sh | sh                 install (asks where, once), then set up
#   curl -fsSL https://agent.qozu.fyi/install.sh | sh -s -- add llama
#
#   install.sh [install]            a new install: the folder, the program, then setup
#   install.sh setup                engines, runtimes and a model; can be run again at any time
#   install.sh add NAME[,NAME]      add engines or runtimes (llama, sdcpp, audiocpp, onnx, trellis, bun, uv, node, pnpm, nub, ...)
#   install.sh rm NAME[,NAME]       remove them
#   install.sh ls                   what exists for this machine and what is installed
#   install.sh update [--check]     move to the newest agent.cpp release and the components it pins
#   install.sh uninstall            remove everything (--keep-data keeps models, notes and settings)
#
# One folder holds everything: bin/ (the program), engines/, runtimes/, models/, workspaces/, scripts/ (this file and components.tsv, of the
# installed release), cache/. Its path is the environment variable AGENT_HOME, which is also what the program reads; <folder>/bin is put on
# PATH. Nothing else is written outside the folder, except those lines in the shell profiles.
#
# Pinned: a release of agent.cpp lists the exact version of every component it works with (components.tsv). add, setup and rm use the list
# of the installed release; only update reads a newer one. Environment: AGENT_HOME, AGENT_VERSION (a tag; install and update), AGENT_BACKEND
# (cpu, vulkan, cuda12, cuda13, rocm, metal), AGENT_YES, AGENT_REPO, AGENT_SCRIPTS and AGENT_RELEASE (an address or folder to read
# instead of GitHub), AGENT_NO_ENV (do not touch the shell profiles), NO_COLOR.
set -eu

TAB="$(printf '\t')"
REPO="${AGENT_REPO:-amitverma-cf/agent.cpp}"

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

# ---- the command and its options
cmd="${AGENT_CMD:-}"
if [ -z "$cmd" ]; then
    cmd=install
    if [ $# -gt 0 ]; then cmd="$1"; shift; fi
fi
yes="${AGENT_YES:-}"; backend="${AGENT_BACKEND:-}"; engines=""; runtimes=""; model=""; no_model=""; no_engines=""; no_runtimes=""
check=""; keep_data=""; names=""; all=""; version="${AGENT_VERSION:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --yes | -y) yes=1 ;;
        --backend) shift; backend="${1:-}" ;;
        --engines) shift; engines="${1:-}" ;;
        --runtimes) shift; runtimes="${1:-}" ;;
        --all) all=1 ;;
        --no-engines) no_engines=1 ;;
        --no-runtimes) no_runtimes=1 ;;
        --model) shift; model="${1:-}" ;;
        --no-model | --no-models) no_model=1 ;;
        --check) check=1 ;;
        --keep-data) keep_data=1 ;;
        --version | --tag) shift; version="${1:-}" ;;
        --home) shift; AGENT_HOME="${1:-}" ;;
        -*) die "unknown option $1" ;;
        *) names="$names,$1" ;;
    esac
    shift
done
names="$(printf '%s' "$names" | sed 's/^,//; s/ /,/g')"

# `agent update` and `agent uninstall` start this script and exit; wait until that process is gone.
if [ -n "${AGENT_WAIT_PID:-}" ]; then
    n=0
    while kill -0 "$AGENT_WAIT_PID" 2>/dev/null && [ "$n" -lt 100 ]; do sleep 0.1 2>/dev/null || sleep 1; n=$((n + 1)); done
fi

ask=""
if [ -z "$yes" ] && [ -r /dev/tty ] && [ -t 1 ]; then ask=1; fi
prompt() { # question, default -> $answer
    printf '  %s ' "$1" >/dev/tty
    read -r answer </dev/tty || answer=""
    [ -n "$answer" ] || answer="${2:-}"
}
confirm() { # question; true when the user says yes (or --yes)
    [ -z "$yes" ] || return 0
    [ -n "$ask" ] || return 1
    prompt "$1 [y/N]" "n"
    case "$answer" in y | Y | yes | YES) return 0 ;; *) return 1 ;; esac
}

# ---- tools
if command -v curl >/dev/null 2>&1; then
    http_get() { curl -fsSL "$1" -o "$2"; }
    http_get_progress() { if [ -t 1 ]; then curl -fL --progress-bar "$1" -o "$2"; else curl -fsSL "$1" -o "$2"; fi; }
elif command -v wget >/dev/null 2>&1; then
    http_get() { wget -q "$1" -O "$2"; }
    http_get_progress() { if [ -t 1 ]; then wget -q --show-progress "$1" -O "$2"; else wget -q "$1" -O "$2"; fi; }
else
    die "curl or wget is needed."
fi
if command -v sha256sum >/dev/null 2>&1; then sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else die "sha256sum or shasum is needed."; fi

# A source is an address or, for tests and offline use, a folder.
get() { # source, destination
    case "$1" in
        http://* | https://*) http_get "$1" "$2" ;;
        file://*) cp "${1#file://}" "$2" ;;
        *) cp "$1" "$2" ;;
    esac
}
get_progress() {
    case "$1" in http://* | https://*) http_get_progress "$1" "$2" ;; *) get "$1" "$2" ;; esac
}

# ---- this machine
case "$(uname -s)" in
    Linux) OS=linux ;;
    Darwin) OS=macos ;;
    *) die "This script is for Linux and macOS. On Windows use: irm https://agent.qozu.fyi/install.ps1 | iex" ;;
esac
case "$(uname -m)" in
    x86_64 | amd64) ARCH=x86_64 ;;
    aarch64 | arm64) ARCH=aarch64 ;;
    *) die "Unsupported CPU: $(uname -m). x86_64 and aarch64 are supported." ;;
esac
LIBC=gnu; GLIBC=""
if [ "$OS" = linux ]; then
    if { ldd --version 2>&1 || true; } | grep -qi musl; then LIBC=musl
    else GLIBC="$({ ldd --version 2>&1 || true; } | head -n 1 | sed -n 's/.* \([0-9][0-9]*\.[0-9][0-9]*\)[^0-9.]*$/\1/p')"; fi
fi

# The backends this machine can run, best first.
detect_backends() {
    list=""
    if [ "$OS" = macos ]; then [ "$ARCH" = aarch64 ] && list="metal"; echo "$list cpu" | sed 's/^ //'; return; fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -n 1)"
        if [ -n "$cuda" ] && [ "$cuda" -ge 13 ]; then list="cuda13 cuda12"; else list="cuda12"; fi
    fi
    if [ -e /dev/kfd ] || [ -d /opt/rocm ]; then list="$list rocm"; fi
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so' && ls /dev/dri/renderD* >/dev/null 2>&1; then
        list="$list vulkan"
    fi
    echo "$list cpu" | sed 's/^ *//'
}
if [ -n "$backend" ]; then BACKENDS="$backend cpu"; else BACKENDS="$(detect_backends)"; fi

# ---- the folder
resolve_root() {
    ROOT="${AGENT_HOME:-}"
    if [ -z "$ROOT" ] && [ "$cmd" = install ]; then
        def="$HOME/.agent.cpp"
        if [ -n "$ask" ]; then
            printf '\n  %sWhere should agent.cpp keep everything?%s\n' "$B" "$R" >/dev/tty
            prompt "Folder [$def]:" "$def"
            ROOT="$answer"
        else
            ROOT="$def"
        fi
    fi
    [ -n "$ROOT" ] || die "AGENT_HOME is not set: agent.cpp is not installed here. Install it first: curl -fsSL https://agent.qozu.fyi/install.sh | sh"
    case "$ROOT" in "~") ROOT="$HOME" ;; "~/"*) ROOT="$HOME/${ROOT#\~/}" ;; esac
    case "$ROOT" in /*) ;; *) die "The folder must be a full path starting with /, not '$ROOT'." ;; esac
    ROOT="$(printf '%s' "$ROOT" | sed 's:/*$::; s:/\./:/:g')"
    [ -n "$ROOT" ] && [ "$ROOT" != "$HOME" ] || die "Choose a folder of its own, not your home folder."
    if [ -e "$ROOT" ] && [ ! -d "$ROOT" ]; then die "$ROOT is a file."; fi
    if [ "$cmd" = install ] && [ -d "$ROOT" ] && [ ! -e "$ROOT/.agent-root" ] && [ -n "$(ls -A "$ROOT" 2>/dev/null)" ]; then
        die "$ROOT already has other files in it. Choose an empty folder or a new one."
    fi
    mkdir -p "$ROOT" 2>/dev/null || die "Cannot create $ROOT."
    ROOT="$(cd "$ROOT" && pwd)"
    BIN="$ROOT/bin"
    export AGENT_HOME="$ROOT"
}

# The folder is remembered as the environment variable AGENT_HOME (and <folder>/bin on PATH), in the lines of the shell profiles that carry
# the comment "# agent.cpp". Run again, the lines are replaced.
profiles() {
    for f in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.zshenv"; do
        if [ -f "$f" ]; then echo "$f"; fi
    done
}
forget_env() {
    [ -z "${AGENT_NO_ENV:-}" ] || return 0
    for f in $(profiles); do
        if grep -q '# agent.cpp$' "$f" 2>/dev/null; then
            t="$f.agent-tmp.$$"
            grep -v '# agent.cpp$' "$f" >"$t" && cat "$t" >"$f"
            rm -f "$t"
        fi
    done
}
remember_env() {
    [ -z "${AGENT_NO_ENV:-}" ] || { info "AGENT_NO_ENV is set: the shell profiles are left alone"; return 0; }
    forget_env
    list="$(profiles)"
    if [ -z "$list" ]; then : >"$HOME/.profile"; list="$HOME/.profile"; fi
    for f in $list; do
        printf '\nexport AGENT_HOME="%s"  # agent.cpp\nexport PATH="$AGENT_HOME/bin:$PATH"  # agent.cpp\n' "$ROOT" >>"$f"
    done
    step "AGENT_HOME is $ROOT, and $ROOT/bin is on PATH (open a new terminal)"
}

# ---- the release and the list of components
release_base() { # the tag whose assets and scripts to use
    if [ -n "${AGENT_RELEASE:-}" ]; then REL="$AGENT_RELEASE"; return; fi
    if [ -n "$TAG" ]; then REL="https://github.com/$REPO/releases/download/$TAG"; else REL="https://github.com/$REPO/releases/latest/download"; fi
}
latest_tag() {
    t="$(mktemp)"
    http_get "https://api.github.com/repos/$REPO/releases/latest" "$t" 2>/dev/null || { rm -f "$t"; return 1; }
    sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' "$t" | head -n 1
    rm -f "$t"
}
TMP=""
cleanup() { if [ -n "$TMP" ]; then rm -rf "$TMP"; fi; }
trap cleanup EXIT INT TERM
make_tmp() {
    mkdir -p "$ROOT/cache/tmp"
    TMP="$(mktemp -d "$ROOT/cache/tmp/install.XXXXXX")"
}

# The components of the installed release (scripts/components.tsv), or, before anything is installed, of the release to be installed.
# The first line "# tag vX.Y.Z" names the release. Columns (tab separated):
#   kind  name  os  arch  backend  version  sha256  strip  dest  url
# sha256 is "@sums" for a file of the agent.cpp release itself (its SHA256SUMS holds the sum); @TAG@ and @RELEASE@ in a url are the tag and
# the release address.
LIST=""
load_list() {
    if [ -r "$ROOT/scripts/components.tsv" ]; then LIST="$ROOT/scripts/components.tsv"; return; fi
    [ -n "$TMP" ] || make_tmp
    LIST="$TMP/components.tsv"
    base="${AGENT_SCRIPTS:-https://raw.githubusercontent.com/$REPO/${TAG:-main}/scripts}"
    get "$base/components.tsv" "$LIST" || die "Cannot read the list of components from $base."
}
lst() { tr -d '\r' <"$LIST"; } # the list as plain lines, whatever its line endings
list_tag() { lst | sed -n 's/^# tag \(.*\)$/\1/p' | head -n 1; }

rows_for() { # kind name -> the rows for this machine
    lst | awk -F'\t' -v k="$1" -v n="$2" -v o="$OS" -v a="$ARCH" '$0 !~ /^#/ && $1==k && $2==n && $3==o && $4==a' 
}
# A row may carry a last column "glibc>=2.43": a build made on a newer system, which loads only where the C library is at least that.
version_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]; }
backend_runs() { # kind name backend -> true when every row of that backend can run here
    [ -z "${FETCHING:-}" ] || return 0 # fetch is for compiling against the files, not for running them
    need="$(rows_for "$1" "$2" | awk -F'\t' -v b="$3" '($5==b || $5=="any") && $11 ~ /^glibc>=/ {sub(/^glibc>=/, "", $11); print $11}' | sort -V | tail -n 1)"
    [ -z "$need" ] || [ -z "$GLIBC" ] || version_ge "$GLIBC" "$need"
}
unmet_for() { # kind name -> why no backend can run here, when a build exists but needs a newer C library
    for b in $BACKENDS; do
        if [ -n "$(rows_for "$1" "$2" | awk -F'	' -v b="$b" '$5==b {print 1; exit}')" ] && ! backend_runs "$1" "$2" "$b"; then
            echo "the upstream $b build needs glibc $(rows_for "$1" "$2" | awk -F'	' -v b="$b" '$5==b && $11 ~ /^glibc>=/ {sub(/^glibc>=/, "", $11); print $11; exit}') or newer; this system has glibc $GLIBC"; return
        fi
    done
}
UNMET=""
pick_backend() { # kind name -> the backend to install, or nothing
    rows="$(rows_for "$1" "$2")"
    [ -n "$rows" ] || return 0
    have="$(printf '%s\n' "$rows" | awk -F'\t' '{print $5}' | sort -u | tr '\n' ' ' | sed 's/ *$//')"
    UNMET=""
    case " $have " in " any ") echo any; return ;; esac
    for b in $BACKENDS; do
        case " $have " in
            *" $b "*)
                if backend_runs "$1" "$2" "$b"; then echo "$b"; return; fi
                UNMET="$b needs a newer C library than glibc $GLIBC" ;;
        esac
    done
}
kind_of() { # a name -> engine, runtime or core
    k="$(lst | awk -F'\t' -v n="$1" '$0 !~ /^#/ && $2==n {print $1; exit}')"
    echo "$k"
}
dir_for() { # kind name
    case "$1" in engine) echo "$ROOT/engines/$2" ;; runtime) echo "$ROOT/runtimes/$2" ;; *) echo "$ROOT/bin" ;; esac
}

SUMS=""
sum_of() { # a row's sha256 field and the file name -> the sum to check
    if [ "$1" != "@sums" ]; then echo "$1"; return; fi
    if [ -z "$SUMS" ]; then
        SUMS="$TMP/SHA256SUMS"
        get "$REL/SHA256SUMS" "$SUMS" || die "Cannot read $REL/SHA256SUMS."
    fi
    s="$(grep -E " \*?$2\$" "$SUMS" | head -n 1 | cut -d' ' -f1)"
    [ -n "$s" ] || die "$2 is not in SHA256SUMS."
    echo "$s"
}

unpack() { # archive destination strip
    ar="$1"; dest="$2"; strip="${3:-0}"
    x="$TMP/x.$$"; rm -rf "$x"; mkdir -p "$x" "$dest"
    case "$ar" in
        *.tar.gz | *.tgz) tar -xzf "$ar" -C "$x" ;;
        *.zip)
            if command -v unzip >/dev/null 2>&1; then unzip -q -o "$ar" -d "$x"
            elif command -v bsdtar >/dev/null 2>&1; then bsdtar -xf "$ar" -C "$x"
            else die "unzip is needed to unpack $(basename "$ar")."; fi ;;
        *) die "Unknown archive type: $ar" ;;
    esac
    src="$x"; i=0
    case "$strip" in
        find:*) # the folder that holds the named file (a runtime's program sits one or two folders down)
            hit="$(find "$x" -maxdepth 4 -name "${strip#find:}" \( -type f -o -type l \) 2>/dev/null | head -n 1)"
            [ -n "$hit" ] || die "$(basename "$ar") does not contain ${strip#find:}."
            src="$(dirname "$hit")" ;;
        *)
            while [ "$i" -lt "$strip" ]; do
                cnt="$(ls -A "$src" | wc -l | tr -d ' ')"
                [ "$cnt" = 1 ] || break
                src="$src/$(ls -A "$src")"; i=$((i + 1))
            done ;;
    esac
    cp -R "$src"/. "$dest"/
    rm -rf "$x"
}

# Installs one component with the backend `$3`: downloads each row, checks its sum, unpacks it next to the others.
install_component() { # kind name backend
    kind="$1"; name="$2"; bk="$3"
    final="$(dir_for "$kind" "$name")"
    stage="$final.new"
    [ "$kind" != core ] || stage="$ROOT/bin.new"
    rm -rf "$stage"; mkdir -p "$stage"
    got=""
    rows_for "$kind" "$name" | while IFS="$TAB" read -r rk rn ro ra b v s st d u need; do
        [ "$b" = any ] || [ "$b" = "$bk" ] || continue
        printf '%s\n' "$b $v $s $st $d $u"
    done >"$TMP/rows.$$"
    [ -s "$TMP/rows.$$" ] || die "$name has no build for $OS $ARCH."
    # Already installed from these very rows (same backend, versions and checksums): nothing to download. AGENT_FORCE=1 downloads again.
    want_ver="$(awk '{printf "%s ", $2}' "$TMP/rows.$$")"; want_sha="$(awk '{printf "%s ", $3}' "$TMP/rows.$$")"
    if [ "$kind" != core ] && [ -z "${AGENT_FORCE:-}" ] && [ -r "$final/.pin" ] && [ "$(sed -n 1p "$final/.pin")" = "$bk" ] &&
        [ "$(sed -n 2p "$final/.pin")" = "$want_ver" ] && [ "$(sed -n 3p "$final/.pin")" = "$want_sha" ]; then
        rm -rf "$stage" "$TMP/rows.$$"
        step "$name is up to date ($bk, $want_ver)"
        return 0
    fi
    while read -r b v s st d u; do
        url="$(printf '%s' "$u" | sed "s|@RELEASE@|$REL|; s|@TAG@|$TAG|g")"
        file="${url##*/}"
        printf '  %s↓%s %s %s\n' "$T" "$R" "$name" "$G($b, $v)$R"
        get_progress "$url" "$TMP/$file" || die "The download of $file failed: $url"
        want="$(sum_of "$s" "$file")"
        [ "$(sha "$TMP/$file")" = "$want" ] || die "SHA-256 mismatch for $file. Nothing was installed."
        case "$d" in . | -) dd="$stage" ;; *) dd="$stage/$d" ;; esac
        unpack "$TMP/$file" "$dd" "$st"
        rm -f "$TMP/$file"
        got="$got$v "
    done <"$TMP/rows.$$"
    rm -f "$TMP/rows.$$"
    if [ "$kind" = core ]; then
        rm -rf "$ROOT/bin.old"; [ ! -d "$ROOT/bin" ] || mv "$ROOT/bin" "$ROOT/bin.old"
        mv "$stage" "$ROOT/bin"; rm -rf "$ROOT/bin.old"
        chmod +x "$ROOT/bin"/* 2>/dev/null || true
    else
        printf '%s\n%s\n%s\n' "$bk" "$got" "$want_sha" >"$stage/.pin"
        rm -rf "$final"; mkdir -p "$(dirname "$final")"; mv "$stage" "$final"
        chmod +x "$final"/* 2>/dev/null || true
    fi
    step "$name installed ($bk)"
}

pinned() { # kind name -> "backend version" of the installed one, or nothing
    d="$(dir_for "$1" "$2")"
    [ -r "$d/.pin" ] || return 0
    printf '%s %s' "$(sed -n 1p "$d/.pin")" "$(sed -n 2p "$d/.pin")"
}

add_names() { # comma separated names
    [ -n "$1" ] || return 0
    load_list
    [ -n "$TMP" ] || make_tmp
    release_base
    for n in $(printf '%s' "$1" | tr ',' ' '); do
        k="$(kind_of "$n")"
        if [ -z "$k" ] || [ "$k" = core ]; then info "$n is not a component of this release (agent ls shows them)"; continue; fi
        b="$(pick_backend "$k" "$n")"
        if [ -z "$b" ]; then
            why="$(unmet_for "$k" "$n")"
            if [ -n "$why" ]; then info "$n cannot run here: $why"; else info "$n has no build for $OS $ARCH yet"; fi
            continue
        fi
        (install_component "$k" "$n" "$b") || { info "$n skipped"; continue; }
        if [ "$k" = engine ] && [ -x "$BIN/agent" ]; then
            if "$BIN/agent" doctor 2>/dev/null | grep -Eq "^ +$n +ok"; then step "$n loads"; else info "$n is installed but did not report ok in agent doctor"; fi
        fi
    done
}

components_of() { # kind -> the names that exist for this machine
    lst | awk -F'\t' -v k="$1" -v o="$OS" -v a="$ARCH" '$0 !~ /^#/ && $1==k && $3==o && $4==a {print $2}' | sort -u | tr '\n' ' '
}

# ---------------------------------------------------------------------------------------------------------------- commands

do_install() {
    resolve_root
    printf '\n  %s%sagent.cpp%s  %sinstaller%s\n  %s────────────────────────────%s\n\n' "$B" "$T" "$R" "$G" "$R" "$G" "$R"
    make_tmp
    TAG="$version"
    if [ -z "$TAG" ] && [ -z "${AGENT_RELEASE:-}" ]; then TAG="$(latest_tag)" || die "Cannot reach GitHub to find the latest release."; fi
    [ -n "$TAG" ] || TAG="local"
    release_base
    # The scripts and the list of components of this release are kept in the folder: every later command uses these, so what is installed
    # stays pinned until update.
    mkdir -p "$ROOT/scripts"
    sbase="${AGENT_SCRIPTS:-https://raw.githubusercontent.com/$REPO/$TAG/scripts}"
    get "$sbase/components.tsv" "$ROOT/scripts/components.tsv" || die "Cannot read the list of components from $sbase."
    get "$sbase/install.sh" "$ROOT/scripts/install.sh" || die "Cannot read the script from $sbase."
    chmod +x "$ROOT/scripts/install.sh"
    LIST="$ROOT/scripts/components.tsv"
    step "$OS $ARCH  ·  backends: $BACKENDS"
    step "Release $(list_tag)"
    install_component core agent any
    : >"$ROOT/.agent-root"
    for d in engines runtimes models workspaces cache logs skills mcp plugins; do mkdir -p "$ROOT/$d"; done
    remember_env
    export PATH="$BIN:$PATH"
    if [ -z "${AGENT_NO_START:-}" ]; then do_setup; else printf '\n  Next: %sagent setup%s\n\n' "$T" "$R"; fi
}

do_setup() {
    resolve_root
    load_list
    make_tmp
    TAG="$(list_tag)"; release_base
    printf '\n  %s%sagent.cpp%s  %ssetup%s\n  %s────────────────────────────%s\n\n' "$B" "$T" "$R" "$G" "$R" "$G" "$R"
    step "Files are kept in $ROOT"

    # Engines: what runs the models. The chat needs llama; the others add images, speech, ONNX and 3D.
    avail="$(components_of engine)"
    chosen="$engines"
    if [ -n "$all" ]; then chosen="$(printf '%s' "$avail" | tr -s ' ' ',' | sed 's/^,//; s/,$//')"; fi
    if [ -z "$chosen" ] && [ -z "$no_engines" ]; then
        if [ -n "$ask" ]; then
            prompt "Install which? (all, $(printf '%s' "$avail" | sed 's/ *$//; s/ /, /g')):" "llama"
            chosen="$answer"; [ "$chosen" != all ] || chosen="$(printf '%s' "$avail" | tr -s ' ' ',' | sed 's/^,//; s/,$//')"
        else chosen=llama; fi
    fi
    [ -z "$no_engines" ] || chosen=""
    add_names "$chosen"

    # Runtimes: private copies of the tools MCP servers and skills need.
    avail="$(components_of runtime)"
    chosen="$runtimes"
    if [ -z "$chosen" ] && [ -z "$no_runtimes" ]; then
        if [ -n "$ask" ]; then
            prompt "Runtimes? (all, none, $(printf '%s' "$avail" | sed 's/ *$//; s/ /, /g')):" "bun,uv"
            chosen="$answer"; [ "$chosen" != none ] || chosen=""
            [ "$chosen" != all ] || chosen="$(printf '%s' "$avail" | tr -s ' ' ',' | sed 's/^,//; s/,$//')"
        else chosen="bun,uv"; fi
    fi
    [ -z "$no_runtimes" ] || chosen=""
    add_names "$chosen"

    # A model: a link or a file, not a list.
    if [ -z "$model" ] && [ -z "$no_model" ] && [ -n "$ask" ]; then
        prompt "Model? (Hugging Face link, owner/repo or a file; Enter to skip):" ""
        model="$answer"
    fi
    model="$(printf '%s' "$model" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^[\"']//" -e "s/[\"']\$//")"
    if [ -n "$model" ]; then
        printf '  %s↓%s Adding %s\n' "$T" "$R" "$model"
        "$BIN/agent" model pull "$model" || die "The model could not be added."
    fi
    have_model=""
    listed="$("$BIN/agent" model ls 2>/dev/null || true)"
    case "$listed" in "" | "no models"*) ;; *) have_model=1 ;; esac
    if [ -n "$have_model" ] && [ -n "$ask" ]; then
        printf '\n  Done. Starting the chat ...\n\n'
        exec "$BIN/agent" </dev/tty
    fi
    if [ -n "$have_model" ]; then printf '\n  Done. Run %sagent%s to start the chat.\n\n' "$T" "$R"
    else printf '\n  Done. Add a model with %sagent model pull <Hugging Face link or file>%s, then run %sagent%s.\n\n' "$T" "$R" "$T" "$R"; fi
}

do_add() {
    resolve_root; load_list; make_tmp
    TAG="$(list_tag)"; release_base
    [ -n "$names" ] || die "Name what to add: $(components_of engine)$(components_of runtime)"
    add_names "$names"
}

do_rm() {
    resolve_root; load_list
    [ -n "$names" ] || die "Name what to remove."
    for n in $(printf '%s' "$names" | tr ',' ' '); do
        k="$(kind_of "$n")"
        d="$(dir_for "$k" "$n")"
        if [ -n "$k" ] && [ "$k" != core ] && [ -d "$d" ]; then rm -rf "$d"; step "$n removed"; else info "$n is not installed"; fi
    done
}

do_ls() {
    resolve_root; load_list
    printf '\n  agent.cpp %s  ·  %s %s  ·  backends: %s\n\n' "$(list_tag)" "$OS" "$ARCH" "$BACKENDS"
    for k in engine runtime; do
        for n in $(components_of $k); do
            p="$(pinned "$k" "$n")"
            b="$(pick_backend "$k" "$n")"
            if [ -n "$p" ]; then printf '  %-10s %-9s installed  %s\n' "$n" "$k" "$p"
            else printf '  %-10s %-9s available  (%s)\n' "$n" "$k" "$b"; fi
        done
    done
    printf '\n'
}

do_update() {
    resolve_root
    [ -r "$ROOT/scripts/components.tsv" ] || die "agent.cpp is not installed in $ROOT."
    LIST="$ROOT/scripts/components.tsv"; have="$(list_tag)"
    make_tmp
    TAG="$version"
    if [ -z "$TAG" ]; then TAG="$(latest_tag)" || die "Cannot reach GitHub to find the latest release."; fi
    if [ "$TAG" = "$have" ]; then step "Up to date ($have)"; [ -z "$check" ] || exit 0; fi
    if [ -n "$check" ]; then step "An update is available: $have -> $TAG"; exit 0; fi
    [ "$TAG" != "$have" ] || exit 0
    release_base
    sbase="${AGENT_SCRIPTS:-https://raw.githubusercontent.com/$REPO/$TAG/scripts}"
    get "$sbase/components.tsv" "$TMP/new.tsv" || die "Cannot read the list of components of $TAG."
    get "$sbase/install.sh" "$TMP/new.sh" || die "Cannot read the script of $TAG."
    step "Updating $have -> $TAG"
    cp "$ROOT/scripts/components.tsv" "$TMP/old.tsv"
    # What is installed is replaced by what the new release pins; the new list and script are kept only when everything worked.
    cp "$TMP/new.tsv" "$ROOT/scripts/components.tsv"; LIST="$ROOT/scripts/components.tsv"
    install_component core agent any
    for k in engine runtime; do
        for n in $(components_of $k); do
            [ -n "$(pinned "$k" "$n")" ] || continue
            b="$(pick_backend "$k" "$n")"
            [ -n "$b" ] || { info "$n has no build in $TAG: kept as it is"; continue; }
            install_component "$k" "$n" "$b"
        done
    done
    cp "$TMP/new.sh" "$ROOT/scripts/install.sh"; chmod +x "$ROOT/scripts/install.sh"
    step "Now at $TAG"
}

do_uninstall() {
    resolve_root
    printf '\n  agent uninstall removes:\n'
    if [ -n "$keep_data" ]; then
        printf '    the program, engines, runtimes and caches in %s\n    (models, workspaces, settings and notes stay)\n' "$ROOT"
    else
        printf '    the whole folder %s  (program, engines, runtimes, models, conversations, notes, logs)\n' "$ROOT"
    fi
    printf '    AGENT_HOME and the PATH entry in your shell profiles\n\n'
    confirm "This cannot be undone. Remove it?" || { info "Nothing was removed (--yes removes without asking)"; exit 1; }
    if [ -n "$keep_data" ]; then
        for d in bin engines runtimes cache scripts; do rm -rf "$ROOT/$d"; done # the marker stays: a new install finds its data again
    else
        case "$ROOT" in "/" | "$HOME" | "") die "Refusing to remove $ROOT." ;; esac
        cd /
        TMP=""
        rm -rf "$ROOT"
    fi
    forget_env
    step "agent.cpp is removed"
}

# fetch NAME DIR: only the upstream parts of an engine (what its plugin is compiled against), for the release workflow. Everything is
# unpacked whole, the header folders included, and no agent.cpp folder is needed.
do_fetch() {
    n="$(printf '%s' "$names" | cut -d, -f1)"; d="$(printf '%s' "$names" | cut -d, -f2)"
    [ -n "$n" ] && [ -n "$d" ] || die "usage: install.sh fetch NAME,DIR  (or: fetch NAME DIR)"
    case "$d" in /*) ;; *) d="$(pwd)/$d" ;; esac
    TMP="$(mktemp -d)"; ROOT="$TMP"; FETCHING=1
    LIST="${AGENT_LIST:-${AGENT_SCRIPTS:-}/components.tsv}"
    [ -r "$LIST" ] || die "set AGENT_LIST to scripts/components.tsv"
    TAG="$(list_tag)"; release_base
    b="$(pick_backend engine "$n")"; [ -n "$b" ] || die "$n has no build for $OS $ARCH."
    mkdir -p "$d"
    lst | awk -F'\t' -v n="$n" -v o="$OS" -v a="$ARCH" -v b="$b" '$0 !~ /^#/ && $1=="engine" && $2==n && $3==o && $4==a && ($5=="any" || $5==b) && $10 !~ /^@RELEASE@/' | while IFS="$TAB" read -r rk rn ro ra rb v s st dd u need; do
        file="${u##*/}"
        get_progress "$u" "$TMP/$file" || die "The download of $file failed."
        [ "$(sha "$TMP/$file")" = "$s" ] || die "SHA-256 mismatch for $file."
        case "$st" in find:*) st=0 ;; esac # whole archive: the header folders are wanted too
        unpack "$TMP/$file" "$d" "$st"
        printf '  %s fetched %s (%s)\n' "$T$R" "$file" "$v"
        : >"$TMP/fetched"
    done
    [ -e "$TMP/fetched" ] || die "$n has no upstream part to fetch in scripts/components.tsv."
}

case "$cmd" in
    fetch) do_fetch ;;
    install) do_install ;;
    setup) do_setup ;;
    add) do_add ;;
    rm | remove) do_rm ;;
    ls | list) do_ls ;;
    update | upgrade) do_update ;;
    uninstall) do_uninstall ;;
    *) die "unknown command '$cmd' (install, setup, add, rm, ls, update, uninstall)" ;;
esac
