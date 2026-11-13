#!/usr/bin/env bash
# Pins ports/agent-cpp to a commit: scripts/update-vcpkg-port.sh [COMMIT]   (default: origin/main)
# Computes the SHA512 of the GitHub archive of that commit and of the simdjson commit its vendor/simdjson points at.
set -euo pipefail
cd "$(dirname "$0")/.."
commit=$(git rev-parse "${1:-origin/main}")
simdjson=$(git ls-tree "$commit" vendor/simdjson | awk '{print $3}')
sum() { curl -fsSL "$1" | sha512sum | cut -d' ' -f1; }
agent_sha=$(sum "https://github.com/amitverma-cf/agent.cpp/archive/${commit}.tar.gz")
simdjson_sha=$(sum "https://github.com/simdjson/simdjson/archive/${simdjson}.tar.gz")
port=ports/agent-cpp/portfile.cmake
python3 - "$port" "$commit" "$agent_sha" "$simdjson" "$simdjson_sha" <<'PY'
import re, sys
path, commit, agent_sha, simdjson, simdjson_sha = sys.argv[1:]
text = open(path, encoding="utf-8").read()
pattern = r"(REPO amitverma-cf/agent\.cpp\n    REF )\w+(\n    SHA512 )\w+"
text, n1 = re.subn(pattern, lambda m: m.group(1) + commit + m.group(2) + agent_sha, text)
pattern = r"(REPO simdjson/simdjson\n    REF )\w+(\n    SHA512 )\w+"
text, n2 = re.subn(pattern, lambda m: m.group(1) + simdjson + m.group(2) + simdjson_sha, text)
assert n1 == 1 and n2 == 1, "the portfile has changed shape"
open(path, "w", encoding="utf-8").write(text)
PY
echo "pinned agent.cpp ${commit} and simdjson ${simdjson}"
