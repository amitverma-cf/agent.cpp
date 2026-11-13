#!/usr/bin/env bash
# Signs a release manifest with the project's Ed25519 release key: scripts/sign-release.sh KEY.pem manifest.json
# Writes manifest.json.sig next to it (base64). Publish both; `agent update` refuses a release without a valid signature.
# The key pair is made once with:  openssl genpkey -algorithm ed25519 -out release-signing-key.pem
# and the public half goes into apps/cli/src/release_verify.hpp:
#   openssl pkey -in release-signing-key.pem -pubout -outform DER | tail -c 32 | base64 -w0
set -euo pipefail
key=${1:?usage: sign-release.sh KEY.pem manifest.json}
manifest=${2:?usage: sign-release.sh KEY.pem manifest.json}
openssl pkeyutl -sign -rawin -inkey "$key" -in "$manifest" | base64 -w0 > "$manifest.sig"
echo "wrote $manifest.sig"
