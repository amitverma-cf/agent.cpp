#pragma once

// Releases are signed: the release carries manifest.json (a SHA-256 for every file) and manifest.json.sig, an Ed25519 signature of the
// manifest's bytes made with the project's release key (scripts/sign-release.sh). `agent update` checks it before downloading anything, so
// a changed manifest, or an archive swapped on the download host, is refused. The public key is part of the program.

#include <string>
#include <string_view>

namespace agent::cli {

// The project's release public key: 32 raw bytes, base64.
inline constexpr std::string_view kReleasePublicKey = "dXlNBUoOgh9rGPnaPb9mdACHtQRJVSJbWcIqqPmRVYs=";

// Base64 (standard alphabet, padding optional, whitespace ignored) to bytes; false when the text is not base64.
bool base64_to_bytes(std::string_view text, std::string &out);

// True when `signature_b64` is a valid Ed25519 signature of `message` under the base64 `public_key_b64`.
bool verify_ed25519(std::string_view message, std::string_view signature_b64, std::string_view public_key_b64);

} // namespace agent::cli
