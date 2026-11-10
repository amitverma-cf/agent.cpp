#pragma once

// Prebuilt engine plugins from the project's GitHub releases (see .github/workflows/release.yml): the release carries
// a manifest.json with a SHA-256 for every file, and each engine as a file of its own named
// agent-engine-<name>-<os>-<arch>.<so|dylib|dll>.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/hub.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

inline constexpr std::string_view kDefaultReleaseRepo = "amitverma-cf/agent.cpp";

struct ReleaseFile {
    std::string name;
    std::string sha256;
    uint64_t size = 0;
};

struct ReleaseManifest {
    std::string version;
    std::string repository;
    std::vector<ReleaseFile> files;
};

Result<ReleaseManifest> parse_release_manifest(std::string_view json);

// "agent-engine-llama-linux-x86_64.so" ("windows" -> .dll, "macos" -> .dylib)
std::string engine_asset_name(std::string_view engine, std::string_view os, std::string_view arch);

// tag: "latest" or a release tag such as "v0.2.0".
std::string release_manifest_url(std::string_view repo, std::string_view tag);
std::string release_asset_url(std::string_view repo, std::string_view tag, std::string_view file);

// Downloads the engine for this machine into <engines_dir>/<engine>/<version>/agent-engine-<engine>.<ext> after
// checking its SHA-256 against the manifest. Returns the plugin path.
Result<std::string> pull_engine(const std::string &engines_dir, std::string_view engine, std::string_view repo, std::string_view tag,
                                const agent::hub::HttpOptions &http, bool (*on_progress)(uint64_t, uint64_t, void *) = nullptr, void *user = nullptr);

} // namespace agent::cli
