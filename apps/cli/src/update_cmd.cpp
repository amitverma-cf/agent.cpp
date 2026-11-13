// `agent update`: looks for a newer release of the CLI (the project's GitHub releases carry a manifest.json with a SHA-256 for every file),
// downloads the archive for this machine, checks it and replaces the agent and agent-engine-host executables next to the running one.
// Engines are not touched: `agent engine pull` fetches those. With no release published it says so and changes nothing.

#include "archive.hpp"
#include "commands.hpp"
#include "engine_release.hpp"
#include "release_verify.hpp"
#include "runtimes.hpp"

#include <agent-cpp/hub.hpp>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

// Puts `source` in place of `target`; the old file stays as <target>.old until the next run (a running executable cannot be deleted on
// Windows, but it can be renamed).
bool replace_file(const fs::path &source, const fs::path &target, std::string &error) {
    std::error_code ec;
    const fs::path old = target.string() + ".old";
    fs::remove(old, ec);
    if (fs::exists(target, ec)) {
        fs::rename(target, old, ec);
        if (ec) {
            error = "cannot move " + target.string() + " aside: " + ec.message();
            return false;
        }
    }
    fs::copy_file(source, target, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        error = "cannot write " + target.string() + ": " + ec.message();
        fs::rename(old, target, ec); // put the old one back
        return false;
    }
    fs::permissions(target, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
    fs::remove(old, ec); // fails on Windows while it runs; the next update removes it
    return true;
}

} // namespace

int cmd_update(const CommandLine &cl, Context &ctx) {
    const std::string repo(cl.get("repo", kDefaultReleaseRepo));
    const std::string tag(cl.get("tag", "latest"));
    const agent::hub::HttpOptions http = agent::hub::http_options_from_env();

    auto text = agent::hub::http_get_text(release_manifest_url(repo, tag), http);
    if (!text.ok) {
        const bool none = text.error.message.find("404") != std::string::npos;
        if (none) {
            std::printf("agent %.*s: no release is published in %s yet, nothing to update.\n", static_cast<int>(kCliVersion.size()),
                        kCliVersion.data(), repo.c_str());
            return 0;
        }
        return report_error(make_error(text.error.code, "cannot look for a release of " + repo + ": " + text.error.message));
    }
    // The release is signed: manifest.json.sig is an Ed25519 signature of the manifest, made with the project's release key.
    if (!cl.has("allow-unsigned")) {
        auto signature = agent::hub::http_get_text(release_manifest_url(repo, tag) + ".sig", http, 4096);
        if (!signature.ok)
            return report_error(make_error(ErrorCode::HttpError, "this release has no manifest.json.sig, so it cannot be checked: " +
                                                                     signature.error.message + " (--allow-unsigned installs it anyway)"));
        if (!verify_ed25519(text.value, signature.value, kReleasePublicKey))
            return report_error(
                make_error(ErrorCode::HttpError,
                           "the signature of this release's manifest is not valid: it was not made with the project's release key"));
    }
    auto manifest = parse_release_manifest(text.value);
    if (!manifest.ok) return report_error(manifest.error);
    const std::string &latest = manifest.value.version;
    if (tag == "latest" && !version_is_newer(latest, kCliVersion)) {
        std::printf("agent %.*s is up to date (latest release: %s).\n", static_cast<int>(kCliVersion.size()), kCliVersion.data(),
                    latest.c_str());
        return 0;
    }
    std::printf("update available: %.*s -> %s\n", static_cast<int>(kCliVersion.size()), kCliVersion.data(), latest.c_str());
    if (cl.has("check")) return 0;

    const std::string os = host_os(), arch = host_arch();
    const std::string asset = cli_archive_name(latest, os, arch);
    const ReleaseFile *file = nullptr;
    for (const ReleaseFile &f : manifest.value.files)
        if (f.name == asset) file = &f;
    if (!file) return report_error(make_error(ErrorCode::KeyNotFound, "release " + latest + " has no " + asset + " for this machine"));

    const fs::path work = fs::path(ctx.has_home ? ctx.home.root : ctx.config_dir) / "cache" / "updates" / latest;
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work, ec);
    const std::string archive = (work / asset).string();
    agent::hub::DownloadOptions opts;
    opts.expected_sha256 = file->sha256;
    opts.expected_size = file->size;
    std::printf("downloading %s ...\n", asset.c_str());
    if (auto d = agent::hub::download_file(release_asset_url(repo, latest, asset), archive, http, opts); !d.ok)
        return report_error(d.error);
    const std::string unpacked = (work / "unpacked").string();
    if (auto x = extract_archive(archive, unpacked, 1); !x.ok) return report_error(x.error);

    const fs::path dir = executable_dir();
    const std::string ext = os == "windows" ? ".exe" : "";
    for (const char *name : {"agent", "agent-engine-host"}) {
        const fs::path source = fs::path(unpacked) / (std::string(name) + ext);
        if (!fs::exists(source, ec)) continue;
        std::string error;
        if (!replace_file(source, dir / (std::string(name) + ext), error))
            return report_error(make_error(ErrorCode::FilesystemError, error));
        std::printf("updated %s\n", (dir / (std::string(name) + ext)).string().c_str());
    }
    fs::remove_all(work, ec);
    std::printf("now at %s; start agent again to use it. Engines were not changed (agent engine pull NAME fetches newer ones).\n",
                latest.c_str());
    return 0;
}

} // namespace agent::cli
