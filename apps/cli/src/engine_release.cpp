#include "engine_release.hpp"

#include "runtimes.hpp" // host_os(), host_arch()

#include <filesystem>
#include <simdjson.h>

namespace agent::cli {

namespace fs = std::filesystem;

Result<ReleaseManifest> parse_release_manifest(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc)) return fail<ReleaseManifest>(ErrorCode::ParseError, "the release manifest is not valid JSON");
    ReleaseManifest m;
    std::string_view v;
    if (!doc["version"].get(v)) m.version = std::string(v);
    if (!doc["repository"].get(v)) m.repository = std::string(v);
    simdjson::dom::array files;
    if (doc["files"].get(files)) return fail<ReleaseManifest>(ErrorCode::ParseError, "the release manifest has no \"files\" list");
    for (simdjson::dom::element e : files) {
        simdjson::dom::object o;
        if (e.get(o)) continue;
        ReleaseFile f;
        if (!o["name"].get(v)) f.name = std::string(v);
        if (!o["sha256"].get(v)) f.sha256 = std::string(v);
        int64_t size = 0;
        if (!o["size"].get(size) && size > 0) f.size = static_cast<uint64_t>(size);
        if (!f.name.empty() && f.sha256.size() == 64) m.files.push_back(std::move(f));
    }
    return ok(std::move(m));
}

std::string engine_asset_name(std::string_view engine, std::string_view os, std::string_view arch) {
    const char *ext = os == "windows" ? "dll" : os == "macos" ? "dylib" : "so";
    return "agent-engine-" + std::string(engine) + "-" + std::string(os) + "-" + std::string(arch) + "." + ext;
}

std::string release_manifest_url(std::string_view repo, std::string_view tag) { return release_asset_url(repo, tag, "manifest.json"); }

std::string release_asset_url(std::string_view repo, std::string_view tag, std::string_view file) {
    if (tag.empty() || tag == "latest") return "https://github.com/" + std::string(repo) + "/releases/latest/download/" + std::string(file);
    return "https://github.com/" + std::string(repo) + "/releases/download/" + std::string(tag) + "/" + std::string(file);
}

Result<std::string> pull_engine(const std::string &engines_dir, std::string_view engine, std::string_view repo, std::string_view tag,
                                const agent::hub::HttpOptions &http, bool (*on_progress)(uint64_t, uint64_t, void *), void *user) {
    if (engine.empty() || engine.find_first_of("/\\.") != std::string_view::npos)
        return fail<std::string>(ErrorCode::InvalidConfig, "invalid engine name '" + std::string(engine) + "'");
    auto text = agent::hub::http_get_text(release_manifest_url(repo, tag), http);
    if (!text.ok)
        return fail<std::string>(text.error.code, "cannot read the release manifest of " + std::string(repo) + ": " + text.error.message);
    auto manifest = parse_release_manifest(text.value);
    if (!manifest.ok) return fail<std::string>(manifest.error.code, manifest.error.message);

    const std::string os = host_os(), arch = host_arch();
    const std::string asset = engine_asset_name(engine, os, arch);
    const ReleaseFile *file = nullptr;
    for (const ReleaseFile &f : manifest.value.files)
        if (f.name == asset) file = &f;
    if (!file)
        return fail<std::string>(ErrorCode::KeyNotFound, "release " + manifest.value.version + " has no " + asset + " (no prebuilt '" +
                                                             std::string(engine) + "' engine for " + os + "-" + arch + ")");

    const std::string version = manifest.value.version.empty() ? std::string(tag) : manifest.value.version;
    const char *ext = os == "windows" ? "dll" : os == "macos" ? "dylib" : "so";
    const fs::path target = fs::path(engines_dir) / std::string(engine) / version / ("agent-engine-" + std::string(engine) + "." + ext);
    std::error_code ec;
    fs::create_directories(target.parent_path(), ec);
    if (ec) return fail<std::string>(ErrorCode::FilesystemError, "cannot create " + target.parent_path().string() + ": " + ec.message());

    agent::hub::DownloadOptions opts;
    opts.expected_sha256 = file->sha256;
    opts.expected_size = file->size;
    opts.on_progress = on_progress;
    opts.progress_user_data = user;
    const std::string repo_name = manifest.value.repository.empty() ? std::string(repo) : manifest.value.repository;
    if (auto d = agent::hub::download_file(release_asset_url(repo_name, version, asset), target.string(), http, opts); !d.ok)
        return fail<std::string>(d.error.code, d.error.message);
    fs::permissions(target, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
    return ok(target.string());
}

} // namespace agent::cli
