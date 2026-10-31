#pragma once

// Hugging Face model hub for the agent CLI: resolve, download (resume + sha256) and catalogue models.
// Link agent_hub. libagent itself has no network dependency; only the CLI links this module.

#include <agent-cpp/agent.hpp>

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::hub {

// -------------------------- Spec resolution --------------------------

// "owner/repo:file.gguf", "owner/repo:Q4_K_M" or "owner/repo" (single .gguf in the repo).
// An optional "@revision" may follow the repo ("owner/repo@main:Q4_K_M").
struct ModelSpec {
    std::string repo;           // "owner/name"
    std::string revision = "main";
    std::string file;           // exact path inside the repo, when the selector looks like a file
    std::string quant;          // quant tag such as "Q4_K_M", when the selector is not a file
};

Result<ModelSpec> parse_spec(std::string_view spec);

// Case-insensitive "filename contains quant tag" test, e.g. ("qwen-q4_k_m.gguf", "Q4_K_M").
bool quant_matches(std::string_view filename, std::string_view quant);

struct RemoteFile {
    std::string path;           // path inside the repo ("rfilename")
    uint64_t size = 0;          // 0 when unknown
    std::string sha256;         // lowercase hex of the LFS object; empty when unknown
};

struct RepoInfo {
    std::string repo;
    std::string license;        // HF licence id ("apache-2.0", "cc-by-nc-4.0", "other", ...); empty if unknown
    std::string license_name;   // cardData.license_name (used with license == "other")
    bool gated = false;
    std::vector<RemoteFile> files;
};

// Parses the body of GET /api/models/<repo>?blobs=true. Pure function, no network.
Result<RepoInfo> parse_repo_info(std::string_view json);

// Picks the file the spec refers to: exact path (or unique basename) match for ModelSpec::file,
// otherwise the .gguf whose name contains ModelSpec::quant (a single-file GGUF wins; otherwise the
// first shard "-00001-of-0000N" of a split GGUF). With neither, the repo must hold one .gguf.
Result<RemoteFile> select_file(const RepoInfo &info, const ModelSpec &spec);

// Split GGUF ("<stem>-00001-of-00003.gguf"): every shard of the set `shard` belongs to, in order.
// A file that isn't a shard yields just itself. Fails when a shard is missing from the repo.
Result<std::vector<RemoteFile>> split_set(const RepoInfo &info, const RemoteFile &shard);

// -------------------------- Licence policy --------------------------

// True for licences that restrict use (non-commercial, research-only, no-derivatives ...). Policy
// (ROADMAP §1.1): pulling such a model requires PullOptions::allow_restricted_license.
bool is_restricted_license(std::string_view license, std::string_view license_name = {});

// -------------------------- Hashing --------------------------

std::string sha256_hex(std::span<const uint8_t> data);
std::string sha256_hex(std::string_view data);
Result<std::string> sha256_file(const std::string &path);

// -------------------------- Catalogue (models.json) --------------------------

struct CatalogEntry {
    std::string name;           // logical name used by "agent model rm" and configs
    std::string repo;
    std::string file;           // path inside the repo
    std::string path;           // local absolute path
    uint64_t size = 0;
    std::string sha256;
    std::string license;
};

std::string catalog_to_json(std::span<const CatalogEntry> entries);
Result<std::vector<CatalogEntry>> catalog_from_json(std::string_view json);
// Missing file -> empty catalogue. Saves atomically (write "<path>.tmp", then rename).
Result<std::vector<CatalogEntry>> load_catalog(const std::string &path);
Result<void> save_catalog(const std::string &path, std::span<const CatalogEntry> entries);

// -------------------------- Network options --------------------------

struct HttpOptions {
    std::string endpoint = "https://huggingface.co"; // HF_ENDPOINT
    std::string token;          // HF_TOKEN; sent only to the endpoint host, never to CDN redirects
    std::string proxy_host;     // from HTTPS_PROXY / https_proxy; empty = direct
    int proxy_port = 0;
    std::string no_proxy;       // NO_PROXY / no_proxy, comma-separated host suffixes
    std::string ca_cert_path;   // AGENT_CA_BUNDLE, SSL_CERT_FILE or CURL_CA_BUNDLE; empty = system store
    int connect_timeout_sec = 30;
    int read_timeout_sec = 60;
    int max_redirects = 10;
};

// Reads the environment variables named above.
HttpOptions http_options_from_env();

struct ProxyAddress {
    std::string host;
    int port = 0;
};
// "http://host:port", "host:port", "http://user:pass@host:port/" -> {host, port}. Port defaults to 80.
Result<ProxyAddress> parse_proxy_url(std::string_view url);

// True when `host` matches an entry of a NO_PROXY list ("*" , "example.com", ".example.com").
bool host_bypasses_proxy(std::string_view host, std::string_view no_proxy);

// Return false to cancel the transfer.
using ProgressFn = bool (*)(uint64_t downloaded, uint64_t total, void *user_data);

// -------------------------- Network operations --------------------------

Result<RepoInfo> fetch_repo_info(std::string_view repo, const HttpOptions &http, std::string_view revision = "main");

struct DownloadOptions {
    uint64_t expected_size = 0;      // 0 = unknown
    std::string expected_sha256;     // empty = skip verification
    ProgressFn on_progress = nullptr;
    void *progress_user_data = nullptr;
    const std::atomic<bool> *cancel = nullptr;
};

// GET `url` (following cross-host redirects) into "<dest>.part", resuming with HTTP Range when a
// partial file exists, then verifies size/sha256 and renames to `dest`. On cancel the .part file is
// kept for a later resume and ErrorCode::Cancelled is returned.
Result<void> download_file(std::string_view url, const std::string &dest, const HttpOptions &http, const DownloadOptions &opts);

// -------------------------- High-level commands --------------------------

// Default: $AGENT_MODELS_DIR, else %LOCALAPPDATA%\agent\models (Windows) or
// $XDG_DATA_HOME/agent/models, else ~/.local/share/agent/models.
std::string default_models_dir();

struct PullOptions {
    std::string models_dir;          // empty = default_models_dir()
    std::string catalog_path;        // empty = <models_dir>/models.json
    std::string name;                // catalogue name; empty = file basename without extension
    bool allow_restricted_license = false;
    HttpOptions http = http_options_from_env();
    ProgressFn on_progress = nullptr;
    void *progress_user_data = nullptr;
    const std::atomic<bool> *cancel = nullptr;
};

// Resolve + licence check + download + verify + record in models.json. The model lands in
// <models_dir>/<owner>/<repo>/<file>. Restricted licences fail with ErrorCode::PermissionDenied
// unless allow_restricted_license is set. Already-present files with matching size are not re-fetched.
Result<CatalogEntry> pull_model(std::string_view spec, const PullOptions &opts);

Result<std::vector<CatalogEntry>> list_models(const std::string &catalog_path);

// Removes the catalogue entry and (when delete_file) the model file. KeyNotFound if absent.
Result<void> remove_model(std::string_view name, const std::string &catalog_path, bool delete_file = true);

} // namespace agent::hub
