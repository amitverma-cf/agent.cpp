#pragma once

// Hugging Face model hub for the agent CLI: resolve, download (resume + sha256) and catalogue models.
// Link agent_hub. libagent itself has no network dependency; only the CLI links this module.

#include <agent-cpp/agent.hpp>
#include <atomic>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::hub {

// -------------------------- Spec resolution --------------------------

// "owner/repo:file.gguf", "owner/repo:Q4_K_M" or "owner/repo" (single .gguf in the repo).
// An optional "@revision" may follow the repo ("owner/repo@main:Q4_K_M").
struct ModelSpec {
    std::string repo; // "owner/name"
    std::string revision = "main";
    std::string file;  // exact path inside the repo, when the selector looks like a file
    std::string quant; // quant tag such as "Q4_K_M", when the selector is not a file
    bool all = false;  // selector "*": every file of the repo (a folder model: TRELLIS, ...)
};

Result<ModelSpec> parse_spec(std::string_view spec);

// Case-insensitive "filename contains quant tag" test, e.g. ("qwen-q4_k_m.gguf", "Q4_K_M").
bool quant_matches(std::string_view filename, std::string_view quant);

struct RemoteFile {
    std::string path;   // path inside the repo ("rfilename")
    uint64_t size = 0;  // 0 when unknown
    std::string sha256; // lowercase hex of the LFS object; empty when unknown
};

struct RepoInfo {
    std::string repo;
    std::string license;      // HF licence id ("apache-2.0", "cc-by-nc-4.0", "other", ...); empty if unknown
    std::string license_name; // cardData.license_name (used with license == "other")
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

// The multimodal projector (mmproj-*.gguf) that goes with a vision / audio model, or nullopt when the repo
// has none or `want` is "none". `want`: "auto" (the projector whose quant matches the model file, else
// Q8_0, else F16, else BF16, else the smallest), or a quant tag / file name fragment such as "f16".
// A model file that is itself a projector never gets another one. Fails when `want` matches nothing.
Result<std::optional<RemoteFile>> select_mmproj(const RepoInfo &info, std::string_view model_file, std::string_view want = "auto");

// -------------------------- Licence policy --------------------------

// True for licences that restrict use (non-commercial, research-only, no-derivatives ...). Policy
// (ROADMAP §1.1): pulling such a model requires PullOptions::allow_restricted_license.
bool is_restricted_license(std::string_view license, std::string_view license_name = {});

// -------------------------- Hashing --------------------------

std::string sha256_hex(std::span<const uint8_t> data);
std::string sha256_hex(std::string_view data);
Result<std::string> sha256_file(const std::string &path);

// -------------------------- Curated registry --------------------------

// Short names for models known to work with the engines here ("qwen3-4b", "glm-ocr", "trellis-coarse").
// A model may need several downloads (a TTS talker and its codec); the first part is the model.
struct RegistryPart {
    std::string_view spec;    // an `agent model pull` spec
    std::string_view include; // for "owner/repo:*" parts: comma separated path filters; otherwise ""
};

struct RegistryEntry {
    std::string_view alias;
    std::string_view task;   // text_gen, vision, ocr, embed, rerank, image_gen, asr, tts, vad, mesh_gen, ...
    std::string_view engine; // llama, sdcpp, audiocpp, onnx, trellis
    std::span<const RegistryPart> parts;
    uint32_t size_mib;               // total download, 0 = not measured
    std::string_view engine_options; // JSON the engine needs to serve this model, "" = none
    std::string_view notes;
    bool tested; // pulled and run end to end with agent.cpp
};

std::span<const RegistryEntry> model_registry();
const RegistryEntry *find_registry_entry(std::string_view alias);

// -------------------------- Catalogue (models.json) --------------------------

struct CatalogEntry {
    std::string name; // logical name used by "agent model rm" and configs
    std::string repo;
    std::string file; // path inside the repo
    std::string path; // local absolute path
    uint64_t size = 0;
    std::string sha256;
    std::string license;
    std::string mmproj;      // local path of the multimodal projector pulled with the model; empty if none
    std::string mmproj_file; // its path inside the repo
};

// engine_options with "mmproj": <path> added (llama's multimodal projector), unless it already has one
// or `mmproj_path` is empty. engine_options is "" or a JSON object.
std::string with_mmproj_option(std::string_view engine_options, std::string_view mmproj_path);

std::string catalog_to_json(std::span<const CatalogEntry> entries);
Result<std::vector<CatalogEntry>> catalog_from_json(std::string_view json);
// Missing file -> empty catalogue. Saves atomically (write "<path>.tmp", then rename).
Result<std::vector<CatalogEntry>> load_catalog(const std::string &path);
Result<void> save_catalog(const std::string &path, std::span<const CatalogEntry> entries);

// -------------------------- Network options --------------------------

struct ProxyAddress {
    std::string host;
    int port = 0;
    std::string user; // percent-decoded; empty = no proxy auth
    std::string pass;
};
// "http://host:port", "host:port", "http://user:pass@host:port/" -> {host, port, user, pass}.
// Port defaults to 80. Credentials are percent-decoded.
Result<ProxyAddress> parse_proxy_url(std::string_view url);

struct HttpOptions {
    std::string endpoint = "https://huggingface.co"; // HF_ENDPOINT
    // Explicit option, or HF_TOKEN / HUGGING_FACE_HUB_TOKEN / the HF token file (see resolve_hf_token).
    // Sent only to hosts accepted by token_allowed_for_host(); never forwarded after a redirect to a CDN.
    std::string token;
    std::string proxy_host; // from HTTPS_PROXY / https_proxy; empty = direct
    int proxy_port = 0;
    std::string proxy_user; // "user:pass@" of HTTPS_PROXY, sent as Proxy-Authorization: Basic
    std::string proxy_pass;
    ProxyAddress http_proxy;  // from HTTP_PROXY / http_proxy, used for plain-http URLs (host empty = direct)
    std::string no_proxy;     // NO_PROXY / no_proxy, comma-separated host suffixes
    std::string ca_cert_path; // AGENT_CA_BUNDLE, SSL_CERT_FILE or CURL_CA_BUNDLE; empty = system store
    int connect_timeout_sec = 30;
    int read_timeout_sec = 60;
    int max_redirects = 10;
};

// Reads the environment variables named above (and the HF token file).
HttpOptions http_options_from_env();

// Token precedence: explicit option, HF_TOKEN, HUGGING_FACE_HUB_TOKEN, then the contents of the token file
// ($HF_TOKEN_PATH, else $HF_HOME/token, else ~/.cache/huggingface/token). Values are trimmed.
std::string resolve_hf_token(std::string_view explicit_token, std::string_view env_hf_token, std::string_view env_legacy_token,
                             std::string_view token_file_contents);

// True when the Authorization header may be sent to `host`: the endpoint host itself, plus hf.co and
// huggingface.co when the endpoint is one of them. Redirect targets (CDN hosts) are pre-signed and get no token.
bool token_allowed_for_host(std::string_view host, std::string_view endpoint);

// True when `host` matches an entry of a NO_PROXY list ("*" , "example.com", ".example.com").
bool host_bypasses_proxy(std::string_view host, std::string_view no_proxy);

// Return false to cancel the transfer.
using ProgressFn = bool (*)(uint64_t downloaded, uint64_t total, void *user_data);

// -------------------------- Network operations --------------------------

Result<RepoInfo> fetch_repo_info(std::string_view repo, const HttpOptions &http, std::string_view revision = "main");

struct DownloadOptions {
    uint64_t expected_size = 0;  // 0 = unknown
    std::string expected_sha256; // empty = skip verification
    ProgressFn on_progress = nullptr;
    void *progress_user_data = nullptr;
    const std::atomic<bool> *cancel = nullptr;
};

// GET `url` (following cross-host redirects) into "<dest>.part", resuming with HTTP Range when a
// partial file exists, then verifies size/sha256 and renames to `dest`. On cancel the .part file is
// kept for a later resume and ErrorCode::Cancelled is returned.
Result<void> download_file(std::string_view url, const std::string &dest, const HttpOptions &http, const DownloadOptions &opts);

// GET `url` (following redirects) and returns the body, for small JSON / text documents (registry search,
// GitHub's API, checksum files). No HF token is sent: this is for third-party hosts. Bodies over `max_bytes`
// fail instead of being cut.
Result<std::string> http_get_text(std::string_view url, const HttpOptions &http, size_t max_bytes = 4u << 20);

// What a POST answered: the status (any status is a reply), the Content-Type and Mcp-Session-Id headers, the body.
struct PostReply {
    int status = 0;
    std::string content_type;
    std::string session_id;
    std::string body;
};
// POSTs `body` (sent as application/json unless a Content-Type header is given) with the headers; no redirects, no retries. Fails only
// when there is no reply at all (connection, TLS, timeout). Bodies over `max_bytes` fail instead of being cut.
Result<PostReply> http_post(std::string_view url, const std::vector<std::pair<std::string, std::string>> &headers, std::string_view body,
                            const HttpOptions &http, int timeout_sec = 30, size_t max_bytes = 8u << 20);

// GET bytes [offset, offset + length) of `url` (following redirects): the first megabytes of a large file, for example a GGUF header.
// A server that ignores Range and sends the whole file is cut after `length` bytes. Fewer bytes come back when the file is shorter.
Result<std::string> http_get_range(std::string_view url, const HttpOptions &http, uint64_t offset, uint64_t length);

// -------------------------- High-level commands --------------------------

// Default: $AGENT_MODELS_DIR, else %LOCALAPPDATA%\agent\models (Windows) or
// $XDG_DATA_HOME/agent/models, else ~/.local/share/agent/models.
std::string default_models_dir();

struct PullOptions {
    std::string models_dir;   // empty = default_models_dir()
    std::string catalog_path; // empty = <models_dir>/models.json
    std::string name;         // catalogue name; empty = file basename without extension
    bool allow_restricted_license = false;
    std::vector<std::string> include; // with "owner/repo:*": keep files whose path contains any of these (empty = all)
    std::string mmproj = "auto";      // "auto", "none", or a quant / file name fragment; see select_mmproj
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

// -------------------------- Model registry --------------------------

// An agent::ModelSpec (see agent::register_model) for a catalogue entry: name = entry.name, config =
// base with model = entry.path and provider = Engine. When base.engine is empty the engine is picked
// from the file type (.gguf -> llama, .onnx -> onnx, .safetensors/.ckpt/.sft -> sdcpp). The memory
// estimate is left to the registry (estimated_bytes = 0), except that a missing file falls back to
// the catalogued size so the registry can still plan.
agent::ModelSpec model_spec_from_entry(const CatalogEntry &entry, const Config &base);
std::vector<agent::ModelSpec> model_specs_from_catalog(std::span<const CatalogEntry> entries, const Config &base);

} // namespace agent::hub
