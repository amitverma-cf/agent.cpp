# Model hub

`#include <agent-cpp/hub.hpp>` · namespace `agent::hub` · link `agent-cpp::hub` (target `agent_hub`)

The model hub downloads models from Hugging Face and keeps a local catalogue (`models.json`). It is a separate library, not part of
`libagent`: `libagent` has no network code. `agent_hub` is built when `AGENT_BUILD_HUB` is on (the default) and needs the OpenSSL
development package (`libssl-dev` on Debian and Ubuntu). It uses the vendored cpp-httplib and simdjson.

All functions return `Result<T>`. Nothing throws. Error checks are left out of the examples below.

The `agent` CLI exposes the same operations as `agent model`. The [`pull` example](../../examples/07-model-pull/pull.cpp) is the library
call in a few lines.

## Command line

```sh
agent model search qwen3                                          # look up repos
agent model info Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M           # files, sizes, licence
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M           # pick the file by quant tag
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf
agent model pull ggml-org/models:tinyllamas/stories260K.gguf --name tiny
agent model ls
agent model rm tiny [--keep-file]
```

Pull flags: `--name NAME`, `--quant Q4_K_M`, `--mmproj auto|none|QUANT|FILE` (or `--no-mmproj`) for the vision projector, and
`--allow-restricted` for restricted licences (see [Licence policy](#licence-policy)).

Ctrl-C cancels a download and keeps `<file>.part`. Running the same command again resumes it.

## Specs

```
owner/repo[@revision][:selector]
```

- A `selector` that contains `.` or `/` is a file path inside the repo. A unique basename also works.
- Any other `selector` is a quant tag such as `Q4_K_M` or `IQ2_XXS`, matched case-insensitively against `.gguf` file names. An exact
  tag beats a longer one: `Q4_K` picks `*-q4_k.gguf` over `*-q4_k_m.gguf`. `mmproj` files are ignored.
- For a split GGUF (`-00001-of-0000N`), the first shard is selected and every shard is downloaded into the same folder. Each is size-
  and sha256-checked. The catalogue points at shard 1, which llama.cpp loads together with the rest, and `remove_model` deletes all shards.
- `*` selects every file of the repo. This is for folder models such as TRELLIS.
- With no selector, the repo must contain exactly one `.gguf`.
- The prefixes `https://huggingface.co/`, `hf.co/` and `hf://` are removed.

```cpp
auto spec = agent::hub::parse_spec("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M");   // ModelSpec{repo, revision, file, quant, all}
auto http = agent::hub::http_options_from_env();
auto info = agent::hub::fetch_repo_info(spec.value.repo, http);                  // RepoInfo: files, sizes, sha256, licence
auto file = agent::hub::select_file(info.value, spec.value);                     // RemoteFile: path, size, sha256
```

### Spec functions

```cpp
Result<ModelSpec> parse_spec(std::string_view spec);
bool quant_matches(std::string_view filename, std::string_view quant);   // case-insensitive "contains" test
Result<RemoteFile> select_file(const RepoInfo &info, const ModelSpec &spec);
Result<std::vector<RemoteFile>> split_set(const RepoInfo &info, const RemoteFile &shard);
Result<std::optional<RemoteFile>> select_mmproj(const RepoInfo &info, std::string_view model_file, std::string_view want = "auto");
```

- `select_file` uses the exact path (or unique basename) for `file`, otherwise the `.gguf` whose name contains `quant`. A single-file GGUF
  wins; otherwise the first shard is chosen.
- `split_set` returns every shard that `shard` belongs to, in order. It fails when a shard is missing from the repo.
- `select_mmproj` returns the multimodal projector (`mmproj-*.gguf`) for a vision or audio model, or `nullopt` when the repo has none or
  `want` is `"none"`. `want = "auto"` prefers the projector whose quant matches the model file, then Q8_0, F16, BF16, and the smallest.
  A quant tag or file-name fragment selects that file. It fails when `want` matches nothing.

### Repo info

```cpp
struct RemoteFile { std::string path; uint64_t size = 0; std::string sha256; };
struct RepoInfo   { std::string repo; std::string license; std::string license_name; bool gated = false; std::vector<RemoteFile> files; };

Result<RepoInfo> fetch_repo_info(std::string_view repo, const HttpOptions &http, std::string_view revision = "main");
Result<RepoInfo> parse_repo_info(std::string_view json);   // pure parser, no network
```

`fetch_repo_info` calls `GET /api/models/<repo>?blobs=true`, or `/revision/<rev>` for another revision. The response gives every file's
size and LFS sha256, and the licence. The licence is read from `cardData.license`, then `cardData.license_name`, then a `license:<id>` tag.
`parse_repo_info` is the pure part and is what the unit tests use.

## Pull, list, remove

```cpp
struct PullOptions {
    std::string models_dir;           // empty = default_models_dir()
    std::string catalog_path;         // empty = <models_dir>/models.json
    std::string name;                 // catalogue name; empty = file name without extension
    bool allow_restricted_license = false;
    std::vector<std::string> include; // "owner/repo:*" only: keep files whose path contains any of these (empty = all)
    std::string mmproj = "auto";      // "auto", "none", or a quant / file name fragment (see select_mmproj)
    HttpOptions http = http_options_from_env();
    ProgressFn on_progress = nullptr; // return false to cancel
    void *progress_user_data = nullptr;
    const std::atomic<bool> *cancel = nullptr;
};

Result<CatalogEntry> pull_model(std::string_view spec, const PullOptions &opts);
Result<std::vector<CatalogEntry>> list_models(const std::string &catalog_path);
Result<void> remove_model(std::string_view name, const std::string &catalog_path, bool delete_file = true);
```

```cpp
agent::hub::PullOptions opts;                 // opts.http is filled from the environment
opts.models_dir  = "/data/models";            // default: default_models_dir()
opts.on_progress = [](uint64_t done, uint64_t total, void *) { /* draw a bar */ return true; };
opts.cancel      = &cancel_flag;              // const std::atomic<bool>*

auto entry = agent::hub::pull_model("ggml-org/models:tinyllamas/stories260K.gguf", opts);
auto all   = agent::hub::list_models("/data/models/models.json");
agent::hub::remove_model("stories260K", "/data/models/models.json");   // also deletes the file
```

`pull_model` runs these steps:

1. Resolve the spec.
2. Apply the licence policy.
3. Download `<endpoint>/<repo>/resolve/<rev>/<file>` to `<models_dir>/<owner>/<repo>/<file>`.
4. Verify the size and sha256.
5. Add or update the entry in the catalogue.

A file that is already present with the right size (and sha256) is not downloaded again. A catalogue name that already points to a
different repo or file is an error. With `mmproj` set (default `"auto"`), the projector is pulled with the model and recorded in
`CatalogEntry::mmproj`.

`remove_model` deletes the catalogue entry, and the file unless `delete_file` is false. It returns `KeyNotFound` when the name is absent.

### Download

```cpp
struct DownloadOptions {
    uint64_t expected_size = 0;      // 0 = unknown
    std::string expected_sha256;     // empty = skip verification
    ProgressFn on_progress = nullptr;
    void *progress_user_data = nullptr;
    const std::atomic<bool> *cancel = nullptr;
};

Result<void> download_file(std::string_view url, const std::string &dest, const HttpOptions &http, const DownloadOptions &opts);
```

`download_file` is the primitive that `pull_model` uses:

- It follows redirects by hand, with a new client for each host. Hugging Face answers with relative redirects for renamed repos and with
  cross-host redirects to its CDN.
- It streams into `<dest>.part`. When a `.part` file exists, it resumes with `Range: bytes=N-`. A `200`, `416` or mismatched
  `Content-Range` reply restarts the download.
- It checks size and sha256 when they are known, then renames the file into place. On a sha256 mismatch it deletes the `.part` file.
- On cancel it returns `ErrorCode::Cancelled` and keeps the `.part` file.

`default_models_dir()` returns the first match below:

| Platform | Directory |
|---|---|
| Any, when set | `$AGENT_MODELS_DIR` |
| Windows | `%LOCALAPPDATA%\agent\models` |
| Linux and macOS | `$XDG_DATA_HOME/agent/models`, else `~/.local/share/agent/models` |

## models.json

```cpp
struct CatalogEntry {
    std::string name;          // logical name used by "agent model rm" and configs
    std::string repo;
    std::string file;          // path inside the repo
    std::string path;          // local absolute path
    uint64_t size = 0;
    std::string sha256;
    std::string license;
    std::string mmproj;        // local path of the projector pulled with the model; empty if none
    std::string mmproj_file;   // its path inside the repo
};

Result<std::vector<CatalogEntry>> load_catalog(const std::string &path);   // missing file = empty catalogue
Result<void> save_catalog(const std::string &path, std::span<const CatalogEntry> entries);
std::string catalog_to_json(std::span<const CatalogEntry> entries);
Result<std::vector<CatalogEntry>> catalog_from_json(std::string_view json);
std::string with_mmproj_option(std::string_view engine_options, std::string_view mmproj_path);
```

```json
{
  "models": [
    {"name": "stories260K", "repo": "ggml-org/models", "file": "tinyllamas/stories260K.gguf",
     "path": "/home/me/.local/share/agent/models/ggml-org/models/tinyllamas/stories260K.gguf",
     "size": 1185376, "sha256": "270cba1b...256d", "license": "unknown"}
  ]
}
```

`save_catalog` writes `<path>.tmp` and renames it, so the file is never half-written. The catalogue is parsed with simdjson.
`with_mmproj_option` adds `"mmproj": <path>` to a JSON options object, unless it already has one or `mmproj_path` is empty.

A JSON Schema for the file is in [`schemas/models.schema.json`](../../schemas/models.schema.json).

## Curated registry

```cpp
struct RegistryPart { std::string_view spec; std::string_view include; };
struct RegistryEntry {
    std::string_view alias;          // "qwen3-4b", "glm-ocr", "trellis-coarse", ...
    std::string_view task;           // text_gen, vision, ocr, embed, rerank, image_gen, asr, tts, vad, mesh_gen, ...
    std::string_view engine;         // llama, sdcpp, audiocpp, onnx, trellis
    std::span<const RegistryPart> parts;
    uint32_t size_mib;               // total download; 0 = not measured
    std::string_view engine_options; // JSON the engine needs for this model; "" = none
    std::string_view notes;
    bool tested;                     // pulled and run end to end with agent.cpp
};

std::span<const RegistryEntry> model_registry();
const RegistryEntry *find_registry_entry(std::string_view alias);
```

A registry entry can need several downloads, such as a speech model and its codec. The first part is the model. Pass each part's `spec`
to `pull_model`. When `include` is set, split it on commas into `PullOptions::include`.

```cpp
if (const auto *e = agent::hub::find_registry_entry("qwen3-4b")) {
    for (const auto &part : e->parts) {
        agent::hub::PullOptions opts;
        // opts.include = split(part.include) when part.include is not empty
        agent::hub::pull_model(part.spec, opts);
    }
}
```

## Licence policy

`pull_model` refuses models whose licence restricts use. It returns `ErrorCode::PermissionDenied`, and the message names the licence.
`is_restricted_license(license, license_name)` treats these as restricted:

- `cc-by-nc*`
- any id or `license_name` that contains `noncommercial`, `non-commercial`, `research`, `academic` or `-nc-`
- a few known ids: `deepfloyd-if-license`, `apple-amlr`, `fair-nc`

For `license: other`, only `license_name` can be checked. Llama and Gemma community licences, OpenRAIL and permissive licences are
allowed. An unknown licence is allowed and recorded as `"unknown"`.

Set `PullOptions::allow_restricted_license` (`--allow-restricted` in the CLI) only when you are entitled to use the model.

```cpp
bool restricted = agent::hub::is_restricted_license("cc-by-nc-4.0");   // true
```

## Hashing

```cpp
std::string sha256_hex(std::span<const uint8_t> data);
std::string sha256_hex(std::string_view data);
Result<std::string> sha256_file(const std::string &path);
```

## Network configuration

```cpp
struct ProxyAddress { std::string host; int port = 0; std::string user; std::string pass; };
Result<ProxyAddress> parse_proxy_url(std::string_view url);   // "http://user:pass@host:port/", "host:port"; port defaults to 80

struct HttpOptions {
    std::string endpoint = "https://huggingface.co";   // HF_ENDPOINT
    std::string token;                                  // see resolve_hf_token
    std::string proxy_host;                             // HTTPS_PROXY
    int proxy_port = 0;
    std::string proxy_user, proxy_pass;
    ProxyAddress http_proxy;                            // HTTP_PROXY, for plain http URLs
    std::string no_proxy;                               // NO_PROXY
    std::string ca_cert_path;                           // AGENT_CA_BUNDLE, SSL_CERT_FILE or CURL_CA_BUNDLE
    int connect_timeout_sec = 30;
    int read_timeout_sec = 60;
    int max_redirects = 10;
};

HttpOptions http_options_from_env();
```

`http_options_from_env()` reads these variables:

| Variable | Use |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | `Authorization: Bearer`. Falls back to the token file (`$HF_TOKEN_PATH`, `$HF_HOME/token`, `~/.cache/huggingface/token`). Set `HttpOptions::token` to override. Sent only to `huggingface.co`, `hf.co` and the `HF_ENDPOINT` host, never to CDN redirect hosts. Never logged. |
| `HF_ENDPOINT` | Mirror base URL (default `https://huggingface.co`) |
| `HTTPS_PROXY` / `https_proxy` | `http://[user:pass@]host:port`. HTTPS goes through `CONNECT`. Credentials are sent as basic proxy auth. |
| `HTTP_PROXY` / `http_proxy` | Same format, for plain `http://` URLs, such as an `http` mirror. |
| `NO_PROXY` / `no_proxy` | Comma-separated hosts or suffixes, matched on dot boundaries, or `*` |
| `AGENT_CA_BUNDLE`, `SSL_CERT_FILE`, `CURL_CA_BUNDLE` | CA bundle for TLS (first one set wins). Default: the system store. |

TLS verification is always on.

```cpp
std::string resolve_hf_token(std::string_view explicit_token, std::string_view env_hf_token, std::string_view env_legacy_token,
                             std::string_view token_file_contents);   // precedence: explicit, HF_TOKEN, legacy, token file
bool token_allowed_for_host(std::string_view host, std::string_view endpoint);
bool host_bypasses_proxy(std::string_view host, std::string_view no_proxy);   // "*", "example.com", ".example.com"
```

Gated repos: when the API answers 401 or 403, the error asks you to accept the licence on the model page and set `HF_TOKEN`. HTTP 407
means the proxy wants credentials.

### Lower-level HTTP

```cpp
using ProgressFn = bool (*)(uint64_t downloaded, uint64_t total, void *user_data);   // return false to cancel

Result<std::string> http_get_text(std::string_view url, const HttpOptions &http, size_t max_bytes = 4u << 20);
Result<std::string> http_get_range(std::string_view url, const HttpOptions &http, uint64_t offset, uint64_t length);
```

`http_get_text` returns the body of a small JSON or text document, for example from a third-party API. No Hugging Face token is sent.
Bodies over `max_bytes` fail instead of being cut. `http_get_range` fetches bytes `[offset, offset + length)`, for example the header of
a large GGUF file.

## Into agent sessions

Catalogue entries become model specs for the [model registry](models.md):

```cpp
agent::ModelSpec model_spec_from_entry(const CatalogEntry &entry, const Config &base);
std::vector<agent::ModelSpec> model_specs_from_catalog(std::span<const CatalogEntry> entries, const Config &base);
```

When `base.engine` is empty, the engine is chosen from the file type: `.gguf` uses llama, `.onnx` uses onnx, and `.safetensors`, `.ckpt`
and `.sft` use sdcpp. A missing file falls back to the catalogued size as the memory estimate. See [models.md](models.md#specs-from-modelsjson).
