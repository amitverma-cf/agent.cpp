# Model hub

`#include <agent-cpp/hub.hpp>` · link `agent_hub`

The model hub downloads models from Hugging Face and keeps a local catalogue (`models.json`).
It belongs to the **CLI layer**: `libagent` has no network dependency and does not link it.
`agent_hub` uses the vendored cpp-httplib (built with `CPPHTTPLIB_NO_EXCEPTIONS`) and OpenSSL
(HTTPS and SHA-256), so building it requires the OpenSSL development package
(`libssl-dev` on Debian/Ubuntu).

All functions return `Result<T>`; nothing throws.

## Command-line example

The `agent` CLI wraps the hub as `agent model pull|ls|rm` (see [../cli.md](../cli.md)). The
standalone `agent_pull` example works the same way:

```sh
agent_pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M            # pick the file by quant tag
agent_pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf
agent_pull ggml-org/models:tinyllamas/stories260K.gguf --dir ./models --name tiny
agent_pull --list [--dir DIR]
agent_pull --rm tiny [--dir DIR] [--keep-file]
```

Ctrl-C cancels a download and keeps `<file>.part`. Running the same command again resumes it.
Restricted licences need `--allow-restricted` (see below).

## Specs

```
owner/repo[@revision][:selector]
```

- `selector` containing `.` or `/` is a **file path** inside the repo. A unique basename also works.
- Any other `selector` is a **quant tag** such as `Q4_K_M` or `IQ2_XXS`. The hub matches it
  case-insensitively against `.gguf` basenames. An exact tag beats a longer one (`Q4_K` picks
  `*-q4_k.gguf` over `*-q4_k_m.gguf`). `mmproj` files are ignored. For a split GGUF
  (`-00001-of-0000N`), the first shard is selected and `pull_model` downloads every shard, each
  size- and sha256-checked, into the same folder. The catalogue points at shard 1, which llama.cpp
  loads together with the rest, and `remove_model` deletes all shards. This is unit-tested, but not
  yet against a real split repo.
- No selector: the repo must contain exactly one `.gguf`.
- The prefixes `https://huggingface.co/`, `hf.co/` and `hf://` are stripped.

```cpp
auto spec = agent::hub::parse_spec("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M");
auto info = agent::hub::fetch_repo_info(spec.value.repo, agent::hub::http_options_from_env());
auto file = agent::hub::select_file(info.value, spec.value);   // path, size, sha256
```

`fetch_repo_info` calls `GET /api/models/<repo>?blobs=true` (or `/revision/<rev>` for another
revision). The response gives every file's size and LFS sha256, plus the licence. The hub reads
the licence from `cardData.license`, then `cardData.license_name`, then a `license:<id>` tag.
`parse_repo_info` is the pure parser and is used by the unit tests.

## Pull, list, remove

```cpp
agent::hub::PullOptions opts;               // opts.http is filled from the environment
opts.models_dir  = "/data/models";          // default: default_models_dir()
opts.on_progress = [](uint64_t done, uint64_t total, void *) { /* ... */ return true; };
opts.cancel      = &cancel_flag;            // const std::atomic<bool>*

auto entry = agent::hub::pull_model("ggml-org/models:tinyllamas/stories260K.gguf", opts);
auto all   = agent::hub::list_models(catalog_path);
agent::hub::remove_model("stories260K", catalog_path);           // also deletes the file
```

`pull_model` runs these steps:

1. Resolve the spec.
2. Apply the licence policy.
3. Download `<endpoint>/<repo>/resolve/<rev>/<file>` to `<models_dir>/<owner>/<repo>/<file>`.
4. Verify the file.
5. Upsert the entry into the catalogue.

If the file is already present with the right size (and sha256), it is not downloaded again.
A catalogue name that already points to a different repo or file is an error.

`download_file(url, dest, http, opts)` is the lower-level primitive:

- It follows redirects by hand, creating a new client for each host. HF answers with relative
  redirects (renamed repos) and cross-host redirects to its CDN.
- It streams into `<dest>.part`. If a `.part` file exists, it resumes with `Range: bytes=N-`. A
  `200`, `416` or mismatched `Content-Range` reply restarts the download cleanly.
- It checks the size and sha256 (OpenSSL EVP) when they are known, then renames the file into
  place. On a sha256 mismatch it deletes the `.part` file.
- On cancel it returns `ErrorCode::Cancelled` and keeps the `.part` file.

`default_models_dir()` returns the first match below:

| Platform | Directory |
|---|---|
| Any, when set | `$AGENT_MODELS_DIR` |
| Windows | `%LOCALAPPDATA%\agent\models` |
| Linux | `$XDG_DATA_HOME/agent/models`, else `~/.local/share/agent/models` |

The catalogue path defaults to `<models_dir>/models.json`.

## models.json

```json
{
  "models": [
    {"name": "stories260K", "repo": "ggml-org/models", "file": "tinyllamas/stories260K.gguf",
     "path": "/home/me/.local/share/agent/models/ggml-org/models/tinyllamas/stories260K.gguf",
     "size": 1185376, "sha256": "270cba1b...256d", "license": "unknown"}
  ]
}
```

The catalogue is written atomically (`.tmp` file, then rename) and parsed with simdjson.
`catalog_to_json` and `catalog_from_json` are exposed for tools.

## Licence policy

ROADMAP §1.1 applies: `pull_model` refuses models whose licence restricts use and returns
`ErrorCode::PermissionDenied`. The error message names the licence. `is_restricted_license(license, license_name)`
treats these as restricted:

- `cc-by-nc*`
- any id or `license_name` containing `noncommercial`, `non-commercial`, `research`, `academic`
  or `-nc-`
- a few known ids (`deepfloyd-if-license`, `apple-amlr`, `fair-nc`)

For `license: other`, only `license_name` can be checked. Llama- and Gemma-style community
licences, OpenRAIL and permissive licences are allowed. An unknown licence is allowed and
recorded as `"unknown"`.

Set `PullOptions::allow_restricted_license` (`--allow-restricted`) only if you are entitled to use
the model.

## Network configuration

`http_options_from_env()` reads:

| Variable | Use |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | `Authorization: Bearer`. Falls back to the token file (`$HF_TOKEN_PATH`, `$HF_HOME/token`, `~/.cache/huggingface/token`). Set `HttpOptions::token` to override. Sent only to `huggingface.co` / `hf.co` (or the `HF_ENDPOINT` host), never to CDN redirect hosts. Never logged. |
| `HF_ENDPOINT` | Mirror base URL (default `https://huggingface.co`) |
| `HTTPS_PROXY` / `https_proxy` | `http://[user:pass@]host:port`. HTTPS goes through `CONNECT`. `user:pass` (percent-decoded) is sent as basic proxy auth. |
| `HTTP_PROXY` / `http_proxy` | Same format, used for plain `http://` URLs (e.g. an `http` `HF_ENDPOINT` mirror). |
| `NO_PROXY` / `no_proxy` | Comma-separated hosts or suffixes (matched on dot boundaries), or `*` |
| `AGENT_CA_BUNDLE`, `SSL_CERT_FILE`, `CURL_CA_BUNDLE` | CA bundle for TLS verification (first one set wins). Default: OpenSSL's system store. |

TLS verification is always on.

Gated repos: if the API answers 401/403, the error asks you to accept the licence on the model page and set `HF_TOKEN`. HTTP 407 means the proxy wants credentials.
