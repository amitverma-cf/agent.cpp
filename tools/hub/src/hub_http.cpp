// Network half of the hub: HF API, streaming downloads with redirects, Range resume and verification.

#include <chrono>
#include <thread>
#include <agent-cpp/hub.hpp>

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#ifndef CPPHTTPLIB_NO_EXCEPTIONS
#define CPPHTTPLIB_NO_EXCEPTIONS
#endif
#include <httplib.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <system_error>

namespace agent::hub {

namespace {

constexpr const char *kUserAgent = "agent.cpp-hub/0.1";
constexpr size_t kMaxApiBody = 64u << 20;
constexpr size_t kMaxErrorBody = 4096;

struct Url {
    std::string scheme;
    std::string host;
    int port = 0;
    std::string target; // path + query
};

Result<Url> parse_url(std::string_view s) {
    Url u;
    const size_t sep = s.find("://");
    if (sep == std::string_view::npos) return fail<Url>(ErrorCode::InvalidConfig, "hub: not an absolute URL: " + std::string(s));
    u.scheme = std::string(s.substr(0, sep));
    for (char &c : u.scheme) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (u.scheme != "https" && u.scheme != "http") return fail<Url>(ErrorCode::InvalidConfig, "hub: unsupported URL scheme: " + u.scheme);
    s.remove_prefix(sep + 3);
    const size_t path_start = s.find_first_of("/?");
    std::string_view authority = s.substr(0, path_start);
    u.target = path_start == std::string_view::npos ? "/" : std::string(s.substr(path_start));
    if (u.target.front() == '?') u.target.insert(0, "/");
    if (const size_t at = authority.rfind('@'); at != std::string_view::npos) authority.remove_prefix(at + 1);
    u.port = u.scheme == "https" ? 443 : 80;
    const size_t colon = authority.rfind(':');
    const size_t bracket = authority.rfind(']');
    if (colon != std::string_view::npos && (bracket == std::string_view::npos || colon > bracket)) {
        int p = 0;
        for (char c : authority.substr(colon + 1)) {
            if (c < '0' || c > '9' || p > 65535) return fail<Url>(ErrorCode::InvalidConfig, "hub: bad port in URL");
            p = p * 10 + (c - '0');
        }
        if (p > 0 && p <= 65535) u.port = p;
        authority = authority.substr(0, colon);
    }
    u.host = std::string(authority);
    if (u.host.empty()) return fail<Url>(ErrorCode::InvalidConfig, "hub: URL without host");
    return ok(std::move(u));
}

// Resolves a Location header against the URL that produced it.
std::string resolve_location(const Url &base, const std::string &location) {
    if (location.find("://") != std::string::npos) return location;
    const std::string origin = base.scheme + "://" + base.host + ":" + std::to_string(base.port);
    if (location.rfind("//", 0) == 0) return base.scheme + ":" + location;
    if (!location.empty() && location.front() == '/') return origin + location;
    std::string dir = base.target.substr(0, base.target.find('?'));
    dir = dir.substr(0, dir.rfind('/') + 1);
    return origin + dir + location;
}

// Percent-encodes everything except RFC 3986 unreserved characters and '/'.
std::string encode_path(std::string_view path) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    for (const char ch : path) {
        const auto c = static_cast<unsigned char>(ch);
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') {
            out += ch;
        } else {
            out += '%';
            out += kHex[c >> 4];
            out += kHex[c & 0xf];
        }
    }
    return out;
}

bool is_redirect(int status) { return status == 301 || status == 302 || status == 303 || status == 307 || status == 308; }

// Per-request callbacks. Plain function pointers + user data: no std::function in our own API.
struct Sink {
    // Called once per final (non-redirect) response, before any body bytes. Return false to abort.
    bool (*on_response)(const httplib::Response &res, void *user) = nullptr;
    // Body bytes of the final response. Return false to abort.
    bool (*on_data)(const char *data, size_t len, void *user) = nullptr;
    void *user = nullptr;
};

struct Reply {
    int status = 0;
    std::string final_url;
    std::string error_body; // first bytes of a non-2xx body, for messages
    bool aborted = false;   // a sink callback returned false
};

// GET with manual redirect handling: HF answers with a relative redirect (renamed repos) or an
// absolute one to a CDN host, so every hop gets a fresh client for its host. The Authorization header
// is only sent to hosts accepted by token_allowed_for_host(); CDN URLs are pre-signed and get none.
Result<Reply> http_get(std::string url, const httplib::Headers &extra, const HttpOptions &http, const Sink &sink) {
    Reply reply;
    for (int hop = 0; hop <= http.max_redirects; ++hop) {
        auto parsed = parse_url(url);
        if (!parsed.ok) return fail<Reply>(parsed.error.code, parsed.error.message);
        const Url &u = parsed.value;

        httplib::Client cli(u.scheme + "://" + u.host + ":" + std::to_string(u.port));
        if (!cli.is_valid()) return fail<Reply>(ErrorCode::NetworkError, "hub: cannot create HTTP client for " + u.host);
        cli.set_follow_location(false);
        cli.set_connection_timeout(http.connect_timeout_sec, 0);
        cli.set_read_timeout(http.read_timeout_sec, 0);
        if (!host_bypasses_proxy(u.host, http.no_proxy)) {
            if (u.scheme == "https" && !http.proxy_host.empty()) {
                cli.set_proxy(http.proxy_host, http.proxy_port);
                if (!http.proxy_user.empty()) cli.set_proxy_basic_auth(http.proxy_user, http.proxy_pass);
            } else if (u.scheme == "http" && !http.http_proxy.host.empty()) {
                cli.set_proxy(http.http_proxy.host, http.http_proxy.port);
                if (!http.http_proxy.user.empty()) cli.set_proxy_basic_auth(http.http_proxy.user, http.http_proxy.pass);
            }
        }
        if (u.scheme == "https") {
            cli.enable_server_certificate_verification(true);
            if (!http.ca_cert_path.empty()) cli.set_ca_cert_path(http.ca_cert_path);
        }

        httplib::Headers headers = extra;
        headers.emplace("User-Agent", kUserAgent);
        if (!http.token.empty() && token_allowed_for_host(u.host, http.endpoint)) headers.emplace("Authorization", "Bearer " + http.token);

        int status = 0;
        std::string location;
        bool final_ok = false;
        reply.error_body.clear();
        auto res = cli.Get(
            u.target, headers,
            [&](const httplib::Response &r) {
                status = r.status;
                if (is_redirect(status)) {
                    location = r.get_header_value("Location");
                    return true;
                }
                final_ok = status >= 200 && status < 300;
                if (sink.on_response && !sink.on_response(r, sink.user)) {
                    reply.aborted = true;
                    return false;
                }
                return true;
            },
            [&](const char *data, size_t len) {
                if (is_redirect(status)) return true;
                if (!final_ok) {
                    const size_t room = kMaxErrorBody - std::min(kMaxErrorBody, reply.error_body.size());
                    reply.error_body.append(data, std::min(room, len));
                    return true;
                }
                if (sink.on_data && !sink.on_data(data, len, sink.user)) {
                    reply.aborted = true;
                    return false;
                }
                return true;
            });

        if (reply.aborted) {
            reply.status = status;
            reply.final_url = url;
            return ok(std::move(reply));
        }
        if (!res) {
            std::string msg = "hub: request to " + u.host + " failed: " + httplib::to_string(res.error());
            if (res.error() == httplib::Error::SSLServerVerification)
                msg += " (set AGENT_CA_BUNDLE / SSL_CERT_FILE to a CA bundle that signs this host or proxy)";
            if (res.error() == httplib::Error::ProxyConnection) msg += " (proxy " + http.proxy_host + ":" + std::to_string(http.proxy_port) + ")";
            return fail<Reply>(ErrorCode::NetworkError, msg);
        }
        if (is_redirect(status)) {
            if (location.empty()) return fail<Reply>(ErrorCode::HttpError, "hub: redirect without Location from " + u.host);
            url = resolve_location(u, location);
            continue;
        }
        reply.status = status;
        reply.final_url = url;
        return ok(std::move(reply));
    }
    return fail<Reply>(ErrorCode::HttpError, "hub: too many redirects");
}

std::string status_message(const Reply &r, std::string_view what) {
    std::string msg = "hub: " + std::string(what) + " returned HTTP " + std::to_string(r.status);
    if (r.status == 401 || r.status == 403)
        msg += " (gated or private repo: accept the licence on the model page on huggingface.co, then set HF_TOKEN to a token "
               "with read access, or run `huggingface-cli login`)";
    if (r.status == 404) msg += " (not found)";
    if (r.status == 407) msg += " (proxy authentication required: put user:pass@ in HTTPS_PROXY)";
    if (!r.error_body.empty()) msg += ": " + r.error_body.substr(0, 300);
    return msg;
}

// -------------------------- download state --------------------------

struct DownloadState {
    std::string part_path;
    std::FILE *file = nullptr;
    uint64_t offset = 0;       // bytes already in the .part file when the request was sent
    uint64_t written = 0;      // bytes written in this request
    uint64_t total = 0;
    bool restarted = false;    // server ignored Range (200) and we truncated
    bool range_mismatch = false;
    bool write_failed = false;
    bool cancelled = false;
    const DownloadOptions *opts = nullptr;
};

uint64_t parse_u64(std::string_view s) {
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') break;
        v = v * 10 + static_cast<uint64_t>(c - '0');
    }
    return v;
}

bool download_on_response(const httplib::Response &res, void *user) {
    auto *st = static_cast<DownloadState *>(user);
    if (res.status == 206) {
        // Content-Range: bytes <start>-<end>/<total>
        const std::string cr = res.get_header_value("Content-Range");
        const size_t sp = cr.find(' ');
        const uint64_t start = sp == std::string::npos ? 0 : parse_u64(std::string_view(cr).substr(sp + 1));
        if (start != st->offset) {
            st->range_mismatch = true;
            return false;
        }
        const size_t slash = cr.find('/');
        if (slash != std::string::npos && cr.compare(slash + 1, 1, "*") != 0) st->total = parse_u64(std::string_view(cr).substr(slash + 1));
        st->file = std::fopen(st->part_path.c_str(), "ab");
    } else if (res.status == 200) {
        if (st->offset != 0) st->restarted = true;
        st->offset = 0;
        st->file = std::fopen(st->part_path.c_str(), "wb");
        if (res.has_header("Content-Length")) st->total = parse_u64(res.get_header_value("Content-Length"));
    } else {
        return true; // error status: http_get collects the body for the message
    }
    if (st->opts->expected_size) st->total = st->opts->expected_size;
    if (!st->file) {
        st->write_failed = true;
        return false;
    }
    return true;
}

bool download_on_data(const char *data, size_t len, void *user) {
    auto *st = static_cast<DownloadState *>(user);
    if (st->opts->cancel && st->opts->cancel->load(std::memory_order_relaxed)) {
        st->cancelled = true;
        return false;
    }
    if (std::fwrite(data, 1, len, st->file) != len) {
        st->write_failed = true;
        return false;
    }
    st->written += len;
    if (st->opts->on_progress && !st->opts->on_progress(st->offset + st->written, st->total, st->opts->progress_user_data)) {
        st->cancelled = true;
        return false;
    }
    return true;
}

uint64_t file_size_or_zero(const std::string &path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(sz);
}

Result<void> finalize_download(const std::string &part, const std::string &dest, const DownloadOptions &opts) {
    std::error_code ec;
    const uint64_t size = file_size_or_zero(part);
    if (opts.expected_size && size != opts.expected_size) {
        return fail(ErrorCode::NetworkError,
                    "hub: size mismatch for " + dest + ": got " + std::to_string(size) + ", expected " + std::to_string(opts.expected_size));
    }
    if (!opts.expected_sha256.empty()) {
        auto digest = sha256_file(part);
        if (!digest.ok) return fail(digest.error.code, digest.error.message);
        if (digest.value != opts.expected_sha256) {
            std::filesystem::remove(part, ec);
            return fail(ErrorCode::NetworkError, "hub: sha256 mismatch for " + dest + ": got " + digest.value + ", expected " + opts.expected_sha256);
        }
    }
    std::filesystem::rename(part, dest, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "hub: cannot move " + part + " to " + dest + ": " + ec.message());
    return ok();
}

} // namespace

// -------------------------- public API --------------------------

Result<RepoInfo> fetch_repo_info(std::string_view repo, const HttpOptions &http, std::string_view revision) {
    std::string url = http.endpoint + "/api/models/" + encode_path(repo);
    if (!revision.empty() && revision != "main") url += "/revision/" + encode_path(revision);
    url += "?blobs=true";

    struct Body {
        std::string text;
    } body;
    Sink sink;
    sink.user = &body;
    sink.on_data = [](const char *data, size_t len, void *user) {
        auto *b = static_cast<Body *>(user);
        if (b->text.size() + len > kMaxApiBody) return false;
        b->text.append(data, len);
        return true;
    };
    auto reply = http_get(url, {}, http, sink);
    if (!reply.ok) return fail<RepoInfo>(reply.error.code, reply.error.message);
    if (reply.value.aborted) return fail<RepoInfo>(ErrorCode::HttpError, "hub: model info for " + std::string(repo) + " is too large");
    if (reply.value.status != 200) return fail<RepoInfo>(ErrorCode::HttpError, status_message(reply.value, "model info for " + std::string(repo)));
    auto info = parse_repo_info(body.text);
    if (info.ok && info.value.repo.empty()) info.value.repo = std::string(repo);
    return info;
}

Result<std::string> http_get_text(std::string_view url, const HttpOptions &http, size_t max_bytes) {
    struct Body {
        std::string text;
        size_t max = 0;
    } body;
    body.max = max_bytes;
    Sink sink;
    sink.user = &body;
    sink.on_data = [](const char *data, size_t len, void *user) {
        auto *b = static_cast<Body *>(user);
        if (b->text.size() + len > b->max) return false;
        b->text.append(data, len);
        return true;
    };
    httplib::Headers headers{{"User-Agent", "agent-cpp"}, {"Accept", "application/json, text/plain, */*"}};
    // A release's CDN can answer 400 / 5xx for the first minutes after it was published, and any network can drop a request: try
    // up to four times, waiting 1, 3 and 8 seconds, before giving up.
    static constexpr int kWaitSeconds[] = {1, 3, 8};
    Result<Reply> reply = fail<Reply>(ErrorCode::NetworkError, "no attempt");
    for (int attempt = 0;; ++attempt) {
        body.text.clear();
        reply = http_get(std::string(url), headers, http, sink);
        const bool transient = !reply.ok || reply.value.status == 400 || reply.value.status == 429 || reply.value.status >= 500;
        if (!transient || attempt >= 3) break;
        std::this_thread::sleep_for(std::chrono::seconds(kWaitSeconds[attempt]));
    }
    if (!reply.ok) return fail<std::string>(reply.error.code, reply.error.message);
    if (reply.value.aborted) return fail<std::string>(ErrorCode::HttpError, "hub: the response from " + std::string(url) + " is larger than " + std::to_string(max_bytes) + " bytes");
    if (reply.value.status != 200) return fail<std::string>(ErrorCode::HttpError, status_message(reply.value, std::string(url)));
    return ok(std::move(body.text));
}

Result<void> download_file(std::string_view url, const std::string &dest, const HttpOptions &http, const DownloadOptions &opts) {
    const std::string part = dest + ".part";
    std::error_code ec;
    const std::filesystem::path dest_path(dest);
    if (dest_path.has_parent_path()) std::filesystem::create_directories(dest_path.parent_path(), ec);

    for (int attempt = 0; attempt < 2; ++attempt) {
        DownloadState st;
        st.part_path = part;
        st.opts = &opts;
        st.offset = file_size_or_zero(part);
        if (opts.expected_size && st.offset > opts.expected_size) {
            std::filesystem::remove(part, ec);
            st.offset = 0;
        }
        if (opts.expected_size && st.offset == opts.expected_size) {
            auto fin = finalize_download(part, dest, opts);
            if (fin.ok) return fin;
            std::filesystem::remove(part, ec); // corrupt complete .part: start over
            st.offset = 0;
        }

        httplib::Headers headers;
        if (st.offset > 0) headers.emplace("Range", "bytes=" + std::to_string(st.offset) + "-");
        Sink sink{.on_response = download_on_response, .on_data = download_on_data, .user = &st};
        auto reply = http_get(std::string(url), headers, http, sink);
        if (st.file) std::fclose(st.file);

        if (!reply.ok) return fail(reply.error.code, reply.error.message + " (partial data kept in " + part + ")");
        if (st.cancelled) return fail(ErrorCode::Cancelled, "hub: download cancelled; partial data kept in " + part);
        if (st.write_failed) return fail(ErrorCode::FilesystemError, "hub: cannot write " + part);
        if (st.range_mismatch || reply.value.status == 416) {
            std::filesystem::remove(part, ec); // server disagrees about the partial file: restart from zero
            continue;
        }
        if (reply.value.status != 200 && reply.value.status != 206) return fail(ErrorCode::HttpError, status_message(reply.value, "download"));
        if (st.total && st.offset + st.written < st.total)
            return fail(ErrorCode::NetworkError, "hub: connection closed early (" + std::to_string(st.offset + st.written) + " of " +
                                                     std::to_string(st.total) + " bytes); run again to resume");
        return finalize_download(part, dest, opts);
    }
    return fail(ErrorCode::HttpError, "hub: server rejected resume twice for " + std::string(url));
}

namespace {

// "owner/repo:*": every file of the repo (a speech or 3D model is a folder of files, not one GGUF), narrowed
// by PullOptions::include (a file is kept when its path contains any of the strings). The catalogue entry
// points at the folder.
Result<CatalogEntry> pull_whole_repo(const ModelSpec &spec, const RepoInfo &ri, const PullOptions &opts) {
    std::string license = ri.license.empty() ? std::string("unknown") : ri.license;
    if (ri.license == "other" && !ri.license_name.empty()) license += " (" + ri.license_name + ")";
    if (is_restricted_license(ri.license, ri.license_name) && !opts.allow_restricted_license)
        return fail<CatalogEntry>(ErrorCode::PermissionDenied, "hub: " + spec.repo + " is under a restricted licence (" + license +
                                                                   "); pass allow_restricted_license (--allow-restricted) if you are entitled to use it");

    std::vector<const RemoteFile *> chosen;
    for (const RemoteFile &f : ri.files) {
        if (f.path == ".gitattributes" || f.path == "README.md" || f.path.rfind(".", 0) == 0) continue;
        bool keep = opts.include.empty();
        for (const std::string &needle : opts.include)
            if (!needle.empty() && f.path.find(needle) != std::string::npos) keep = true;
        if (keep) chosen.push_back(&f);
    }
    if (chosen.empty()) return fail<CatalogEntry>(ErrorCode::KeyNotFound, "hub: no files in " + spec.repo + " match the include filter");

    const std::string models_dir = opts.models_dir.empty() ? default_models_dir() : opts.models_dir;
    const std::string catalog_path = opts.catalog_path.empty() ? (std::filesystem::path(models_dir) / "models.json").string() : opts.catalog_path;
    std::error_code ec;
    const auto repo_dir = std::filesystem::absolute(std::filesystem::path(models_dir) / std::filesystem::path(spec.repo), ec).lexically_normal();

    uint64_t total = 0;
    for (const RemoteFile *f : chosen) {
        const std::string dest = std::filesystem::absolute(repo_dir / f->path, ec).lexically_normal().string();
        if (!(std::filesystem::is_regular_file(dest, ec) && (f->size == 0 || file_size_or_zero(dest) == f->size))) {
            const std::string url = opts.http.endpoint + "/" + encode_path(spec.repo) + "/resolve/" + encode_path(spec.revision) + "/" + encode_path(f->path);
            DownloadOptions dl{.expected_size = f->size,
                               .expected_sha256 = f->sha256,
                               .on_progress = opts.on_progress,
                               .progress_user_data = opts.progress_user_data,
                               .cancel = opts.cancel};
            auto r = download_file(url, dest, opts.http, dl);
            if (!r.ok) return fail<CatalogEntry>(r.error.code, r.error.message);
        }
        total += f->size ? f->size : file_size_or_zero(dest);
    }

    std::string name = opts.name;
    if (name.empty()) name = spec.repo.substr(spec.repo.find('/') + 1);
    CatalogEntry entry{.name = name, .repo = spec.repo, .file = "*", .path = repo_dir.string(), .size = total, .license = license};
    auto cat = load_catalog(catalog_path);
    if (!cat.ok) return fail<CatalogEntry>(cat.error.code, cat.error.message);
    auto existing = std::find_if(cat.value.begin(), cat.value.end(), [&](const CatalogEntry &e) { return e.name == name; });
    if (existing != cat.value.end()) *existing = entry;
    else cat.value.push_back(entry);
    auto saved = save_catalog(catalog_path, cat.value);
    if (!saved.ok) return fail<CatalogEntry>(saved.error.code, saved.error.message);
    return ok(std::move(entry));
}

} // namespace

Result<CatalogEntry> pull_model(std::string_view spec_text, const PullOptions &opts) {
    auto spec = parse_spec(spec_text);
    if (!spec.ok) return fail<CatalogEntry>(spec.error.code, spec.error.message);

    auto info = fetch_repo_info(spec.value.repo, opts.http, spec.value.revision);
    if (!info.ok) return fail<CatalogEntry>(info.error.code, info.error.message);
    if (spec.value.all) return pull_whole_repo(spec.value, info.value, opts);

    auto file = select_file(info.value, spec.value);
    if (!file.ok) return fail<CatalogEntry>(file.error.code, file.error.message);
    // A split GGUF downloads every shard next to the first; the catalogue points at shard 1, which
    // is what llama.cpp loads (it finds the rest by name).
    auto shards = split_set(info.value, file.value);
    if (!shards.ok) return fail<CatalogEntry>(shards.error.code, shards.error.message);

    const RepoInfo &ri = info.value;
    std::string license = ri.license.empty() ? std::string("unknown") : ri.license;
    if (ri.license == "other" && !ri.license_name.empty()) license += " (" + ri.license_name + ")";
    if (is_restricted_license(ri.license, ri.license_name) && !opts.allow_restricted_license)
        return fail<CatalogEntry>(ErrorCode::PermissionDenied, "hub: " + spec.value.repo + " is under a restricted licence (" + license +
                                                                   "); pass allow_restricted_license (--allow-restricted) if you are entitled to use it");

    const std::string models_dir = opts.models_dir.empty() ? default_models_dir() : opts.models_dir;
    const std::string catalog_path = opts.catalog_path.empty() ? (std::filesystem::path(models_dir) / "models.json").string() : opts.catalog_path;
    const std::string repo_dir_name = spec.value.repo; // "owner/name" -> <models_dir>/owner/name
    std::filesystem::path dest = std::filesystem::path(models_dir) / std::filesystem::path(repo_dir_name) / std::filesystem::path(file.value.path);
    std::error_code ec;
    dest = std::filesystem::absolute(dest, ec).lexically_normal();

    std::string name = opts.name;
    if (name.empty()) name = std::filesystem::path(file.value.path).stem().string();

    auto cat = load_catalog(catalog_path);
    if (!cat.ok) return fail<CatalogEntry>(cat.error.code, cat.error.message);
    auto existing = std::find_if(cat.value.begin(), cat.value.end(), [&](const CatalogEntry &e) { return e.name == name; });
    if (existing != cat.value.end() && (existing->repo != spec.value.repo || existing->file != file.value.path))
        return fail<CatalogEntry>(ErrorCode::InvalidConfig, "hub: name '" + name + "' is already used by " + existing->repo + ":" + existing->file +
                                                                "; choose another name or remove it first");

    CatalogEntry entry{.name = name,
                       .repo = spec.value.repo,
                       .file = file.value.path,
                       .path = dest.string(),
                       .size = file.value.size,
                       .sha256 = file.value.sha256,
                       .license = license};

    const auto repo_dir = std::filesystem::path(models_dir) / std::filesystem::path(repo_dir_name);
    uint64_t total_size = 0;
    for (const RemoteFile &shard : shards.value) {
        const std::string shard_path = (shard.path == file.value.path) ? entry.path : std::filesystem::absolute(repo_dir / shard.path, ec).lexically_normal().string();
        bool have = false;
        if (std::filesystem::is_regular_file(shard_path, ec) && (shard.size == 0 || file_size_or_zero(shard_path) == shard.size)) {
            have = true;
            const bool known_good = shard.path == file.value.path && existing != cat.value.end() && existing->sha256 == shard.sha256;
            if (!shard.sha256.empty() && !known_good) {
                auto digest = sha256_file(shard_path);
                have = digest.ok && digest.value == shard.sha256;
            }
        }
        if (!have) {
            const std::string url = opts.http.endpoint + "/" + encode_path(spec.value.repo) + "/resolve/" + encode_path(spec.value.revision) + "/" +
                                    encode_path(shard.path);
            DownloadOptions dl{.expected_size = shard.size,
                               .expected_sha256 = shard.sha256,
                               .on_progress = opts.on_progress,
                               .progress_user_data = opts.progress_user_data,
                               .cancel = opts.cancel};
            auto r = download_file(url, shard_path, opts.http, dl);
            if (!r.ok) return fail<CatalogEntry>(r.error.code, r.error.message);
        }
        total_size += shard.size ? shard.size : file_size_or_zero(shard_path);
    }
    // The multimodal projector of a vision / audio / OCR model lives in the same repo; fetch it too so
    // the model works without a second command.
    auto proj = select_mmproj(ri, file.value.path, opts.mmproj);
    if (!proj.ok) return fail<CatalogEntry>(proj.error.code, proj.error.message);
    if (proj.value) {
        const RemoteFile &p = *proj.value;
        const std::string proj_path = std::filesystem::absolute(repo_dir / p.path, ec).lexically_normal().string();
        if (!(std::filesystem::is_regular_file(proj_path, ec) && (p.size == 0 || file_size_or_zero(proj_path) == p.size))) {
            const std::string url = opts.http.endpoint + "/" + encode_path(spec.value.repo) + "/resolve/" + encode_path(spec.value.revision) + "/" +
                                    encode_path(p.path);
            DownloadOptions dl{.expected_size = p.size,
                               .expected_sha256 = p.sha256,
                               .on_progress = opts.on_progress,
                               .progress_user_data = opts.progress_user_data,
                               .cancel = opts.cancel};
            auto r = download_file(url, proj_path, opts.http, dl);
            if (!r.ok) return fail<CatalogEntry>(r.error.code, r.error.message);
        }
        entry.mmproj = proj_path;
        entry.mmproj_file = p.path;
    }
    if (shards.value.size() > 1) entry.size = total_size;
    if (entry.size == 0) entry.size = file_size_or_zero(entry.path);

    if (existing != cat.value.end()) *existing = entry;
    else cat.value.push_back(entry);
    auto saved = save_catalog(catalog_path, cat.value);
    if (!saved.ok) return fail<CatalogEntry>(saved.error.code, saved.error.message);
    return ok(std::move(entry));
}

} // namespace agent::hub
