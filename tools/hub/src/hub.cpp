// Offline half of the hub: spec parsing, file selection, licence policy, sha256, models.json.
// The network half lives in hub_http.cpp so that this file stays cheap to compile and test.

#include <agent-cpp/hub.hpp>

#include <openssl/evp.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <simdjson.h>
#include <system_error>

namespace agent::hub {

namespace {

char lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

std::string to_lower(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = lower(c);
    return out;
}

bool is_alnum(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }

// Case-insensitive search; returns npos when absent.
size_t ifind(std::string_view hay, std::string_view needle, size_t from = 0) {
    if (needle.empty() || needle.size() > hay.size()) return std::string_view::npos;
    for (size_t i = from; i + needle.size() <= hay.size(); ++i) {
        size_t j = 0;
        while (j < needle.size() && lower(hay[i + j]) == lower(needle[j])) ++j;
        if (j == needle.size()) return i;
    }
    return std::string_view::npos;
}

bool ends_with_ci(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && ifind(s.substr(s.size() - suffix.size()), suffix) == 0;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

std::string_view basename_of(std::string_view path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

bool valid_repo_part(std::string_view part) {
    if (part.empty() || part == "." || part == "..") return false;
    return std::all_of(part.begin(), part.end(), [](char c) { return is_alnum(c) || c == '-' || c == '_' || c == '.'; });
}

// A repo-relative file path that cannot escape the destination directory.
bool valid_repo_file(std::string_view file) {
    if (file.empty() || file.front() == '/' || file.find('\\') != std::string_view::npos || file.find(':') != std::string_view::npos)
        return false;
    size_t start = 0;
    while (start <= file.size()) {
        size_t end = file.find('/', start);
        if (end == std::string_view::npos) end = file.size();
        const std::string_view seg = file.substr(start, end - start);
        if (seg.empty() || seg == "." || seg == "..") return false;
        start = end + 1;
    }
    return true;
}

bool is_gguf(std::string_view path) { return ends_with_ci(path, ".gguf"); }
bool is_mmproj(std::string_view path) { return ifind(basename_of(path), "mmproj") != std::string_view::npos; }
// "model-00001-of-00003.gguf"
bool is_split_shard(std::string_view path) { return ifind(basename_of(path), "-of-0") != std::string_view::npos; }

// "<prefix>-00002-of-00005.gguf" -> prefix, 2, 5. false when path isn't a well-formed shard name.
bool parse_shard(std::string_view path, std::string_view &prefix, int &index, int &total) {
    constexpr std::string_view kExt = ".gguf";
    if (path.size() < kExt.size() + 15 || !ends_with_ci(path, kExt)) return false;
    const std::string_view stem = path.substr(0, path.size() - kExt.size());
    // stem ends with "-NNNNN-of-NNNNN" (5 digits each).
    const std::string_view tail = stem.substr(stem.size() - 15);
    if (tail[0] != '-' || tail.substr(6, 4) != "-of-") return false;
    auto digits = [](std::string_view d, int &out) {
        out = 0;
        for (char c : d) {
            if (c < '0' || c > '9') return false;
            out = out * 10 + (c - '0');
        }
        return true;
    };
    if (!digits(tail.substr(1, 5), index) || !digits(tail.substr(10, 5), total) || index < 1 || total < 1 || index > total) return false;
    prefix = stem.substr(0, stem.size() - 15);
    return true;
}

// Quant match whose boundaries are not part of a longer tag: "Q4_K" must not pick "Q4_K_M".
bool quant_matches_exact(std::string_view filename, std::string_view quant) {
    const std::string_view name = basename_of(filename);
    for (size_t pos = ifind(name, quant); pos != std::string_view::npos; pos = ifind(name, quant, pos + 1)) {
        const bool left_ok = pos == 0 || !is_alnum(name[pos - 1]);
        const size_t after = pos + quant.size();
        const bool right_ok = after == name.size() || name[after] == '.' || name[after] == '-';
        if (left_ok && right_ok) return true;
    }
    return false;
}

std::string join_paths(const std::vector<const RemoteFile *> &files) {
    std::string out;
    for (const RemoteFile *f : files) {
        if (!out.empty()) out += ", ";
        out += f->path;
    }
    return out;
}

void append_json_string(std::string &out, std::string_view s) {
    out += '"';
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            } else {
                out += ch;
            }
        }
    }
    out += '"';
}

std::string get_string_field(simdjson::dom::element obj, std::string_view key) {
    std::string_view v;
    if (obj[key].get(v)) return {};
    return std::string(v);
}

uint64_t get_uint_field(simdjson::dom::element obj, std::string_view key) {
    uint64_t v = 0;
    if (obj[key].get(v)) return 0;
    return v;
}

} // namespace

// -------------------------- Spec resolution --------------------------

Result<ModelSpec> parse_spec(std::string_view spec) {
    spec = trim(spec);
    for (std::string_view prefix : {"https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/", "hf://"})
        if (spec.substr(0, prefix.size()) == prefix) spec.remove_prefix(prefix.size());
    if (spec.empty()) return fail<ModelSpec>(ErrorCode::InvalidConfig, "empty model spec");

    ModelSpec out;
    std::string_view repo = spec;
    std::string_view selector;
    if (const size_t colon = spec.find(':'); colon != std::string_view::npos) {
        repo = spec.substr(0, colon);
        selector = trim(spec.substr(colon + 1));
        if (selector.empty()) return fail<ModelSpec>(ErrorCode::InvalidConfig, "model spec has an empty selector after ':'");
    }
    if (const size_t at = repo.find('@'); at != std::string_view::npos) {
        const std::string_view rev = repo.substr(at + 1);
        repo = repo.substr(0, at);
        if (rev.empty() || !std::all_of(rev.begin(), rev.end(), [](char c) { return is_alnum(c) || c == '-' || c == '_' || c == '.'; }))
            return fail<ModelSpec>(ErrorCode::InvalidConfig, "invalid revision in model spec");
        out.revision = std::string(rev);
    }
    while (!repo.empty() && repo.back() == '/') repo.remove_suffix(1);
    const size_t slash = repo.find('/');
    if (slash == std::string_view::npos || repo.find('/', slash + 1) != std::string_view::npos || !valid_repo_part(repo.substr(0, slash)) ||
        !valid_repo_part(repo.substr(slash + 1)))
        return fail<ModelSpec>(ErrorCode::InvalidConfig, "model spec must look like owner/repo[:file-or-quant], got '" + std::string(spec) + "'");
    out.repo = std::string(repo);

    if (!selector.empty()) {
        if (selector.find('.') != std::string_view::npos || selector.find('/') != std::string_view::npos) {
            if (!valid_repo_file(selector)) return fail<ModelSpec>(ErrorCode::InvalidConfig, "invalid file path in model spec");
            out.file = std::string(selector);
        } else {
            out.quant = std::string(selector);
        }
    }
    return ok(std::move(out));
}

bool quant_matches(std::string_view filename, std::string_view quant) {
    return !quant.empty() && ifind(basename_of(filename), quant) != std::string_view::npos;
}

Result<RepoInfo> parse_repo_info(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc) || !doc.is_object())
        return fail<RepoInfo>(ErrorCode::ParseError, "hub: model info is not a JSON object");

    RepoInfo info;
    info.repo = get_string_field(doc, "id");

    bool gated_bool = false;
    std::string_view gated_str;
    if (!doc["gated"].get(gated_bool)) info.gated = gated_bool;
    else if (!doc["gated"].get(gated_str)) info.gated = !gated_str.empty() && gated_str != "false";

    simdjson::dom::element card;
    if (!doc["cardData"].get(card) && card.is_object()) {
        std::string_view lic;
        simdjson::dom::array lic_arr;
        if (!card["license"].get(lic)) info.license = std::string(lic);
        else if (!card["license"].get(lic_arr))
            for (simdjson::dom::element e : lic_arr)
                if (!e.get(lic)) { info.license = std::string(lic); break; }
        info.license_name = get_string_field(card, "license_name");
    }
    if (info.license.empty()) {
        simdjson::dom::array tags;
        if (!doc["tags"].get(tags))
            for (simdjson::dom::element t : tags) {
                std::string_view tag;
                if (!t.get(tag) && tag.substr(0, 8) == "license:") { info.license = std::string(tag.substr(8)); break; }
            }
    }

    simdjson::dom::array siblings;
    if (!doc["siblings"].get(siblings))
        for (simdjson::dom::element s : siblings) {
            RemoteFile f;
            f.path = get_string_field(s, "rfilename");
            if (f.path.empty()) continue;
            f.size = get_uint_field(s, "size");
            simdjson::dom::element lfs;
            if (!s["lfs"].get(lfs) && lfs.is_object()) {
                f.sha256 = to_lower(get_string_field(lfs, "sha256"));
                if (f.sha256.empty()) {
                    // Tree API shape: lfs.oid holds the sha256.
                    f.sha256 = to_lower(get_string_field(lfs, "oid"));
                }
                if (f.size == 0) f.size = get_uint_field(lfs, "size");
            }
            info.files.push_back(std::move(f));
        }
    return ok(std::move(info));
}

Result<RemoteFile> select_file(const RepoInfo &info, const ModelSpec &spec) {
    if (!spec.file.empty()) {
        std::vector<const RemoteFile *> by_base;
        for (const RemoteFile &f : info.files) {
            if (f.path == spec.file) return ok(f);
            if (basename_of(f.path) == spec.file) by_base.push_back(&f);
        }
        if (by_base.size() == 1) return ok(*by_base.front());
        if (by_base.size() > 1)
            return fail<RemoteFile>(ErrorCode::InvalidConfig, "hub: '" + spec.file + "' is ambiguous in " + spec.repo + ": " + join_paths(by_base));
        return fail<RemoteFile>(ErrorCode::KeyNotFound, "hub: file '" + spec.file + "' not found in " + spec.repo);
    }

    std::vector<const RemoteFile *> ggufs;
    for (const RemoteFile &f : info.files)
        if (is_gguf(f.path) && !is_mmproj(f.path)) ggufs.push_back(&f);

    if (spec.quant.empty()) {
        if (ggufs.size() == 1) return ok(*ggufs.front());
        if (ggufs.empty()) return fail<RemoteFile>(ErrorCode::KeyNotFound, "hub: no .gguf file in " + spec.repo);
        return fail<RemoteFile>(ErrorCode::InvalidConfig,
                                "hub: " + spec.repo + " has several .gguf files; add ':<file>' or ':<QUANT>' (" + join_paths(ggufs) + ")");
    }

    std::vector<const RemoteFile *> loose, exact;
    for (const RemoteFile *f : ggufs) {
        if (!quant_matches(f->path, spec.quant)) continue;
        loose.push_back(f);
        if (quant_matches_exact(f->path, spec.quant)) exact.push_back(f);
    }
    std::vector<const RemoteFile *> &pool = exact.empty() ? loose : exact;
    if (pool.empty())
        return fail<RemoteFile>(ErrorCode::KeyNotFound, "hub: no .gguf matching '" + spec.quant + "' in " + spec.repo +
                                                            (ggufs.empty() ? std::string() : " (available: " + join_paths(ggufs) + ")"));
    std::vector<const RemoteFile *> whole, first_shards;
    for (const RemoteFile *f : pool) {
        std::string_view prefix;
        int index = 0, total = 0;
        if (!is_split_shard(f->path)) whole.push_back(f);
        else if (parse_shard(f->path, prefix, index, total) && index == 1) first_shards.push_back(f);
    }
    if (whole.empty()) {
        if (first_shards.size() == 1) return ok(*first_shards.front());
        if (first_shards.empty())
            return fail<RemoteFile>(ErrorCode::InvalidConfig, "hub: '" + spec.quant + "' in " + spec.repo +
                                                                  " has shards but no first shard (-00001-of-N): " + join_paths(pool));
        return fail<RemoteFile>(ErrorCode::InvalidConfig, "hub: '" + spec.quant + "' is ambiguous in " + spec.repo + ": " + join_paths(first_shards));
    }
    if (whole.size() > 1)
        return fail<RemoteFile>(ErrorCode::InvalidConfig, "hub: '" + spec.quant + "' is ambiguous in " + spec.repo + ": " + join_paths(whole));
    return ok(*whole.front());
}

Result<std::vector<RemoteFile>> split_set(const RepoInfo &info, const RemoteFile &shard) {
    std::string_view prefix;
    int index = 0, total = 0;
    if (!parse_shard(shard.path, prefix, index, total)) return ok(std::vector<RemoteFile>{shard});
    std::vector<RemoteFile> set;
    set.reserve(static_cast<size_t>(total));
    char suffix[32];
    for (int i = 1; i <= total; ++i) {
        std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, total);
        const std::string want = std::string(prefix) + suffix;
        auto it = std::find_if(info.files.begin(), info.files.end(), [&](const RemoteFile &f) { return f.path == want; });
        if (it == info.files.end())
            return fail<std::vector<RemoteFile>>(ErrorCode::KeyNotFound, "hub: split GGUF is missing shard " + want);
        set.push_back(*it);
    }
    return ok(std::move(set));
}

// -------------------------- Licence policy --------------------------

bool is_restricted_license(std::string_view license, std::string_view license_name) {
    auto restricted = [](std::string_view text) {
        const std::string s = to_lower(text);
        if (s.empty()) return false;
        if (s.rfind("cc-by-nc", 0) == 0) return true; // cc-by-nc-4.0, cc-by-nc-sa-4.0, cc-by-nc-nd-4.0, ...
        for (std::string_view word : {"noncommercial", "non-commercial", "non commercial", "research", "academic", "-nc-"})
            if (s.find(word) != std::string::npos) return true;
        // Known HF licence ids with research-only or non-commercial terms.
        for (std::string_view id : {"deepfloyd-if-license", "apple-amlr", "fair-nc"})
            if (s == id) return true;
        return false;
    };
    if (restricted(license)) return true;
    // "other" (custom licence): the only hint is the free-form licence name.
    return restricted(license_name);
}

// -------------------------- Hashing --------------------------

namespace {

std::string to_hex(const unsigned char *digest, unsigned int len) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(static_cast<size_t>(len) * 2, '0');
    for (unsigned int i = 0; i < len; ++i) {
        out[2 * i] = kHex[digest[i] >> 4];
        out[2 * i + 1] = kHex[digest[i] & 0xf];
    }
    return out;
}

struct EvpCtx {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    ~EvpCtx() { EVP_MD_CTX_free(ctx); }
    EvpCtx() = default;
    EvpCtx(const EvpCtx &) = delete;
    EvpCtx &operator=(const EvpCtx &) = delete;
};

} // namespace

std::string sha256_hex(std::span<const uint8_t> data) {
    EvpCtx c;
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    if (!c.ctx || EVP_DigestInit_ex(c.ctx, EVP_sha256(), nullptr) != 1 || EVP_DigestUpdate(c.ctx, data.data(), data.size()) != 1 ||
        EVP_DigestFinal_ex(c.ctx, digest, &len) != 1)
        return {};
    return to_hex(digest, len);
}

std::string sha256_hex(std::string_view data) {
    return sha256_hex(std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(data.data()), data.size()));
}

Result<std::string> sha256_file(const std::string &path) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return fail<std::string>(ErrorCode::FilesystemError, "hub: cannot open " + path);
    EvpCtx c;
    bool good = c.ctx && EVP_DigestInit_ex(c.ctx, EVP_sha256(), nullptr) == 1;
    std::vector<unsigned char> buf(1 << 20);
    while (good) {
        const size_t n = std::fread(buf.data(), 1, buf.size(), f);
        if (n > 0 && EVP_DigestUpdate(c.ctx, buf.data(), n) != 1) good = false;
        if (n < buf.size()) {
            if (std::ferror(f)) good = false;
            break;
        }
    }
    std::fclose(f);
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int len = 0;
    if (!good || EVP_DigestFinal_ex(c.ctx, digest, &len) != 1) return fail<std::string>(ErrorCode::FilesystemError, "hub: failed to hash " + path);
    return ok(to_hex(digest, len));
}

// -------------------------- Catalogue --------------------------

std::string catalog_to_json(std::span<const CatalogEntry> entries) {
    std::string out = "{\n  \"models\": [";
    bool first = true;
    for (const CatalogEntry &e : entries) {
        out += first ? "\n" : ",\n";
        first = false;
        out += "    {\"name\": ";
        append_json_string(out, e.name);
        out += ", \"repo\": ";
        append_json_string(out, e.repo);
        out += ", \"file\": ";
        append_json_string(out, e.file);
        out += ", \"path\": ";
        append_json_string(out, e.path);
        out += ", \"size\": ";
        out += std::to_string(e.size);
        out += ", \"sha256\": ";
        append_json_string(out, e.sha256);
        out += ", \"license\": ";
        append_json_string(out, e.license);
        out += "}";
    }
    out += first ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

Result<std::vector<CatalogEntry>> catalog_from_json(std::string_view json) {
    using R = std::vector<CatalogEntry>;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc)) return fail<R>(ErrorCode::ParseError, "hub: models.json is not valid JSON");
    simdjson::dom::array models;
    if (doc["models"].get(models)) return fail<R>(ErrorCode::ParseError, "hub: models.json has no \"models\" array");
    R out;
    for (simdjson::dom::element m : models) {
        if (!m.is_object()) return fail<R>(ErrorCode::ParseError, "hub: models.json entry is not an object");
        CatalogEntry e;
        e.name = get_string_field(m, "name");
        e.repo = get_string_field(m, "repo");
        e.file = get_string_field(m, "file");
        e.path = get_string_field(m, "path");
        e.size = get_uint_field(m, "size");
        e.sha256 = get_string_field(m, "sha256");
        e.license = get_string_field(m, "license");
        if (e.name.empty()) return fail<R>(ErrorCode::ParseError, "hub: models.json entry without a name");
        out.push_back(std::move(e));
    }
    return ok(std::move(out));
}

Result<std::vector<CatalogEntry>> load_catalog(const std::string &path) {
    using R = std::vector<CatalogEntry>;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return ok(R{});
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail<R>(ErrorCode::FilesystemError, "hub: cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (trim(text).empty()) return ok(R{});
    return catalog_from_json(text);
}

Result<void> save_catalog(const std::string &path, std::span<const CatalogEntry> entries) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return fail(ErrorCode::FilesystemError, "hub: cannot write " + tmp);
        const std::string json = catalog_to_json(entries);
        out.write(json.data(), static_cast<std::streamsize>(json.size()));
        if (!out.flush()) return fail(ErrorCode::FilesystemError, "hub: failed writing " + tmp);
    }
    std::filesystem::rename(tmp, p, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "hub: cannot replace " + path + ": " + ec.message());
    return ok();
}

// -------------------------- Environment --------------------------

namespace {

std::string env_or_empty(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::string first_env(std::initializer_list<const char *> names) {
    for (const char *n : names) {
        std::string v = env_or_empty(n);
        if (!v.empty()) return v;
    }
    return {};
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string percent_decode(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && hex_value(s[i + 1]) >= 0 && hex_value(s[i + 2]) >= 0) {
            out.push_back(static_cast<char>(hex_value(s[i + 1]) * 16 + hex_value(s[i + 2])));
            i += 2;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::string read_small_file(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::string s(512, '\0');
    f.read(s.data(), static_cast<std::streamsize>(s.size()));
    s.resize(static_cast<size_t>(f.gcount()));
    return s;
}

std::string hf_token_file_path() {
    if (std::string p = env_or_empty("HF_TOKEN_PATH"); !p.empty()) return p;
    if (std::string h = env_or_empty("HF_HOME"); !h.empty()) return (std::filesystem::path(h) / "token").string();
#ifdef _WIN32
    const std::string home = env_or_empty("USERPROFILE");
#else
    const std::string home = env_or_empty("HOME");
#endif
    if (home.empty()) return {};
    return (std::filesystem::path(home) / ".cache" / "huggingface" / "token").string();
}

} // namespace

std::string resolve_hf_token(std::string_view explicit_token, std::string_view env_hf_token, std::string_view env_legacy_token,
                             std::string_view token_file_contents) {
    for (std::string_view v : {explicit_token, env_hf_token, env_legacy_token, token_file_contents}) {
        v = trim(v);
        if (!v.empty()) return std::string(v);
    }
    return {};
}

bool token_allowed_for_host(std::string_view host, std::string_view endpoint) {
    const std::string h = to_lower(host);
    std::string e = to_lower(endpoint);
    if (const size_t scheme = e.find("://"); scheme != std::string::npos) e.erase(0, scheme + 3);
    if (const size_t slash = e.find('/'); slash != std::string::npos) e.erase(slash);
    if (e.find(']') == std::string::npos) {
        if (const size_t colon = e.rfind(':'); colon != std::string::npos) e.erase(colon);
    }
    if (h.empty() || e.empty()) return false;
    if (h == e) return true;
    const bool hf = e == "huggingface.co" || e == "hf.co";
    return hf && (h == "huggingface.co" || h == "hf.co");
}

Result<ProxyAddress> parse_proxy_url(std::string_view url) {
    url = trim(url);
    if (const size_t scheme = url.find("://"); scheme != std::string_view::npos) url.remove_prefix(scheme + 3);
    if (const size_t slash = url.find('/'); slash != std::string_view::npos) url = url.substr(0, slash);
    ProxyAddress out;
    if (const size_t at = url.rfind('@'); at != std::string_view::npos) {
        const std::string_view cred = url.substr(0, at);
        url.remove_prefix(at + 1);
        const size_t colon = cred.find(':');
        out.user = percent_decode(cred.substr(0, colon));
        if (colon != std::string_view::npos) out.pass = percent_decode(cred.substr(colon + 1));
    }
    if (url.empty()) return fail<ProxyAddress>(ErrorCode::InvalidConfig, "hub: empty proxy URL");

    out.port = 80;
    std::string_view host = url;
    std::string_view port;
    if (url.front() == '[') { // [::1]:3128
        const size_t close = url.find(']');
        if (close == std::string_view::npos) return fail<ProxyAddress>(ErrorCode::InvalidConfig, "hub: bad proxy URL");
        host = url.substr(1, close - 1);
        if (close + 1 < url.size() && url[close + 1] == ':') port = url.substr(close + 2);
    } else if (const size_t colon = url.rfind(':'); colon != std::string_view::npos) {
        host = url.substr(0, colon);
        port = url.substr(colon + 1);
    }
    if (!port.empty()) {
        int p = 0;
        for (char c : port) {
            if (c < '0' || c > '9' || p > 65535) return fail<ProxyAddress>(ErrorCode::InvalidConfig, "hub: bad proxy port");
            p = p * 10 + (c - '0');
        }
        if (p <= 0 || p > 65535) return fail<ProxyAddress>(ErrorCode::InvalidConfig, "hub: bad proxy port");
        out.port = p;
    }
    if (host.empty()) return fail<ProxyAddress>(ErrorCode::InvalidConfig, "hub: empty proxy host");
    out.host = std::string(host);
    return ok(std::move(out));
}

bool host_bypasses_proxy(std::string_view host, std::string_view no_proxy) {
    const std::string h = to_lower(host);
    size_t start = 0;
    while (start <= no_proxy.size()) {
        size_t end = no_proxy.find(',', start);
        if (end == std::string_view::npos) end = no_proxy.size();
        std::string entry = to_lower(trim(no_proxy.substr(start, end - start)));
        start = end + 1;
        if (entry.empty()) continue;
        if (entry == "*") return true;
        if (entry.rfind("*.", 0) == 0) entry.erase(0, 1); // "*.example.com" -> ".example.com"
        if (entry.front() == '.') {
            if (h.size() > entry.size() && h.compare(h.size() - entry.size(), entry.size(), entry) == 0) return true;
            if (h == entry.substr(1)) return true;
        } else if (h == entry || (h.size() > entry.size() && h.compare(h.size() - entry.size(), entry.size(), entry) == 0 &&
                                  h[h.size() - entry.size() - 1] == '.')) {
            return true;
        }
    }
    return false;
}

HttpOptions http_options_from_env() {
    HttpOptions o;
    if (std::string ep = env_or_empty("HF_ENDPOINT"); !ep.empty()) {
        while (!ep.empty() && ep.back() == '/') ep.pop_back();
        o.endpoint = ep;
    }
    const std::string env_token = env_or_empty("HF_TOKEN");
    const std::string legacy_token = env_or_empty("HUGGING_FACE_HUB_TOKEN");
    std::string file_token;
    if (trim(env_token).empty() && trim(legacy_token).empty()) {
        if (const std::string path = hf_token_file_path(); !path.empty()) file_token = read_small_file(path);
    }
    o.token = resolve_hf_token({}, env_token, legacy_token, file_token);
    if (const std::string proxy = first_env({"HTTPS_PROXY", "https_proxy"}); !proxy.empty()) {
        auto p = parse_proxy_url(proxy);
        if (p.ok) {
            o.proxy_host = p.value.host;
            o.proxy_port = p.value.port;
            o.proxy_user = p.value.user;
            o.proxy_pass = p.value.pass;
        }
    }
    if (const std::string proxy = first_env({"HTTP_PROXY", "http_proxy"}); !proxy.empty()) {
        auto p = parse_proxy_url(proxy);
        if (p.ok) o.http_proxy = p.value;
    }
    o.no_proxy = first_env({"NO_PROXY", "no_proxy"});
    o.ca_cert_path = first_env({"AGENT_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"});
    return o;
}

std::string default_models_dir() {
    if (std::string d = env_or_empty("AGENT_MODELS_DIR"); !d.empty()) return d;
#ifdef _WIN32
    if (std::string d = env_or_empty("LOCALAPPDATA"); !d.empty()) return (std::filesystem::path(d) / "agent" / "models").string();
    return (std::filesystem::path(env_or_empty("USERPROFILE")) / "agent" / "models").string();
#else
    if (std::string d = env_or_empty("XDG_DATA_HOME"); !d.empty()) return (std::filesystem::path(d) / "agent" / "models").string();
    return (std::filesystem::path(env_or_empty("HOME")) / ".local" / "share" / "agent" / "models").string();
#endif
}

// -------------------------- Catalogue commands --------------------------

Result<std::vector<CatalogEntry>> list_models(const std::string &catalog_path) { return load_catalog(catalog_path); }

agent::ModelSpec model_spec_from_entry(const CatalogEntry &entry, const Config &base) {
    agent::ModelSpec spec;
    spec.name = entry.name;
    spec.config = base;
    spec.config.provider = AiProvider::Engine;
    spec.config.model = entry.path;
    if (spec.config.engine.empty()) {
        std::string ext = std::filesystem::path(entry.path).extension().string();
        for (char &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext == ".gguf") spec.config.engine = "llama";
        else if (ext == ".onnx") spec.config.engine = "onnx";
        else if (ext == ".safetensors" || ext == ".ckpt" || ext == ".sft") spec.config.engine = "sdcpp";
    }
    std::error_code ec;
    if (!std::filesystem::exists(entry.path, ec)) spec.estimated_bytes = entry.size;
    return spec;
}

std::vector<agent::ModelSpec> model_specs_from_catalog(std::span<const CatalogEntry> entries, const Config &base) {
    std::vector<agent::ModelSpec> out;
    out.reserve(entries.size());
    for (const auto &e : entries) out.push_back(model_spec_from_entry(e, base));
    return out;
}

Result<void> remove_model(std::string_view name, const std::string &catalog_path, bool delete_file) {
    auto cat = load_catalog(catalog_path);
    if (!cat.ok) return fail(cat.error.code, cat.error.message);
    auto it = std::find_if(cat.value.begin(), cat.value.end(), [&](const CatalogEntry &e) { return e.name == name; });
    if (it == cat.value.end()) return fail(ErrorCode::KeyNotFound, "hub: no model named '" + std::string(name) + "' in " + catalog_path);
    if (delete_file && !it->path.empty()) {
        // A split GGUF is catalogued by its first shard; delete every shard of the set.
        std::vector<std::string> paths{it->path};
        std::string_view prefix;
        int index = 0, total = 0;
        if (parse_shard(it->path, prefix, index, total)) {
            paths.clear();
            char suffix[32];
            for (int i = 1; i <= total; ++i) {
                std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, total);
                paths.push_back(std::string(prefix) + suffix);
            }
        }
        for (const std::string &p : paths) {
            std::error_code ec;
            std::filesystem::remove(p, ec);
            if (ec) return fail(ErrorCode::FilesystemError, "hub: cannot delete " + p + ": " + ec.message());
            std::filesystem::remove(p + ".part", ec);
        }
    }
    cat.value.erase(it);
    return save_catalog(catalog_path, cat.value);
}

} // namespace agent::hub
