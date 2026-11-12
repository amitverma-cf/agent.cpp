#include "runtimes.hpp"

#include "archive.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <openssl/evp.h>
#include <simdjson.h>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

bool is_hex64(std::string_view s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string exe_name(std::string_view name) {
#ifdef _WIN32
    return std::string(name) + ".exe";
#else
    return std::string(name);
#endif
}

// File to look for in <runtimes>/<name>/.
std::string entry_name(std::string_view name) { return name == "pnpm" ? std::string("bin/pnpm.cjs") : exe_name(name); }

// "npx", "npx.cmd", "C:\x\npx.exe" -> "npx"
std::string command_stem(const std::string &command) {
    std::string stem = fs::path(command).filename().string();
    for (const char *ext : {".exe", ".cmd", ".bat"})
        if (lower(stem).ends_with(ext)) stem.resize(stem.size() - std::string_view(ext).size());
    return lower(stem);
}

} // namespace

std::string host_os() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}

std::string host_arch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    return "x86_64";
#endif
}

bool is_runtime_name(std::string_view name) { return name == "bun" || name == "uv" || name == "node" || name == "pnpm" || name == "nub"; }

std::vector<std::string> runtimes_for_js(std::string_view js_runtime) {
    if (js_runtime == "node") return {"node", "pnpm"};
    if (js_runtime == "nub") return {"nub"};
    return {"bun"};
}

Result<RuntimeAsset> runtime_asset(std::string_view name, std::string_view version, std::string_view os, std::string_view arch) {
    if (os != "linux" && os != "macos" && os != "windows")
        return fail<RuntimeAsset>(ErrorCode::InvalidConfig, "unsupported operating system: " + std::string(os));
    if (arch != "x86_64" && arch != "aarch64")
        return fail<RuntimeAsset>(ErrorCode::InvalidConfig, "unsupported CPU architecture: " + std::string(arch));
    const bool latest = version.empty() || version == "latest";
    RuntimeAsset a;
    if (name == "bun") {
        if (os == "windows" && arch == "aarch64") return fail<RuntimeAsset>(ErrorCode::InvalidConfig, "bun has no Windows arm64 build");
        const std::string target = std::string(os == "macos" ? "darwin" : os) + "-" + (arch == "x86_64" ? "x64" : "aarch64");
        a.archive_name = "bun-" + target + ".zip";
        const std::string base = latest ? "https://github.com/oven-sh/bun/releases/latest/download/"
                                        : "https://github.com/oven-sh/bun/releases/download/bun-v" + std::string(version) + "/";
        a.url = base + a.archive_name;
        a.checksum_url = base + "SHASUMS256.txt";
        a.binary = exe_name("bun");
        return ok(std::move(a));
    }
    if (name == "uv") {
        std::string triple;
        if (os == "linux") triple = std::string(arch) + "-unknown-linux-gnu";
        else if (os == "macos") triple = std::string(arch) + "-apple-darwin";
        else triple = std::string(arch) + "-pc-windows-msvc";
        a.archive_name = "uv-" + triple + (os == "windows" ? ".zip" : ".tar.gz");
        const std::string base = latest ? "https://github.com/astral-sh/uv/releases/latest/download/"
                                        : "https://github.com/astral-sh/uv/releases/download/" + std::string(version) + "/";
        a.url = base + a.archive_name;
        a.checksum_url = a.url + ".sha256";
        a.binary = exe_name("uv");
        return ok(std::move(a));
    }
    if (name == "nub") {
        const std::string target = std::string(os == "macos"     ? "darwin"
                                               : os == "windows" ? "win32"
                                                                 : "linux") +
                                   "-" + (arch == "x86_64" ? "x64" : "arm64");
        a.archive_name = "nub-" + target + (os == "windows" ? ".zip" : ".tar.gz");
        const std::string base = latest ? "https://github.com/nubjs/nub/releases/latest/download/"
                                        : "https://github.com/nubjs/nub/releases/download/v" + std::string(version) + "/";
        a.url = base + a.archive_name;
        a.checksum_url = a.url + ".sha256";
        a.binary = exe_name("nub");
        return ok(std::move(a));
    }
    if (name == "node") {
        // The archive's name carries the version, so it is read from the checksum file (see find_node_archive).
        std::string v(version);
        if (!v.empty() && v[0] == 'v') v.erase(0, 1);
        const std::string dir = latest ? "latest-v24.x" : "v" + v;
        a.url = "https://nodejs.org/dist/" + dir + "/";
        a.checksum_url = a.url + "SHASUMS256.txt";
        a.binary = exe_name("node");
        return ok(std::move(a));
    }
    return fail<RuntimeAsset>(ErrorCode::InvalidConfig, "unknown runtime '" + std::string(name) + "' (bun, uv, node, pnpm or nub)");
}

std::string find_sha256(std::string_view text, std::string_view file_name) {
    std::string only; // the single hash of a one-line file
    size_t hashes = 0;
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string_view::npos) end = text.size();
        std::string_view line = text.substr(at, end - at);
        at = end + 1;
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.remove_suffix(1);
        size_t sp = line.find_first_of(" \t");
        const std::string_view first = line.substr(0, sp);
        if (!is_hex64(first)) continue;
        ++hashes;
        only = lower(std::string(first));
        if (sp == std::string_view::npos) continue;
        std::string_view rest = line.substr(sp);
        while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t' || rest.front() == '*')) rest.remove_prefix(1);
        if (fs::path(std::string(rest)).filename().string() == file_name) return lower(std::string(first));
    }
    return hashes == 1 && text.find(file_name) == std::string_view::npos ? only : std::string();
}

std::string find_node_archive(std::string_view shasums, std::string_view os, std::string_view arch) {
    const std::string suffix = std::string("-") +
                               (os == "macos"     ? "darwin"
                                : os == "windows" ? "win"
                                                  : "linux") +
                               "-" + (arch == "x86_64" ? "x64" : "arm64") + (os == "windows" ? ".zip" : ".tar.gz");
    size_t at = 0;
    while (at < shasums.size()) {
        size_t end = shasums.find('\n', at);
        if (end == std::string_view::npos) end = shasums.size();
        std::string_view line = shasums.substr(at, end - at);
        at = end + 1;
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.remove_suffix(1);
        const size_t sp = line.find_last_of(" \t*");
        if (sp == std::string_view::npos) continue;
        const std::string_view file = line.substr(sp + 1);
        if (file.starts_with("node-v") && file.ends_with(suffix)) return std::string(file);
    }
    return {};
}

std::string runtime_binary_path(const HomeLayout &home, std::string_view name) {
    const fs::path p = fs::path(home.runtimes) / std::string(name) / entry_name(name);
    std::error_code ec;
    return fs::is_regular_file(p, ec) ? p.string() : std::string();
}

namespace {

// The directory under `root` (at most three levels down) that holds `binary`.
fs::path find_binary_dir(const fs::path &root, const std::string &binary, int depth = 3) {
    std::error_code ec;
    if (fs::is_regular_file(root / binary, ec)) return root;
    if (depth <= 0) return {};
    for (const auto &entry : fs::directory_iterator(root, ec))
        if (entry.is_directory(ec))
            if (fs::path found = find_binary_dir(entry.path(), binary, depth - 1); !found.empty()) return found;
    return {};
}

// SHA-512 of a file as the "sha512-<base64>" string the npm registry publishes.
std::string npm_integrity(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha512(), nullptr) != 1) return {};
    std::vector<char> buf(1 << 16);
    while (in.read(buf.data(), static_cast<std::streamsize>(buf.size())) || in.gcount() > 0)
        EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(in.gcount()));
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned len = 0;
    EVP_DigestFinal_ex(ctx, md, &len);
    EVP_MD_CTX_free(ctx);
    std::string out((len + 2) / 3 * 4, '=');
    static constexpr char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (unsigned i = 0; i < len; i += 3) {
        const unsigned v = (md[i] << 16) | (i + 1 < len ? md[i + 1] << 8 : 0) | (i + 2 < len ? md[i + 2] : 0);
        out[o++] = tbl[(v >> 18) & 63];
        out[o++] = tbl[(v >> 12) & 63];
        if (i + 1 < len) out[o++] = tbl[(v >> 6) & 63];
        if (i + 2 < len) out[o++] = tbl[v & 63];
    }
    return "sha512-" + out;
}

Result<void> place(const HomeLayout &home, const std::string &name, const fs::path &bin_dir, const fs::path &staging,
                   const std::string &entry) {
    std::error_code ec;
    const fs::path target = fs::path(home.runtimes) / name;
    fs::remove_all(target, ec);
    fs::rename(bin_dir, target, ec);
    fs::remove_all(staging, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot move the runtime into " + target.string() + ": " + ec.message());
    if (name != "pnpm")
        fs::permissions(target / entry, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);
    return ok();
}

} // namespace

Result<std::string> install_runtime(const HomeLayout &home, std::string_view name, std::string_view version,
                                    const agent::hub::HttpOptions &http, const InstallProgress &progress) {
    if (!is_runtime_name(name))
        return fail<std::string>(ErrorCode::InvalidConfig, "unknown runtime '" + std::string(name) + "' (bun, uv, node, pnpm or nub)");
    std::error_code ec;
    const fs::path downloads = fs::path(home.cache) / "downloads";
    fs::create_directories(downloads, ec);
    fs::create_directories(home.runtimes, ec);
    const std::string runtime(name);
    const fs::path staging = fs::path(home.runtimes) / (".staging-" + runtime);
    fs::remove_all(staging, ec);
    fs::create_directories(staging, ec);
    agent::hub::DownloadOptions opts;
    opts.on_progress = progress.on_progress;
    opts.progress_user_data = progress.user;

    if (name == "pnpm") { // the pnpm package from the npm registry, run by the private node; verified by the registry's SHA-512
        const std::string v = version.empty() || version == "latest" ? "latest" : std::string(version);
        auto meta = agent::hub::http_get_text("https://registry.npmjs.org/pnpm/" + v, http);
        if (!meta.ok) return fail<std::string>(meta.error.code, "cannot read pnpm's registry entry: " + meta.error.message);
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        simdjson::dom::object dist;
        std::string_view tarball, integrity;
        if (parser.parse(meta.value).get(doc) || doc["dist"].get(dist) || dist["tarball"].get(tarball) || dist["integrity"].get(integrity))
            return fail<std::string>(ErrorCode::ParseError, "pnpm's registry entry has no tarball / integrity");
        const fs::path archive = downloads / "pnpm.tgz";
        if (auto d = agent::hub::download_file(tarball, archive.string(), http, opts); !d.ok)
            return fail<std::string>(d.error.code, d.error.message);
        if (npm_integrity(archive.string()) != integrity) {
            fs::remove(archive, ec);
            return fail<std::string>(ErrorCode::SandboxViolation, "the pnpm package does not match the registry's SHA-512; not installed");
        }
        // Registry tarballs hold everything under "package/".
        if (auto x = extract_archive(archive.string(), staging.string(), 1); !x.ok) return fail<std::string>(x.error.code, x.error.message);
        if (!fs::is_regular_file(staging / "bin" / "pnpm.cjs", ec))
            return fail<std::string>(ErrorCode::ParseError, "the pnpm package has no bin/pnpm.cjs");
        if (auto p = place(home, runtime, staging, staging, "bin/pnpm.cjs"); !p.ok) return fail<std::string>(p.error.code, p.error.message);
        fs::remove(archive, ec);
        return ok((fs::path(home.runtimes) / "pnpm" / "bin" / "pnpm.cjs").string());
    }

    auto asset = runtime_asset(name, version, host_os(), host_arch());
    if (!asset.ok) return fail<std::string>(asset.error.code, asset.error.message);
    auto sums = agent::hub::http_get_text(asset.value.checksum_url, http);
    if (!sums.ok) return fail<std::string>(sums.error.code, "cannot fetch the checksums for " + runtime + ": " + sums.error.message);
    if (name == "node") {
        asset.value.archive_name = find_node_archive(sums.value, host_os(), host_arch());
        if (asset.value.archive_name.empty())
            return fail<std::string>(ErrorCode::KeyNotFound,
                                     "no Node.js build for " + host_os() + "-" + host_arch() + " in " + asset.value.checksum_url);
        asset.value.url += asset.value.archive_name;
    }
    const std::string expected = find_sha256(sums.value, asset.value.archive_name);
    if (expected.empty())
        return fail<std::string>(ErrorCode::ParseError, "no SHA-256 for " + asset.value.archive_name + " in " + asset.value.checksum_url);
    const fs::path archive = downloads / asset.value.archive_name;
    opts.expected_sha256 = expected;
    if (auto d = agent::hub::download_file(asset.value.url, archive.string(), http, opts); !d.ok)
        return fail<std::string>(d.error.code, d.error.message);
    if (auto x = extract_archive(archive.string(), staging.string(), 0); !x.ok) {
        fs::remove_all(staging, ec);
        return fail<std::string>(x.error.code, x.error.message);
    }
    const fs::path bin_dir = find_binary_dir(staging, asset.value.binary);
    if (bin_dir.empty()) {
        fs::remove_all(staging, ec);
        return fail<std::string>(ErrorCode::ParseError, asset.value.archive_name + " does not contain " + asset.value.binary);
    }
    if (auto p = place(home, runtime, bin_dir, staging, asset.value.binary); !p.ok) return fail<std::string>(p.error.code, p.error.message);
    std::ofstream meta(fs::path(home.runtimes) / runtime / "runtime.json");
    meta << "{\"name\":\"" << name << "\",\"version\":\"" << (version.empty() ? "latest" : version) << "\",\"url\":\"" << asset.value.url
         << "\",\"sha256\":\"" << expected << "\"}\n";
    fs::remove(archive, ec);
    return ok((fs::path(home.runtimes) / runtime / asset.value.binary).string());
}

Result<void> remove_runtime(const HomeLayout &home, std::string_view name) {
    if (!is_runtime_name(name))
        return fail(ErrorCode::InvalidConfig, "unknown runtime '" + std::string(name) + "' (bun, uv, node, pnpm or nub)");
    std::error_code ec;
    const fs::path target = fs::path(home.runtimes) / std::string(name);
    if (!fs::exists(target, ec)) return fail(ErrorCode::KeyNotFound, std::string(name) + " is not installed");
    fs::remove_all(target, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot remove " + target.string() + ": " + ec.message());
    return ok();
}

std::map<std::string, std::string> runtime_env(const HomeLayout &home) { return runtime_env_in(home.cache); }

std::map<std::string, std::string> runtime_env_in(const std::string &cache_root) {
    const fs::path c(cache_root);
    std::map<std::string, std::string> env = {
        // bun
        {"BUN_INSTALL", (c / "bun" / "install").string()},
        {"BUN_INSTALL_CACHE_DIR", (c / "bun" / "packages").string()},
        {"BUN_RUNTIME_TRANSPILER_CACHE_PATH", (c / "bun" / "transpiler").string()},
        // uv
        {"UV_CACHE_DIR", (c / "uv").string()},
        {"UV_TOOL_DIR", (c / "uv-tools").string()},
        {"UV_TOOL_BIN_DIR", (c / "uv-bin").string()},
        {"UV_PYTHON_INSTALL_DIR", (c / "uv-python").string()},
        // pnpm / node / nub (npm_config_* is how pnpm reads its settings; npm itself is never run)
        {"PNPM_HOME", (c / "pnpm-home").string()},
        {"npm_config_store_dir", (c / "pnpm-store").string()},
        {"npm_config_cache", (c / "npm-cache").string()},
        {"npm_config_update_notifier", "false"},
        {"NODE_COMPILE_CACHE", (c / "node-compile").string()},
        // whatever else wants a home or a temp folder
        {"HOME", (c / "home").string()},
        {"XDG_CACHE_HOME", (c / "xdg" / "cache").string()},
        {"XDG_DATA_HOME", (c / "xdg" / "data").string()},
        {"XDG_STATE_HOME", (c / "xdg" / "state").string()},
        {"XDG_CONFIG_HOME", (c / "xdg" / "config").string()},
        {"TMPDIR", (c / "tmp").string()},
#ifdef _WIN32
        {"USERPROFILE", (c / "home").string()},
        {"APPDATA", (c / "home" / "AppData" / "Roaming").string()},
        {"LOCALAPPDATA", (c / "home" / "AppData" / "Local").string()},
        {"TEMP", (c / "tmp").string()},
        {"TMP", (c / "tmp").string()},
#endif
    };
    return env;
}

LauncherRewrite rewrite_launcher(McpServerConfig &server, const HomeLayout &home, const std::string &js_runtime) {
    LauncherRewrite result;
    const std::string stem = command_stem(server.command);
    const bool js_command =
        stem == "npx" || stem == "bunx" || stem == "npm" || stem == "node" || stem == "bun" || stem == "nub" || stem == "pnpm";
    const bool py_command = stem == "uvx" || stem == "uv";
    if (!js_command && !py_command) return result;
    if (fs::path(server.command).is_absolute() && server.command.starts_with(home.runtimes)) return result; // already rewritten

    auto missing = [&](const std::string &what) {
        result.note = "'" + server.name + "' runs " + server.command + ", but the private " + what +
                      " is not installed; using the system command. Run `agent runtime install " + what + "`.";
        return result;
    };
    std::vector<std::string> args;
    std::string command;
    if (py_command) {
        command = runtime_binary_path(home, "uv");
        if (command.empty()) return missing("uv");
        if (stem == "uvx") args = {"tool", "run"};
        args.insert(args.end(), server.args.begin(), server.args.end());
    } else {
        auto dlx_args = [&](std::vector<std::string> prefix) {
            for (const std::string &a : server.args)
                if (a != "-y" && a != "--yes") prefix.push_back(a); // the runner never asks before installing
            return prefix;
        };
        auto pnpm_style = [&](std::vector<std::string> prefix) { // npm's subcommands in pnpm's words
            std::vector<std::string> out = std::move(prefix);
            for (size_t i = 0; i < server.args.size(); ++i) {
                std::string a = server.args[i];
                if (i == 0) {
                    if (a == "i" || a == "install" || a == "ci") a = "install";
                    else if (a == "run-script") a = "run";
                    else if (a == "exec" || a == "x") a = "dlx";
                }
                out.push_back(std::move(a));
            }
            return out;
        };
        if (js_runtime == "node") {
            const std::string node = runtime_binary_path(home, "node"), pnpm = runtime_binary_path(home, "pnpm");
            if (node.empty()) return missing("node");
            command = node;
            if (stem == "node") args = server.args;
            else if (stem == "npx" || stem == "bunx") {
                if (pnpm.empty()) return missing("pnpm");
                args = dlx_args({pnpm, "dlx"});
            } else if (stem == "npm" || stem == "pnpm") {
                if (pnpm.empty()) return missing("pnpm");
                args = pnpm_style({pnpm});
            } else return result; // bun / nub written explicitly: left alone
        } else if (js_runtime == "nub") {
            command = runtime_binary_path(home, "nub");
            if (command.empty()) return missing("nub");
            if (stem == "npx" || stem == "bunx") args = dlx_args({"dlx"});
            else if (stem == "npm" || stem == "pnpm") args = pnpm_style({});
            else if (stem == "nub") args = server.args;
            else return result; // node / bun written explicitly: left alone
        } else {                // bun
            command = runtime_binary_path(home, "bun");
            if (command.empty()) return missing("bun");
            if (stem == "npx" || stem == "bunx") args = dlx_args({"x"});
            else if (stem == "npm" || stem == "pnpm") {
                args = pnpm_style({});
                if (!args.empty() && args[0] == "dlx") args[0] = "x"; // bun's name for dlx
            } else args = server.args;                                // node, bun
        }
    }
    const std::string cache_root = server.working_directory.empty() ? home.cache : (fs::path(server.working_directory) / "cache").string();
    server.command = command;
    server.args = std::move(args);
    for (const auto &[key, value] : runtime_env_in(cache_root)) server.env.emplace(key, value); // an explicit value in the file wins
    std::error_code ec;
    for (const char *dir : {"home", "tmp", "xdg", "bun", "uv", "pnpm-home", "npm-cache"})
        fs::create_directories(fs::path(cache_root) / dir, ec);
    result.rewritten = true;
    return result;
}

} // namespace agent::cli
