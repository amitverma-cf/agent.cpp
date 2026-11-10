#include "runtimes.hpp"

#include "archive.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>

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

Result<RuntimeAsset> runtime_asset(std::string_view name, std::string_view version, std::string_view os, std::string_view arch) {
    if (os != "linux" && os != "macos" && os != "windows") return fail<RuntimeAsset>(ErrorCode::InvalidConfig, "unsupported operating system: " + std::string(os));
    if (arch != "x86_64" && arch != "aarch64") return fail<RuntimeAsset>(ErrorCode::InvalidConfig, "unsupported CPU architecture: " + std::string(arch));
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
    return fail<RuntimeAsset>(ErrorCode::InvalidConfig, "unknown runtime '" + std::string(name) + "' (bun or uv)");
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

std::string runtime_binary_path(const HomeLayout &home, std::string_view name) {
    const fs::path p = fs::path(home.runtimes) / std::string(name) / exe_name(name);
    std::error_code ec;
    return fs::is_regular_file(p, ec) ? p.string() : std::string();
}

namespace {

// The directory under `root` (at most two levels down) that holds `binary`.
fs::path find_binary_dir(const fs::path &root, const std::string &binary) {
    std::error_code ec;
    if (fs::is_regular_file(root / binary, ec)) return root;
    for (const auto &entry : fs::directory_iterator(root, ec)) {
        if (entry.is_directory(ec) && fs::is_regular_file(entry.path() / binary, ec)) return entry.path();
    }
    return {};
}

} // namespace

Result<std::string> install_runtime(const HomeLayout &home, std::string_view name, std::string_view version, const agent::hub::HttpOptions &http,
                                    const InstallProgress &progress) {
    auto asset = runtime_asset(name, version, host_os(), host_arch());
    if (!asset.ok) return fail<std::string>(asset.error.code, asset.error.message);

    auto sums = agent::hub::http_get_text(asset.value.checksum_url, http);
    if (!sums.ok) return fail<std::string>(sums.error.code, "cannot fetch the checksum for " + asset.value.archive_name + ": " + sums.error.message);
    const std::string expected = find_sha256(sums.value, asset.value.archive_name);
    if (expected.empty()) return fail<std::string>(ErrorCode::ParseError, "no SHA-256 for " + asset.value.archive_name + " in " + asset.value.checksum_url);

    std::error_code ec;
    const fs::path downloads = fs::path(home.cache) / "downloads";
    fs::create_directories(downloads, ec);
    const fs::path archive = downloads / asset.value.archive_name;
    agent::hub::DownloadOptions opts;
    opts.expected_sha256 = expected;
    opts.on_progress = progress.on_progress;
    opts.progress_user_data = progress.user;
    if (auto d = agent::hub::download_file(asset.value.url, archive.string(), http, opts); !d.ok) return fail<std::string>(d.error.code, d.error.message);

    const fs::path staging = fs::path(home.runtimes) / (".staging-" + std::string(name));
    fs::remove_all(staging, ec);
    fs::create_directories(staging, ec);
    if (auto x = extract_archive(archive.string(), staging.string(), 0); !x.ok) {
        fs::remove_all(staging, ec);
        return fail<std::string>(x.error.code, x.error.message);
    }
    const fs::path bin_dir = find_binary_dir(staging, asset.value.binary);
    if (bin_dir.empty()) {
        fs::remove_all(staging, ec);
        return fail<std::string>(ErrorCode::ParseError, asset.value.archive_name + " does not contain " + asset.value.binary);
    }

    const fs::path target = fs::path(home.runtimes) / std::string(name);
    fs::remove_all(target, ec);
    fs::rename(bin_dir, target, ec);
    if (ec) {
        fs::remove_all(staging, ec);
        return fail<std::string>(ErrorCode::FilesystemError, "cannot move the runtime into " + target.string() + ": " + ec.message());
    }
    fs::remove_all(staging, ec);
    fs::permissions(target / asset.value.binary, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec, fs::perm_options::add, ec);

    std::ofstream meta(target / "runtime.json");
    meta << "{\"name\":\"" << name << "\",\"version\":\"" << (version.empty() ? "latest" : version) << "\",\"url\":\"" << asset.value.url << "\",\"sha256\":\"" << expected << "\"}\n";
    fs::remove(archive, ec);
    return ok((target / asset.value.binary).string());
}

Result<void> remove_runtime(const HomeLayout &home, std::string_view name) {
    if (name != "bun" && name != "uv") return fail(ErrorCode::InvalidConfig, "unknown runtime '" + std::string(name) + "' (bun or uv)");
    std::error_code ec;
    const fs::path target = fs::path(home.runtimes) / std::string(name);
    if (!fs::exists(target, ec)) return fail(ErrorCode::KeyNotFound, std::string(name) + " is not installed");
    fs::remove_all(target, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot remove " + target.string() + ": " + ec.message());
    return ok();
}

std::map<std::string, std::string> runtime_env(const HomeLayout &home) { return runtime_env_in(home.cache); }

std::map<std::string, std::string> runtime_env_in(const std::string &cache_root) {
    const fs::path cache(cache_root);
    return {
        {"BUN_INSTALL", (cache / "bun" / "install").string()},
        {"BUN_INSTALL_CACHE_DIR", (cache / "bun" / "packages").string()},
        {"BUN_RUNTIME_TRANSPILER_CACHE_PATH", (cache / "bun" / "transpiler").string()},
        {"UV_CACHE_DIR", (cache / "uv").string()},
        {"UV_TOOL_DIR", (cache / "uv-tools").string()},
        {"UV_TOOL_BIN_DIR", (cache / "uv-bin").string()},
        {"UV_PYTHON_INSTALL_DIR", (cache / "uv-python").string()},
    };
}

LauncherRewrite rewrite_launcher(McpServerConfig &server, const HomeLayout &home, const std::string &cache_root) {
    LauncherRewrite result;
    const std::string stem = command_stem(server.command);
    const bool wants_bun = stem == "npx" || stem == "bunx" || stem == "bun" || stem == "npm" || stem == "node";
    const bool wants_uv = stem == "uvx" || stem == "uv";
    if (!wants_bun && !wants_uv) return result;

    const char *runtime = wants_bun ? "bun" : "uv";
    const std::string binary = runtime_binary_path(home, runtime);
    if (binary.empty()) {
        result.note = "'" + server.name + "' runs " + server.command + ", but no private " + runtime + " is installed; using the system command. Run `agent runtime install " + runtime + "`.";
        return result;
    }
    std::vector<std::string> args;
    if (stem == "npx" || stem == "bunx") {
        args.push_back("x");
        for (const std::string &a : server.args)
            if (a != "-y" && a != "--yes") args.push_back(a); // bun x never asks before installing
    } else if (stem == "npm") { // npm install|i|add -> bun install|add, run -> bun run, exec|x -> bun x
        args = server.args;
        if (!args.empty()) {
            if (args[0] == "i" || args[0] == "install" || args[0] == "ci") args[0] = "install";
            else if (args[0] == "run-script") args[0] = "run";
            else if (args[0] == "exec" || args[0] == "x") args[0] = "x";
        }
    } else if (stem == "uvx") {
        args = {"tool", "run"};
        args.insert(args.end(), server.args.begin(), server.args.end());
    } else {
        args = server.args;
    }
    server.command = binary;
    server.args = std::move(args);
    for (const auto &[key, value] : runtime_env_in(cache_root.empty() ? home.cache : cache_root)) server.env.emplace(key, value); // an explicit value in the file wins
    result.rewritten = true;
    return result;
}

} // namespace agent::cli
