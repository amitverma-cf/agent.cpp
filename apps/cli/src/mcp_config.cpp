#include "mcp_config.hpp"

#include "runtimes.hpp"

#include <simdjson.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

bool valid_server_name(std::string_view n) {
    if (n.empty() || n.size() > 64) return false;
    return std::all_of(n.begin(), n.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-'; });
}

// Takes the lookup result itself: converting a missing field to an element would throw.
std::vector<std::string> string_list(simdjson::simdjson_result<simdjson::dom::element> field) {
    std::vector<std::string> out;
    simdjson::dom::array arr;
    if (field.get(arr)) return out;
    for (simdjson::dom::element e : arr) {
        std::string_view s;
        if (!e.get(s)) out.emplace_back(s);
    }
    return out;
}

} // namespace

Result<McpFile> parse_mcp_config(std::string_view json, const std::string &source) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc)) return fail<McpFile>(ErrorCode::ParseError, source + " is not valid JSON");
    simdjson::dom::object servers;
    if (doc["mcpServers"].get(servers) && doc["servers"].get(servers))
        return fail<McpFile>(ErrorCode::ParseError, source + " has no \"mcpServers\" object");

    McpFile file;
    for (auto field : servers) {
        const std::string name(field.key);
        simdjson::dom::object o;
        if (!valid_server_name(name)) {
            file.warnings.push_back(source + ": skipped server '" + name + "': the name must be 1-64 letters, digits, '_' or '-'");
            continue;
        }
        if (field.value.get(o)) {
            file.warnings.push_back(source + ": skipped server '" + name + "': it is not an object");
            continue;
        }
        McpServerConfig c;
        c.name = name;
        std::string_view v;
        if (!o["command"].get(v)) c.command = std::string(v);
        if (c.command.empty()) {
            file.warnings.push_back(source + ": skipped server '" + name + "': no \"command\"");
            continue;
        }
        c.args = string_list(o["args"]);
        if (!o["cwd"].get(v)) c.working_directory = std::string(v);
        int64_t timeout = 0;
        if (!o["timeout_ms"].get(timeout) && timeout > 0) c.timeout_ms = static_cast<int>(timeout);
        simdjson::dom::object env;
        if (!o["env"].get(env))
            for (auto e : env)
                if (!e.value.get(v)) c.env[std::string(e.key)] = std::string(v);

        // "sandbox" replaces the defaults entirely; remember that it was given.
        simdjson::dom::object sb;
        if (!o["sandbox"].get(sb)) {
            c.sandbox.enabled = true;
            bool enabled = true;
            if (!sb["enabled"].get(enabled)) c.sandbox.enabled = enabled;
            if (!sb["network"].get(v)) c.sandbox.network = (v == "deny") ? SandboxNetwork::Deny : SandboxNetwork::Allow;
            c.sandbox.read_only_paths = string_list(sb["read"]);
            c.sandbox.read_write_paths = string_list(sb["write"]);
            c.sandbox.memory_limit_bytes = 0;
        }
        c.sandbox.filesystem = true;
        if (!o["sandbox"].get(sb)) file.own_sandbox.push_back(name);
        file.servers.push_back(std::move(c));
    }
    return ok(std::move(file));
}

void apply_mcp_defaults(McpServerConfig &server, const std::string &data_dir, const std::string &runtimes_dir, const std::string & /*cache_dir*/, bool had_sandbox_object) {
    if (server.working_directory.empty()) server.working_directory = data_dir;
    if (had_sandbox_object) return;
    server.sandbox.enabled = true;
    server.sandbox.network = SandboxNetwork::Allow;
    server.sandbox.read_write_paths = {data_dir}; // nothing outside its own folder: package caches live in <data>/cache
    std::error_code ec;
    if (fs::is_directory(runtimes_dir, ec)) server.sandbox.read_only_paths = {runtimes_dir};
}

namespace {

std::string slurp(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void add_file(const std::string &path, const std::string &data_root, const Context &ctx, std::vector<McpServerConfig> &out, std::vector<std::string> &warnings) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return;
    auto parsed = parse_mcp_config(slurp(path), path);
    if (!parsed.ok) {
        warnings.push_back(parsed.error.message);
        return;
    }
    for (const std::string &w : parsed.value.warnings) warnings.push_back(w);
    const std::vector<std::string> &own_sandbox = parsed.value.own_sandbox;
    const std::string runtimes = ctx.has_home ? ctx.home.runtimes : std::string();
    const std::string cache = ctx.has_home ? ctx.home.cache : (fs::path(data_root) / "cache").string();
    for (McpServerConfig &s : parsed.value.servers) {
        const std::string data_dir = (fs::path(data_root) / s.name / "data").string();
        fs::create_directories(data_dir, ec);
        const bool had = std::find(own_sandbox.begin(), own_sandbox.end(), s.name) != own_sandbox.end();
        apply_mcp_defaults(s, data_dir, runtimes, cache, had);
        if (ctx.has_home) { // npx / uvx run on the private bun / uv when they are installed
            const LauncherRewrite launcher = rewrite_launcher(s, ctx.home, ctx.config.js_runtime); // caches stay in the server's own folder (its working directory)
            if (!launcher.note.empty()) warnings.push_back(launcher.note);
        }
        // A project entry replaces a global one with the same name.
        out.erase(std::remove_if(out.begin(), out.end(), [&](const McpServerConfig &e) { return e.name == s.name; }), out.end());
        out.push_back(std::move(s));
    }
}

} // namespace

namespace {

// <dir>/mcp.d/*.json: servers added with `agent mcp add`, one file each (sorted by name).
void add_dropins(const fs::path &dir, const std::string &data_root, const Context &ctx, std::vector<McpServerConfig> &out, std::vector<std::string> &warnings) {
    std::error_code ec;
    std::vector<std::string> files;
    for (const auto &entry : fs::directory_iterator(dir / "mcp.d", ec))
        if (entry.is_regular_file(ec) && entry.path().extension() == ".json") files.push_back(entry.path().string());
    std::sort(files.begin(), files.end());
    for (const std::string &f : files) add_file(f, data_root, ctx, out, warnings);
}

} // namespace

std::string mcp_dropin_path(const Context &ctx, bool project, std::string_view server_name) {
    const fs::path base = project ? fs::path(project_layout(ctx.project_root).agents) : fs::path(ctx.home.mcp);
    return (base / "mcp.d" / (std::string(server_name) + ".json")).string();
}

std::vector<McpServerConfig> collect_mcp_servers(const Context &ctx, std::vector<std::string> &warnings) {
    std::vector<McpServerConfig> out;
    if (ctx.has_home) {
        add_file((fs::path(ctx.home.mcp) / "mcp.json").string(), ctx.home.plugins, ctx, out, warnings);
        add_dropins(ctx.home.mcp, ctx.home.plugins, ctx, out, warnings);
    }
    if (!ctx.project_root.empty()) {
        const ProjectLayout project = project_layout(ctx.project_root);
        add_file(project.mcp_config, (fs::path(project.local) / "mcp").string(), ctx, out, warnings);
        add_dropins(project.agents, (fs::path(project.local) / "mcp").string(), ctx, out, warnings);
    }
    return out;
}

} // namespace agent::cli
