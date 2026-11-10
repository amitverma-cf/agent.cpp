#include "registries.hpp"

#include "cli_core.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string percent_encode(std::string_view s, bool keep_slash = false) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || (keep_slash && c == '/')) {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

std::string str_field(simdjson::dom::object o, const char *key) {
    std::string_view v;
    if (o[key].get(v)) return {};
    return std::string(v);
}

bool bool_field(simdjson::dom::object o, const char *key) {
    bool v = false;
    if (o[key].get(v)) return false;
    return v;
}

bool safe_relative(std::string_view p) {
    if (p.empty() || p[0] == '/' || p.find('\\') != std::string_view::npos || (p.size() > 1 && p[1] == ':')) return false;
    size_t start = 0;
    while (start <= p.size()) {
        size_t end = p.find('/', start);
        if (end == std::string_view::npos) end = p.size();
        const std::string_view part = p.substr(start, end - start);
        if (part == ".." || part.empty()) return false;
        start = end + 1;
    }
    return true;
}

std::string safe_name(std::string s) {
    for (char &c : s)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') c = '-';
    if (s.size() > 64) s.resize(64);
    return s.empty() ? std::string("server") : s;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string quote(std::string_view s) { return "\"" + json_escape(s) + "\""; }

} // namespace

// ---------------------------------------------------------------- skills.sh

std::string skill_search_url(std::string_view query) { return "https://skills.sh/api/search?q=" + percent_encode(query); }

Result<std::vector<SkillHit>> parse_skill_search(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc)) return fail<std::vector<SkillHit>>(ErrorCode::ParseError, "skills.sh: the answer is not valid JSON");
    simdjson::dom::array skills;
    if (doc["skills"].get(skills)) return fail<std::vector<SkillHit>>(ErrorCode::ParseError, "skills.sh: the answer has no \"skills\" list");
    std::vector<SkillHit> out;
    for (simdjson::dom::element e : skills) {
        simdjson::dom::object o;
        if (e.get(o)) continue;
        SkillHit h;
        h.id = str_field(o, "id");
        h.source = str_field(o, "source");
        h.skill_id = str_field(o, "skillId");
        h.name = str_field(o, "name");
        int64_t installs = 0;
        if (!o["installs"].get(installs) && installs > 0) h.installs = static_cast<uint64_t>(installs);
        if (h.id.empty() || h.source.empty()) continue;
        if (h.skill_id.empty()) h.skill_id = h.name;
        out.push_back(std::move(h));
    }
    return ok(std::move(out));
}

Result<SkillHit> parse_skill_id(std::string_view id) {
    const size_t a = id.find('/');
    const size_t b = a == std::string_view::npos ? a : id.find('/', a + 1);
    if (a == std::string_view::npos || b == std::string_view::npos || a == 0 || b == a + 1 || b + 1 >= id.size())
        return fail<SkillHit>(ErrorCode::InvalidConfig, "a skill id looks like owner/repo/skill-name, got '" + std::string(id) + "'");
    SkillHit h;
    h.id = std::string(id);
    h.source = std::string(id.substr(0, b));
    h.skill_id = std::string(id.substr(b + 1));
    h.name = h.skill_id;
    if (!safe_relative(h.source) || !safe_relative(h.skill_id)) return fail<SkillHit>(ErrorCode::InvalidConfig, "invalid skill id '" + std::string(id) + "'");
    return ok(std::move(h));
}

Result<SkillFiles> find_skill_in_tree(std::string_view tree_json, std::string_view skill_id) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(tree_json).get(doc)) return fail<SkillFiles>(ErrorCode::ParseError, "GitHub: the repository listing is not valid JSON");
    simdjson::dom::array tree;
    if (doc["tree"].get(tree)) return fail<SkillFiles>(ErrorCode::ParseError, "GitHub: the repository listing has no \"tree\"");
    bool truncated = false;
    if (!doc["truncated"].get(truncated) && truncated) return fail<SkillFiles>(ErrorCode::HttpError, "GitHub: the repository is too large to list");

    struct Blob {
        std::string path;
        uint64_t size;
    };
    std::vector<Blob> blobs;
    for (simdjson::dom::element e : tree) {
        simdjson::dom::object o;
        if (e.get(o) || str_field(o, "type") != "blob") continue;
        int64_t size = 0;
        (void)o["size"].get(size);
        blobs.push_back({str_field(o, "path"), static_cast<uint64_t>(std::max<int64_t>(0, size))});
    }
    // Skill folders: the parent of every SKILL.md whose folder is named like the skill.
    std::string best_dir;
    bool found = false;
    const std::string leaf = fs::path(std::string(skill_id)).filename().string();
    for (const Blob &b : blobs) {
        std::string dir;
        if (b.path == "SKILL.md") dir = "";
        else if (b.path.ends_with("/SKILL.md")) dir = b.path.substr(0, b.path.size() - 9);
        else continue;
        const std::string base = fs::path(dir).filename().string();
        if (base != leaf && !(dir.empty() && !found)) continue;
        if (!found || (!dir.empty() && (best_dir.empty() || dir.size() < best_dir.size()))) {
            best_dir = dir;
            found = true;
        }
    }
    if (!found) return fail<SkillFiles>(ErrorCode::KeyNotFound, "no folder named '" + leaf + "' with a SKILL.md in that repository");

    SkillFiles out;
    out.dir = best_dir;
    const std::string prefix = best_dir.empty() ? std::string() : best_dir + "/";
    for (const Blob &b : blobs) {
        if (!prefix.empty() && !b.path.starts_with(prefix)) continue;
        out.files.push_back(b.path);
        out.bytes += b.size;
    }
    return ok(std::move(out));
}

Result<SkillFiles> plan_skill(const SkillHit &hit, const agent::hub::HttpOptions &http) {
    auto tree = agent::hub::http_get_text("https://api.github.com/repos/" + hit.source + "/git/trees/HEAD?recursive=1", http, 32u << 20);
    if (!tree.ok) return fail<SkillFiles>(tree.error.code, "cannot list " + hit.source + ": " + tree.error.message);
    auto found = find_skill_in_tree(tree.value, hit.skill_id);
    if (!found.ok) return found;
    constexpr size_t kMaxFiles = 200;
    constexpr uint64_t kMaxBytes = 20u << 20;
    if (found.value.files.size() > kMaxFiles) return fail<SkillFiles>(ErrorCode::InvalidConfig, "the skill has " + std::to_string(found.value.files.size()) + " files (limit " + std::to_string(kMaxFiles) + ")");
    if (found.value.bytes > kMaxBytes) return fail<SkillFiles>(ErrorCode::InvalidConfig, "the skill is larger than 20 MiB");
    return found;
}

Result<SkillInstall> install_skill(const SkillHit &hit, const SkillFiles &planned, const std::string &skills_dir, const agent::hub::HttpOptions &http) {
    const auto found = ok(planned);

    const std::string leaf = safe_name(fs::path(hit.skill_id).filename().string());
    std::error_code ec;
    const fs::path staging = fs::path(skills_dir) / ("." + leaf + ".installing");
    const fs::path target = fs::path(skills_dir) / leaf;
    fs::remove_all(staging, ec);
    fs::create_directories(staging, ec);

    SkillInstall result;
    std::string manifest;
    const std::string prefix = found.value.dir.empty() ? std::string() : found.value.dir + "/";
    for (const std::string &path : found.value.files) {
        const std::string rel = path.substr(prefix.size());
        if (!safe_relative(rel)) {
            fs::remove_all(staging, ec);
            return fail<SkillInstall>(ErrorCode::PermissionDenied, "refusing the file '" + path + "'");
        }
        auto body = agent::hub::http_get_text("https://raw.githubusercontent.com/" + hit.source + "/HEAD/" + percent_encode(path, true), http, 8u << 20);
        if (!body.ok) {
            fs::remove_all(staging, ec);
            return fail<SkillInstall>(body.error.code, "cannot download " + path + ": " + body.error.message);
        }
        const fs::path out = staging / fs::path(rel);
        fs::create_directories(out.parent_path(), ec);
        {
            std::ofstream f(out, std::ios::binary | std::ios::trunc);
            f.write(body.value.data(), static_cast<std::streamsize>(body.value.size()));
            if (!f) {
                fs::remove_all(staging, ec);
                return fail<SkillInstall>(ErrorCode::FilesystemError, "cannot write " + out.string());
            }
        }
        manifest += std::string(manifest.empty() ? "" : ",") + "{\"path\":" + quote(rel) + ",\"sha256\":\"" + agent::hub::sha256_hex(body.value) + "\"}";
        result.files.push_back(rel);
        const std::string ext = lower(fs::path(rel).extension().string());
        static const std::set<std::string> docs = {".md", ".txt", ".json", ".yaml", ".yml", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".csv", ".toml", ""};
        if (!docs.count(ext) && rel != "LICENSE") result.scripts.push_back(rel);
    }
    {
        std::ofstream meta(staging / ".source.json");
        meta << "{\"id\":" << quote(hit.id) << ",\"repository\":\"https://github.com/" << hit.source << "\",\"directory\":" << quote(found.value.dir)
             << ",\"installed_at\":" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count() << ",\"files\":[" << manifest << "]}\n";
    }
    fs::remove_all(target, ec);
    fs::rename(staging, target, ec);
    if (ec) {
        fs::remove_all(staging, ec);
        return fail<SkillInstall>(ErrorCode::FilesystemError, "cannot move the skill into " + target.string() + ": " + ec.message());
    }
    result.dir = target.string();
    return ok(std::move(result));
}

// ---------------------------------------------------------------- MCP registry

std::string mcp_search_url(std::string_view query, int limit) {
    return "https://registry.modelcontextprotocol.io/v0.1/servers?search=" + percent_encode(query) + "&limit=" + std::to_string(limit);
}

Result<std::vector<McpHit>> parse_mcp_search(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc)) return fail<std::vector<McpHit>>(ErrorCode::ParseError, "MCP registry: the answer is not valid JSON");
    simdjson::dom::array servers;
    if (doc["servers"].get(servers)) return fail<std::vector<McpHit>>(ErrorCode::ParseError, "MCP registry: the answer has no \"servers\" list");
    std::vector<McpHit> out;
    std::map<std::string, size_t> index;
    for (simdjson::dom::element e : servers) {
        simdjson::dom::object entry, server;
        if (e.get(entry) || entry["server"].get(server)) continue;
        McpHit h;
        h.name = str_field(server, "name");
        if (h.name.empty()) continue;
        h.description = str_field(server, "description");
        h.version = str_field(server, "version");
        simdjson::dom::object repo;
        if (!server["repository"].get(repo)) h.repository = str_field(repo, "url");
        simdjson::dom::array packages;
        if (!server["packages"].get(packages)) {
            for (simdjson::dom::element pe : packages) {
                simdjson::dom::object po;
                if (pe.get(po)) continue;
                McpPackage p;
                p.type = str_field(po, "registryType");
                p.identifier = str_field(po, "identifier");
                p.version = str_field(po, "version");
                simdjson::dom::object transport;
                if (!po["transport"].get(transport)) p.transport = str_field(transport, "type");
                simdjson::dom::array args;
                if (!po["packageArguments"].get(args)) {
                    for (simdjson::dom::element ae : args) {
                        simdjson::dom::object ao;
                        if (ae.get(ao)) continue;
                        const std::string type = str_field(ao, "type"), value = str_field(ao, "value");
                        if (value.empty()) continue;
                        if (type == "named") p.arguments.push_back(str_field(ao, "name") + "=" + value);
                        else p.arguments.push_back(value);
                    }
                }
                simdjson::dom::array vars;
                if (!po["environmentVariables"].get(vars)) {
                    for (simdjson::dom::element ve : vars) {
                        simdjson::dom::object vo;
                        if (ve.get(vo)) continue;
                        McpEnvVar v;
                        v.name = str_field(vo, "name");
                        v.description = str_field(vo, "description");
                        v.default_value = str_field(vo, "default");
                        v.secret = bool_field(vo, "isSecret");
                        v.required = bool_field(vo, "isRequired");
                        if (!v.name.empty()) p.env.push_back(std::move(v));
                    }
                }
                if (!p.identifier.empty()) h.packages.push_back(std::move(p));
            }
        }
        simdjson::dom::array remotes;
        if (!server["remotes"].get(remotes))
            for (simdjson::dom::element re : remotes) {
                simdjson::dom::object ro;
                if (!re.get(ro) && !str_field(ro, "url").empty()) h.remotes.push_back(str_field(ro, "url"));
            }

        bool latest = true; // a missing flag means the only version listed
        simdjson::dom::object meta, official;
        if (!entry["_meta"].get(meta) && !meta["io.modelcontextprotocol.registry/official"].get(official)) {
            bool flag = false;
            if (!official["isLatest"].get(flag)) latest = flag;
        }
        // One row per server: the first version seen, replaced when a version flagged latest turns up.
        auto it = index.find(h.name);
        if (it == index.end()) {
            index[h.name] = out.size();
            out.push_back(std::move(h));
        } else if (latest) {
            out[it->second] = std::move(h);
        }
    }
    return ok(std::move(out));
}

Result<McpEntry> mcp_entry_for(const McpHit &hit) {
    for (const McpPackage &p : hit.packages) {
        if (!p.transport.empty() && p.transport != "stdio") continue;
        std::string command;
        std::vector<std::string> args;
        const std::string pinned = p.version.empty() ? p.identifier : p.identifier + "@" + p.version;
        if (p.type == "npm") {
            command = "npx";
            args = {"-y", pinned};
        } else if (p.type == "pypi") {
            command = "uvx";
            args = {pinned};
        } else {
            continue;
        }
        args.insert(args.end(), p.arguments.begin(), p.arguments.end());

        McpEntry entry;
        std::string slash = hit.name.substr(hit.name.rfind('/') == std::string::npos ? 0 : hit.name.rfind('/') + 1);
        entry.server_name = safe_name(slash);
        std::string env;
        for (const McpEnvVar &v : p.env) {
            std::string value;
            if (v.secret) {
                std::string secret = lower(v.name);
                for (char &c : secret)
                    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' && c != '.') c = '_';
                value = "${secret:" + secret + "}";
                entry.secrets.push_back(secret);
            } else if (!v.default_value.empty()) {
                value = v.default_value;
            } else if (v.required) {
                entry.to_fill.push_back(v.name);
            } else {
                continue;
            }
            env += std::string(env.empty() ? "" : ",\n        ") + quote(v.name) + ": " + quote(value);
        }
        std::string args_json;
        for (size_t i = 0; i < args.size(); ++i) args_json += std::string(i ? ", " : "") + quote(args[i]);
        entry.json = "{\n  \"mcpServers\": {\n    " + quote(entry.server_name) + ": {\n      \"command\": " + quote(command) + ",\n      \"args\": [" + args_json + "]";
        if (!env.empty()) entry.json += ",\n      \"env\": {\n        " + env + "\n      }";
        entry.json += "\n    }\n  }\n}\n";
        return ok(std::move(entry));
    }
    std::string why = "no package of '" + hit.name + "' can be started by agent.cpp (it needs an npm or pypi package that speaks stdio)";
    if (!hit.remotes.empty()) why += "; it is also offered as a hosted server at " + hit.remotes[0] + ", which agent.cpp does not connect to yet";
    return fail<McpEntry>(ErrorCode::InvalidConfig, why);
}

} // namespace agent::cli
