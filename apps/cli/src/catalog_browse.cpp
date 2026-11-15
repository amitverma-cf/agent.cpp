#include "catalog_browse.hpp"

#include "mcp_config.hpp"
#include "registries.hpp"

#include <agent-cpp/hub.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

bool valid_name(std::string_view n) {
    if (n.empty() || n.size() > 64) return false;
    return std::all_of(n.begin(), n.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-' || c == '.'; }) &&
           n.front() != '.';
}

std::string slurp(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// A context that finds the project of `workspace`, which the chat can be pointed at after the program started.
Context with_workspace(const Context &ctx, const std::string &workspace) {
    Context c = ctx;
    if (c.project_root.empty() && !workspace.empty()) c.project_root = find_project_root(workspace);
    return c;
}

// Where a server added here is written: the global mcp.d, or the project's when there is no home.
fs::path dropin_for(const Context &ctx, const std::string &workspace, std::string_view name, std::string_view suffix = ".json") {
    const fs::path base = ctx.has_home ? fs::path(ctx.home.mcp) / "mcp.d" : fs::path(project_layout(workspace).agents) / "mcp.d";
    return base / (std::string(name) + std::string(suffix));
}

std::string json_list(const std::vector<std::string> &items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? ", " : "") + std::string("\"") + json_escape(items[i]) + "\"";
    return out + "]";
}

Result<std::string> write_server(const Context &ctx, const std::string &workspace, const std::string &name, const std::string &body,
                                 const std::string &after) {
    if (!valid_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "the server name may use letters, digits, '-' and '_'");
    const fs::path path = dropin_for(ctx, workspace, name);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    fs::remove(dropin_for(ctx, workspace, name, ".json.disabled"), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "{\"mcpServers\": {\"" << json_escape(name) << "\": " << body << "}}\n";
    if (!out.flush()) return fail<std::string>(ErrorCode::FilesystemError, "cannot write " + path.string());
    return ok("Added the MCP server '" + name + "'. " + after);
}

constexpr std::string_view kStartNote = "It starts with the next message and runs in the sandbox; its tools appear then.";

} // namespace

std::span<const CuratedMcp> curated_mcp() {
    static const CuratedMcp list[] = {
        {"filesystem",
         "Read, write and search the files of your project folder.",
         "npx",
         {"-y", "@modelcontextprotocol/server-filesystem"},
         true,
         ""},
        {"memory",
         "Remember people, places and facts between chats as a small knowledge graph.",
         "npx",
         {"-y", "@modelcontextprotocol/server-memory"},
         false,
         ""},
        {"sequential-thinking",
         "Think a hard problem through in steps and revise them.",
         "npx",
         {"-y", "@modelcontextprotocol/server-sequential-thinking"},
         false,
         ""},
        {"fetch", "Read web pages and turn them into plain text.", "uvx", {"mcp-server-fetch"}, false, ""},
        {"git", "Look at the history, changes and branches of a git folder.", "uvx", {"mcp-server-git"}, false, ""},
        {"time", "The current time and conversion between time zones.", "uvx", {"mcp-server-time"}, false, ""},
        {"playwright",
         "Drive a web browser: open pages, click, fill forms, read what is shown.",
         "npx",
         {"-y", "@playwright/mcp@latest"},
         false,
         ""},
        {"github",
         "Repositories, issues and pull requests on GitHub (needs a personal access token).",
         "npx",
         {"-y", "@modelcontextprotocol/server-github"},
         false,
         "GITHUB_PERSONAL_ACCESS_TOKEN"},
    };
    return list;
}

Result<McpSpec> parse_mcp_spec(std::string_view text, std::string_view name) {
    McpSpec spec;
    std::vector<std::string> words;
    {
        std::string w;
        char quote = 0;
        for (const char c : text) {
            if (quote) {
                if (c == quote) quote = 0;
                else w += c;
            } else if (c == '"' || c == '\'') {
                quote = c;
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                if (!w.empty()) words.push_back(std::move(w));
                w.clear();
            } else {
                w += c;
            }
        }
        if (!w.empty()) words.push_back(std::move(w));
    }
    if (words.empty()) return fail<McpSpec>(ErrorCode::InvalidConfig, "give a package, a command or an address");
    std::string base;
    if (words[0].rfind("http://", 0) == 0 || words[0].rfind("https://", 0) == 0) {
        spec.url = words[0];
        std::string host = words[0].substr(words[0].find("//") + 2);
        host = host.substr(0, host.find_first_of("/:?"));
        if (host.rfind("www.", 0) == 0 || host.rfind("mcp.", 0) == 0) host.erase(0, host.find('.') + 1);
        base = host.substr(0, host.find('.'));
    } else if (words.size() == 1 && (words[0][0] == '@' || words[0].find('/') == std::string::npos) && words[0] != "npx" &&
               words[0] != "uvx" && words[0] != "node" && words[0] != "python" && words[0] != "python3" && words[0] != "bun" &&
               words[0].find('.') == std::string::npos) {
        spec.command = "npx"; // a bare package name: an npm package
        spec.args = {"-y", words[0]};
        base = words[0];
    } else if (words.size() == 1 && words[0][0] == '@') {
        spec.command = "npx";
        spec.args = {"-y", words[0]};
        base = words[0];
    } else {
        spec.command = words[0];
        spec.args.assign(words.begin() + 1, words.end());
        if (spec.command == "npx" && (spec.args.empty() || spec.args[0] != "-y")) spec.args.insert(spec.args.begin(), "-y");
        for (const std::string &a : spec.args)
            if (!a.empty() && a[0] != '-') {
                base = a;
                break;
            }
        if (base.empty()) base = spec.command;
    }
    if (name.empty()) {
        // @modelcontextprotocol/server-filesystem -> filesystem; mcp-server-time -> time
        std::string n = base.substr(base.rfind('/') == std::string::npos ? 0 : base.rfind('/') + 1);
        if (n.find('@', 1) != std::string::npos) n = n.substr(0, n.find('@', 1));
        for (const char *prefix : {"server-", "mcp-server-", "mcp-"})
            if (n.rfind(prefix, 0) == 0) {
                n.erase(0, std::string_view(prefix).size());
                break;
            }
        for (char &c : n)
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '-';
        spec.name = n.empty() ? std::string("server") : n.substr(0, 64);
    } else {
        spec.name = std::string(name);
    }
    if (!valid_name(spec.name)) return fail<McpSpec>(ErrorCode::InvalidConfig, "the server name may use letters, digits, '-' and '_'");
    return ok(std::move(spec));
}

Result<std::string> mcp_add_curated(const Context &ctx, const std::string &workspace, std::string_view id, std::string_view folder) {
    for (const CuratedMcp &c : curated_mcp()) {
        if (c.id != id) continue;
        std::vector<std::string> args(c.args.begin(), c.args.end());
        if (c.needs_folder) args.push_back(folder.empty() ? workspace : std::string(folder));
        std::string body = "{\"command\": \"" + std::string(c.command) + "\", \"args\": " + json_list(args);
        std::string after(kStartNote);
        if (!c.secret.empty()) {
            body += ", \"env\": {\"" + std::string(c.secret) + "\": \"${secret:" + std::string(c.id) + "}\"}";
            after = "It needs a key named '" + std::string(c.id) + "' kept in the credential store before it can start.";
        }
        body += "}";
        return write_server(ctx, workspace, std::string(c.id), body, after);
    }
    return fail<std::string>(ErrorCode::KeyNotFound, "'" + std::string(id) + "' is not in the built-in list");
}

Result<std::string> mcp_add_spec(const Context &ctx, const std::string &workspace, const McpSpec &spec) {
    std::string body;
    if (!spec.url.empty()) body = "{\"url\": \"" + json_escape(spec.url) + "\"}";
    else body = "{\"command\": \"" + json_escape(spec.command) + "\", \"args\": " + json_list(spec.args) + "}";
    return write_server(ctx, workspace, spec.name, body, std::string(kStartNote));
}

std::vector<McpInstalled> list_mcp(const Context &ctx, const std::string &workspace) {
    const Context c = with_workspace(ctx, workspace);
    std::vector<McpInstalled> out;
    std::vector<std::string> warnings;
    for (const McpServerConfig &s : collect_mcp_servers(c, warnings)) {
        McpInstalled m;
        m.name = s.name;
        m.detail = s.url.empty() ? s.command : s.url;
        for (const std::string &a : s.args) m.detail += " " + a;
        std::error_code ec;
        const bool dropin = (c.has_home && fs::exists(mcp_dropin_path(c, false, s.name), ec)) ||
                            (!c.project_root.empty() && fs::exists(mcp_dropin_path(c, true, s.name), ec));
        m.removable = dropin;
        m.where = dropin ? "added here" : "mcp.json";
        out.push_back(std::move(m));
    }
    // Switched off: the file stays as NAME.json.disabled.
    std::vector<fs::path> dirs;
    if (c.has_home) dirs.push_back(fs::path(c.home.mcp) / "mcp.d");
    if (!c.project_root.empty()) dirs.push_back(fs::path(project_layout(c.project_root).agents) / "mcp.d");
    for (const fs::path &d : dirs) {
        std::error_code ec;
        for (const auto &e : fs::directory_iterator(d, ec)) {
            const std::string file = e.path().filename().string();
            if (!file.ends_with(".json.disabled")) continue;
            McpInstalled m;
            m.name = file.substr(0, file.size() - 14);
            m.detail = "switched off";
            m.where = "added here";
            m.enabled = false;
            m.removable = true;
            out.push_back(std::move(m));
        }
    }
    return out;
}

Result<std::string> mcp_add_entry_json(const Context &ctx, const std::string &workspace, std::string_view name, const std::string &json) {
    if (!valid_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "the server name may use letters, digits, '-' and '_'");
    const fs::path path = dropin_for(ctx, workspace, name);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << json;
    if (!out.flush()) return fail<std::string>(ErrorCode::FilesystemError, "cannot write " + path.string());
    return ok("Added the MCP server '" + std::string(name) + "'. " + std::string(kStartNote) + " If it needs a key, it is named in " +
              path.string() + ".");
}

Result<std::string> mcp_set_enabled(const Context &ctx, const std::string &workspace, std::string_view name, bool enabled) {
    const Context c = with_workspace(ctx, workspace);
    std::vector<fs::path> candidates;
    if (c.has_home) candidates.push_back(fs::path(mcp_dropin_path(c, false, name)));
    if (!c.project_root.empty()) candidates.push_back(fs::path(mcp_dropin_path(c, true, name)));
    for (const fs::path &on : candidates) {
        fs::path off = on;
        off += ".disabled";
        std::error_code ec;
        if (enabled && fs::exists(off, ec)) {
            fs::rename(off, on, ec);
            return ec ? fail<std::string>(ErrorCode::FilesystemError, ec.message())
                      : ok("Switched on '" + std::string(name) + "'. " + std::string(kStartNote));
        }
        if (!enabled && fs::exists(on, ec)) {
            fs::rename(on, off, ec);
            return ec ? fail<std::string>(ErrorCode::FilesystemError, ec.message())
                      : ok("Switched off '" + std::string(name) + "'; its tools are gone from the next message.");
        }
    }
    return fail<std::string>(ErrorCode::KeyNotFound,
                             "'" + std::string(name) +
                                 "' was not added here; servers written into mcp.json are changed by editing that file");
}

Result<std::string> mcp_remove_named(const Context &ctx, const std::string &workspace, std::string_view name) {
    const Context c = with_workspace(ctx, workspace);
    if (!valid_name(name)) return fail<std::string>(ErrorCode::InvalidConfig, "that is not a server name");
    bool removed = false;
    std::error_code ec;
    for (bool project : {false, true}) {
        if (project ? c.project_root.empty() : !c.has_home) continue;
        const fs::path on = mcp_dropin_path(c, project, name);
        fs::path off = on;
        off += ".disabled";
        removed = fs::remove(on, ec) || removed;
        removed = fs::remove(off, ec) || removed;
    }
    if (!removed)
        return fail<std::string>(ErrorCode::KeyNotFound,
                                 "'" + std::string(name) +
                                     "' was not added here; servers written into mcp.json are removed by editing that file");
    return ok("Removed '" + std::string(name) + "'; its tools are gone from the next message.");
}

// ---------------------------------------------------------------- skills

std::span<const CuratedSkill> curated_skills() {
    static const CuratedSkill list[] = {
        {"github/awesome-copilot/git-commit", "Write clear commit messages in the conventional format."},
        {"github/awesome-copilot/conventional-commit", "Commit messages that follow the Conventional Commits style."},
        {"anthropics/skills/pdf", "Read, fill, merge and split PDF files."},
        {"anthropics/skills/docx", "Create and edit Word documents."},
        {"anthropics/skills/xlsx", "Create and edit spreadsheets."},
        {"anthropics/skills/pptx", "Create and edit slide decks."},
        {"anthropics/skills/frontend-design", "Build good-looking web pages and interfaces."},
        {"anthropics/skills/skill-creator", "Write a new skill of your own."},
    };
    return list;
}

std::vector<std::string> skill_dirs(const Context &ctx, const std::string &workspace) {
    std::vector<std::string> dirs;
    if (ctx.has_home) dirs.push_back(ctx.home.skills);
    if (!workspace.empty()) dirs.push_back((fs::path(workspace) / ".agents" / "skills").string());
    return dirs;
}

namespace {

std::string skill_description(const fs::path &md) {
    std::ifstream in(md);
    std::string line;
    for (int n = 0; n < 40 && std::getline(in, line); ++n) {
        if (line.rfind("description:", 0) != 0) continue;
        std::string d = line.substr(12);
        const size_t a = d.find_first_not_of(" \t\"'");
        if (a == std::string::npos) return {};
        d = d.substr(a);
        while (!d.empty() && (d.back() == '\r' || d.back() == '"' || d.back() == '\'' || d.back() == ' ')) d.pop_back();
        return d.size() > 160 ? d.substr(0, 157) + "..." : d;
    }
    return {};
}

fs::path find_skill_folder(const Context &ctx, const std::string &workspace, std::string_view name) {
    if (!valid_name(name)) return {};
    for (const std::string &d : skill_dirs(ctx, workspace)) {
        const fs::path p = fs::path(d) / std::string(name);
        std::error_code ec;
        if (fs::exists(p / "SKILL.md", ec) || fs::exists(p / "SKILL.md.off", ec)) return p;
    }
    return {};
}

} // namespace

std::vector<SkillInstalled> list_skills(const Context &ctx, const std::string &workspace) {
    std::vector<SkillInstalled> out;
    const auto dirs = skill_dirs(ctx, workspace);
    for (size_t i = 0; i < dirs.size(); ++i) {
        std::error_code ec;
        std::vector<fs::path> folders;
        for (const auto &e : fs::directory_iterator(dirs[i], ec))
            if (e.is_directory(ec) && !e.path().filename().string().starts_with(".")) folders.push_back(e.path());
        std::sort(folders.begin(), folders.end());
        for (const fs::path &f : folders) {
            const bool on = fs::exists(f / "SKILL.md", ec);
            if (!on && !fs::exists(f / "SKILL.md.off", ec)) continue;
            SkillInstalled s;
            s.name = f.filename().string();
            s.where = ctx.has_home && i == 0 ? "every project" : "this project";
            s.enabled = on;
            s.from_registry = fs::exists(f / ".source.json", ec);
            s.description = skill_description(f / (on ? "SKILL.md" : "SKILL.md.off"));
            out.push_back(std::move(s));
        }
    }
    return out;
}

Result<std::string> skill_install_id(const Context &ctx, const std::string &workspace, std::string_view id, bool project) {
    const auto dirs = skill_dirs(ctx, workspace);
    if (dirs.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "there is no folder to keep skills in yet");
    const std::string dir = project || !ctx.has_home ? dirs.back() : dirs.front();
    auto hit = parse_skill_id(id);
    if (!hit.ok) return fail<std::string>(hit.error.code, "a skill is named owner/repo/skill, for example anthropics/skills/pdf");
    const auto http = agent::hub::http_options_from_env();
    auto plan = plan_skill(hit.value, http);
    if (!plan.ok) return fail<std::string>(plan.error.code, plan.error.message);
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto done = install_skill(hit.value, plan.value, dir, http);
    if (!done.ok) return fail<std::string>(done.error.code, done.error.message);
    return ok("Installed the skill '" + hit.value.skill_id + "' (" + std::to_string(done.value.files.size()) +
              " files). It is offered to the agent from the next message.");
}

Result<std::string> skill_set_enabled(const Context &ctx, const std::string &workspace, std::string_view name, bool enabled) {
    const fs::path folder = find_skill_folder(ctx, workspace, name);
    if (folder.empty()) return fail<std::string>(ErrorCode::KeyNotFound, "no skill named '" + std::string(name) + "'");
    std::error_code ec;
    if (enabled) fs::rename(folder / "SKILL.md.off", folder / "SKILL.md", ec);
    else fs::rename(folder / "SKILL.md", folder / "SKILL.md.off", ec);
    if (ec && ec != std::errc::no_such_file_or_directory) return fail<std::string>(ErrorCode::FilesystemError, ec.message());
    return ok(std::string(enabled ? "Switched on" : "Switched off") + " the skill '" + std::string(name) + "' (from the next message).");
}

Result<std::string> skill_remove_named(const Context &ctx, const std::string &workspace, std::string_view name) {
    const fs::path folder = find_skill_folder(ctx, workspace, name);
    if (folder.empty()) return fail<std::string>(ErrorCode::KeyNotFound, "no skill named '" + std::string(name) + "'");
    std::error_code ec;
    fs::remove_all(folder, ec);
    if (ec) return fail<std::string>(ErrorCode::FilesystemError, ec.message());
    return ok("Removed the skill '" + std::string(name) + "'.");
}

} // namespace agent::cli
