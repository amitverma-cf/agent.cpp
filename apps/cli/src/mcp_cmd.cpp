// agent mcp: the MCP servers configured for this home and project.

#include "commands.hpp"
#include "mcp_config.hpp"
#include "registries.hpp"

#include <agent-cpp/hub.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

int mcp_ls(Context &ctx) {
    std::vector<std::string> warnings;
    const auto servers = collect_mcp_servers(ctx, warnings);
    for (const std::string &w : warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    if (servers.empty()) {
        std::printf("no MCP servers configured. Add them under \"mcpServers\" in %s%s%s.\n",
                    ctx.has_home ? (ctx.home.mcp + "/mcp.json").c_str() : "the home's mcp/mcp.json", ctx.project_root.empty() ? "" : " or ",
                    ctx.project_root.empty() ? "" : (ctx.project_root + "/.agents/mcp.json").c_str());
        return 0;
    }
    for (const McpServerConfig &s : servers) {
        std::string keys;
        for (const auto &entry : s.env) {
            const std::string &value = entry.second;
            size_t at = 0;
            while ((at = value.find("${secret:", at)) != std::string::npos) {
                const size_t end = value.find('}', at);
                if (end == std::string::npos) break;
                keys += (keys.empty() ? "" : ", ") + value.substr(at + 9, end - at - 9);
                at = end;
            }
        }
        std::printf("%-16s %s", s.name.c_str(), s.command.c_str());
        for (const std::string &a : s.args) std::printf(" %s", a.c_str());
        std::printf("\n    sandbox: %s%s", s.sandbox.enabled ? "on" : "OFF",
                    s.sandbox.enabled && s.sandbox.network == SandboxNetwork::Deny ? ", no network" : "");
        if (!keys.empty()) std::printf("   keys: %s", keys.c_str());
        std::printf("\n    data folder: %s\n", s.working_directory.c_str());
    }
    return 0;
}

int mcp_search(const std::string &query) {
    auto body = agent::hub::http_get_text(mcp_search_url(query, 30), agent::hub::http_options_from_env());
    if (!body.ok) return report_error(body.error);
    auto hits = parse_mcp_search(body.value);
    if (!hits.ok) return report_error(hits.error);
    if (hits.value.empty()) {
        std::printf("nothing found for '%s'\n", query.c_str());
        return 0;
    }
    for (const McpHit &h : hits.value) {
        auto entry = mcp_entry_for(h);
        std::printf("%-48s %s  %s\n    %s\n", h.name.c_str(), h.version.c_str(),
                    entry.ok ? "[can start here]" : "[hosted or container only]", h.description.substr(0, 110).c_str());
    }
    std::printf("\nadd one with: agent mcp add <name>\n");
    return 0;
}

int mcp_add(const CommandLine &cl, Context &ctx) {
    const bool project = cl.has("project");
    if (project ? ctx.project_root.empty() : !ctx.has_home) {
        std::fprintf(stderr, "agent mcp add: %s\n",
                     project ? "this folder has no .agents project (run `agent init`)" : "no home is set (run `agent setup`)");
        return 1;
    }
    const std::string wanted = cl.positionals[0];
    auto body = agent::hub::http_get_text(mcp_search_url(wanted, 30), agent::hub::http_options_from_env());
    if (!body.ok) return report_error(body.error);
    auto hits = parse_mcp_search(body.value);
    if (!hits.ok) return report_error(hits.error);
    const McpHit *chosen = nullptr;
    for (const McpHit &h : hits.value)
        if (h.name == wanted) chosen = &h;
    if (!chosen) {
        std::fprintf(stderr, "agent mcp add: '%s' is not an exact registry name. Search first: agent mcp search %s\n", wanted.c_str(),
                     wanted.c_str());
        return 1;
    }
    auto entry = mcp_entry_for(*chosen);
    if (!entry.ok) return report_error(entry.error);

    std::printf("%s %s\n  %s\n  repository: %s\n\nconfig:\n%s", chosen->name.c_str(), chosen->version.c_str(), chosen->description.c_str(),
                chosen->repository.c_str(), entry.value.json.c_str());
    std::printf("It will run in the sandbox: it writes only in its own data folder, and keys are handed to it by name.\n");
    if (!cl.has("yes") && !ask_yes_no("Add it?")) {
        std::printf("cancelled\n");
        return 1;
    }
    const std::string path = mcp_dropin_path(ctx, project, entry.value.server_name);
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << entry.value.json;
        if (!out) {
            std::fprintf(stderr, "agent mcp add: cannot write %s\n", path.c_str());
            return 1;
        }
    }
    std::printf("added '%s' (%s)\n", entry.value.server_name.c_str(), path.c_str());
    for (const std::string &s : entry.value.secrets)
        std::printf("  needs the key '%s':  agent secret set %s   then   agent secret bind %s mcp:%s\n", s.c_str(), s.c_str(), s.c_str(),
                    entry.value.server_name.c_str());
    for (const std::string &f : entry.value.to_fill) std::printf("  edit the file and set the required value %s\n", f.c_str());
    return 0;
}

int mcp_rm(const CommandLine &cl, Context &ctx) {
    for (bool project : {false, true}) {
        if (project ? ctx.project_root.empty() : !ctx.has_home) continue;
        const std::string path = mcp_dropin_path(ctx, project, cl.positionals[0]);
        std::error_code ec;
        if (fs::remove(path, ec)) {
            std::printf("removed '%s' (%s)\n", cl.positionals[0].c_str(), path.c_str());
            return 0;
        }
    }
    std::fprintf(stderr,
                 "agent mcp rm: '%s' was not added with `agent mcp add`. Servers written by hand are removed by editing mcp.json.\n",
                 cl.positionals[0].c_str());
    return 1;
}

} // namespace

int cmd_mcp(const CommandLine &cl, Context &ctx) {
    const std::string &sub = cl.subcommand;
    if (sub == "search") return mcp_search(cl.positionals[0]);
    if (sub == "add") return mcp_add(cl, ctx);
    if (sub == "rm") return mcp_rm(cl, ctx);
    if (sub == "serve") return mcp_serve(cl, ctx);
    return mcp_ls(ctx);
}

} // namespace agent::cli
