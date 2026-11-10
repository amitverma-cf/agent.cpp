// agent mcp: the MCP servers configured for this home and project.

#include "commands.hpp"
#include "mcp_config.hpp"

#include <cstdio>

namespace agent::cli {

int cmd_mcp(const CommandLine &, Context &ctx) {
    std::vector<std::string> warnings;
    const auto servers = collect_mcp_servers(ctx, warnings);
    for (const std::string &w : warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    if (servers.empty()) {
        std::printf("no MCP servers configured. Add them under \"mcpServers\" in %s%s%s.\n", ctx.has_home ? (ctx.home.mcp + "/mcp.json").c_str() : "the home's mcp/mcp.json",
                    ctx.project_root.empty() ? "" : " or ", ctx.project_root.empty() ? "" : (ctx.project_root + "/.agents/mcp.json").c_str());
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
        std::printf("\n    sandbox: %s%s", s.sandbox.enabled ? "on" : "OFF", s.sandbox.enabled && s.sandbox.network == SandboxNetwork::Deny ? ", no network" : "");
        if (!keys.empty()) std::printf("   keys: %s", keys.c_str());
        std::printf("\n    data folder: %s\n", s.working_directory.c_str());
    }
    return 0;
}

} // namespace agent::cli
