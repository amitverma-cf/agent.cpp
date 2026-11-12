// agent sandbox: paths outside the workspace that sandboxed tools may use, and the prompt that asks for new ones.

#include "commands.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string grants_file(const Context &ctx, bool project) {
    if (project && !ctx.project_root.empty()) return (fs::path(ctx.project_root) / ".agents" / "sandbox.json").string();
    return ((ctx.has_home ? fs::path(ctx.home.root) : fs::path(ctx.config_dir)) / "sandbox-grants.json").string();
}

std::string quote(std::string_view s) { return "\"" + json_escape(s) + "\""; }

} // namespace

std::vector<PathGrant> load_grants(const std::string &file) {
    std::vector<PathGrant> out;
    std::ifstream in(file, std::ios::binary);
    if (!in) return out;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::dom::array list;
    if (parser.parse(text).get(doc) || doc["grants"].get(list)) return out;
    for (simdjson::dom::element e : list) {
        simdjson::dom::object o;
        std::string_view path, consumer;
        bool write = false;
        if (e.get(o) || o["path"].get(path) || path.empty()) continue;
        if (o["for"].get(consumer)) consumer = "*";
        if (o["write"].get(write)) write = false;
        out.push_back(PathGrant{.path = std::string(path), .write = write, .consumer = std::string(consumer)});
    }
    return out;
}

Result<void> save_grants(const std::string &file, const std::vector<PathGrant> &grants) {
    std::error_code ec;
    fs::create_directories(fs::path(file).parent_path(), ec);
    std::string out = "{\n  \"grants\": [";
    for (size_t i = 0; i < grants.size(); ++i)
        out += std::string(i ? "," : "") + "\n    {\"path\": " + quote(grants[i].path) +
               ", \"write\": " + (grants[i].write ? "true" : "false") + ", \"for\": " + quote(grants[i].consumer) + "}";
    out += grants.empty() ? "]\n}\n" : "\n  ]\n}\n";
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    f << out;
    if (!f) return fail(ErrorCode::FilesystemError, "cannot write " + file);
    return ok();
}

std::vector<PathGrant> load_all_grants(const Context &ctx) {
    std::vector<PathGrant> all = load_grants(grants_file(ctx, false));
    if (!ctx.project_root.empty()) {
        std::vector<PathGrant> project = load_grants(grants_file(ctx, true));
        all.insert(all.end(), project.begin(), project.end());
    }
    return all;
}

// The question a sandboxed tool's request becomes. Without a terminal there is nobody to ask: the answer is no, and the
// message says how to grant it ahead of time.
PathGateDecision cli_path_gate(const PathAccessRequest &request, void *user) {
    auto *ctx = static_cast<Context *>(user);
    const std::string path(request.path), consumer(request.consumer);
    if (!stdin_is_tty()) {
        std::fprintf(stderr, "[sandbox] %s wants to %s %s: not allowed. Grant it with: agent sandbox grant \"%s\"%s --for %s\n",
                     consumer.c_str(), request.write ? "write" : "read", path.c_str(), path.c_str(), request.write ? " --write" : "",
                     consumer.c_str());
        return PathGateDecision::Deny;
    }
    std::fprintf(stderr,
                 "\n[sandbox] %s wants to %s %s, which is outside the workspace.\n  [o] allow once   [a] allow always   [n] no (default): ",
                 consumer.c_str(), request.write ? "WRITE" : "read", path.c_str());
    std::fflush(stderr);
    char line[64] = {0};
    if (!std::fgets(line, sizeof(line), stdin)) return PathGateDecision::Deny;
    if (line[0] == 'o' || line[0] == 'O') return PathGateDecision::AllowOnce;
    if (line[0] == 'a' || line[0] == 'A') {
        const std::string file = grants_file(*ctx, false);
        std::vector<PathGrant> grants = load_grants(file);
        grants.push_back(PathGrant{.path = path, .write = request.write, .consumer = consumer});
        if (auto r = save_grants(file, grants); !r.ok)
            std::fprintf(stderr, "[sandbox] could not save the grant: %s\n", r.error.message.c_str());
        else std::fprintf(stderr, "[sandbox] saved in %s\n", file.c_str());
        return PathGateDecision::AllowAlways;
    }
    return PathGateDecision::Deny;
}

int cmd_sandbox(const CommandLine &cl, Context &ctx) {
    const bool project = cl.has("project");
    if (project && ctx.project_root.empty()) {
        std::fprintf(stderr, "agent sandbox: this folder has no .agents project (run `agent init`)\n");
        return 1;
    }
    const std::string file = grants_file(ctx, project);
    const std::string &sub = cl.subcommand;
    if (sub == "ls" || sub.empty()) {
        std::printf("workspace and the system's standard read-only folders are always available. Beyond that:\n");
        bool any = false;
        for (bool p : {false, true}) {
            if (p && ctx.project_root.empty()) continue;
            for (const PathGrant &g : load_grants(grants_file(ctx, p))) {
                std::printf("  %-5s %-12s %s   (%s)\n", g.write ? "write" : "read", g.consumer.c_str(), g.path.c_str(),
                            p ? "this project" : "global");
                any = true;
            }
        }
        if (!any)
            std::printf("  nothing granted. A tool that needs a path outside the workspace asks first; or `agent sandbox grant PATH "
                        "[--write] [--for run_command|files|mcp:NAME]`\n");
        return 0;
    }
    if (cl.positionals.empty()) {
        std::fprintf(stderr, "agent sandbox %s: give a path\n", sub.c_str());
        return 2;
    }
    std::error_code ec;
    const std::string path = fs::weakly_canonical(fs::absolute(cl.positionals[0], ec), ec).string();
    std::vector<PathGrant> grants = load_grants(file);
    if (sub == "grant") {
        const std::string consumer = std::string(cl.get("for", "*"));
        const bool write = cl.has("write");
        for (PathGrant &g : grants)
            if (g.path == path && g.consumer == consumer) { // upgrade or keep, never two entries for the same thing
                g.write = g.write || write;
                if (auto r = save_grants(file, grants); !r.ok) return report_error(r.error);
                std::printf("updated: %s %s for %s\n", g.write ? "write" : "read", path.c_str(), consumer.c_str());
                return 0;
            }
        grants.push_back(PathGrant{.path = path, .write = write, .consumer = consumer});
        if (auto r = save_grants(file, grants); !r.ok) return report_error(r.error);
        std::printf("granted: %s %s for %s (%s)\n", write ? "write" : "read", path.c_str(), consumer.c_str(), file.c_str());
        return 0;
    }
    if (sub == "revoke") {
        const size_t before = grants.size();
        grants.erase(std::remove_if(grants.begin(), grants.end(), [&](const PathGrant &g) { return g.path == path; }), grants.end());
        if (grants.size() == before) {
            std::fprintf(stderr, "agent sandbox revoke: nothing was granted for %s in %s\n", path.c_str(), file.c_str());
            return 1;
        }
        if (auto r = save_grants(file, grants); !r.ok) return report_error(r.error);
        std::printf("revoked %s\n", path.c_str());
        return 0;
    }
    std::fprintf(stderr, "agent sandbox: unknown subcommand '%s'\n", sub.c_str());
    return 2;
}

} // namespace agent::cli
