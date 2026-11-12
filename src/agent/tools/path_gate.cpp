#include "path_gate.hpp"

#include "../../hal/sandbox.hpp"

#include <algorithm>
#include <cctype>
#include <mutex>

namespace agent::tools {

namespace fs = std::filesystem;

namespace {

std::mutex g_mu; // grant lists are read and extended from tool threads (parallel graph branches)

fs::path canonical_or_normal(const fs::path &p) {
    std::error_code ec;
    fs::path c = fs::weakly_canonical(p, ec);
    return ec ? p.lexically_normal() : c;
}

bool inside(const fs::path &child, const fs::path &parent) {
    auto c = child.begin(), p = parent.begin();
    for (; p != parent.end(); ++p, ++c) {
        if (c == child.end()) return false;
#ifdef _WIN32
        std::string a = c->string(), b = p->string();
        std::transform(a.begin(), a.end(), a.begin(), [](unsigned char x) { return static_cast<char>(std::tolower(x)); });
        std::transform(b.begin(), b.end(), b.begin(), [](unsigned char x) { return static_cast<char>(std::tolower(x)); });
        if (a != b) return false;
#else
        if (*c != *p) return false;
#endif
    }
    return true;
}

bool consumer_matches(const PathGrant &g, std::string_view consumer) { return g.consumer == "*" || g.consumer == consumer; }

} // namespace

bool grant_covers(const std::vector<PathGrant> &grants, std::string_view consumer, const fs::path &path, bool write) {
    const fs::path target = canonical_or_normal(path);
    for (const PathGrant &g : grants)
        if (consumer_matches(g, consumer) && (g.write || !write) && inside(target, canonical_or_normal(g.path))) return true;
    return false;
}

bool path_access_allowed(const WorkspaceContext &ctx, std::string_view consumer, const fs::path &path, bool write) {
    if (ctx.grants) {
        std::lock_guard<std::mutex> lock(g_mu);
        if (grant_covers(*ctx.grants, consumer, path, write)) return true;
    }
    if (!ctx.gate) return false;
    const std::string text = path.string();
    const PathGateDecision decision = ctx.gate(PathAccessRequest{.consumer = consumer, .path = text, .write = write}, ctx.gate_user_data);
    if (decision == PathGateDecision::Deny) return false;
    if (decision == PathGateDecision::AllowAlways && ctx.grants) {
        std::lock_guard<std::mutex> lock(g_mu);
        ctx.grants->push_back(PathGrant{.path = canonical_or_normal(path).string(), .write = write, .consumer = std::string(consumer)});
    }
    return true;
}

void apply_grants_to_policy(const WorkspaceContext &ctx, std::string_view consumer, SandboxPolicy &policy) {
    if (!ctx.grants) return;
    std::lock_guard<std::mutex> lock(g_mu);
    for (const PathGrant &g : *ctx.grants)
        if (consumer_matches(g, consumer)) (g.write ? policy.read_write_paths : policy.read_only_paths).push_back(g.path);
}

std::vector<fs::path> external_paths_in_command(std::string_view command, const fs::path &workspace) {
    std::vector<fs::path> out;
    const fs::path ws = canonical_or_normal(workspace);
    const std::vector<std::string> system = hal::sandbox_system_read_only_paths(true);
    size_t i = 0;
    while (i < command.size()) {
        // a token starts after whitespace, a quote, '=', '<', '>' or ';' and looks like an absolute path or ~/...
        const char c = command[i];
        const bool boundary = i == 0 || std::isspace(static_cast<unsigned char>(command[i - 1])) ||
                              std::string_view("\"'=<>;|&(").find(command[i - 1]) != std::string_view::npos;
        const bool unix_abs = c == '/' && i + 1 < command.size() && command[i + 1] != '/' && command[i + 1] != ' ';
        const bool home = c == '~' && i + 1 < command.size() && command[i + 1] == '/';
        const bool drive = std::isalpha(static_cast<unsigned char>(c)) && i + 2 < command.size() && command[i + 1] == ':' &&
                           (command[i + 2] == '\\' || command[i + 2] == '/');
        if (boundary && (unix_abs || home || drive)) {
            size_t j = i;
            while (j < command.size() && !std::isspace(static_cast<unsigned char>(command[j])) &&
                   std::string_view("\"'<>;|&)").find(command[j]) == std::string_view::npos)
                ++j;
            std::string token(command.substr(i, j - i));
            if (home) {
                const char *h = std::getenv("HOME");
                if (!h) h = std::getenv("USERPROFILE");
                if (h) token = std::string(h) + token.substr(1);
            }
            const fs::path p = canonical_or_normal(fs::path(token));
            bool skip = inside(p, ws) || token == "/dev/null";
            for (const std::string &s : system)
                if (!skip && inside(p, fs::path(s))) skip = true;
#ifdef _WIN32
            {
                std::string low = p.generic_string();
                std::transform(low.begin(), low.end(), low.begin(), [](unsigned char x) { return static_cast<char>(std::tolower(x)); });
                if (low.find("/windows") == 2 || low.find("/program files") == 2) skip = true;
            }
#endif
            if (!skip && std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
            i = j;
            continue;
        }
        ++i;
    }
    return out;
}

bool command_looks_like_write(std::string_view command) {
    if (command.find('>') != std::string_view::npos) return true;
    static constexpr std::string_view verbs[] = {"cp ",    "mv ",      "rm ",       "mkdir",        "touch ",  "tee ",
                                                 "sed -i", "chmod ",   "chown ",    "ln ",          "rmdir",   "install ",
                                                 "dd ",    "truncate", "git clone", "git checkout", "curl -o", "wget "};
    for (std::string_view v : verbs)
        if (command.find(v) != std::string_view::npos) return true;
    return false;
}

} // namespace agent::tools
