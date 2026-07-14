#include "hal/path.hpp"

#include <cctype>
#include <system_error>

namespace agent::hal {
namespace {

bool is_prefix_path(const std::filesystem::path &root, const std::filesystem::path &candidate) {
    auto root_it = root.begin();
    auto cand_it = candidate.begin();
    for (; root_it != root.end() && cand_it != candidate.end(); ++root_it, ++cand_it) {
        if (*root_it != *cand_it) return false;
    }
    return root_it == root.end();
}

} // namespace

Result<std::filesystem::path> resolve_under_workspace(std::string_view workspace_dir, std::string_view user_path, bool sandbox) {
    if (workspace_dir.empty()) return fail<std::filesystem::path>(ErrorCode::InvalidConfig, "workspace_dir is empty.");

    std::filesystem::path ws{std::string(workspace_dir)};
    std::filesystem::path raw{std::string(user_path)};

    std::filesystem::path joined;
    if (raw.empty() || raw == ".") joined = ws;
    else if (raw.is_absolute()) joined = raw;
    else joined = ws / raw;

    std::error_code ec;
    auto canonical_ws = std::filesystem::weakly_canonical(ws, ec);
    if (ec) canonical_ws = std::filesystem::absolute(ws, ec);
    if (ec) return fail<std::filesystem::path>(ErrorCode::FilesystemError, "Cannot resolve workspace: " + ec.message());

    auto canonical = std::filesystem::weakly_canonical(joined, ec);
    if (ec) canonical = std::filesystem::absolute(joined, ec);
    if (ec) return fail<std::filesystem::path>(ErrorCode::FilesystemError, "Cannot resolve path: " + ec.message());

    if (sandbox && !is_prefix_path(canonical_ws, canonical)) {
        return fail<std::filesystem::path>(ErrorCode::SandboxViolation,
                                           "Path is outside workspace_dir (sandbox mode is enabled): " + canonical.string());
    }

    return ok(std::move(canonical));
}

bool is_workspace_root(std::string_view workspace_dir, const std::filesystem::path &path) {
    std::error_code ec;
    auto canonical_ws = std::filesystem::weakly_canonical(std::filesystem::path(std::string(workspace_dir)), ec);
    if (ec) return false;
    auto canonical_path = std::filesystem::weakly_canonical(path, ec);
    if (ec) return false;
    return canonical_ws == canonical_path;
}

bool is_safe_identifier(std::string_view name) {
    if (name.empty() || name.size() > 64) return false;
    char c0 = name[0];
    if (!(std::isalpha(static_cast<unsigned char>(c0)) || c0 == '_')) return false;
    for (size_t i = 1; i < name.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(name[i]);
        if (!(std::isalnum(c) || c == '_')) return false;
    }
    return true;
}

} // namespace agent::hal
