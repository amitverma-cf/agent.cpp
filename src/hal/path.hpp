#pragma once

#include <agent-cpp/agent.hpp>

#include <filesystem>
#include <string>
#include <string_view>

namespace agent::hal {

Result<std::filesystem::path> resolve_under_workspace(std::string_view workspace_dir,
                                                      std::string_view user_path, bool sandbox);

bool is_workspace_root(std::string_view workspace_dir, const std::filesystem::path &path);

bool is_safe_identifier(std::string_view name);

} // namespace agent::hal
