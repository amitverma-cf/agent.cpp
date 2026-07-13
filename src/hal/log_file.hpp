#pragma once

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

namespace agent::hal {

std::shared_ptr<FILE> open_session_log(std::string_view workspace_dir);

void write_log_line(FILE *file, std::string_view level_tag, std::string_view message);

} // namespace agent::hal
