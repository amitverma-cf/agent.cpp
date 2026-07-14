#include "hal/log_file.hpp"

#include "hal/time.hpp"

#include <cstdio>
#include <filesystem>

namespace agent::hal {

std::shared_ptr<FILE> open_session_log(std::string_view workspace_dir) {
    std::string stamp = format_utc_now("agent_%Y-%m-%d_%H-%M-%S.log");
    if (stamp.empty()) stamp = "agent_session.log";

    auto log_path = (std::filesystem::path(std::string(workspace_dir)) / "logs" / stamp).string();
    FILE *f = std::fopen(log_path.c_str(), "a");
    if (!f) return {};
    return std::shared_ptr<FILE>(f, [](FILE *fp) {
        if (fp) std::fclose(fp);
    });
}

void write_log_line(FILE *file, std::string_view level_tag, std::string_view message) {
    if (!file) return;
    std::string ts = format_utc_now("%Y-%m-%dT%H:%M:%SZ");
    if (ts.empty()) ts = "?";
    std::fprintf(file, "[%s] [%.*s] %.*s\n", ts.c_str(), static_cast<int>(level_tag.size()), level_tag.data(),
                 static_cast<int>(message.size()), message.data());
    std::fflush(file);
}

} // namespace agent::hal
