#include "hal/time.hpp"

#include "hal/platform.hpp"

#include <chrono>

namespace agent::hal {

bool utc_tm(std::time_t t, std::tm &out) {
#if defined(AGENT_HAL_WINDOWS)
    return gmtime_s(&out, &t) == 0;
#else
    return gmtime_r(&t, &out) != nullptr;
#endif
}

std::string format_utc(std::time_t t, const char *fmt) {
    std::tm tm_buf{};
    if (!utc_tm(t, tm_buf)) return {};
    char buf[64];
    if (std::strftime(buf, sizeof(buf), fmt, &tm_buf) == 0) return {};
    return std::string(buf);
}

std::string format_utc_now(const char *fmt) {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    return format_utc(t, fmt);
}

} // namespace agent::hal
