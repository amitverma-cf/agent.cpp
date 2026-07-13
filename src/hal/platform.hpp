#pragma once

#if defined(_WIN32)
#define AGENT_HAL_WINDOWS 1
#elif defined(__APPLE__)
#define AGENT_HAL_MACOS 1
#define AGENT_HAL_POSIX 1
#else
#define AGENT_HAL_LINUX 1
#define AGENT_HAL_POSIX 1
#endif

namespace agent::hal {

inline constexpr const char *platform_name() {
#if defined(AGENT_HAL_WINDOWS)
    return "windows";
#elif defined(AGENT_HAL_MACOS)
    return "macos";
#else
    return "linux";
#endif
}

} // namespace agent::hal
