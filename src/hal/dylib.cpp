#include "hal/dylib.hpp"

#include "hal/platform.hpp"

#if defined(AGENT_HAL_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace agent::hal {

#if defined(AGENT_HAL_WINDOWS)

void *open_library(std::string_view path, std::string &error) {
    int n = MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), nullptr, 0);
    std::wstring wide(static_cast<size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), wide.data(), n);

    // Search the plugin's own directory for its dependencies (e.g. ggml.dll next to the engine).
    HMODULE h = LoadLibraryExW(wide.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h) error = "LoadLibrary failed with error " + std::to_string(GetLastError());
    return reinterpret_cast<void *>(h);
}

void *find_symbol(void *library, const char *name) {
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
}

#else

void *open_library(std::string_view path, std::string &error) {
    // RTLD_LOCAL keeps each plugin's symbols (its own ggml, ORT, ...) out of the global namespace.
    void *h = dlopen(std::string(path).c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char *msg = dlerror();
        error = msg ? msg : "dlopen failed";
    }
    return h;
}

void *find_symbol(void *library, const char *name) { return dlsym(library, name); }

#endif

} // namespace agent::hal
