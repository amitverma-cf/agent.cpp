#pragma once

#include <string>
#include <string_view>

namespace agent::hal {

// Opens a shared library. Returns nullptr and fills `error` on failure. Handles are never closed:
// engine plugins stay loaded for the life of the process.
void *open_library(std::string_view path, std::string &error);

void *find_symbol(void *library, const char *name);

} // namespace agent::hal
