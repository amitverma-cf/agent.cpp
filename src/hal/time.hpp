#pragma once

#include <ctime>
#include <string>

namespace agent::hal {

bool utc_tm(std::time_t t, std::tm &out);

std::string format_utc(std::time_t t, const char *fmt);

std::string format_utc_now(const char *fmt);

} // namespace agent::hal
