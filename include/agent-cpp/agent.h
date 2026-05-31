#pragma once

#include <string>
#include <string_view>

namespace agent {

struct Session {};

Session init();

std::string run(Session &session, std::string_view prompt);

} // namespace agent