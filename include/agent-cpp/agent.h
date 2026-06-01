#pragma once

#include <string>
#include <string_view>

namespace agent {

enum Provider {
    Mock,
    LlamaCpp,
    OpenAICompatible,
};

struct Config {
    Provider provider = Provider::Mock;
    std::string base_url;
    std::string api_key;
    std::string model;
};

struct Session {
    Config config;
};

Session init(const Config &config);

std::string run(Session &session, std::string_view prompt);

} // namespace agent