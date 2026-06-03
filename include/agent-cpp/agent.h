#pragma once

#include <string>
#include <string_view>
#include <functional>

namespace agent {

enum class Provider {
    Mock,
    LlamaCpp,
    OpenAICompatible,
};

struct Config {
    Provider provider;
    std::string base_url;
    std::string api_key;
    std::string model;
};

struct Session {
    Config config;
};

Session init(const Config &config);

std::string generate_text(Session &session, std::string_view prompt);
void stream_text(Session &session, std::string_view prompt, const std::function<void(std::string_view)> &on_token);

} // namespace agent