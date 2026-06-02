#include <agent-cpp/agent.h>
#include <httplib.h>
#include <json.hpp>

namespace agent {

Session init(const Config &config) {
    return {.config = config};
}

static std::string run_openai_compatible(Session &session, std::string_view prompt) {
    httplib::Client client(session.config.base_url);
    nlohmann::json request = {{"model", session.config.model}, {"messages", {{{"role", "user"}, {"content", prompt}}}}};
    httplib::Headers headers;
    if (!session.config.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + session.config.api_key);
    }
    auto response = client.Post("/v1/chat/completions", headers, request.dump(), "application/json");
    if (!response) {
        return "Error: Internal HTTP Client request failed completely.";
    }
    if (response->status != 200) {
        return "HTTP Error " + std::to_string(response->status) + ": " + response->body;
    }
    auto parsed = nlohmann::json::parse(response->body);
    return parsed["choices"][0]["message"]["content"];
}

std::string run(Session &session, std::string_view prompt) {
    switch (session.config.provider) {
    case Provider::Mock:
        return "Mock Response";

    case Provider::LlamaCpp:
        return "not implemented";

    case Provider::OpenAICompatible:
        return run_openai_compatible(session, prompt);

    default:
        return std::string(prompt);
    }
}

} // namespace agent