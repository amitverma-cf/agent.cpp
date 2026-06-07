#include <agent-cpp/agent.h>

#ifdef AGENT_HAS_OPENAICOMPATIBLE
#include <httplib.h>
#include <json.hpp>

namespace agent {

std::string generate_text_openai_compatible(Session &session, std::string_view prompt) {
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

void stream_text_openai_compatible(Session &session, std::string_view prompt,
                                   const std::function<void(std::string_view)> &on_token) {
    httplib::Client client(session.config.base_url);
    nlohmann::json request = {
        {"model", session.config.model}, {"stream", true}, {"messages", {{{"role", "user"}, {"content", prompt}}}}};
    httplib::Headers headers = {{"Authorization", "Bearer " + session.config.api_key}};
    std::string buffer;
    auto response = client.Post("/v1/chat/completions", headers, request.dump(), "application/json",
                                [&](const char *data, size_t len) {
                                    buffer.append(data, len);
                                    size_t pos;
                                    while ((pos = buffer.find("\n")) != std::string::npos) {
                                        std::string line = buffer.substr(0, pos);
                                        buffer.erase(0, pos + 1);
                                        if (!line.starts_with("data: "))
                                            continue;
                                        std::string payload = line.substr(6);
                                        if (payload == "[DONE]")
                                            return false;
                                        auto json = nlohmann::json::parse(payload, nullptr, false);
                                        if (json.is_discarded())
                                            continue;
                                        auto &delta = json["choices"][0]["delta"]["content"];
                                        if (delta.is_string()) {
                                            on_token(delta.get<std::string>());
                                        }
                                    }
                                    return true;
                                });
    if (!response) {
        if (response.error() == httplib::Error::Canceled) {
            return;
        }
        std::cout << "Error: Internal HTTP Client request failed completely.\n";
        return;
    }
    if (response->status != 200) {
        std::cout << "HTTP Error " + std::to_string(response->status) + ": " + response->body + "\n";
    }
}

} // namespace agent
#endif