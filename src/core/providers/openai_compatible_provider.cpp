#include "../utils/utils.h"

#include <agent-cpp/agent.h>

#ifdef AGENT_HAS_OPENAICOMPATIBLE
#include <httplib.h>
#include <json.hpp>

namespace agent::providers {

namespace {

httplib::Headers make_headers(const Config &config) {
    httplib::Headers headers;
    if (!config.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + config.api_key);
    }
    return headers;
}

Result<std::string> extract_message_content(const nlohmann::json &parsed) {
    if (!parsed.is_object()) {
        return fail<std::string>(ErrorCode::ParseError, "OpenAI-compatible response root was not an object.");
    }

    const auto choices = parsed.find("choices");
    if (choices == parsed.end() || !choices->is_array() || choices->empty()) {
        return fail<std::string>(ErrorCode::ParseError, "OpenAI-compatible response did not contain choices.");
    }

    const auto &first_choice = choices->front();
    if (!first_choice.is_object()) {
        return fail<std::string>(ErrorCode::ParseError, "OpenAI-compatible response choice was not an object.");
    }

    const auto message = first_choice.find("message");
    if (message == first_choice.end() || !message->is_object()) {
        return fail<std::string>(ErrorCode::ParseError, "OpenAI-compatible response did not contain a message object.");
    }

    const auto content = message->find("content");
    if (content == message->end() || !content->is_string()) {
        return fail<std::string>(ErrorCode::ParseError, "OpenAI-compatible response did not contain message content.");
    }

    return ok(content->get<std::string>());
}

} // namespace

Result<void> init_openai_compatible(Session &session) {
    if (session.config.base_url.empty()) {
        return fail(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires base_url.");
    }
    if (session.config.model.empty()) {
        return fail(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires model.");
    }
    utils::log(session, LogLevel::Debug, "OpenAI-compatible provider configured.");
    return ok();
}

Result<std::string> generate_text_openai_compatible(Session &session, std::string_view prompt) {
    if (session.config.base_url.empty()) {
        return fail<std::string>(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires base_url.");
    }
    if (session.config.model.empty()) {
        return fail<std::string>(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires model.");
    }

    httplib::Client client(session.config.base_url);
    nlohmann::json request = {{"model", session.config.model},
                              {"messages", {{{"role", "user"}, {"content", std::string(prompt)}}}}};
    httplib::Headers headers = make_headers(session.config);
    auto response = client.Post("/v1/chat/completions", headers, request.dump(), "application/json");
    if (!response) {
        utils::log(session, LogLevel::Error, "OpenAI-compatible request failed.");
        return fail<std::string>(ErrorCode::NetworkError, "OpenAI-compatible request failed.");
    }
    if (response->status != 200) {
        return fail<std::string>(ErrorCode::HttpError,
                                 "HTTP Error " + std::to_string(response->status) + ": " + response->body);
    }

    auto parsed = nlohmann::json::parse(response->body, nullptr, false);
    if (parsed.is_discarded()) {
        return fail<std::string>(ErrorCode::ParseError, "Failed to parse OpenAI-compatible response JSON.");
    }

    return extract_message_content(parsed);
}

Result<void> stream_text_openai_compatible(Session &session, std::string_view prompt, TokenCallback on_token,
                                           void *user_data) {
    if (session.config.base_url.empty()) {
        return fail(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires base_url.");
    }
    if (session.config.model.empty()) {
        return fail(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires model.");
    }

    httplib::Client client(session.config.base_url);
    nlohmann::json request = {{"model", session.config.model},
                              {"stream", true},
                              {"messages", {{{"role", "user"}, {"content", std::string(prompt)}}}}};
    httplib::Headers headers = make_headers(session.config);
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
                                        if (!payload.empty() && payload.back() == '\r') {
                                            payload.pop_back();
                                        }
                                        if (payload == "[DONE]")
                                            return false;
                                        auto json = nlohmann::json::parse(payload, nullptr, false);
                                        if (json.is_discarded())
                                            continue;
                                        auto &delta = json["choices"][0]["delta"]["content"];
                                        if (delta.is_string()) {
                                            std::string token = delta.get<std::string>();
                                            on_token(token, user_data);
                                        }
                                    }
                                    return true;
                                });
    if (!response) {
        if (response.error() == httplib::Error::Canceled) {
            return ok();
        }
        utils::log(session, LogLevel::Error, "OpenAI-compatible stream request failed.");
        return fail(ErrorCode::NetworkError, "OpenAI-compatible stream request failed.");
    }
    if (response->status != 200) {
        return fail(ErrorCode::HttpError, "HTTP Error " + std::to_string(response->status) + ": " + response->body);
    }
    return ok();
}

} // namespace agent
#endif
