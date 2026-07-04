#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>

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

Result<GenerationResult> extract_message_content(const nlohmann::json &parsed) {
    if (!parsed.is_object()) {
        return fail<GenerationResult>(ErrorCode::ParseError, "OpenAI-compatible response root was not an object.");
    }

    const auto choices = parsed.find("choices");
    if (choices == parsed.end() || !choices->is_array() || choices->empty()) {
        return fail<GenerationResult>(ErrorCode::ParseError, "OpenAI-compatible response did not contain choices.");
    }

    const auto &first_choice = choices->front();
    if (!first_choice.is_object()) {
        return fail<GenerationResult>(ErrorCode::ParseError, "OpenAI-compatible response choice was not an object.");
    }

    const auto message = first_choice.find("message");
    if (message == first_choice.end() || !message->is_object()) {
        return fail<GenerationResult>(ErrorCode::ParseError, "OpenAI-compatible response did not contain a message object.");
    }

    const auto content = message->find("content");
    if (content == message->end() || !content->is_string()) {
        return fail<GenerationResult>(ErrorCode::ParseError, "OpenAI-compatible response did not contain message content.");
    }

    Usage usage;
    if (parsed.contains("usage") && parsed["usage"].is_object()) {
        auto &u = parsed["usage"];
        if (u.contains("prompt_tokens") && u["prompt_tokens"].is_number_integer()) {
            usage.prompt_tokens = u["prompt_tokens"].get<int>();
        }
        if (u.contains("completion_tokens") && u["completion_tokens"].is_number_integer()) {
            usage.completion_tokens = u["completion_tokens"].get<int>();
        }
        if (u.contains("total_tokens") && u["total_tokens"].is_number_integer()) {
            usage.total_tokens = u["total_tokens"].get<int>();
        }
    }

    return ok(GenerationResult{.text = content->get<std::string>(), .usage = usage});
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

Result<GenerationResult> generate_text_openai_compatible(Session &session, std::string_view prompt) {
    if (session.config.base_url.empty()) {
        return fail<GenerationResult>(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires base_url.");
    }
    if (session.config.model.empty()) {
        return fail<GenerationResult>(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires model.");
    }

    httplib::Client client(session.config.base_url);
    client.set_connection_timeout(30);
    client.set_read_timeout(60);
    nlohmann::json request = {{"model", session.config.model},
                              {"messages", {{{"role", "user"}, {"content", std::string(prompt)}}}},
                              {"max_tokens", session.config.max_tokens},
                              {"temperature", session.config.temperature}};
    httplib::Headers headers = make_headers(session.config);
    auto response = client.Post("/v1/chat/completions", headers, request.dump(), "application/json");
    if (!response) {
        utils::log(session, LogLevel::Error, "OpenAI-compatible request failed.");
        return fail<GenerationResult>(ErrorCode::NetworkError, "OpenAI-compatible request failed.");
    }
    if (response->status != 200) {
        return fail<GenerationResult>(ErrorCode::HttpError,
                                 "HTTP Error " + std::to_string(response->status) + ": " + response->body);
    }

    auto parsed = nlohmann::json::parse(response->body, nullptr, false);
    if (parsed.is_discarded()) {
        return fail<GenerationResult>(ErrorCode::ParseError, "Failed to parse OpenAI-compatible response JSON.");
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
    client.set_connection_timeout(30);
    client.set_read_timeout(60);
    nlohmann::json request = {{"model", session.config.model},
                              {"stream", true},
                              {"messages", {{{"role", "user"}, {"content", std::string(prompt)}}}},
                              {"max_tokens", session.config.max_tokens},
                              {"temperature", session.config.temperature}};
    httplib::Headers headers = make_headers(session.config);
    std::string buffer;
    ErrorCode stream_err = ErrorCode::Ok;
    std::string stream_err_msg;
    auto response = client.Post(
        "/v1/chat/completions", headers, request.dump(), "application/json", [&](const char *data, size_t len) {
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
                if (json.is_discarded()) {
                    stream_err = ErrorCode::ParseError;
                    stream_err_msg = "Malformed JSON chunk in OpenAI compatible stream.";
                    return false;
                }
                if (json.is_object() && json.contains("choices") && json["choices"].is_array() &&
                    !json["choices"].empty()) {
                    auto &choice = json["choices"][0];
                    if (choice.is_object() && choice.contains("delta") && choice["delta"].is_object()) {
                        auto &delta = choice["delta"];
                        if (delta.contains("content") && delta["content"].is_string()) {
                            std::string token = delta["content"].get<std::string>();
                            on_token(token, user_data);
                        }
                    }
                }
            }
            return true;
        });
    if (!response) {
        if (stream_err != ErrorCode::Ok) {
            return fail(stream_err, std::move(stream_err_msg));
        }
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

} // namespace agent::providers
#endif
