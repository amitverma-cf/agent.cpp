#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>

#ifdef AGENT_HAS_OPENAICOMPATIBLE
#include <cstring>
#include <httplib.h>
#include <simdjson.h>
#include <vector>

namespace agent::providers {

namespace {

httplib::Headers make_headers(const Config &config) {
    httplib::Headers headers;
    if (!config.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + config.api_key);
    }
    return headers;
}

Result<ChatResponse> extract_response(Session &session, simdjson::ondemand::document &doc) {
    simdjson::ondemand::array choices;
    auto choices_err = doc["choices"].get_array().get(choices);
    if (choices_err) {
        return fail<ChatResponse>(ErrorCode::ParseError, "OpenAI-compatible response did not contain choices.");
    }

    simdjson::ondemand::object first_choice;
    auto choice_err = choices.at(0).get_object().get(first_choice);
    if (choice_err) {
        return fail<ChatResponse>(ErrorCode::ParseError, "OpenAI-compatible response choice was not an object.");
    }

    simdjson::ondemand::object message;
    auto msg_err = first_choice["message"].get_object().get(message);
    if (msg_err) {
        return fail<ChatResponse>(ErrorCode::ParseError,
                                  "OpenAI-compatible response did not contain a message object.");
    }

    std::string_view role = "assistant";
    std::string_view content_view = "";
    std::span<ToolCallView> tool_calls;
    std::string_view tool_call_id = "";

    for (auto field : message) {
        auto key_res = field.unescaped_key();
        if (key_res.error())
            continue;
        std::string_view key_str = key_res.value();

        if (key_str == "role") {
            std::string_view val_str;
            if (!field.value().get_string().get(val_str)) {
                role = val_str;
            }
        } else if (key_str == "content") {
            std::string_view val_str;
            if (!field.value().get_string().get(val_str)) {
                content_view = val_str;
            }
        } else if (key_str == "tool_call_id") {
            std::string_view val_str;
            if (!field.value().get_string().get(val_str)) {
                tool_call_id = val_str;
            }
        } else if (key_str == "tool_calls") {
            simdjson::ondemand::array tcs_json;
            if (!field.value().get_array().get(tcs_json)) {
                std::vector<ToolCallView> temp_tcs;
                for (auto tc_item : tcs_json) {
                    simdjson::ondemand::object tc;
                    if (tc_item.get_object().get(tc))
                        continue;

                    std::string_view id_val;
                    if (tc["id"].get_string().get(id_val))
                        continue;

                    simdjson::ondemand::object func;
                    if (tc["function"].get_object().get(func))
                        continue;

                    std::string_view name_val;
                    if (func["name"].get_string().get(name_val))
                        continue;

                    std::string_view args_val;
                    if (func["arguments"].get_string().get(args_val))
                        continue;

                    temp_tcs.push_back(ToolCallView{id_val, name_val, args_val});
                }
                if (!temp_tcs.empty()) {
                    tool_calls = session.arena.allocate_span<ToolCallView>(temp_tcs.size());
                    std::copy(temp_tcs.begin(), temp_tcs.end(), tool_calls.begin());
                }
            }
        }
    }

    Usage usage;
    simdjson::ondemand::object u;
    if (!doc["usage"].get_object().get(u)) {
        int64_t prompt_tokens = 0;
        if (!u["prompt_tokens"].get_int64().get(prompt_tokens)) {
            usage.prompt_tokens = static_cast<int>(prompt_tokens);
        }
        int64_t completion_tokens = 0;
        if (!u["completion_tokens"].get_int64().get(completion_tokens)) {
            usage.completion_tokens = static_cast<int>(completion_tokens);
        }
        int64_t total_tokens = 0;
        if (!u["total_tokens"].get_int64().get(total_tokens)) {
            usage.total_tokens = static_cast<int>(total_tokens);
        }
    }

    MessageView msg_view;
    msg_view.role = role;
    msg_view.content = content_view;
    msg_view.tool_calls = tool_calls;
    msg_view.tool_call_id = tool_call_id;

    return ok(ChatResponse{.message = msg_view, .usage = usage});
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

Result<ChatResponse> execute_turn_openai_compatible(Session &session, const ChatRequest &request) {
    if (session.config.base_url.empty()) {
        return fail<ChatResponse>(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires base_url.");
    }
    if (session.config.model.empty()) {
        return fail<ChatResponse>(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires model.");
    }

    httplib::Client client(session.config.base_url);
    client.set_connection_timeout(30);
    client.set_read_timeout(60);

    const int effective_max_tokens = request.max_tokens > 0 ? request.max_tokens : session.config.max_tokens;
    const float effective_temperature = request.temperature >= 0.0f ? request.temperature : session.config.temperature;

    std::string request_body = "{";
    request_body += "\"model\":\"" + utils::escape_json_string(session.config.model) + "\",";
    request_body += "\"max_tokens\":" + std::to_string(effective_max_tokens) + ",";
    request_body += "\"temperature\":" + std::to_string(effective_temperature) + ",";

    request_body += "\"messages\":[";
    for (size_t i = 0; i < request.messages.size(); ++i) {
        if (i > 0)
            request_body += ",";
        const auto &msg = request.messages[i];
        request_body += "{\"role\":\"" + utils::escape_json_string(msg.role) + "\",";
        request_body += "\"content\":\"" + utils::escape_json_string(msg.content) + "\"";
        if (!msg.tool_calls.empty()) {
            request_body += ",\"tool_calls\":[";
            for (size_t j = 0; j < msg.tool_calls.size(); ++j) {
                if (j > 0)
                    request_body += ",";
                const auto &tc = msg.tool_calls[j];
                request_body += "{\"id\":\"" + utils::escape_json_string(tc.id) + "\",";
                request_body += "\"type\":\"function\",";
                request_body += "\"function\":{";
                request_body += "\"name\":\"" + utils::escape_json_string(tc.name) + "\",";
                request_body += "\"arguments\":\"" + utils::escape_json_string(tc.arguments) + "\"";
                request_body += "}}";
            }
            request_body += "]";
        }
        if (!msg.tool_call_id.empty()) {
            request_body += ",\"tool_call_id\":\"" + utils::escape_json_string(msg.tool_call_id) + "\"";
        }
        request_body += "}";
    }
    request_body += "]";

    if (!session.config.tools.empty()) {
        request_body += ",\"tools\":[";
        for (size_t i = 0; i < session.config.tools.size(); ++i) {
            if (i > 0)
                request_body += ",";
            const auto &tool = session.config.tools[i];
            request_body += "{\"type\":\"function\",";
            request_body += "\"function\":{";
            request_body += "\"name\":\"" + utils::escape_json_string(tool.name) + "\",";
            request_body += "\"description\":\"" + utils::escape_json_string(tool.description) + "\"";
            if (!tool.parameter_schema.empty()) {
                request_body += ",\"parameters\":" + std::string(tool.parameter_schema);
            }
            request_body += "}}";
        }
        request_body += "]";
    }

    request_body += "}";

    httplib::Headers headers = make_headers(session.config);

    if (request.stream) {
        std::string stream_body = "{\"stream\":true,";
        stream_body.append(request_body.data() + 1, request_body.size() - 1);

        std::string accumulated_text;
        std::string buffer;
        ErrorCode stream_err = ErrorCode::Ok;
        std::string stream_err_msg;

        auto response = client.Post(
            "/v1/chat/completions", headers, stream_body, "application/json", [&](const char *data, size_t len) {
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

                    size_t orig_len = payload.size();
                    payload.append(simdjson::SIMDJSON_PADDING, '\0');

                    simdjson::padded_string_view padded_payload(payload.data(), orig_len, payload.size());
                    simdjson::ondemand::parser stream_parser;
                    simdjson::ondemand::document chunk_doc;
                    auto err = stream_parser.iterate(padded_payload).get(chunk_doc);
                    if (err) {
                        stream_err = ErrorCode::ParseError;
                        stream_err_msg = "Malformed JSON chunk in OpenAI compatible stream.";
                        return false;
                    }

                    simdjson::ondemand::array choices;
                    if (!chunk_doc["choices"].get_array().get(choices)) {
                        simdjson::ondemand::object first_choice;
                        if (!choices.at(0).get_object().get(first_choice)) {
                            simdjson::ondemand::object delta;
                            if (!first_choice["delta"].get_object().get(delta)) {
                                std::string_view token;
                                if (!delta["content"].get_string().get(token)) {
                                    accumulated_text += token;
                                    if (request.on_token) {
                                        request.on_token(token, request.token_user_data);
                                    }
                                }
                            }
                        }
                    }
                }
                return true;
            });

        if (!response) {
            if (stream_err != ErrorCode::Ok) {
                return fail<ChatResponse>(stream_err, std::move(stream_err_msg));
            }
            if (response.error() == httplib::Error::Canceled) {
            } else {
                std::string err_msg =
                    "OpenAI-compatible stream request failed. httplib error: " + httplib::to_string(response.error());
                utils::log(session, LogLevel::Error, err_msg);
                return fail<ChatResponse>(ErrorCode::NetworkError, std::move(err_msg));
            }
        }

        std::string_view content_view = session.arena.allocate_string(accumulated_text);
        MessageView msg_view;
        msg_view.role = "assistant";
        msg_view.content = content_view;
        msg_view.tool_calls = {};
        msg_view.tool_call_id = "";

        return ok(ChatResponse{.message = msg_view, .usage = {}});
    }

    auto response = client.Post("/v1/chat/completions", headers, request_body, "application/json");
    if (!response) {
        std::string err_msg =
            "OpenAI-compatible request failed. httplib error: " + httplib::to_string(response.error());
        utils::log(session, LogLevel::Error, err_msg);
        return fail<ChatResponse>(ErrorCode::NetworkError, std::move(err_msg));
    }
    if (response->status != 200) {
        return fail<ChatResponse>(ErrorCode::HttpError,
                                  "HTTP Error " + std::to_string(response->status) + ": " + response->body);
    }

    size_t resp_size = response->body.size();
    void *arena_mem = session.arena.allocate(resp_size + simdjson::SIMDJSON_PADDING);
    if (!arena_mem) {
        return fail<ChatResponse>(ErrorCode::DecodeFailed, "Out of memory in Arena to store response.");
    }
    std::memcpy(arena_mem, response->body.data(), resp_size);
    std::memset(static_cast<char *>(arena_mem) + resp_size, 0, simdjson::SIMDJSON_PADDING);

    simdjson::padded_string_view padded_json(static_cast<const char *>(arena_mem), resp_size,
                                             resp_size + simdjson::SIMDJSON_PADDING);
    simdjson::ondemand::document doc;
    auto error = session.json_parser.iterate(padded_json).get(doc);
    if (error) {
        return fail<ChatResponse>(ErrorCode::ParseError, "Failed to parse OpenAI-compatible response JSON.");
    }

    return extract_response(session, doc);
}

} // namespace agent::providers
#endif
