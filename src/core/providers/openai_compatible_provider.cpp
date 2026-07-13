#include "../executor/ambient_turn.hpp"
#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>

#ifdef AGENT_HAS_OPENAICOMPATIBLE
#include <chrono>
#include <cstring>
#include <httplib.h>
#include <map>
#include <simdjson.h>
#include <thread>
#include <vector>

namespace agent::providers {

namespace {

httplib::Headers make_headers(const Config &config) {
    httplib::Headers headers;
    if (!config.api_key.empty())
        headers.emplace("Authorization", "Bearer " + config.api_key);
    return headers;
}

Result<InferResponse> extract_response(Session &session, simdjson::ondemand::document &doc) {
    simdjson::ondemand::array choices;
    if (doc["choices"].get_array().get(choices))
        return fail<InferResponse>(ErrorCode::ParseError,
                                   "OpenAI response missing 'choices'.");

    simdjson::ondemand::object first_choice;
    if (choices.at(0).get_object().get(first_choice))
        return fail<InferResponse>(ErrorCode::ParseError, "OpenAI response choice not an object.");

    simdjson::ondemand::object message;
    if (first_choice["message"].get_object().get(message))
        return fail<InferResponse>(ErrorCode::ParseError, "OpenAI response missing 'message'.");

    std::string_view role = "assistant";
    std::string_view content_view = "";
    std::span<ToolCallView> tool_calls;
    std::string_view tool_call_id = "";

    Arena &arena = ambient::current_arena(session);

    for (auto field : message) {
        auto key_res = field.unescaped_key();
        if (key_res.error())
            continue;
        std::string_view key = key_res.value();

        if (key == "role") {
            std::string_view v;
            if (!field.value().get_string().get(v))
                role = v;
        } else if (key == "content") {
            std::string_view v;
            if (!field.value().get_string().get(v))
                content_view = v;
        } else if (key == "tool_call_id") {
            std::string_view v;
            if (!field.value().get_string().get(v))
                tool_call_id = v;
        } else if (key == "tool_calls") {
            simdjson::ondemand::array tcs_json;
            if (field.value().get_array().get(tcs_json))
                continue;

            std::vector<ToolCallView> tmp;
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

                tmp.push_back(ToolCallView{id_val, name_val, args_val});
            }
            if (!tmp.empty()) {
                tool_calls = arena.allocate_span<ToolCallView>(tmp.size());
                std::copy(tmp.begin(), tmp.end(), tool_calls.begin());
            }
        }
    }

    TokenUsage usage;
    simdjson::ondemand::object u;
    if (!doc["usage"].get_object().get(u)) {
        int64_t v = 0;
        if (!u["prompt_tokens"].get_int64().get(v))
            usage.prompt_tokens = static_cast<int>(v);
        v = 0;
        if (!u["completion_tokens"].get_int64().get(v))
            usage.completion_tokens = static_cast<int>(v);
        v = 0;
        if (!u["total_tokens"].get_int64().get(v))
            usage.total_tokens = static_cast<int>(v);
    }

    MessageView mv;
    mv.role = role;
    mv.content = content_view;
    mv.tool_calls = tool_calls;
    mv.tool_call_id = tool_call_id;
    return ok(InferResponse{.message = mv, .usage = usage});
}

bool is_retriable(int http_status, httplib::Error http_err) {
    if (http_err != httplib::Error::Success && http_err != httplib::Error::Canceled)
        return true;
    return http_status == 429 || http_status == 500 || http_status == 502 ||
           http_status == 503 || http_status == 504;
}

struct StreamedToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

}

Result<void> init_openai_compatible(Session &session) {
    if (session.config.base_url.empty())
        return fail(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires base_url.");
    if (session.config.model.empty())
        return fail(ErrorCode::InvalidConfig, "OpenAI-compatible provider requires model.");
    utils::log(session, LogLevel::Debug, "OpenAI-compatible provider configured.");
    return ok();
}

Result<InferResponse> infer_openai_compatible(Session &session, const InferRequest &request) {
    if (session.config.base_url.empty())
        return fail<InferResponse>(ErrorCode::InvalidConfig,
                                   "OpenAI-compatible provider requires base_url.");
    if (session.config.model.empty())
        return fail<InferResponse>(ErrorCode::InvalidConfig,
                                   "OpenAI-compatible provider requires model.");

    httplib::Client client(session.config.base_url);
    client.set_connection_timeout(30);
    client.set_read_timeout(120);

    const int eff_max_tokens = request.max_tokens > 0 ? request.max_tokens : session.config.max_tokens;
    const float eff_temp = request.temperature >= 0.0f ? request.temperature : session.config.temperature;

    std::string body = "{";
    body += "\"model\":\"" + utils::escape_json_string(session.config.model) + "\",";
    body += "\"max_tokens\":" + std::to_string(eff_max_tokens) + ",";
    body += "\"temperature\":" + std::to_string(eff_temp) + ",";
    body += "\"messages\":[";
    for (size_t i = 0; i < request.messages.size(); ++i) {
        if (i > 0)
            body += ",";
        const auto &msg = request.messages[i];
        body += "{\"role\":\"" + utils::escape_json_string(msg.role) + "\",";
        body += "\"content\":\"" + utils::escape_json_string(msg.content) + "\"";
        if (!msg.tool_calls.empty()) {
            body += ",\"tool_calls\":[";
            for (size_t j = 0; j < msg.tool_calls.size(); ++j) {
                if (j > 0)
                    body += ",";
                const auto &tc = msg.tool_calls[j];
                body += "{\"id\":\"" + utils::escape_json_string(tc.id) + "\",";
                body += "\"type\":\"function\",\"function\":{";
                body += "\"name\":\"" + utils::escape_json_string(tc.name) + "\",";
                body += "\"arguments\":\"" + utils::escape_json_string(tc.arguments) + "\"}}";
            }
            body += "]";
        }
        if (!msg.tool_call_id.empty())
            body += ",\"tool_call_id\":\"" + utils::escape_json_string(msg.tool_call_id) + "\"";
        body += "}";
    }
    body += "]";
    if (!session.config.tools.empty()) {
        body += ",\"tools\":[";
        for (size_t i = 0; i < session.config.tools.size(); ++i) {
            if (i > 0)
                body += ",";
            const auto &t = session.config.tools[i];
            body += "{\"type\":\"function\",\"function\":{";
            body += "\"name\":\"" + utils::escape_json_string(t.name) + "\",";
            body += "\"description\":\"" + utils::escape_json_string(t.description) + "\"";
            if (!t.parameter_schema.empty())
                body += ",\"parameters\":" + std::string(t.parameter_schema);
            body += "}}";
        }
        body += "]";
    }
    body += "}";

    httplib::Headers headers = make_headers(session.config);
    const int max_attempts = std::max(1, session.config.retry_max_attempts);
    const int base_delay_ms = std::max(100, session.config.retry_base_delay_ms);

    if (request.stream) {
        std::string stream_body = "{\"stream\":true,";
        stream_body.append(body.data() + 1, body.size() - 1);

        std::string accumulated;
        std::map<int64_t, StreamedToolCall> tool_calls_by_index;
        std::string buf;
        ErrorCode err_code = ErrorCode::Ok;
        std::string err_msg;

        auto resp = client.Post(
            "/v1/chat/completions", headers, stream_body, "application/json",
            [&](const char *data, size_t len) {
                buf.append(data, len);
                size_t pos;
                while ((pos = buf.find('\n')) != std::string::npos) {
                    std::string line = buf.substr(0, pos);
                    buf.erase(0, pos + 1);
                    if (!line.starts_with("data: "))
                        continue;
                    std::string payload = line.substr(6);
                    if (!payload.empty() && payload.back() == '\r')
                        payload.pop_back();
                    if (payload == "[DONE]")
                        return false;

                    size_t orig = payload.size();
                    payload.append(simdjson::SIMDJSON_PADDING, '\0');
                    simdjson::padded_string_view pv(payload.data(), orig, payload.size());
                    simdjson::ondemand::parser p;
                    simdjson::ondemand::document chunk;
                    if (p.iterate(pv).get(chunk)) {
                        err_code = ErrorCode::ParseError;
                        err_msg = "Malformed JSON in stream chunk.";
                        return false;
                    }
                    simdjson::ondemand::array choices;
                    if (!chunk["choices"].get_array().get(choices)) {
                        simdjson::ondemand::object fc;
                        if (!choices.at(0).get_object().get(fc)) {
                            simdjson::ondemand::object delta;
                            if (!fc["delta"].get_object().get(delta)) {
                                for (auto field : delta) {
                                    auto key_res = field.unescaped_key();
                                    if (key_res.error())
                                        continue;
                                    std::string_view key = key_res.value();

                                    if (key == "content") {
                                        std::string_view tok;
                                        if (!field.value().get_string().get(tok)) {
                                            accumulated += tok;
                                            if (request.on_token)
                                                request.on_token(tok, request.token_user_data);
                                        }
                                    } else if (key == "tool_calls") {
                                        simdjson::ondemand::array tcs;
                                        if (field.value().get_array().get(tcs))
                                            continue;
                                        for (auto tc_item : tcs) {
                                            simdjson::ondemand::object tc;
                                            if (tc_item.get_object().get(tc))
                                                continue;
                                            int64_t idx = 0;
                                            if (tc["index"].get_int64().get(idx))
                                                continue;
                                            auto &acc = tool_calls_by_index[idx];

                                            std::string_view id_val;
                                            if (!tc["id"].get_string().get(id_val))
                                                acc.id = id_val;

                                            simdjson::ondemand::object func;
                                            if (!tc["function"].get_object().get(func)) {
                                                std::string_view name_val;
                                                if (!func["name"].get_string().get(name_val))
                                                    acc.name = name_val;
                                                std::string_view args_frag;
                                                if (!func["arguments"].get_string().get(args_frag))
                                                    acc.arguments += args_frag;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                return true;
            });

        if (err_code != ErrorCode::Ok)
            return fail<InferResponse>(err_code, std::move(err_msg));
        if (!resp && resp.error() != httplib::Error::Canceled) {
            return fail<InferResponse>(ErrorCode::NetworkError,
                                       "Stream request failed: " + httplib::to_string(resp.error()));
        }

        Arena &arena = ambient::current_arena(session);
        std::string_view cv = arena.allocate_string(accumulated);

        std::span<ToolCallView> tool_calls;
        if (!tool_calls_by_index.empty()) {
            tool_calls = arena.allocate_span<ToolCallView>(tool_calls_by_index.size());
            size_t i = 0;
            for (const auto &[idx, tc] : tool_calls_by_index) {
                tool_calls[i].id = arena.allocate_string(tc.id);
                tool_calls[i].name = arena.allocate_string(tc.name);
                tool_calls[i].arguments = arena.allocate_string(tc.arguments);
                ++i;
            }
        }

        MessageView mv;
        mv.role = "assistant";
        mv.content = cv;
        mv.tool_calls = tool_calls;
        return ok(InferResponse{.message = mv, .usage = {}});
    }

    int delay_ms = base_delay_ms;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        auto resp = client.Post("/v1/chat/completions", headers, body, "application/json");

        bool net_fail = !resp;
        int status = net_fail ? 0 : resp->status;
        httplib::Error herr = net_fail ? resp.error() : httplib::Error::Success;

        if (!net_fail && status == 200) {
            Arena &arena = ambient::current_arena(session);
            size_t rsz = resp->body.size();
            void *arena_mem = arena.allocate(rsz + simdjson::SIMDJSON_PADDING);
            if (!arena_mem)
                return fail<InferResponse>(ErrorCode::DecodeFailed, "Arena OOM for response.");
            std::memcpy(arena_mem, resp->body.data(), rsz);
            std::memset(static_cast<char *>(arena_mem) + rsz, 0, simdjson::SIMDJSON_PADDING);

            simdjson::padded_string_view pv(static_cast<const char *>(arena_mem), rsz,
                                            rsz + simdjson::SIMDJSON_PADDING);
            simdjson::ondemand::document doc;
            if (ambient::json_parser.iterate(pv).get(doc))
                return fail<InferResponse>(ErrorCode::ParseError, "Failed to parse response JSON.");

            return extract_response(session, doc);
        }

        if (attempt + 1 < max_attempts && is_retriable(status, herr)) {
            std::string warn = "OpenAI request failed (attempt " + std::to_string(attempt + 1) +
                               "/" + std::to_string(max_attempts) + "), retrying in " +
                               std::to_string(delay_ms) + "ms.";
            utils::log(session, LogLevel::Warning, warn);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            delay_ms *= 2;
            continue;
        }

        if (net_fail)
            return fail<InferResponse>(ErrorCode::NetworkError,
                                       "HTTP request failed: " + httplib::to_string(herr));
        return fail<InferResponse>(ErrorCode::HttpError,
                                   "HTTP " + std::to_string(status) + ": " + resp->body);
    }

    return fail<InferResponse>(ErrorCode::NetworkError, "All retry attempts exhausted.");
}

}
#endif
