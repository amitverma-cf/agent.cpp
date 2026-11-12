#include "upstream_engine.hpp"

#ifdef AGENT_LLAMA_UPSTREAM

#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "llama.h"
#include "log.h"
#include "server-context.h"
#include "server-http.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>

namespace upstream {

namespace {

using njson = nlohmann::ordered_json;

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

std::string base64(const unsigned char *data, size_t n) {
    static constexpr char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const unsigned v = (data[i] << 16) | (i + 1 < n ? data[i + 1] << 8 : 0) | (i + 2 < n ? data[i + 2] : 0);
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < n ? tbl[(v >> 6) & 63] : '=';
        out += i + 2 < n ? tbl[v & 63] : '=';
    }
    return out;
}

bool read_file(const std::string &path, std::string &bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

std::string audio_format(std::string_view mime) {
    if (mime.find("mpeg") != std::string_view::npos || mime.find("mp3") != std::string_view::npos) return "mp3";
    if (mime.find("flac") != std::string_view::npos) return "flac";
    return "wav";
}

} // namespace

bool gguf_has_mtp(const std::string &path) {
    try {
        for (auto type : common_speculative_types_from_gguf(path))
            if (type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) return true;
    } catch (...) {}
    return false;
}

bool gguf_is_decision(const std::string &path) {
    gguf_init_params ip{true, nullptr};
    gguf_context *g = gguf_init_from_file(path.c_str(), ip);
    if (!g) return false;
    bool found = false;
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch >= 0 && gguf_get_kv_type(g, arch) == GGUF_TYPE_STRING) {
        const std::string prefix = std::string(gguf_get_val_str(g, arch)) + ".decision.";
        for (int64_t i = 0, n = gguf_get_n_kv(g); i < n && !found; ++i)
            found = std::strncmp(gguf_get_key(g, i), prefix.c_str(), prefix.size()) == 0;
    }
    gguf_free(g);
    return found;
}

struct Engine::Impl {
    common_params params;
    std::unique_ptr<server_context> ctx;
    std::unique_ptr<server_routes> routes;
    std::thread loop;
    bool mtp = false;
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}

Engine::~Engine() {
    if (impl_ && impl_->ctx) {
        try {
            impl_->ctx->terminate();
        } catch (...) {}
        if (impl_->loop.joinable()) impl_->loop.join();
    }
}

bool Engine::mtp_active() const { return impl_->mtp; }

int Engine::count_tokens(const std::string &text) const {
    const llama_model *model = llama_get_model(impl_->ctx->get_llama_context());
    const int n = llama_tokenize(llama_model_get_vocab(model), text.data(), static_cast<int32_t>(text.size()), nullptr, 0, false, false);
    return n < 0 ? -n : n;
}

std::unique_ptr<Engine> Engine::create(const Settings &s, std::string &error) {
    std::unique_ptr<Engine> engine(new Engine());
    Impl &im = *engine->impl_;
    try {
        std::vector<std::string> args = {"agent-engine-llama",
                                         "-m",
                                         s.model,
                                         "-c",
                                         std::to_string(std::max(1, s.context) * std::max(1, s.parallel)),
                                         "-np",
                                         std::to_string(std::max(1, s.parallel)),
                                         "--no-webui",
                                         "--no-warmup"};
        if (s.threads > 0) args.insert(args.end(), {"-t", std::to_string(s.threads)});
        if (s.gpu_layers >= 0) args.insert(args.end(), {"-ngl", std::to_string(s.gpu_layers)});
        if (!s.mmproj.empty()) args.insert(args.end(), {"--mmproj", s.mmproj});
        for (const auto &[path, scale] : s.loras) args.insert(args.end(), {"--lora-scaled", path, std::to_string(scale)});
        if (!s.kv_type.empty() && s.kv_type != "auto") args.insert(args.end(), {"-ctk", s.kv_type, "-ctv", s.kv_type});
        im.mtp = s.mtp && gguf_has_mtp(s.model);
        if (im.mtp) args.insert(args.end(), {"--spec-type", "draft-mtp", "--spec-draft-n-max", std::to_string(std::max(1, s.draft_max))});

        std::vector<char *> argv;
        for (std::string &a : args) argv.push_back(a.data());
        if (!common_params_parse(static_cast<int>(argv.size()), argv.data(), im.params, LLAMA_EXAMPLE_SERVER)) {
            error = "llama.cpp did not accept the settings for its own engine";
            return nullptr;
        }
        common_init();
        im.ctx = std::make_unique<server_context>();
        im.routes = std::make_unique<server_routes>(im.params, *im.ctx); // before load_model(): it registers a callback
        if (!im.ctx->load_model(im.params)) {
            error = "llama.cpp's engine could not load " + s.model;
            return nullptr;
        }
        im.routes->update_meta(*im.ctx);
        server_context *ctx = im.ctx.get();
        im.loop = std::thread([ctx] {
            try {
                ctx->start_loop();
            } catch (...) {}
        });
    } catch (const std::exception &e) {
        error = std::string("llama.cpp's engine failed to start: ") + e.what();
        return nullptr;
    } catch (...) {
        error = "llama.cpp's engine failed to start";
        return nullptr;
    }
    return engine;
}

namespace {

void read_usage(const njson &j, Result &out) {
    if (j.contains("usage") && j["usage"].is_object()) {
        const njson &u = j["usage"];
        out.prompt_tokens = u.value("prompt_tokens", out.prompt_tokens);
        out.completion_tokens = u.value("completion_tokens", out.completion_tokens);
    }
    if (j.contains("timings") && j["timings"].is_object()) {
        const njson &t = j["timings"];
        out.prompt_ms = t.value("prompt_ms", out.prompt_ms);
        out.predicted_ms = t.value("predicted_ms", out.predicted_ms);
        out.cached_tokens = t.value("cache_n", out.cached_tokens);
        out.drafted = t.value("draft_n", out.drafted);
        out.accepted = t.value("draft_n_accepted", out.accepted);
        out.prompt_tokens = t.value("prompt_n", out.prompt_tokens);
        out.completion_tokens = t.value("predicted_n", out.completion_tokens);
    }
}

// One streamed event (a chat.completion.chunk) or the whole non-streamed answer.
void apply_choice(const njson &choice, bool delta, Result &out, const std::function<void(const std::string &)> &on_text) {
    const njson &m = delta ? choice.value("delta", njson::object()) : choice.value("message", njson::object());
    if (m.contains("content") && m["content"].is_string()) {
        const std::string piece = m["content"].get<std::string>();
        out.content += piece;
        if (delta && on_text && !piece.empty()) on_text(piece);
    }
    if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) out.reasoning += m["reasoning_content"].get<std::string>();
    if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
        for (const njson &tc : m["tool_calls"]) {
            const size_t index = tc.contains("index") ? tc["index"].get<size_t>() : out.calls.size();
            if (index >= out.calls.size()) out.calls.resize(index + 1);
            ToolCall &call = out.calls[index];
            if (tc.contains("id") && tc["id"].is_string()) call.id = tc["id"].get<std::string>();
            if (tc.contains("function") && tc["function"].is_object()) {
                const njson &f = tc["function"];
                if (f.contains("name") && f["name"].is_string()) call.name += f["name"].get<std::string>();
                if (f.contains("arguments") && f["arguments"].is_string()) call.arguments += f["arguments"].get<std::string>();
            }
        }
    }
    if (choice.contains("finish_reason") && choice["finish_reason"].is_string())
        out.finish_reason = choice["finish_reason"].get<std::string>();
}

} // namespace

bool Engine::chat(const std::string &body, const volatile int32_t *cancel, const std::function<void(const std::string &)> &on_text,
                  Result &out, std::string &error) {
    out = Result{};
    try {
        const std::function<bool()> stop = [cancel] { return cancel && *cancel != 0; };
        server_http_req req{{}, {}, "/v1/chat/completions", "", body, {}, stop};
        server_http_res_ptr res = impl_->routes->post_chat_completions(req);
        if (!res) {
            error = "llama.cpp's engine returned no answer";
            return false;
        }
        auto fail_with = [&](const std::string &raw) {
            try {
                const njson j = njson::parse(raw);
                error = j.contains("error") ? j["error"].value("message", raw) : raw;
            } catch (...) { error = raw; }
            return false;
        };
        if (!res->is_stream()) {
            if (res->status != 200) return fail_with(res->data);
            const njson j = njson::parse(res->data);
            if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty())
                apply_choice(j["choices"][0], false, out, on_text);
            read_usage(j, out);
            if (on_text && !out.content.empty()) on_text(out.content);
            res->on_complete();
            return true;
        }
        std::string pending;
        bool more = true;
        while (more) {
            std::string chunk;
            more = res->next(chunk);
            pending += chunk;
            for (size_t at = pending.find("\n\n"); at != std::string::npos; at = pending.find("\n\n")) {
                std::string event = pending.substr(0, at);
                pending.erase(0, at + 2);
                for (size_t line = 0; line < event.size();) {
                    size_t end = event.find('\n', line);
                    if (end == std::string::npos) end = event.size();
                    std::string text = event.substr(line, end - line);
                    line = end + 1;
                    if (text.rfind("data:", 0) != 0) continue;
                    text.erase(0, 5);
                    while (!text.empty() && text.front() == ' ') text.erase(0, 1);
                    if (text.empty() || text == "[DONE]") continue;
                    const njson j = njson::parse(text, nullptr, false);
                    if (j.is_discarded()) continue;
                    if (j.contains("error")) {
                        error = j["error"].value("message", text);
                        res->on_complete();
                        return false;
                    }
                    if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty())
                        apply_choice(j["choices"][0], true, out, on_text);
                    read_usage(j, out);
                }
            }
        }
        res->on_complete();
        return true;
    } catch (const std::exception &e) { error = std::string("llama.cpp's engine: ") + e.what(); } catch (...) {
        error = "llama.cpp's engine failed";
    }
    return false;
}

bool Engine::systemone(const std::string &body, std::string &out, std::string &error) {
    try {
        server_http_req req{{}, {}, "/v1/systemone", "", body, {}, std::function<bool()>([] { return false; })};
        server_http_res_ptr res = impl_->routes->post_systemone(req);
        if (!res) {
            error = "llama.cpp's engine returned no answer";
            return false;
        }
        out = res->data;
        if (res->status != 200) {
            try {
                const njson j = njson::parse(out);
                error = j.contains("error") ? j["error"].value("message", out) : out;
            } catch (...) { error = out; }
            return false;
        }
        res->on_complete();
        return true;
    } catch (const std::exception &e) { error = std::string("llama.cpp's engine: ") + e.what(); } catch (...) {
        error = "llama.cpp's engine failed";
    }
    return false;
}

std::string chat_body_from_request(const agent_request *req, int max_tokens, float temperature, bool stream, std::string &error) {
    njson body = njson::object();
    njson messages = njson::array();
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        njson msg = njson::object();
        msg["role"] = std::string(sv(m.role));
        njson parts = njson::array();
        std::string text;
        bool media = false;
        for (size_t p = 0; p < m.n_parts; ++p) {
            const agent_part &part = m.parts[p];
            if (part.kind == AGENT_PART_TEXT) {
                text += std::string(sv(part.text));
                parts.push_back({{"type", "text"}, {"text", std::string(sv(part.text))}});
            } else if (part.kind == AGENT_PART_FILE || part.kind == AGENT_PART_BYTES) {
                std::string bytes;
                if (part.kind == AGENT_PART_FILE) {
                    if (!read_file(std::string(sv(part.path)), bytes)) {
                        error = "cannot read " + std::string(sv(part.path));
                        return {};
                    }
                } else if (part.data && part.size) {
                    bytes.assign(static_cast<const char *>(part.data), part.size);
                }
                const std::string mime(sv(part.mime));
                const std::string encoded = base64(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
                media = true;
                if (mime.rfind("audio/", 0) == 0)
                    parts.push_back({{"type", "input_audio"}, {"input_audio", {{"data", encoded}, {"format", audio_format(mime)}}}});
                else
                    parts.push_back(
                        {{"type", "image_url"},
                         {"image_url", {{"url", "data:" + (mime.empty() ? std::string("image/png") : mime) + ";base64," + encoded}}}});
            }
        }
        if (media) msg["content"] = parts;
        else msg["content"] = text;
        if (m.n_tool_calls) {
            njson calls = njson::array();
            for (size_t c = 0; c < m.n_tool_calls; ++c) {
                const agent_tool_call &tc = m.tool_calls[c];
                calls.push_back({{"id", std::string(sv(tc.id))},
                                 {"type", "function"},
                                 {"function", {{"name", std::string(sv(tc.name))}, {"arguments", std::string(sv(tc.arguments))}}}});
            }
            msg["tool_calls"] = calls;
        }
        if (sv(m.tool_call_id).size()) msg["tool_call_id"] = std::string(sv(m.tool_call_id));
        messages.push_back(std::move(msg));
    }
    body["messages"] = std::move(messages);
    if (req->tools_json.size) {
        const njson tools = njson::parse(sv(req->tools_json), nullptr, false);
        if (tools.is_array() && !tools.empty()) {
            njson list = njson::array();
            for (const njson &t : tools) list.push_back({{"type", "function"}, {"function", t}}); // {name, description, parameters}
            body["tools"] = std::move(list);
            body["parallel_tool_calls"] = true;
        }
    }
    if (max_tokens > 0) body["max_tokens"] = max_tokens;
    if (temperature >= 0.0f) body["temperature"] = temperature;
    body["stream"] = stream;
    if (stream) body["stream_options"] = {{"include_usage", true}};
    body["chat_template_kwargs"] = {{"enable_thinking", false}};
    return body.dump();
}

} // namespace upstream

#else // built without llama.cpp's server engine

namespace upstream {

bool gguf_has_mtp(const std::string &) { return false; }
bool gguf_is_decision(const std::string &) { return false; }
bool Engine::systemone(const std::string &, std::string &, std::string &error) {
    error = "llama.cpp's own engine is not built into this plugin";
    return false;
}
std::unique_ptr<Engine> Engine::create(const Settings &, std::string &error) {
    error = "this build of the llama plugin does not include llama.cpp's own engine (AGENT_LLAMA_UPSTREAM=OFF)";
    return nullptr;
}
struct Engine::Impl {};
Engine::Engine() {}
Engine::~Engine() {}
bool Engine::mtp_active() const { return false; }
int Engine::count_tokens(const std::string &text) const { return static_cast<int>(text.size() / 4 + 1); }
bool Engine::chat(const std::string &, const volatile int32_t *, const std::function<void(const std::string &)> &, Result &,
                  std::string &error) {
    error = "llama.cpp's own engine is not built into this plugin";
    return false;
}
std::string chat_body_from_request(const agent_request *, int, float, bool, std::string &error) {
    error = "llama.cpp's own engine is not built into this plugin";
    return {};
}

} // namespace upstream

#endif
