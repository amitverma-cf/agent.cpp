// The C API of the library (include/agent-cpp/agent_c.h): a thin layer over agent.hpp. No C++ type crosses it.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/agent_c.h>
#include <atomic>
#include <cstring>
#include <new>
#include <string>

struct agent_c_session {
    agent::Session session;
};

struct agent_c_conversation {
    agent::Conversation conversation;
};

namespace {

void set_text(char *dst, size_t cap, std::string_view text) {
    if (!dst || cap == 0) return;
    const size_t n = text.size() < cap - 1 ? text.size() : cap - 1;
    std::memcpy(dst, text.data(), n);
    dst[n] = '\0';
}

int fail_with(const agent::Error &e, char *err, size_t err_cap) {
    set_text(err, err_cap, e.message);
    return e.code == agent::ErrorCode::Cancelled ? AGENT_C_CANCELLED : AGENT_C_ERROR;
}

int fail_text(std::string_view message, char *err, size_t err_cap) {
    set_text(err, err_cap, message);
    return AGENT_C_ERROR;
}

// The runtime backends are started once and live until the process ends.
void ensure_runtime() {
    static agent::AgentRuntime runtime;
    (void)runtime;
}

struct TokenForward {
    agent_c_token_fn fn;
    void *user;
};

void forward_token(std::string_view token, void *user) {
    auto *f = static_cast<TokenForward *>(user);
    if (f->fn) f->fn(token.data(), token.size(), f->user);
}

} // namespace

extern "C" {

const char *agent_c_version(void) { return AGENT_LIB_VERSION; }

agent_c_session *agent_c_open(const char *engine, const char *model_path, const char *options_json, const char *workspace_dir,
                              int context_window, int max_tokens, char *err, size_t err_cap) {
    if (!engine || !*engine) {
        set_text(err, err_cap, "agent_c_open: engine is required");
        return nullptr;
    }
    if (!workspace_dir || !*workspace_dir) {
        set_text(err, err_cap, "agent_c_open: workspace_dir is required");
        return nullptr;
    }
    ensure_runtime();
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = engine;
    if (model_path) cfg.model = model_path;
    if (options_json && *options_json) cfg.engine_options = options_json;
    cfg.workspace_dir = workspace_dir;
    if (context_window > 0) cfg.context_window = context_window;
    if (max_tokens > 0) cfg.max_tokens = max_tokens;
    auto opened = agent::init(cfg);
    if (!opened.ok) {
        set_text(err, err_cap, opened.error.message);
        return nullptr;
    }
    auto *handle = new (std::nothrow) agent_c_session{std::move(opened.value)};
    if (!handle) set_text(err, err_cap, "agent_c_open: out of memory");
    return handle;
}

void agent_c_close(agent_c_session *session) { delete session; }

agent_c_conversation *agent_c_conversation_new(const char *id) {
    auto *c = new (std::nothrow) agent_c_conversation();
    if (c && id) c->conversation.id = id;
    return c;
}

void agent_c_conversation_free(agent_c_conversation *conversation) { delete conversation; }

int agent_c_run_turn(agent_c_session *session, agent_c_conversation *conversation, const char *prompt, agent_c_token_fn on_token,
                     void *user_data, char *reply, size_t reply_cap, size_t *needed, char *err, size_t err_cap) {
    if (!session || !conversation || !prompt)
        return fail_text("agent_c_run_turn: session, conversation and prompt are required", err, err_cap);
    TokenForward forward{on_token, user_data};
    auto r = agent::run_turn(session->session, conversation->conversation, prompt, on_token != nullptr, on_token ? forward_token : nullptr,
                             &forward);
    if (!r.ok) return fail_with(r.error, err, err_cap);
    if (needed) *needed = r.value.size() + 1;
    set_text(reply, reply_cap, r.value);
    return reply && reply_cap > r.value.size() ? AGENT_C_OK : AGENT_C_BUFFER_TOO_SMALL;
}

void agent_c_cancel(agent_c_session *session) {
    if (session) agent::request_cancel(session->session);
}

int agent_c_embed(agent_c_session *session, const char *text, float *out, size_t out_cap, size_t *dims, char *err, size_t err_cap) {
    if (!session || !text) return fail_text("agent_c_embed: session and text are required", err, err_cap);
#if !AGENT_WITH_RAG
    return fail_text("agent_c_embed: this build of agent.cpp was made without AGENT_WITH_RAG", err, err_cap);
#else
    const std::string_view texts[] = {text};
    auto r = agent::embed(session->session, texts);
    if (!r.ok) return fail_with(r.error, err, err_cap);
    if (r.value.empty()) return fail_text("agent_c_embed: the engine returned no embedding", err, err_cap);
    const std::vector<float> &v = r.value[0];
    if (dims) *dims = v.size();
    if (!out || out_cap < v.size()) return AGENT_C_BUFFER_TOO_SMALL;
    std::memcpy(out, v.data(), v.size() * sizeof(float));
    return AGENT_C_OK;
#endif
}

int agent_c_run_task(agent_c_session *session, uint32_t task, const char *text, const char *media_path, char *text_out, size_t text_cap,
                     char *file_out, size_t file_cap, char *err, size_t err_cap) {
    if (!session || task == 0) return fail_text("agent_c_run_task: session and a task bit are required", err, err_cap);
    agent::MessageView message{.role = "user", .content = text ? std::string_view(text) : std::string_view()};
    agent::MediaView media[1];
    if (media_path && *media_path) {
        media[0] = agent::MediaView{media_path, ""};
        message.media = std::span<const agent::MediaView>(media, 1);
    }
    agent::InferRequest request;
    request.messages = std::span<const agent::MessageView>(&message, 1);
    auto r = agent::run_task(session->session, task, request);
    if (!r.ok) return fail_with(r.error, err, err_cap);
    set_text(text_out, text_cap, r.value.message.content);
    set_text(file_out, file_cap, r.value.message.media.empty() ? std::string_view() : r.value.message.media[0].path);
    return AGENT_C_OK;
}

} // extern "C"
