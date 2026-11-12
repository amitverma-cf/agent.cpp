// docs/guides/writing-an-engine.md: a complete engine plugin (agent-engine-reverse).
//
//   AGENT_CAP_TEXT_GEN   reverses the text of the last user message, streamed one character per chunk
//   AGENT_CAP_IMAGE_GEN  writes a small PPM gradient to output_dir (a FILE part), with progress + cancel
//
// It depends on nothing but <agent-cpp/engine_abi.h>: no libagent, no C++ types across the boundary.

#include <agent-cpp/engine_abi.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>

namespace {

// One per create(). The engine sets no AGENT_ENGINE_FLAG_CONCURRENT, so the host never calls run() on
// the same instance from two threads at once and the per-instance output buffers below are safe.
struct ReverseInstance {
    int step_delay_ms = 0; // options.step_delay_ms: slows IMAGE_GEN down (to demo progress/cancel)
    std::string reply;     // owned output, valid until free_response() or the next run()
    std::string image_path;
    std::string summary;
    agent_part parts[2]{};
    std::string error;
};

std::string_view view(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }
agent_str to_c(std::string_view s) { return agent_str{s.data(), s.size()}; }

// Integer value of "key": in a JSON object, or fallback. A real engine would use a JSON parser; the
// ABI only passes the text.
int json_int(std::string_view json, std::string_view key, int fallback) {
    const std::string needle = "\"" + std::string(key) + "\":";
    const size_t at = json.find(needle);
    if (at == std::string_view::npos) return fallback;
    return std::atoi(std::string(json.substr(at + needle.size(), 16)).c_str());
}

// The concatenated TEXT parts of the last user message.
std::string last_user_text(const agent_request *req) {
    std::string text;
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        if (view(m.role) != "user") continue;
        text.clear();
        for (size_t p = 0; p < m.n_parts; ++p)
            if (m.parts[p].kind == AGENT_PART_TEXT) text += view(m.parts[p].text);
    }
    return text;
}

int32_t reverse_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    // config_json: {"model": ..., "context_window": ..., "options": {...Config::engine_options...}}
    const std::string_view cfg = view(config_json);
    auto *inst = new ReverseInstance();
    inst->step_delay_ms = json_int(cfg, "step_delay_ms", 0);
    if (inst->step_delay_ms < 0) {
        delete inst;
        std::snprintf(err, err_cap, "reverse: step_delay_ms must be >= 0");
        return AGENT_ERR_INVALID_CONFIG;
    }
    *out_instance = inst;
    return AGENT_OK;
}

void reverse_destroy(void *instance) { delete static_cast<ReverseInstance *>(instance); }

int32_t run_text(ReverseInstance *inst, const agent_request *req, agent_response *resp) {
    const std::string text = last_user_text(req);
    inst->reply.clear();
    // Walk backwards one UTF-8 code point at a time so multi-byte characters stay intact.
    for (size_t end = text.size(); end > 0;) {
        size_t start = end - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) --start;
        if (req->cancel && *req->cancel) {
            inst->error = "reverse: cancelled";
            return AGENT_ERR_CANCELLED;
        }
        const std::string_view ch(text.data() + start, end - start);
        if (req->on_chunk) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, ch.data(), ch.size());
        inst->reply += ch;
        end = start;
    }
    inst->parts[0] = agent_part{};
    inst->parts[0].kind = AGENT_PART_TEXT;
    inst->parts[0].text = to_c(inst->reply);
    resp->message.parts = inst->parts;
    resp->message.n_parts = 1;
    resp->usage = agent_usage{static_cast<int32_t>(text.size()), static_cast<int32_t>(inst->reply.size())};
    return AGENT_OK;
}

int32_t run_image(ReverseInstance *inst, const agent_request *req, agent_response *resp) {
    constexpr int kSize = 32;
    // FILE outputs go to output_dir (<workspace>/artifacts). Under process isolation that is the only
    // directory the sandboxed host may write to.
    inst->image_path = std::string(view(req->output_dir)) + "/reverse-gradient.ppm";
    FILE *f = std::fopen(inst->image_path.c_str(), "wb");
    if (!f) {
        inst->error = "reverse: cannot write " + inst->image_path;
        return AGENT_ERR_FILESYSTEM;
    }
    std::fprintf(f, "P6\n%d %d\n255\n", kSize, kSize);
    for (int y = 0; y < kSize; ++y) {
        if (req->cancel && *req->cancel) { // honour cancel between steps
            std::fclose(f);
            std::remove(inst->image_path.c_str());
            inst->error = "reverse: cancelled";
            return AGENT_ERR_CANCELLED;
        }
        for (int x = 0; x < kSize; ++x) {
            const unsigned char px[3] = {static_cast<unsigned char>(x * 8), static_cast<unsigned char>(y * 8), 128};
            std::fwrite(px, 1, 3, f);
        }
        if (req->on_progress) req->on_progress(req->user_data, static_cast<uint64_t>(y + 1), kSize);
        if (inst->step_delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(inst->step_delay_ms));
    }
    std::fclose(f);

    inst->summary = "generated image: " + inst->image_path;
    inst->parts[0] = agent_part{};
    inst->parts[0].kind = AGENT_PART_FILE;
    inst->parts[0].path = to_c(inst->image_path);
    inst->parts[0].mime = to_c("image/x-portable-pixmap");
    inst->parts[1] = agent_part{};
    inst->parts[1].kind = AGENT_PART_TEXT;
    inst->parts[1].text = to_c(inst->summary);
    resp->message.parts = inst->parts;
    resp->message.n_parts = 2;
    return AGENT_OK;
}

int32_t reverse_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<ReverseInstance *>(instance);
    inst->error.clear();
    // Never `*resp = agent_response{}`: that would clear struct_size, and against an older host with a
    // smaller agent_response it would write past the end. agent_response_reset keeps struct_size.
    agent_response_reset(resp);
    resp->message.role = to_c("assistant");

    // Fields appended after the v2 baseline must be guarded, e.g.:
    //   if (AGENT_HAS_FIELD(req, agent_request, some_future_field)) ... req->some_future_field ...
    // Everything this engine reads is part of the baseline, so no check is needed here.

    switch (req->task) {
    case AGENT_CAP_TEXT_GEN: return run_text(inst, req, resp);
    case AGENT_CAP_IMAGE_GEN: return run_image(inst, req, resp);
    default: inst->error = "reverse: unsupported task " + std::to_string(req->task); return AGENT_ERR_UNSUPPORTED;
    }
}

void reverse_free_response(void *, agent_response *resp) { agent_response_reset(resp); }

const char *reverse_last_error(void *instance) { return static_cast<ReverseInstance *>(instance)->error.c_str(); }

// Optional: used by the core for context-window accounting. Here: one token per whitespace-separated word.
int32_t reverse_count_tokens(void *, agent_str text, int32_t *out_count) {
    int32_t n = 0;
    bool in_word = false;
    for (char c : view(text)) {
        const bool space = c == ' ' || c == '\n' || c == '\t' || c == '\r';
        if (!space && !in_word) ++n;
        in_word = !space;
    }
    *out_count = n;
    return AGENT_OK;
}

const agent_engine_v1 kReverseEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1), // struct_size: the size this plugin was compiled with
    "reverse",
    "1.0",
    AGENT_CAP_TEXT_GEN | AGENT_CAP_IMAGE_GEN,
    0, // flags: run() serialized by the host, plain-text tool instructions
    reverse_create,
    reverse_destroy,
    reverse_run,
    reverse_free_response,
    reverse_last_error,
    reverse_count_tokens,
    nullptr, // reset_cache: no KV cache
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kReverseEngine; }
