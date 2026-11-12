// ocr(): drive a session's engine with AGENT_CAP_OCR through providers::infer_task, in a per-thread
// arena like embed() / decide(), so it can run from inside a tool call of another session's turn.

#include "../providers/provider_ops.hpp"
#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <atomic>
#include <string>

namespace agent {

namespace {

Arena &ocr_arena(size_t extra) {
    thread_local Arena arena;
    thread_local size_t capacity = 0;
    const size_t need = 64 * 1024 + extra;
    if (capacity < need) {
        arena = Arena(need);
        capacity = need;
    }
    arena.reset();
    return arena;
}

} // namespace

Result<std::string> ocr(Session &session, const Media &image, std::string_view prompt) {
    if (image.path.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "ocr: image path must not be empty.");
    if (std::atomic_ref<int32_t>(*session.cancel_flag).load(std::memory_order_relaxed) != 0)
        return fail<std::string>(ErrorCode::Cancelled, "ocr: cancelled.");

    std::string params;
    if (!prompt.empty()) {
        params = "{\"prompt\":\"";
        utils::escape_json_string(prompt, params);
        params += "\"}";
    }
    // The response text is copied into the arena: leave room for a long page.
    Arena &arena = ocr_arena(params.size() + image.path.size() + 256 * 1024);
    const MediaView media{.path = image.path, .mime = image.mime};
    const MessageView msg{.role = "user", .media = std::span<const MediaView>(&media, 1)};
    InferRequest req;
    req.messages = std::span<const MessageView>(&msg, 1);
    req.cancel = session.cancel_flag.get();
    req.params_json = params;
    auto res = providers::infer_task(session, AGENT_CAP_OCR, req, arena);
    if (!res.ok) return fail<std::string>(res.error.code, res.error.message);
    return ok(std::string(res.value.message.content));
}

} // namespace agent
