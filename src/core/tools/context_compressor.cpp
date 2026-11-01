#include "../executor/ambient_turn.hpp"
#include "../providers/provider_ops.hpp"
#include "../utils/utils.hpp"
#include "tools_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <mutex>
#include <simdjson.h>

namespace agent {

namespace {

constexpr size_t kSummaryArenaBase = 256 * 1024;

// The summary request is built and answered in a per-thread arena, not the Session's per-turn arena:
// compress_context also runs as a tool, in the middle of run_turn, whose message and tool-call views
// live in that arena.
Arena &summary_arena(size_t prompt_bytes) {
    thread_local Arena arena;
    thread_local size_t capacity = 0;
    const size_t need = kSummaryArenaBase + 2 * prompt_bytes;
    if (capacity < need) {
        arena = Arena(need);
        capacity = need;
    }
    arena.reset();
    return arena;
}

// A name no tool has: while it is the only allowed tool, the summary request carries no tool
// definitions (the summarizer must answer with text, not call tools).
constexpr std::string_view kNoTools[] = {"\x01no-tools"};

struct AllowedToolsScope {
    std::span<const std::string_view> saved = ambient::allowed_tools;
    AllowedToolsScope() { ambient::allowed_tools = kNoTools; }
    ~AllowedToolsScope() { ambient::allowed_tools = saved; }
    AllowedToolsScope(const AllowedToolsScope &) = delete;
    AllowedToolsScope &operator=(const AllowedToolsScope &) = delete;
};

Result<std::string> summarize(Session &session, std::string_view prompt) {
    MessageView views[2]{};
    views[0].role = "system";
    views[0].content = "You are a concise summarizer. Output only the summary, no preamble.";
    views[1].role = "user";
    views[1].content = prompt;
    InferRequest req{.messages = std::span<const MessageView>(views, 2), .max_tokens = 512, .cancel = session.cancel_flag.get()};

    AllowedToolsScope no_tools;
    if (session.config.provider == AiProvider::Engine) {
        // Always a chat request, whatever Config::default_task says; fails cleanly (and compression is
        // skipped) on an engine without TEXT_GEN.
        auto res = providers::infer_task(session, AGENT_CAP_TEXT_GEN, req, summary_arena(prompt.size()));
        if (!res.ok) return fail<std::string>(res.error.code, res.error.message);
        return ok(std::string(res.value.message.content));
    }
    auto res = infer(session, req);
    if (!res.ok) return fail<std::string>(res.error.code, res.error.message);
    return ok(std::string(res.value.message.content));
}

} // namespace

Result<void> compress_context(Session &session, FlowMemory &state, size_t keep_recent) {
    // A summary left by an earlier compression is compressed again, not kept as the system prompt.
    bool has_system = !state.history.empty() && state.history[0].role == "system" &&
                      !std::string_view(state.history[0].content).starts_with(tools::kCompressedHistoryPrefix);
    size_t start = has_system ? 1 : 0;

    if (state.history.size() <= start + keep_recent + 2) {
        utils::log(session, LogLevel::Debug, "compress_context: history too short to compress, skipping.");
        return ok();
    }

    size_t compress_end = state.history.size() - keep_recent;
    // Called as a tool, the last message is the assistant message whose tool calls are running: keep it
    // (its results are appended right after).
    const Message &last = state.history.back();
    if (last.role == "assistant" && !last.tool_calls.empty()) compress_end = std::min(compress_end, state.history.size() - 1);
    // Never separate tool results from the assistant message that requested them.
    while (compress_end > start && compress_end < state.history.size() && state.history[compress_end].role == "tool") --compress_end;
    if (compress_end < start + 2) {
        utils::log(session, LogLevel::Debug, "compress_context: nothing old enough to compress, skipping.");
        return ok();
    }

    std::string prompt = "Summarize the following conversation concisely. "
                         "Preserve all key facts, decisions, code snippets, and tool outputs:\n\n";
    for (size_t i = start; i < compress_end; ++i) {
        const auto &msg = state.history[i];
        prompt += msg.role;
        prompt += ": ";
        if (!msg.content.empty()) {
            prompt += msg.content;
            prompt += ' ';
        }
        for (const auto &tc : msg.tool_calls) {
            prompt += "[tool:";
            prompt += tc.name;
            prompt += ' ';
            prompt += tc.arguments;
            prompt += "] ";
        }
        // Files the summary drops from the context: keep at least their names.
        for (const auto &m : msg.media) {
            prompt += "[file: ";
            prompt += m.path;
            prompt += " (";
            prompt += m.mime;
            prompt += ")] ";
        }
        prompt += '\n';
    }

    auto res = summarize(session, prompt);
    if (!res.ok) {
        utils::log(session, LogLevel::Warning,
                   "compress_context: LLM summarization failed, skipping compression. "
                   "The pruner will handle overflow. Error: " +
                       res.error.message);
        return ok();
    }

    Message summary_msg;
    summary_msg.role = "system"; // run_turn merges leading system messages into one
    summary_msg.content = std::string(tools::kCompressedHistoryPrefix) + res.value;

    auto it_start = state.history.begin() + static_cast<ptrdiff_t>(start);
    auto it_end = state.history.begin() + static_cast<ptrdiff_t>(compress_end);
    state.history.erase(it_start, it_end);
    state.history.insert(state.history.begin() + static_cast<ptrdiff_t>(start), std::move(summary_msg));

    state.stats.compressions++;
    {
        std::lock_guard<std::mutex> lock(*session.stats_mutex);
        session.stats.total_compressions++;
    }

    trigger_event(session, EventType::OnCompression, std::span<const uint8_t>());
    utils::log(session, LogLevel::Info, "compress_context: history compressed successfully.");
    return ok();
}

namespace tools {
namespace {

Result<std::string> context_compressor_callback(std::string_view arguments, void *user_data) {
    auto *session_cell = static_cast<Session **>(user_data);
    Session *session = (session_cell && *session_cell) ? *session_cell : nullptr;
    if (!session) return fail<std::string>(ErrorCode::InvalidConfig, "compress_context: session reference is null");
    if (!ambient::active_memory) return fail<std::string>(ErrorCode::InvalidConfig, "compress_context: no active conversation");

    size_t keep_recent = 6;
    if (!arguments.empty()) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(arguments.data(), arguments.size());
        simdjson::ondemand::document doc;
        if (!parser.iterate(json).get(doc)) {
            simdjson::ondemand::object obj;
            if (!doc.get_object().get(obj)) {
                for (auto field : obj) {
                    auto k = field.unescaped_key();
                    if (k.error()) continue;
                    if (k.value() == "keep_recent") {
                        int64_t v;
                        if (!field.value().get_int64().get(v) && v >= 0) keep_recent = static_cast<size_t>(v);
                    }
                }
            }
        }
    }

    auto res = compress_context(*session, *ambient::active_memory, keep_recent);
    if (!res.ok) return fail<std::string>(res.error.code, res.error.message);
    return ok(std::string("Context compressed."));
}

} // namespace

Tool get_context_compressor_tool(Session **session_cell) {
    return Tool{
        .name = "compress_context",
        .description = "Summarize and compress the active conversation's older history to free context "
                       "window space. Optional: keep_recent (integer, number of most recent messages to "
                       "keep uncompressed, default 6).",
        .parameter_schema =
            R"({"type":"object","properties":{"keep_recent":{"type":"integer","description":"messages to keep uncompressed, default 6"}}})",
        .callback = context_compressor_callback,
        .user_data = session_cell,
    };
}

} // namespace tools
} // namespace agent
