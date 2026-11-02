// decide() / decide_score() / decide_yes_no(): drive a session's engine with AGENT_CAP_DECIDE through
// providers::infer_task. Like embed() / rerank(), the request and response live in a per-thread arena
// (not the session's per-turn arena), so these can run from inside a tool call of another session's turn.

#include "../providers/provider_ops.hpp"
#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <atomic>
#include <string>

namespace agent {

namespace {

constexpr size_t kArenaBase = 64 * 1024;

Arena &decide_arena(size_t extra) {
    thread_local Arena arena;
    thread_local size_t capacity = 0;
    const size_t need = kArenaBase + extra;
    if (capacity < need) {
        arena = Arena(need);
        capacity = need;
    }
    arena.reset();
    return arena;
}

Result<Decision> run_decide(Session &session, std::string_view question, const std::string &params, const DecideOptions &opts) {
    if (question.empty()) return fail<Decision>(ErrorCode::InvalidConfig, "decide: question must not be empty.");
    if (std::atomic_ref<int32_t>(*session.cancel_flag).load(std::memory_order_relaxed) != 0)
        return fail<Decision>(ErrorCode::Cancelled, "decide: cancelled.");

    Arena &arena = decide_arena(params.size() + question.size() + opts.system.size());
    auto media = arena.allocate_span<MediaView>(opts.media.size());
    if (media.size() != opts.media.size()) return fail<Decision>(ErrorCode::OutOfMemory, "decide: arena full.");
    for (size_t i = 0; i < opts.media.size(); ++i) media[i] = MediaView{.path = opts.media[i].path, .mime = opts.media[i].mime};

    MessageView msgs[2];
    size_t n = 0;
    if (!opts.system.empty()) msgs[n++] = MessageView{.role = "system", .content = opts.system};
    msgs[n++] = MessageView{.role = "user", .content = question, .media = media};

    InferRequest req;
    req.messages = std::span<const MessageView>(msgs, n);
    req.cancel = session.cancel_flag.get();
    req.params_json = params;
    auto res = providers::infer_task(session, AGENT_CAP_DECIDE, req, arena);
    if (!res.ok) return fail<Decision>(res.error.code, res.error.message);

    for (const TensorView &t : res.value.output_tensors) {
        if (t.name != "probs") continue;
        if (t.dtype != DType::Float32 || t.data.size() % sizeof(float) != 0 || t.data.empty())
            return fail<Decision>(ErrorCode::DecodeFailed, "decide: engine returned a malformed \"probs\" tensor.");
        Decision d;
        const auto *f = reinterpret_cast<const float *>(t.data.data());
        d.probs.assign(f, f + t.data.size() / sizeof(float));
        for (size_t i = 1; i < d.probs.size(); ++i)
            if (d.probs[i] > d.probs[d.choice]) d.choice = i;
        d.confidence = d.probs[d.choice];
        d.score = d.confidence;
        return ok(std::move(d));
    }
    return fail<Decision>(ErrorCode::DecodeFailed, "decide: engine returned no \"probs\" tensor.");
}

std::string common_params(const DecideOptions &opts) { return opts.instructions ? std::string() : std::string(",\"instructions\":false"); }

} // namespace

Result<Decision> decide(Session &session, std::string_view question, std::span<const std::string_view> options, const DecideOptions &opts) {
    if (options.size() < 2) return fail<Decision>(ErrorCode::InvalidConfig, "decide: needs at least two options.");
    std::string params = "{\"mode\":\"choice\",\"options\":[";
    for (size_t i = 0; i < options.size(); ++i) {
        if (options[i].empty()) return fail<Decision>(ErrorCode::InvalidConfig, "decide: options must not be empty.");
        if (i) params += ',';
        params += '"';
        utils::escape_json_string(options[i], params);
        params += '"';
    }
    params += ']' + common_params(opts) + '}';
    auto d = run_decide(session, question, params, opts);
    if (!d.ok) return d;
    if (d.value.probs.size() != options.size())
        return fail<Decision>(ErrorCode::DecodeFailed, "decide: engine returned " + std::to_string(d.value.probs.size()) +
                                                           " probabilities for " + std::to_string(options.size()) + " options.");
    d.value.options.assign(options.begin(), options.end());
    return d;
}

Result<Decision> decide_score(Session &session, std::string_view question, int min, int max, const DecideOptions &opts) {
    if (min >= max || max - min > 100)
        return fail<Decision>(ErrorCode::InvalidConfig, "decide_score: needs min < max and at most 101 levels.");
    const std::string params =
        "{\"mode\":\"score\",\"scale\":[" + std::to_string(min) + "," + std::to_string(max) + "]" + common_params(opts) + "}";
    auto d = run_decide(session, question, params, opts);
    if (!d.ok) return d;
    const size_t levels = static_cast<size_t>(max - min) + 1;
    if (d.value.probs.size() != levels)
        return fail<Decision>(ErrorCode::DecodeFailed, "decide_score: engine returned " + std::to_string(d.value.probs.size()) +
                                                           " probabilities for " + std::to_string(levels) + " levels.");
    double expected = 0.0;
    for (size_t i = 0; i < levels; ++i) {
        d.value.options.push_back(std::to_string(min + static_cast<int>(i)));
        expected += static_cast<double>(d.value.probs[i]) * (min + static_cast<double>(i));
    }
    d.value.score = static_cast<float>(expected);
    return d;
}

Result<float> decide_yes_no(Session &session, std::string_view question, const DecideOptions &opts) {
    auto d = run_decide(session, question, "{\"mode\":\"yesno\"" + common_params(opts) + "}", opts);
    if (!d.ok) return fail<float>(d.error.code, d.error.message);
    if (d.value.probs.size() != 2)
        return fail<float>(ErrorCode::DecodeFailed,
                           "decide_yes_no: engine returned " + std::to_string(d.value.probs.size()) + " probabilities instead of 2.");
    return ok(d.value.probs[0]);
}

} // namespace agent
