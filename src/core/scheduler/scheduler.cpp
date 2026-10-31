#include "../executor/ambient_turn.hpp"
#include "../providers/provider_ops.hpp"
#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <thread>
#include <vector>

namespace agent {

Result<void> AgentScheduler::spawn(std::string id, Flow flow, void *context) {
    for (const auto &e : subagents_)
        if (e.id == id) return fail(ErrorCode::InvalidConfig, "AgentScheduler: subagent id already registered: " + id);

    SubAgent entry;
    entry.id = std::move(id);
    entry.flow = std::move(flow);
    entry.context = context;
    entry.done = false;
    entry.result.id = entry.id;
    subagents_.push_back(std::move(entry));
    return ok();
}

Result<void> AgentScheduler::pump(Session &session) {
    auto finish = [&](SubAgent &entry, Result<bool> res) {
        if (!res.ok) {
            entry.done = true;
            entry.result.ok = false;
            entry.result.error = res.error;
            entry.result.stats = entry.flow.memory.stats;
            trigger_event(session, EventType::OnSubAgentComplete, std::span<const uint8_t>());
            utils::log(session, LogLevel::Warning, "AgentScheduler: subagent '" + entry.id + "' failed: " + res.error.message);
        } else if (!res.value) {
            entry.done = true;
            entry.result.ok = true;
            entry.result.stats = entry.flow.memory.stats;
            trigger_event(session, EventType::OnSubAgentComplete, std::span<const uint8_t>());
            utils::log(session, LogLevel::Info, "AgentScheduler: subagent '" + entry.id + "' completed.");
        }
    };

    const bool provider_can_run_parallel = providers::supports_parallel(session);

    if (provider_can_run_parallel) {
        const int batch_size = std::max(1, session.config.max_parallel_subagents);

        std::vector<SubAgent *> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        for (auto &entry : subagents_) {
            if (entry.done) continue;
            batch.push_back(&entry);
            if (static_cast<int>(batch.size()) >= batch_size) break;
        }

        std::vector<Result<bool>> results(batch.size());
        std::vector<std::thread> workers;
        workers.reserve(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            SubAgent *entry = batch[i];
            Result<bool> *out = &results[i];
            size_t slot = i; // indexes Session::worker_arenas
            workers.emplace_back([&session, entry, out, slot]() {
                ambient::worker_arena_index = slot;
                *out = step_flow(session, entry->context, entry->flow);
            });
        }
        for (auto &w : workers) w.join();

        for (size_t i = 0; i < batch.size(); ++i) {
            utils::log(session, LogLevel::Debug, "AgentScheduler: stepped subagent '" + batch[i]->id + "' (parallel).");
            finish(*batch[i], results[i]);
        }
    } else {
        for (auto &entry : subagents_) {
            if (entry.done) continue;

            utils::log(session, LogLevel::Debug, "AgentScheduler: stepping subagent '" + entry.id + "'.");

            auto res = step_flow(session, entry.context, entry.flow);
            finish(entry, res);
        }
    }

    return ok();
}

Result<void> AgentScheduler::run_until_done(Session &session) {
    while (!all_done()) {
        auto res = pump(session);
        if (!res.ok) return res;
    }
    return ok();
}

bool AgentScheduler::all_done() const {
    for (const auto &e : subagents_)
        if (!e.done) return false;
    return true;
}

std::vector<SubAgentResult> AgentScheduler::results() const {
    std::vector<SubAgentResult> out;
    out.reserve(subagents_.size());
    for (const auto &e : subagents_) out.push_back(e.result);
    return out;
}

} // namespace agent
