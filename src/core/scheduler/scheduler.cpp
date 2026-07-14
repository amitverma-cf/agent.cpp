#include "../executor/ambient_turn.hpp"
#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <chrono>
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

Result<void> AgentScheduler::add_cron(std::string id, CronTask::TaskFn fn, void *user_data, std::chrono::milliseconds delay,
                                      std::chrono::milliseconds interval) {
    for (const auto &e : cron_jobs_)
        if (e.task.id == id) return fail(ErrorCode::InvalidConfig, "AgentScheduler: cron id already registered: " + id);

    CronJob entry;
    entry.task.id = std::move(id);
    entry.task.callback = fn;
    entry.task.user_data = user_data;
    entry.task.next_fire = std::chrono::steady_clock::now() + delay;
    entry.task.interval = interval;
    entry.done = false;
    cron_jobs_.push_back(std::move(entry));
    return ok();
}

Result<void> AgentScheduler::cancel_cron(std::string_view id) {
    for (auto &e : cron_jobs_) {
        if (e.task.id == id) {
            e.done = true;
            return ok();
        }
    }
    return fail(ErrorCode::KeyNotFound, "AgentScheduler: cron job not found: " + std::string(id));
}

Result<void> AgentScheduler::pump(Session &session) {
    auto now = std::chrono::steady_clock::now();

    for (auto &entry : cron_jobs_) {
        if (entry.done || now < entry.task.next_fire) continue;

        utils::log(session, LogLevel::Debug, "AgentScheduler: cron '" + entry.task.id + "' fired.");
        trigger_event(session, EventType::OnCronFire, std::span<const uint8_t>());

        if (entry.task.callback) {
            auto res = entry.task.callback(session, entry.task.user_data);
            if (!res.ok)
                utils::log(session, LogLevel::Warning, "AgentScheduler: cron '" + entry.task.id + "' failed: " + res.error.message);
        }

        if (entry.task.interval.count() > 0) {
            entry.task.next_fire += entry.task.interval;
            now = std::chrono::steady_clock::now();
            if (entry.task.next_fire < now) entry.task.next_fire = now + entry.task.interval;
        } else {
            entry.done = true;
        }
    }

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

    const bool provider_can_run_parallel =
        session.config.provider == AiProvider::OpenAICompatible || session.config.provider == AiProvider::LlamaCpp;

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
    return pump(session);
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
