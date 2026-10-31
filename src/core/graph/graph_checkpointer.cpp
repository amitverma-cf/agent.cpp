// Graph checkpointers: the in-memory CheckpointerOps, the StoreOps adapter used when a GraphRun names
// no checkpointer, and the Async background writer.

#include "graph_internal.hpp"

#include <agent-cpp/agent.hpp>
#include <charconv>
#include <map>
#include <utility>

namespace agent {

namespace {

// ---- in-memory checkpointer ----

struct MemCheckpoint {
    std::string blob;
    std::string metadata;
    int64_t timestamp_ms = 0;
};

struct MemCheckpointer {
    std::mutex mutex;
    std::map<std::string, std::map<uint64_t, MemCheckpoint>, std::less<>> threads;
    std::map<std::string, std::vector<GraphEvent>, std::less<>> events;
};

Result<std::shared_ptr<void>> mem_open(const CheckpointerConfig &) {
    return ok<std::shared_ptr<void>>(std::make_shared<MemCheckpointer>());
}

Result<void> mem_put(void *state, std::string_view thread_id, uint64_t step, std::string_view blob, std::string_view metadata) {
    auto *s = static_cast<MemCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    auto it = s->threads.find(thread_id);
    if (it == s->threads.end()) it = s->threads.emplace(std::string(thread_id), std::map<uint64_t, MemCheckpoint>{}).first;
    it->second.insert_or_assign(
        step, MemCheckpoint{.blob = std::string(blob), .metadata = std::string(metadata), .timestamp_ms = graph::now_ms()});
    return ok();
}

Result<std::string> no_checkpoint(std::string_view thread_id) {
    return fail<std::string>(ErrorCode::KeyNotFound, "no checkpoint for thread '" + std::string(thread_id) + "'");
}

Result<std::string> mem_get_latest(void *state, std::string_view thread_id) {
    auto *s = static_cast<MemCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    auto it = s->threads.find(thread_id);
    if (it == s->threads.end() || it->second.empty()) return no_checkpoint(thread_id);
    return ok(it->second.rbegin()->second.blob);
}

Result<std::string> mem_get(void *state, std::string_view thread_id, uint64_t step) {
    auto *s = static_cast<MemCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    auto it = s->threads.find(thread_id);
    if (it == s->threads.end()) return no_checkpoint(thread_id);
    auto cp = it->second.find(step);
    if (cp == it->second.end())
        return fail<std::string>(ErrorCode::KeyNotFound,
                                 "no checkpoint at step " + std::to_string(step) + " of thread '" + std::string(thread_id) + "'");
    return ok(cp->second.blob);
}

Result<std::vector<GraphCheckpointInfo>> mem_list(void *state, std::string_view thread_id) {
    auto *s = static_cast<MemCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    std::vector<GraphCheckpointInfo> out;
    auto it = s->threads.find(thread_id);
    if (it == s->threads.end()) return ok(std::move(out));
    for (const auto &[step, cp] : it->second)
        out.push_back(GraphCheckpointInfo{.step = step, .timestamp_ms = cp.timestamp_ms, .metadata = cp.metadata});
    return ok(std::move(out));
}

Result<void> mem_delete_thread(void *state, std::string_view thread_id) {
    auto *s = static_cast<MemCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    auto it = s->threads.find(thread_id);
    if (it != s->threads.end()) s->threads.erase(it);
    auto ev = s->events.find(thread_id);
    if (ev != s->events.end()) s->events.erase(ev);
    return ok();
}

Result<void> mem_append_events(void *state, std::string_view thread_id, std::span<const GraphEvent> events) {
    auto *s = static_cast<MemCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    auto it = s->events.find(thread_id);
    if (it == s->events.end()) it = s->events.emplace(std::string(thread_id), std::vector<GraphEvent>{}).first;
    it->second.insert(it->second.end(), events.begin(), events.end());
    return ok();
}

Result<std::vector<GraphEvent>> mem_list_events(void *state, std::string_view thread_id) {
    auto *s = static_cast<MemCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    auto it = s->events.find(thread_id);
    if (it == s->events.end()) return ok(std::vector<GraphEvent>{});
    return ok(it->second);
}

constexpr CheckpointerOps kInMemoryCheckpointer{.open = mem_open,
                                                .put = mem_put,
                                                .get_latest = mem_get_latest,
                                                .get = mem_get,
                                                .list = mem_list,
                                                .delete_thread = mem_delete_thread,
                                                .append_events = mem_append_events,
                                                .list_events = mem_list_events};

// ---- StoreOps adapter (state = the Session) ----
// checkpoint:<t>:<step> and checkpoint:<t>:latest hold the blob (the v1 layout).
// checkpoint_meta:<t>:<step> holds "<timestamp_ms> <metadata>", checkpoint_meta:<t>:latest the
// highest step (StoreOps cannot list keys, so list() probes 0..latest). StoreOps cannot delete
// either: delete_thread overwrites the keys with "" and an empty value reads as missing.

std::string meta_key(std::string_view thread_id, std::string_view suffix) {
    std::string key = "checkpoint_meta:";
    key += thread_id;
    key += ':';
    key += suffix;
    return key;
}

Result<std::string> store_get_nonempty(Session &session, const std::string &key) {
    auto got = retrieve_memory(session, key);
    if (got.ok && got.value.empty()) return fail<std::string>(ErrorCode::KeyNotFound, "Key not found: " + key);
    return got;
}

Result<void> store_put(void *state, std::string_view thread_id, uint64_t step, std::string_view blob, std::string_view metadata) {
    Session &session = *static_cast<Session *>(state);
    const std::string step_str = std::to_string(step);
    auto put = store_memory(session, graph::checkpoint_key(thread_id, step_str), blob);
    if (!put.ok) return put;
    put = store_memory(session, graph::checkpoint_key(thread_id, "latest"), blob);
    if (!put.ok) return put;
    put = store_memory(session, meta_key(thread_id, step_str), std::to_string(graph::now_ms()) + " " + std::string(metadata));
    if (!put.ok) return put;
    return store_memory(session, meta_key(thread_id, "latest"), step_str);
}

Result<std::string> store_get_latest(void *state, std::string_view thread_id) {
    auto got = store_get_nonempty(*static_cast<Session *>(state), graph::checkpoint_key(thread_id, "latest"));
    if (!got.ok && got.error.code == ErrorCode::KeyNotFound) return no_checkpoint(thread_id);
    return got;
}

Result<std::string> store_get(void *state, std::string_view thread_id, uint64_t step) {
    return store_get_nonempty(*static_cast<Session *>(state), graph::checkpoint_key(thread_id, std::to_string(step)));
}

Result<uint64_t> store_latest_step(Session &session, std::string_view thread_id) {
    auto latest = store_get_nonempty(session, meta_key(thread_id, "latest"));
    if (!latest.ok) return fail<uint64_t>(latest.error.code, latest.error.message);
    uint64_t step = 0;
    auto [ptr, ec] = std::from_chars(latest.value.data(), latest.value.data() + latest.value.size(), step);
    if (ec != std::errc())
        return fail<uint64_t>(ErrorCode::ParseError, "checkpoint index of thread '" + std::string(thread_id) + "' is corrupt");
    return ok(step);
}

Result<std::vector<GraphCheckpointInfo>> store_list(void *state, std::string_view thread_id) {
    Session &session = *static_cast<Session *>(state);
    std::vector<GraphCheckpointInfo> out;
    auto latest = store_latest_step(session, thread_id);
    if (!latest.ok) {
        if (latest.error.code == ErrorCode::KeyNotFound) return ok(std::move(out));
        return fail<std::vector<GraphCheckpointInfo>>(latest.error.code, latest.error.message);
    }
    for (uint64_t step = 0; step <= latest.value; ++step) {
        auto meta = store_get_nonempty(session, meta_key(thread_id, std::to_string(step)));
        if (!meta.ok) {
            if (meta.error.code == ErrorCode::KeyNotFound) continue;
            return fail<std::vector<GraphCheckpointInfo>>(meta.error.code, meta.error.message);
        }
        GraphCheckpointInfo info;
        info.step = step;
        const size_t space = meta.value.find(' ');
        std::from_chars(meta.value.data(), meta.value.data() + std::min(space, meta.value.size()), info.timestamp_ms);
        if (space != std::string::npos) info.metadata = meta.value.substr(space + 1);
        out.push_back(std::move(info));
    }
    return ok(std::move(out));
}

// Event log: graph_events:<t>:<n> holds the n-th appended batch (a JSON array), graph_events:<t>:count
// the number of batches.
std::string events_key(std::string_view thread_id, std::string_view suffix) {
    std::string key = "graph_events:";
    key += thread_id;
    key += ':';
    key += suffix;
    return key;
}

Result<uint64_t> store_event_batches(Session &session, std::string_view thread_id) {
    auto count = store_get_nonempty(session, events_key(thread_id, "count"));
    if (!count.ok) {
        if (count.error.code == ErrorCode::KeyNotFound) return ok<uint64_t>(0);
        return fail<uint64_t>(count.error.code, count.error.message);
    }
    uint64_t n = 0;
    std::from_chars(count.value.data(), count.value.data() + count.value.size(), n);
    return ok(n);
}

Result<void> store_append_events(void *state, std::string_view thread_id, std::span<const GraphEvent> events) {
    Session &session = *static_cast<Session *>(state);
    auto n = store_event_batches(session, thread_id);
    if (!n.ok) return fail(n.error.code, n.error.message);
    std::string json;
    graph::encode_events(events, json);
    auto put = store_memory(session, events_key(thread_id, std::to_string(n.value)), json);
    if (!put.ok) return put;
    return store_memory(session, events_key(thread_id, "count"), std::to_string(n.value + 1));
}

Result<std::vector<GraphEvent>> store_list_events(void *state, std::string_view thread_id) {
    using R = Result<std::vector<GraphEvent>>;
    Session &session = *static_cast<Session *>(state);
    auto n = store_event_batches(session, thread_id);
    if (!n.ok) return fail<std::vector<GraphEvent>>(n.error.code, n.error.message);
    std::vector<GraphEvent> out;
    for (uint64_t i = 0; i < n.value; ++i) {
        auto batch = retrieve_memory(session, events_key(thread_id, std::to_string(i)));
        if (!batch.ok) return R{.ok = false, .value = {}, .error = batch.error};
        auto decoded = graph::decode_events(batch.value, out);
        if (!decoded.ok) return R{.ok = false, .value = {}, .error = decoded.error};
    }
    return ok(std::move(out));
}

Result<void> store_delete_thread(void *state, std::string_view thread_id) {
    Session &session = *static_cast<Session *>(state);
    if (store_get_nonempty(session, events_key(thread_id, "count")).ok) {
        auto put = store_memory(session, events_key(thread_id, "count"), "");
        if (!put.ok) return put;
    }
    auto latest = store_latest_step(session, thread_id);
    if (latest.ok) {
        for (uint64_t step = 0; step <= latest.value; ++step) {
            const std::string s = std::to_string(step);
            if (store_get_nonempty(session, meta_key(thread_id, s)).ok) {
                auto put = store_memory(session, meta_key(thread_id, s), "");
                if (!put.ok) return put;
                put = store_memory(session, graph::checkpoint_key(thread_id, s), "");
                if (!put.ok) return put;
            }
        }
        auto put = store_memory(session, meta_key(thread_id, "latest"), "");
        if (!put.ok) return put;
    } else if (latest.error.code != ErrorCode::KeyNotFound) {
        return fail(latest.error.code, latest.error.message);
    }
    if (retrieve_memory(session, graph::checkpoint_key(thread_id, "latest")).ok)
        return store_memory(session, graph::checkpoint_key(thread_id, "latest"), "");
    return ok();
}

constexpr CheckpointerOps kStoreCheckpointer{.open = nullptr,
                                             .put = store_put,
                                             .get_latest = store_get_latest,
                                             .get = store_get,
                                             .list = store_list,
                                             .delete_thread = store_delete_thread,
                                             .append_events = store_append_events,
                                             .list_events = store_list_events};

bool complete(const CheckpointerOps *ops) {
    return ops->put && ops->get_latest && ops->get && ops->list && ops->delete_thread && !ops->append_events == !ops->list_events;
}

} // namespace

const CheckpointerOps *in_memory_checkpointer() { return &kInMemoryCheckpointer; }

Result<Checkpointer> open_checkpointer(const CheckpointerOps *ops, const CheckpointerConfig &config, size_t async_queue_capacity) {
    if (!ops) return fail<Checkpointer>(ErrorCode::InvalidConfig, "open_checkpointer: ops is null");
    if (!ops->open || !complete(ops))
        return fail<Checkpointer>(ErrorCode::InvalidConfig,
                                  "open_checkpointer: CheckpointerOps is missing one of open/put/get_latest/get/list/delete_thread, or "
                                  "sets only one of append_events/list_events");
    auto opened = ops->open(config);
    if (!opened.ok) return fail<Checkpointer>(opened.error.code, opened.error.message);
    Checkpointer cp;
    cp.ops = ops;
    cp.state = std::move(opened.value);
    void *raw = cp.state.get();
    cp.writer = std::make_shared<graph::AsyncWriter>(ops, cp.state, raw, async_queue_capacity == 0 ? 1 : async_queue_capacity);
    return ok(std::move(cp));
}

namespace graph {

// ---- AsyncWriter ----

AsyncWriter::AsyncWriter(const CheckpointerOps *ops, std::shared_ptr<void> owner, void *state, size_t capacity)
    : ops_(ops),
      owner_(std::move(owner)),
      state_(state),
      capacity_(capacity) {}

AsyncWriter::~AsyncWriter() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_work_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void AsyncWriter::put(std::string thread_id, uint64_t step, std::string blob, std::string metadata) {
    enqueue(Job{.thread_id = std::move(thread_id),
                .step = step,
                .blob = std::move(blob),
                .metadata = std::move(metadata),
                .events = {},
                .is_events = false});
}

void AsyncWriter::append_events(std::string thread_id, std::vector<GraphEvent> events) {
    if (!ops_->append_events || events.empty()) return;
    enqueue(Job{.thread_id = std::move(thread_id), .step = 0, .blob = {}, .metadata = {}, .events = std::move(events), .is_events = true});
}

void AsyncWriter::enqueue(Job job) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!started_) {
        started_ = true;
        thread_ = std::thread([this]() { loop(); });
    }
    cv_space_.wait(lock, [this]() { return queue_.size() < capacity_; });
    queue_.push_back(std::move(job));
    ++queued_;
    lock.unlock();
    cv_work_.notify_one();
}

Result<void> AsyncWriter::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    const uint64_t target = queued_;
    cv_done_.wait(lock, [this, target]() { return written_ >= target; });
    if (error_.code == ErrorCode::Ok) return ok();
    Error e = std::move(error_);
    error_ = {};
    return Result<void>{.ok = false, .error = std::move(e)};
}

void AsyncWriter::loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        cv_work_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
        if (queue_.empty()) break; // stop_ and drained
        Job job = std::move(queue_.front());
        queue_.pop_front();
        cv_space_.notify_one();
        lock.unlock();
        auto put = job.is_events ? ops_->append_events(state_, job.thread_id, job.events)
                                 : ops_->put(state_, job.thread_id, job.step, job.blob, job.metadata);
        lock.lock();
        if (!put.ok && error_.code == ErrorCode::Ok) error_ = put.error;
        ++written_;
        cv_done_.notify_all();
    }
}

Result<Sink> resolve_sink(Session &session, const Checkpointer &checkpointer, bool want_writer) {
    Sink sink;
    if (checkpointer.ops) {
        if (!checkpointer.state) return fail<Sink>(ErrorCode::InvalidConfig, "graph checkpointer is not open (use open_checkpointer)");
        if (!complete(checkpointer.ops))
            return fail<Sink>(ErrorCode::InvalidConfig, "graph checkpointer is missing one of put/get_latest/get/list/delete_thread, or "
                                                        "sets only one of append_events/list_events");
        sink.ops = checkpointer.ops;
        sink.state = checkpointer.state.get();
        sink.writer = std::static_pointer_cast<AsyncWriter>(checkpointer.writer);
    } else {
        if (!session.memory_state) return fail<Sink>(ErrorCode::ProviderInitFailed, "Memory store not initialized.");
        sink.ops = &kStoreCheckpointer;
        sink.state = &session;
    }
    if (want_writer && !sink.writer) sink.writer = std::make_shared<AsyncWriter>(sink.ops, nullptr, sink.state, 64);
    return ok(std::move(sink));
}

} // namespace graph

Result<std::vector<GraphCheckpointInfo>> list_graph_checkpoints(Session &session, std::string_view thread_id,
                                                                const Checkpointer &checkpointer) {
    auto sink = graph::resolve_sink(session, checkpointer, false);
    if (!sink.ok) return fail<std::vector<GraphCheckpointInfo>>(sink.error.code, sink.error.message);
    return sink.value.ops->list(sink.value.state, thread_id.empty() ? std::string_view("default") : thread_id);
}

Result<std::vector<GraphEvent>> list_graph_events(Session &session, std::string_view thread_id, const Checkpointer &checkpointer) {
    using R = Result<std::vector<GraphEvent>>;
    auto sink = graph::resolve_sink(session, checkpointer, false);
    if (!sink.ok) return R{.ok = false, .value = {}, .error = sink.error};
    if (!sink.value.ops->list_events)
        return fail<std::vector<GraphEvent>>(ErrorCode::UnsupportedProvider, "graph checkpointer has no event log");
    return sink.value.ops->list_events(sink.value.state, thread_id.empty() ? std::string_view("default") : thread_id);
}

Result<void> delete_graph_thread(Session &session, std::string_view thread_id, const Checkpointer &checkpointer) {
    auto sink = graph::resolve_sink(session, checkpointer, false);
    if (!sink.ok) return fail(sink.error.code, sink.error.message);
    return sink.value.ops->delete_thread(sink.value.state, thread_id.empty() ? std::string_view("default") : thread_id);
}

} // namespace agent
