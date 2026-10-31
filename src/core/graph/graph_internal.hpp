#pragma once

#include <agent-cpp/agent.hpp>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>

namespace agent::graph {

// Store keys. Scoped (user:/app:) values live under "graph_state:<key>"; the list of known scoped
// keys is a JSON array under kScopedIndexKey because StoreOps has no key listing.
inline constexpr std::string_view kScopedIndexKey = "graph_state_keys";
inline constexpr std::string_view kScopedValuePrefix = "graph_state:";
// Runtime-private per-thread keys (loop counters). Carried in checkpoints, never logged.
inline constexpr std::string_view kInternalPrefix = "__";

inline bool is_temp_key(std::string_view key) { return key.starts_with("temp:"); }
inline bool is_scoped_key(std::string_view key) { return key.starts_with("user:") || key.starts_with("app:"); }
inline bool is_internal_key(std::string_view key) { return key.starts_with(kInternalPrefix); }

int64_t now_ms();

std::string checkpoint_key(std::string_view thread_id, std::string_view suffix);
std::string_view status_name(GraphRunStatus status);

// Serialises the resumable part of run (step, next node, status, interrupt + pending tool calls,
// non-temp state values, Agent node histories) as JSON.
std::string encode_checkpoint(const GraphRun &run, std::string_view graph_name);
// Restores what encode_checkpoint wrote into out (state, step, next_node, status, interrupt fields,
// pending_tool_calls). Agent histories get FlowMemory ids "<thread_id>:<node>".
Result<void> decode_checkpoint(std::string_view json, GraphRun &out);

// Loads every persisted user:/app: key into state and fills index with the known scoped keys.
Result<void> load_scoped_state(Session &session, GraphState &state, std::set<std::string, std::less<>> &index);
Result<void> persist_scoped_value(Session &session, std::string_view key, std::string_view value,
                                  std::set<std::string, std::less<>> &index);

// Background writer behind GraphDurability::Async: one thread per checkpointer, a bounded FIFO of
// puts, flushed by the runtime before run_graph / resume_graph return. The thread starts on first
// use and is joined (after draining the queue) when the writer is destroyed.
class AsyncWriter {
  public:
    AsyncWriter(const CheckpointerOps *ops, std::shared_ptr<void> owner, void *state, size_t capacity);
    ~AsyncWriter();
    AsyncWriter(const AsyncWriter &) = delete;
    AsyncWriter &operator=(const AsyncWriter &) = delete;

    // Blocks while the queue is full.
    void put(std::string thread_id, uint64_t step, std::string blob, std::string metadata);
    // Waits until everything queued so far is written; returns the first error since the last flush.
    Result<void> flush();

  private:
    struct Job {
        std::string thread_id;
        uint64_t step = 0;
        std::string blob;
        std::string metadata;
    };
    void loop();

    const CheckpointerOps *ops_;
    std::shared_ptr<void> owner_; // keeps the checkpointer state alive
    void *state_;
    size_t capacity_;

    std::mutex mutex_;
    std::condition_variable cv_work_, cv_space_, cv_done_;
    std::deque<Job> queue_;
    uint64_t queued_ = 0, written_ = 0;
    bool stop_ = false, started_ = false;
    Error error_;
    std::thread thread_;
};

// The checkpointer one run_graph / resume_graph call writes to: the GraphRun's Checkpointer, or the
// Session's StoreOps behind an adapter (state = the Session).
struct Sink {
    const CheckpointerOps *ops = nullptr;
    void *state = nullptr;
    std::shared_ptr<AsyncWriter> writer; // shared with the Checkpointer, or owned by this call
};

// want_writer: create an AsyncWriter for this call when the Checkpointer has none.
Result<Sink> resolve_sink(Session &session, const Checkpointer &checkpointer, bool want_writer);

} // namespace agent::graph
