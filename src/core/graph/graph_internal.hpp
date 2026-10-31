#pragma once

#include <agent-cpp/agent.hpp>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>

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

} // namespace agent::graph
