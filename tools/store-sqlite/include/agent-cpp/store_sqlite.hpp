#pragma once

// Optional SQLite-backed storage for agent.cpp (used by the agent CLI). Link agent_store_sqlite.

#include <agent-cpp/agent.hpp>

namespace agent::store {

// SQLite key-value store with a bounded write-back cache. Set MemoryConfig::store to this.
// MemoryConfig::db_path: file path, ":memory:" for an ephemeral db, or empty for
// <workspace>/memory/db.sqlite3.
const StoreOps *sqlite_store();

} // namespace agent::store

namespace agent {

// sqlite-vec backed DataSource for vector search. Query JSON: {"vector": [...], "top_k": N}.
Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(std::string name, std::string db_path, int dimensions);

Result<void> sqlite_vec_insert(DataSource &source, std::string_view text, std::span<const float> embedding);

} // namespace agent
