#pragma once

// Optional SQLite-backed storage for agent.cpp (used by the agent CLI). Link agent_store_sqlite.

#include <agent-cpp/agent.hpp>

namespace agent::store {

// SQLite key-value store with a bounded write-back cache. Set MemoryConfig::store to this.
// MemoryConfig::db_path: file path, ":memory:" for an ephemeral db, or empty for
// <workspace>/memory/db.sqlite3.
const StoreOps *sqlite_store();

// sqlite-vec vector store for RAG (RagIndex::store). VectorStoreConfig::path: file path, or empty /
// ":memory:". Tables <name>_vectors (vec0, cosine distance) and <name>_chunks; scores are
// 1 - cosine distance. An existing table keeps the dimensions it was created with.
const VectorStoreOps *sqlite_vector_store();

// SQLite graph checkpointer (open with agent::open_checkpointer). CheckpointerConfig::path: file
// path, or empty / ":memory:". WAL journal; tables graph_checkpoints keyed by (thread_id, step) and
// graph_events (the persisted event log) keyed by (thread_id, seq).
// Safe to share between threads; several processes may open the same file.
const CheckpointerOps *sqlite_checkpointer();

} // namespace agent::store

namespace agent {

// sqlite-vec backed DataSource for vector search. Query JSON: {"vector": [...], "top_k": N}.
Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(std::string name, std::string db_path, int dimensions);

Result<void> sqlite_vec_insert(DataSource &source, std::string_view text, std::span<const float> embedding);

} // namespace agent
