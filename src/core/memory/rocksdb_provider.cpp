#include "memory_ops.hpp"
#include <filesystem>
#include <system_error>

#ifdef AGENT_HAS_ROCKSDB
#include <rocksdb/db.h>
#include <rocksdb/options.h>

namespace agent::memory {

struct RocksDBState {
    rocksdb::DB* db = nullptr;

    ~RocksDBState() {
        if (db) {
            db->Close();
            delete db;
        }
    }
};

Result<void> init_rocksdb(Session &session) {
    if (session.config.memory.db_path.empty()) {
        return fail(ErrorCode::InvalidConfig, "RocksDB requires a valid db_path in MemoryConfig.");
    }

    std::error_code ec;
    std::filesystem::create_directories(session.config.memory.db_path, ec);
    if (ec) {
        return fail(ErrorCode::FilesystemError, "Failed to create RocksDB directory: " + ec.message());
    }

    rocksdb::Options options;
    options.IncreaseParallelism();
    options.OptimizeLevelStyleCompaction();
    options.create_if_missing = true;
    
    // Memory tuning for edge AI (Limit background RAM usage)
    options.write_buffer_size = 64 * 1024 * 1024; // 64MB MemTable

    rocksdb::DB* db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, session.config.memory.db_path, &db);

    if (!status.ok()) {
        return fail(ErrorCode::ProviderInitFailed, "Failed to open RocksDB: " + status.ToString());
    }

    auto state = std::make_shared<RocksDBState>();
    state->db = db;
    session.memory_state = state;

    return ok();
}

Result<void> store_rocksdb(Session &session, std::string_view key, std::string_view value) {
    auto state = std::static_pointer_cast<RocksDBState>(session.memory_state);
    if (!state || !state->db) return fail(ErrorCode::ProviderInitFailed, "RocksDB not initialized");

    rocksdb::WriteOptions write_options;
    // CRITICAL: sync = false means the thread returns instantly. 
    // Data lives in RAM and flushes to SSD asynchronously.
    write_options.sync = false; 

    rocksdb::Status s = state->db->Put(write_options, rocksdb::Slice(key.data(), key.size()), rocksdb::Slice(value.data(), value.size()));
    
    if (!s.ok()) {
        return fail(ErrorCode::FilesystemError, "RocksDB Write Failed: " + s.ToString());
    }
    return ok();
}

Result<std::string> retrieve_rocksdb(Session &session, std::string_view key) {
    auto state = std::static_pointer_cast<RocksDBState>(session.memory_state);
    if (!state || !state->db) return fail<std::string>(ErrorCode::ProviderInitFailed, "RocksDB not initialized");

    std::string value;
    rocksdb::ReadOptions read_options;
    rocksdb::Status s = state->db->Get(read_options, rocksdb::Slice(key.data(), key.size()), &value);

    if (s.IsNotFound()) {
        return fail<std::string>(ErrorCode::KeyNotFound, "Key not found in RocksDB.");
    } else if (!s.ok()) {
        return fail<std::string>(ErrorCode::FilesystemError, "RocksDB Read Failed: " + s.ToString());
    }

    return ok(std::move(value));
}

Result<void> clear_rocksdb(Session &session) {
    auto state = std::static_pointer_cast<RocksDBState>(session.memory_state);
    if (!state || !state->db) return fail(ErrorCode::ProviderInitFailed, "RocksDB not initialized");

    std::string path = session.config.memory.db_path;
    delete state->db;
    state->db = nullptr;

    rocksdb::Options options;
    rocksdb::Status s_destroy = rocksdb::DestroyDB(path, options);
    if (!s_destroy.ok()) {
        // Log/handle error but try to reopen
    }
    
    options.create_if_missing = true;
    rocksdb::Status s = rocksdb::DB::Open(options, path, &state->db);
    
    if (!s.ok()) {
        return fail(ErrorCode::ProviderInitFailed, "Failed to recreate RocksDB after clear: " + s.ToString());
    }

    return ok();
}

} // namespace agent::memory
#endif
