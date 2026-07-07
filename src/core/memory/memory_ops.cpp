#include "memory_ops.hpp"

namespace agent::memory {

Result<void> init_in_memory(Session &session);
Result<void> store_in_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_in_memory(Session &session, std::string_view key);
Result<void> clear_in_memory(Session &session);

#ifdef AGENT_HAS_ROCKSDB
Result<void> init_rocksdb(Session &session);
Result<void> store_rocksdb(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_rocksdb(Session &session, std::string_view key);
Result<void> clear_rocksdb(Session &session);
#endif

namespace {
constexpr MemoryOps kMemoryOps[] = {
    MemoryOps{
        .provider = MemoryProvider::InMemory,
        .init = init_in_memory,
        .store = store_in_memory,
        .retrieve = retrieve_in_memory,
        .clear = clear_in_memory
    },
#ifdef AGENT_HAS_ROCKSDB
    MemoryOps{
        .provider = MemoryProvider::RocksDB,
        .init = init_rocksdb,
        .store = store_rocksdb,
        .retrieve = retrieve_rocksdb,
        .clear = clear_rocksdb
    }
#endif
};
} // namespace

const MemoryOps *find_memory_ops(MemoryProvider provider) {
    for (const auto &ops : kMemoryOps) {
        if (ops.provider == provider) {
            return &ops;
        }
    }
    return nullptr;
}

} // namespace agent::memory
