#include "memory_ops.hpp"

namespace agent::memory {

Result<void> init_in_memory(Session &session);
Result<void> store_in_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_in_memory(Session &session, std::string_view key);
Result<void> clear_in_memory(Session &session);

namespace {
constexpr MemoryOps kMemoryOps[] = {
    MemoryOps{
        .provider = MemoryProvider::InMemory,
        .init = init_in_memory,
        .store = store_in_memory,
        .retrieve = retrieve_in_memory,
        .clear = clear_in_memory
    }
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
