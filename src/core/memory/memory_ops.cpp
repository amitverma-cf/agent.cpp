#include "memory_ops.hpp"

#include <agent-cpp/agent.hpp>

namespace agent::memory {

Result<void> init_sqlite(Session &session);
Result<void> store_sqlite(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_sqlite(Session &session, std::string_view key);
Result<void> clear_sqlite(Session &session);

namespace {
constexpr MemoryOps kMemoryOps[] = {MemoryOps{.provider = MemoryProvider::Sqlite,
                                              .init = init_sqlite,
                                              .store = store_sqlite,
                                              .retrieve = retrieve_sqlite,
                                              .clear = clear_sqlite}};
}

const MemoryOps *find_memory_ops(MemoryProvider provider) {
    for (const auto &ops : kMemoryOps) {
        if (ops.provider == provider) {
            return &ops;
        }
    }
    return nullptr;
}

}
