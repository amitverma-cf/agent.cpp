#pragma once

#include <agent-cpp/agent.hpp>
#include <string_view>

namespace agent::memory {

using MemoryInitFn = Result<void> (*)(Session &session);

using MemoryStoreFn = Result<void> (*)(Session &session, std::string_view key, std::string_view value);

using MemoryRetrieveFn = Result<std::string> (*)(Session &session, std::string_view key);

using MemoryClearFn = Result<void> (*)(Session &session);

struct MemoryOps {
    MemoryProvider provider = MemoryProvider::Sqlite;
    MemoryInitFn init = nullptr;
    MemoryStoreFn store = nullptr;
    MemoryRetrieveFn retrieve = nullptr;
    MemoryClearFn clear = nullptr;
};

const MemoryOps *find_memory_ops(MemoryProvider provider);

}
