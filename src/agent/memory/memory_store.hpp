#pragma once

#include <agent-cpp/agent.hpp>

namespace agent::memory {

// Built-in StoreOps used when MemoryConfig::store is nullptr.
const StoreOps *in_memory_store();

inline const StoreOps *active_store(const Session &session) {
    return session.config.memory.store ? session.config.memory.store : in_memory_store();
}

} // namespace agent::memory
