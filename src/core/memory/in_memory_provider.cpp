#include "memory_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <mutex>
#include <unordered_map>

namespace agent::memory {

struct InMemoryState {
    std::unordered_map<std::string, std::string> db;
    std::mutex mutex;
};

Result<void> init_in_memory(Session &session) {
    session.memory_state = std::make_shared<InMemoryState>();
    return ok();
}

Result<void> store_in_memory(Session &session, std::string_view key, std::string_view value) {
    auto state = std::static_pointer_cast<InMemoryState>(session.memory_state);
    if (!state)
        return fail(ErrorCode::ProviderInitFailed, "Memory state not initialized");

    std::lock_guard<std::mutex> lock(state->mutex);
    state->db[std::string(key)] = std::string(value);
    return ok();
}

Result<std::string> retrieve_in_memory(Session &session, std::string_view key) {
    auto state = std::static_pointer_cast<InMemoryState>(session.memory_state);
    if (!state)
        return fail<std::string>(ErrorCode::ProviderInitFailed, "Memory state not initialized");

    std::lock_guard<std::mutex> lock(state->mutex);
    auto it = state->db.find(std::string(key));
    if (it == state->db.end()) {
        return fail<std::string>(ErrorCode::KeyNotFound, "Key not found");
    }
    return ok(it->second);
}

Result<void> clear_in_memory(Session &session) {
    auto state = std::static_pointer_cast<InMemoryState>(session.memory_state);
    if (!state)
        return fail(ErrorCode::ProviderInitFailed, "Memory state not initialized");

    std::lock_guard<std::mutex> lock(state->mutex);
    state->db.clear();
    return ok();
}

} // namespace agent::memory
