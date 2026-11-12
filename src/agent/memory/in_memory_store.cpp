#include "memory_store.hpp"

#include <agent-cpp/agent.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

namespace agent::memory {

namespace {

struct InMemoryState {
    std::mutex mutex;
    std::unordered_map<std::string, std::string> kv;
};

Result<std::shared_ptr<void>> open_in_memory(const MemoryConfig &, std::string_view) {
    return ok<std::shared_ptr<void>>(std::make_shared<InMemoryState>());
}

Result<void> put_in_memory(void *state, std::string_view key, std::string_view value) {
    auto *s = static_cast<InMemoryState *>(state);
    std::lock_guard<std::mutex> lk(s->mutex);
    s->kv.insert_or_assign(std::string(key), std::string(value));
    return ok();
}

Result<std::string> get_in_memory(void *state, std::string_view key) {
    auto *s = static_cast<InMemoryState *>(state);
    std::lock_guard<std::mutex> lk(s->mutex);
    auto it = s->kv.find(std::string(key));
    if (it == s->kv.end()) return fail<std::string>(ErrorCode::KeyNotFound, "Key not found: " + std::string(key));
    return ok(it->second);
}

Result<void> clear_in_memory(void *state) {
    auto *s = static_cast<InMemoryState *>(state);
    std::lock_guard<std::mutex> lk(s->mutex);
    s->kv.clear();
    return ok();
}

constexpr StoreOps kInMemoryStore{.open = open_in_memory, .put = put_in_memory, .get = get_in_memory, .clear = clear_in_memory};

} // namespace

const StoreOps *in_memory_store() { return &kInMemoryStore; }

} // namespace agent::memory
