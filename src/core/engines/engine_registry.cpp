#include "hal/dylib.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <mutex>
#include <string>
#include <vector>

namespace agent {

namespace {

std::mutex &registry_mutex() {
    static std::mutex m;
    return m;
}

std::vector<const agent_engine_v1 *> &registry() {
    static std::vector<const agent_engine_v1 *> engines;
    return engines;
}

const agent_engine_v1 *find_locked(std::string_view name) {
    for (const agent_engine_v1 *e : registry())
        if (e->name && name == e->name) return e;
    return nullptr;
}

Result<void> validate(const agent_engine_v1 *engine) {
    if (!engine) return fail(ErrorCode::InvalidConfig, "Engine is null.");
    if (engine->abi_version != AGENT_ENGINE_ABI_VERSION)
        return fail(ErrorCode::UnsupportedProvider, "Engine ABI version " + std::to_string(engine->abi_version) +
                                                        " does not match host ABI version " + std::to_string(AGENT_ENGINE_ABI_VERSION) + ".");
    if (!engine->name || !*engine->name) return fail(ErrorCode::InvalidConfig, "Engine has no name.");
    if (!engine->create || !engine->destroy || !engine->run || !engine->free_response || !engine->last_error)
        return fail(ErrorCode::InvalidConfig, std::string("Engine '") + engine->name + "' is missing a required function.");
    return ok();
}

} // namespace

Result<void> register_engine(const agent_engine_v1 *engine) {
    if (auto v = validate(engine); !v.ok) return v;
    std::lock_guard<std::mutex> lk(registry_mutex());
    const agent_engine_v1 *existing = find_locked(engine->name);
    if (existing == engine) return ok();
    if (existing) return fail(ErrorCode::InvalidConfig, std::string("An engine named '") + engine->name + "' is already registered.");
    registry().push_back(engine);
    return ok();
}

Result<std::string> load_engine_plugin(std::string_view path) {
    std::string error;
    void *lib = hal::open_library(path, error);
    if (!lib) return fail<std::string>(ErrorCode::ProviderInitFailed, "Failed to load engine plugin '" + std::string(path) + "': " + error);

    auto get = reinterpret_cast<agent_engine_get_v1_fn>(hal::find_symbol(lib, AGENT_ENGINE_ENTRY_POINT));
    if (!get)
        return fail<std::string>(ErrorCode::ProviderInitFailed,
                                 "'" + std::string(path) + "' does not export " AGENT_ENGINE_ENTRY_POINT ".");

    const agent_engine_v1 *engine = get();
    if (auto r = register_engine(engine); !r.ok) return fail<std::string>(r.error.code, r.error.message);
    return ok(std::string(engine->name));
}

const agent_engine_v1 *find_engine(std::string_view name) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    return find_locked(name);
}

} // namespace agent
