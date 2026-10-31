#include "hal/dylib.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <simdjson.h>
#include <sstream>
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

Result<std::vector<std::string>> load_engine_manifest(std::string_view manifest_path) {
    using R = Result<std::vector<std::string>>;
    const std::filesystem::path manifest(manifest_path);
    std::ifstream in(manifest, std::ios::binary);
    if (!in) return fail<std::vector<std::string>>(ErrorCode::FilesystemError, "Cannot open engine manifest: " + manifest.string());
    std::ostringstream text;
    text << in.rdbuf();

    simdjson::ondemand::parser parser;
    simdjson::padded_string json(text.str());
    simdjson::ondemand::document doc;
    simdjson::ondemand::array engines;
    if (parser.iterate(json).get(doc) || doc["engines"].get_array().get(engines))
        return fail<std::vector<std::string>>(ErrorCode::ParseError, manifest.string() + ": expected {\"engines\": [...]}.");

    std::vector<std::string> loaded;
    for (auto entry : engines) {
        std::string_view path_sv;
        if (entry["path"].get_string().get(path_sv) || path_sv.empty())
            return fail<std::vector<std::string>>(ErrorCode::ParseError, manifest.string() + ": every engine needs a \"path\".");
        std::filesystem::path plugin(path_sv);
        if (plugin.is_relative()) plugin = manifest.parent_path() / plugin;
        auto name = load_engine_plugin(plugin.string());
        if (!name.ok) return fail<std::vector<std::string>>(name.error.code, name.error.message);
        loaded.push_back(std::move(name.value));
    }
    return R{.ok = true, .value = std::move(loaded), .error = {}};
}

const agent_engine_v1 *find_engine(std::string_view name) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    return find_locked(name);
}

} // namespace agent
