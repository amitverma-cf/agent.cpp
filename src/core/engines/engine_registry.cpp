#include "engine_registry.hpp"

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

// Where plugins came from, by engine name (load_engine_plugin / load_engine_manifest), and the
// engines.json isolation preference. Process-isolated manifest entries are recorded here without
// being loaded into this process.
struct PluginRecord {
    std::string name;
    std::string path;
    EngineIsolation isolation = EngineIsolation::Default;
};

std::vector<PluginRecord> &plugin_records() {
    static std::vector<PluginRecord> records;
    return records;
}

void record_plugin(std::string_view name, std::string_view path, EngineIsolation isolation) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    for (PluginRecord &r : plugin_records())
        if (r.name == name) {
            r.path = std::string(path);
            if (isolation != EngineIsolation::Default) r.isolation = isolation;
            return;
        }
    plugin_records().push_back(PluginRecord{std::string(name), std::string(path), isolation});
}

// "agent-engine-llama.so" -> "llama"; other file names -> their stem.
std::string name_from_plugin_path(const std::filesystem::path &path) {
    std::string stem = path.stem().string();
    constexpr std::string_view prefix = "agent-engine-";
    return stem.rfind(prefix, 0) == 0 ? stem.substr(prefix.size()) : stem;
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
    record_plugin(engine->name, path, EngineIsolation::Default);
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

        // Optional "isolation": "process" | "inprocess" (see Config::engine_isolation).
        EngineIsolation isolation = EngineIsolation::Default;
        std::string_view isolation_sv;
        if (!entry["isolation"].get_string().get(isolation_sv)) {
            if (isolation_sv == "process") isolation = EngineIsolation::Process;
            else if (isolation_sv == "inprocess") isolation = EngineIsolation::InProcess;
            else
                return fail<std::vector<std::string>>(ErrorCode::ParseError,
                                                      manifest.string() + ": \"isolation\" must be \"process\" or \"inprocess\", got \"" +
                                                          std::string(isolation_sv) + "\".");
        }

        if (isolation == EngineIsolation::Process) {
            // Not loaded into this process: agent-engine-host loads it when a session uses it. The
            // name comes from "name", else from the file name (agent-engine-<name>.<ext>).
            std::error_code ec;
            if (!std::filesystem::is_regular_file(plugin, ec))
                return fail<std::vector<std::string>>(ErrorCode::FilesystemError,
                                                      manifest.string() + ": engine plugin not found: " + plugin.string());
            std::string_view name_sv;
            std::string name =
                (!entry["name"].get_string().get(name_sv) && !name_sv.empty()) ? std::string(name_sv) : name_from_plugin_path(plugin);
            record_plugin(name, plugin.string(), isolation);
            loaded.push_back(std::move(name));
            continue;
        }

        auto name = load_engine_plugin(plugin.string());
        if (!name.ok) return fail<std::vector<std::string>>(name.error.code, name.error.message);
        if (isolation != EngineIsolation::Default) record_plugin(name.value, plugin.string(), isolation);
        loaded.push_back(std::move(name.value));
    }
    return R{.ok = true, .value = std::move(loaded), .error = {}};
}

const agent_engine_v1 *find_engine(std::string_view name) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    return find_locked(name);
}

namespace engines {

bool find_plugin_record(std::string_view name, std::string &path, EngineIsolation &isolation) {
    std::lock_guard<std::mutex> lk(registry_mutex());
    for (const PluginRecord &r : plugin_records())
        if (r.name == name) {
            path = r.path;
            isolation = r.isolation;
            return true;
        }
    return false;
}

} // namespace engines

} // namespace agent
