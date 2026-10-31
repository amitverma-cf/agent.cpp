#pragma once

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>

namespace agent::engines {

// The plugin file an engine name was loaded or listed from (load_engine_plugin, load_engine_manifest)
// and its engines.json "isolation" (Default when the manifest had none). false for engines that were
// never seen as plugins (register_engine with a linked-in table).
bool find_plugin_record(std::string_view name, std::string &path, EngineIsolation &isolation);

} // namespace agent::engines
