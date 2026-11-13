#pragma once

#include <agent-cpp/agent.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace agent::engines {

// The plugin file an engine name was loaded or listed from (load_engine_plugin, load_engine_manifest)
// and its engines.json "isolation" (Default when the manifest had none). false for engines that were
// never seen as plugins (register_engine with a linked-in table).
// The folders added with agent::engines::search_path(), in the order given.
std::vector<std::string> extra_search_dirs();

bool find_plugin_record(std::string_view name, std::string &path, EngineIsolation &isolation);

} // namespace agent::engines
