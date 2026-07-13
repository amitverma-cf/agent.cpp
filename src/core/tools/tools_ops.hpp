#pragma once

#include <agent-cpp/agent.hpp>
#include <string_view>

namespace agent::tools {

using ToolsInitFn = Result<void> (*)(Session &session);

using ToolsExecuteFn = Result<std::string> (*)(Session &session, std::string_view name, std::string_view arguments);

struct ToolsOps {
    ToolsInitFn init = nullptr;
    ToolsExecuteFn execute = nullptr;
};

const ToolsOps *find_tools_ops();

}
