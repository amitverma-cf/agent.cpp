# The `agent` command-line app. agent_cli_core holds the testable logic (argument parsing,
# config/engines.json, model/engine resolution, slash commands) and is linked by test_agent too.
add_library(agent_cli_core STATIC
    src/cli_core.cpp
    src/cli_platform.cpp
)
target_include_directories(agent_cli_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agent_cli_core PUBLIC agent)
if(WIN32)
    target_link_libraries(agent_cli_core PRIVATE shell32)
endif()
target_compile_options(agent_cli_core PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

# The library target is already called `agent`, so the executable target gets another name and
# OUTPUT_NAME agent. It lands in the top of the build tree, where it finds engine plugins under
# engines/<name>/ (installed layout: <prefix>/bin/agent + <prefix>/lib/agent/engines/).
add_executable(agent_cli_bin
    src/main.cpp
    src/commands.cpp
    src/chat.cpp
)
set_target_properties(agent_cli_bin PROPERTIES OUTPUT_NAME agent RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(agent_cli_bin PRIVATE agent_cli_core agent_hub agent_store_sqlite)
target_compile_options(agent_cli_bin PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent_cli_bin PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
