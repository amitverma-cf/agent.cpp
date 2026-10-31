# The `agent` command-line app. agent_cli_core holds the testable logic (argument parsing,
# config/engines.json, model/engine resolution, slash commands) and is linked by test_agent too.
add_library(agent_cli_core STATIC
    src/cli_core.cpp
    src/cli_platform.cpp
)
target_include_directories(agent_cli_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agent_cli_core PUBLIC agent)
target_include_directories(agent_cli_core PRIVATE ${AGENT_SIMDJSON_INCLUDE_DIR})
# The CLI version string comes from project(VERSION) in the top-level CMakeLists.txt.
target_compile_definitions(agent_cli_core PUBLIC AGENT_CLI_VERSION="${PROJECT_VERSION}")
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
# The engine host runs engines out of process (the default for `agent chat` / `agent run`).
add_dependencies(agent_cli_bin agent-engine-host)

if(AGENT_INSTALL)
    # A static libagent leaves the binary with no RPATH at all; a shared one is found in ../lib.
    if(BUILD_SHARED_LIBS)
        if(APPLE)
            set_target_properties(agent_cli_bin PROPERTIES INSTALL_RPATH "@executable_path/../${CMAKE_INSTALL_LIBDIR}")
        else()
            set_target_properties(agent_cli_bin PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        endif()
    endif()
    install(TARGETS agent_cli_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
