# The `agent` command-line app. agent_cli_core holds the testable logic (argument parsing,
# config/engines.json, model/engine resolution, slash commands) and is linked by test_agent too.
add_library(agent_cli_core STATIC
    src/cli_core.cpp
    src/cli_platform.cpp
    src/home.cpp
    src/secrets.cpp
    src/mcp_config.cpp
    src/archive.cpp
    src/runtimes.cpp
    src/engine_release.cpp
    src/registries.cpp
    src/doctor.cpp
    src/status_view.cpp
    src/schedule.cpp
    src/waiting.cpp
    src/audio.cpp
    src/voice_core.cpp
)
target_include_directories(agent_cli_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agent_cli_core PUBLIC agent)
# Secrets: sha256 from the hub, AES-GCM / scrypt from OpenSSL, and the OS credential store.
target_link_libraries(agent_cli_core PRIVATE agent_hub OpenSSL::Crypto)
if(WIN32)
    target_link_libraries(agent_cli_core PRIVATE advapi32)
elseif(APPLE)
    target_link_libraries(agent_cli_core PRIVATE "-framework Security" "-framework CoreFoundation")
endif()
target_include_directories(agent_cli_core PRIVATE ${AGENT_SIMDJSON_INCLUDE_DIR})
# The CLI version string comes from project(VERSION) in the top-level CMakeLists.txt.
target_compile_definitions(agent_cli_core PUBLIC AGENT_CLI_VERSION="${PROJECT_VERSION}")
if(WIN32)
    target_link_libraries(agent_cli_core PRIVATE shell32)
endif()
target_compile_options(agent_cli_core PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

# Microphone / speaker for `agent voice` through miniaudio (vendor/miniaudio, single header). Only the
# CLI links it; libagent stays free of audio dependencies. OFF keeps just the WAV file backend.
option(AGENT_CLI_AUDIO_DEVICE "Build sound device support (miniaudio) into the agent CLI" ON)
if(AGENT_CLI_AUDIO_DEVICE)
    target_sources(agent_cli_core PRIVATE src/miniaudio_impl.c)
    target_include_directories(agent_cli_core SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/vendor/miniaudio)
    set_source_files_properties(src/miniaudio_impl.c PROPERTIES COMPILE_OPTIONS "-w")
    find_package(Threads REQUIRED)
    target_link_libraries(agent_cli_core PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    if(UNIX AND NOT APPLE)
        target_link_libraries(agent_cli_core PRIVATE m)
    endif()
    if(APPLE)
        target_link_libraries(agent_cli_core PRIVATE "-framework CoreFoundation" "-framework CoreAudio" "-framework AudioToolbox")
    endif()
else()
    target_compile_definitions(agent_cli_core PRIVATE AGENT_CLI_NO_AUDIO_DEVICE)
endif()

# The full-screen chat interface (RTXUI, MIT, vendor/RTXUI: HTML and CSS templates for the terminal). OFF keeps only the plain line-based chat.
# RTXUI needs C++23, so the interface is its own small library at that standard; the rest of the CLI stays at C++20.
option(AGENT_CLI_TUI "Build the full-screen chat interface (RTXUI) into the agent CLI" ON)
if(AGENT_CLI_TUI)
    set(RTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(RTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(RTXUI_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(RTXUI_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
    set(_agent_saved_shared "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS OFF) # linked into the program, not shipped as a second library
    add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/RTXUI ${CMAKE_BINARY_DIR}/vendor-rtxui EXCLUDE_FROM_ALL)
    set(BUILD_SHARED_LIBS "${_agent_saved_shared}")
    add_library(agent_cli_tui STATIC src/tui.cpp)
    target_include_directories(agent_cli_tui PRIVATE src)
    target_link_libraries(agent_cli_tui PUBLIC rtxui)
    set_target_properties(agent_cli_tui PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(agent_cli_core PRIVATE agent_cli_tui)
    target_compile_definitions(agent_cli_core PUBLIC AGENT_CLI_HAS_TUI)
endif()

# The library target is already called `agent`, so the executable target gets another name and
# OUTPUT_NAME agent. It lands in the top of the build tree, where it finds engine plugins under
# engines/<name>/ (installed layout: <prefix>/bin/agent + <prefix>/lib/agent/engines/).
add_executable(agent_cli_bin
    src/main.cpp
    src/commands.cpp
    src/chat.cpp
    src/voice.cpp
    src/task.cpp
    src/secret_cmd.cpp
    src/doctor_cmd.cpp
    src/mcp_cmd.cpp
    src/runtime_cmd.cpp
    src/skill_cmd.cpp
    src/sandbox_cmd.cpp
    src/schedule_cmd.cpp
    src/wake_cmd.cpp
)
set_target_properties(agent_cli_bin PROPERTIES OUTPUT_NAME agent RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(agent_cli_bin PRIVATE agent_cli_core agent_hub agent_store_sqlite)
target_include_directories(agent_cli_bin PRIVATE ${AGENT_SIMDJSON_INCLUDE_DIR})
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
