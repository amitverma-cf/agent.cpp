cmake_minimum_required(VERSION 3.21)
# The single version source: the CLI version string, the package config version and the release
# tarball names all come from here.
project("agent.cpp" VERSION 0.1.2 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(AGENT_WARNINGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(AGENT_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror=return-type -Wno-missing-field-initializers)
endif()
option(AGENT_ENABLE_SANITIZERS "Enable Address and Undefined Behavior Sanitizers" OFF)
set(SANITIZER_FLAGS "")
if(AGENT_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
    endif()
endif()

# libagent, agent_store_sqlite and agent_hub follow BUILD_SHARED_LIBS (default: static). Engine
# plugins are always shared modules and the CLI's internal agent_cli_core is always static.
option(BUILD_SHARED_LIBS "Build libagent, agent_store_sqlite and agent_hub as shared libraries" OFF)
option(AGENT_INSTALL "Generate install rules and the agent-cpp CMake package" ${PROJECT_IS_TOP_LEVEL})
# Engine plugins install here; the CLI looks in <exe>/../lib/agent/engines, so this is fixed.
set(AGENT_INSTALL_ENGINEDIR "lib/agent/engines")
# simdjson is an implementation detail: it is compiled into libagent but not part of the public
# headers, so it is a private include path of each target that parses JSON.
set(AGENT_SIMDJSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/simdjson/singleheader)

# Installs an engine plugin into lib/agent/engines/, plus the runtime libraries it loads through
# its $ORIGIN RPATH (ONNX Runtime, sherpa-onnx, ...). A symlink whose target is in the list too is
# installed as a symlink; any other symlink is replaced by a copy of its target under the link's
# name, so the shipped directory has no dangling links.
function(agent_install_engine target)
    if(NOT AGENT_INSTALL)
        return()
    endif()
    install(TARGETS ${target}
        LIBRARY DESTINATION ${AGENT_INSTALL_ENGINEDIR}
        RUNTIME DESTINATION ${AGENT_INSTALL_ENGINEDIR}
    )
    foreach(_lib ${ARGN})
        get_filename_component(_name "${_lib}" NAME)
        file(REAL_PATH "${_lib}" _real)
        get_filename_component(_real_name "${_real}" NAME)
        get_filename_component(_dir "${_lib}" DIRECTORY)
        if(IS_SYMLINK "${_lib}" AND "${_dir}/${_real_name}" IN_LIST ARGN)
            install(FILES "${_lib}" DESTINATION ${AGENT_INSTALL_ENGINEDIR})
        else()
            install(FILES "${_real}" DESTINATION ${AGENT_INSTALL_ENGINEDIR} RENAME "${_name}")
        endif()
    endforeach()
endfunction()

if(BUILD_SHARED_LIBS AND WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

option(AGENT_ENABLE_COVERAGE "Instrument builds with gcov coverage flags" OFF)
set(COVERAGE_FLAGS "")
if(AGENT_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(COVERAGE_FLAGS --coverage)
    endif()
endif()

add_library(agent
    src/agent/agent.cpp
    src/agent/arena.cpp
    src/agent/events.cpp
    src/agent/executor/turn_executor.cpp
    src/agent/executor/ambient_turn.cpp
    src/agent/graph/graph_builders.cpp
    src/agent/graph/graph_checkpoint.cpp
    src/agent/graph/graph_checkpointer.cpp
    src/agent/graph/graph_executor.cpp
    src/agent/graph/graph_json.cpp
    src/agent/mcp/mcp_client.cpp
    src/agent/models/model_registry.cpp
    src/agent/data/data_source_registry.cpp
    src/agent/skills/skills_registry.cpp
    src/agent/providers/provider_ops.cpp
    src/agent/providers/engine_provider.cpp
    src/agent/engines/engine_registry.cpp
    src/agent/engines/engine_remote.cpp
    src/agent/engines/engine_shm.cpp
    src/agent/utils/utils.cpp
    src/agent/memory/in_memory_store.cpp
    src/agent/tools/tools_ops.cpp
    src/agent/tools/filesystem_tools.cpp
    src/agent/tools/terminal_tools.cpp
    src/agent/tools/path_gate.cpp
    src/agent/tools/context_compressor.cpp
    src/agent/tools/lazy_tools.cpp
    src/agent/rag/engine_tasks.cpp
    src/agent/rag/in_memory_vector_store.cpp
    src/agent/rag/chunker.cpp
    src/agent/rag/rag_pipeline.cpp
    src/agent/tasks/decide.cpp
    src/agent/tasks/ocr.cpp
    src/agent/sandbox/sandbox_policy.cpp
    src/hal/time.cpp
    src/hal/path.cpp
    src/hal/process.cpp
    src/hal/child_process.cpp
    src/hal/log_file.cpp
    src/hal/dylib.cpp
    src/hal/sandbox.cpp
    vendor/simdjson/singleheader/simdjson.cpp
)
add_library(agent-cpp::agent ALIAS agent)
# Header-only plugin ABI for engine plugins built against the installed package
# (find_package(agent-cpp) -> agent-cpp::engine_abi, agent_add_engine(); see examples/08-engine-plugin-template/).
add_library(agent_engine_abi INTERFACE)
add_library(agent-cpp::engine_abi ALIAS agent_engine_abi)
target_include_directories(agent_engine_abi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(agent PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(agent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${AGENT_SIMDJSON_INCLUDE_DIR}
)
target_compile_features(agent PUBLIC cxx_std_20)
target_compile_options(agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

set_source_files_properties(
    vendor/simdjson/singleheader/simdjson.cpp
    PROPERTIES COMPILE_OPTIONS "-w"
)

find_package(Threads REQUIRED)
target_link_libraries(agent PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(agent PRIVATE advapi32 userenv winmm) # token / ACL calls of the Windows sandbox
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # shm_open / shm_unlink (engine IPC shared memory); part of libc since glibc 2.34.
    target_link_libraries(agent PRIVATE rt)
endif()

# Optional SQLite storage (KV store + sqlite-vec data source). Used by the CLI; library users opt in
# by linking agent_store_sqlite. libagent itself has no SQLite dependency.
add_library(agent_store_sqlite
    apps/store-sqlite/src/sqlite_checkpointer.cpp
    apps/store-sqlite/src/sqlite_store.cpp
    apps/store-sqlite/src/sqlite_vec_data_source.cpp
    vendor/sqlite/sqlite3.c
    vendor/sqlite-vec/sqlite-vec.c
)
add_library(agent-cpp::store_sqlite ALIAS agent_store_sqlite)
target_include_directories(agent_store_sqlite PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/apps/store-sqlite/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(agent_store_sqlite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${AGENT_SIMDJSON_INCLUDE_DIR})
target_include_directories(agent_store_sqlite SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite-vec
)
set_source_files_properties(
    vendor/sqlite/sqlite3.c
    vendor/sqlite-vec/sqlite-vec.c
    PROPERTIES COMPILE_OPTIONS "-w"
)
target_compile_definitions(agent_store_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_CORE SQLITE_VEC_STATIC SQLITE_ENABLE_FTS5)
target_link_libraries(agent_store_sqlite PUBLIC agent PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(agent_store_sqlite PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent_store_sqlite PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

# Optional Hugging Face model hub (downloads + models.json catalogue). Used by the CLI only;
# libagent has no network dependency. Needs OpenSSL for HTTPS and sha256.
option(AGENT_BUILD_HUB "Build the agent_hub model downloader (requires OpenSSL)" ON)
if(AGENT_BUILD_HUB)
find_package(OpenSSL REQUIRED)
add_library(agent_hub
    apps/hub/src/hub.cpp
    apps/hub/src/hub_http.cpp
    apps/hub/src/registry.cpp
)
add_library(agent-cpp::hub ALIAS agent_hub)
target_include_directories(agent_hub PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/apps/hub/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(agent_hub PRIVATE ${AGENT_SIMDJSON_INCLUDE_DIR})
target_include_directories(agent_hub SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib)
target_compile_definitions(agent_hub PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT CPPHTTPLIB_NO_EXCEPTIONS)
target_link_libraries(agent_hub PUBLIC agent PRIVATE OpenSSL::SSL OpenSSL::Crypto Threads::Threads)
if(APPLE)
    # httplib reads the system certificate store through these (CPPHTTPLIB_OPENSSL_SUPPORT on macOS)
    target_link_libraries(agent_hub PUBLIC "-framework CoreFoundation" "-framework Security")
endif()
if(WIN32)
    target_link_libraries(agent_hub PRIVATE ws2_32 crypt32)
endif()
target_compile_options(agent_hub PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent_hub PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
endif()

# The `agent` CLI binary (apps/cli): setup wizard, engine/model manager, chat TUI.
option(AGENT_BUILD_CLI "Build the agent CLI binary (apps/cli; requires AGENT_BUILD_HUB)" ON)
if(AGENT_BUILD_CLI AND NOT AGENT_BUILD_HUB)
    message(WARNING "AGENT_BUILD_CLI needs AGENT_BUILD_HUB; the agent CLI will not be built")
    set(AGENT_BUILD_CLI OFF)
endif()
if(AGENT_BUILD_CLI)
    add_subdirectory(apps/cli)
endif()

option(AGENT_ENGINE_LLAMA "Build the llama.cpp engine plugin (src/engines/llama)" ON)
option(AGENT_ENGINE_ONNX "Build the ONNX Runtime engine plugin (src/engines/onnx; needs a prebuilt ONNX Runtime, see ONNXRUNTIME_ROOT)" OFF)
# Heavy (builds stable-diffusion.cpp + its own ggml), so off by default for now.
option(AGENT_ENGINE_SDCPP "Build the stable-diffusion.cpp image generation engine plugin (src/engines/sdcpp)" OFF)
# Heavy (builds trellis2.cpp + its own ggml), so off by default.
option(AGENT_ENGINE_TRELLIS "Build the TRELLIS.2 image-to-3D engine plugin (src/engines/trellis)" OFF)
# audio.cpp (Apache-2.0): TTS, ASR, VAD, diarization and more from one library; builds its own ggml, so off by default.
option(AGENT_ENGINE_AUDIOCPP "Build the audio.cpp speech engine plugin (src/engines/audiocpp)" OFF)
# ggml's AMX kernels break when another in-process engine touches the AMX tile state; see
# cmake/ggml_no_amx.cmake. Only matters for GGML_NATIVE builds on AMX CPUs.
option(AGENT_GGML_AMX "Keep Intel AMX in the engines' ggml (safe only with process-isolated engines)" OFF)

add_executable(basic examples/01-hello-agent/basic.cpp)
target_link_libraries(basic PRIVATE agent)
target_compile_options(basic PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(basic PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

if(AGENT_BUILD_HUB)
    add_executable(agent_pull examples/07-model-pull/agent_pull.cpp)
    target_link_libraries(agent_pull PRIVATE agent_hub)
    target_compile_options(agent_pull PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
    target_link_options(agent_pull PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
endif()

# Out-of-process engine host (EngineIsolation::Process). Lands next to the agent CLI and the tests,
# where the core looks for it.
add_executable(agent-engine-host apps/engine-host/engine_host.cpp)
target_include_directories(agent-engine-host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agent-engine-host PRIVATE agent Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(agent-engine-host PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(agent-engine-host PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent-engine-host PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
if(AGENT_INSTALL)
    # agent / libagent look for the host next to the running executable, so it goes beside the CLI.
    if(BUILD_SHARED_LIBS AND NOT APPLE)
        set_target_properties(agent-engine-host PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    elseif(BUILD_SHARED_LIBS)
        set_target_properties(agent-engine-host PROPERTIES INSTALL_RPATH "@executable_path/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(TARGETS agent-engine-host RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_subdirectory(src/engines/mock)
if(AGENT_ENGINE_LLAMA)
    add_subdirectory(src/engines/llama)
endif()
if(AGENT_ENGINE_ONNX)
    add_subdirectory(src/engines/onnx)
endif()
if(AGENT_ENGINE_SDCPP)
    add_subdirectory(src/engines/sdcpp)
endif()
if(AGENT_ENGINE_TRELLIS)
    add_subdirectory(src/engines/trellis)
endif()
if(AGENT_ENGINE_AUDIOCPP)
    add_subdirectory(src/engines/audiocpp)
endif()

enable_testing()
add_executable(test_agent
    vendor/catch2/catch_amalgamated.cpp
    tests/test_main.cpp
    tests/unit/test_providers.cpp
    tests/unit/test_memory.cpp
    tests/unit/test_utils.cpp
    tests/unit/test_onnx.cpp
    tests/unit/test_data_sources.cpp
    tests/unit/test_sqlite_vec.cpp
    tests/unit/test_filesystem_tools.cpp
    tests/unit/test_terminal_tools.cpp
    tests/unit/test_hal_path.cpp
    tests/unit/test_context_compressor.cpp
    tests/unit/test_lazy_tools.cpp
    tests/unit/test_tools_ops.cpp
    tests/unit/test_tool_parts.cpp
    tests/unit/test_builtin_tools_graph.cpp
    tests/unit/test_events.cpp
    tests/unit/test_arena.cpp
    tests/unit/test_skills.cpp
    tests/unit/test_permissions.cpp
    tests/unit/test_engines.cpp
    tests/unit/test_mock_all.cpp
    tests/unit/test_engine_host.cpp
    tests/unit/test_engine_sdcpp.cpp
    tests/unit/test_graph.cpp
    tests/unit/test_rag.cpp
    tests/unit/test_decide.cpp
    tests/unit/test_ocr.cpp
    tests/unit/test_graph_json.cpp
    tests/unit/test_graph_durability.cpp
    tests/unit/test_mcp.cpp
    tests/unit/test_sandbox.cpp
    tests/unit/test_engine_sandbox.cpp
    tests/unit/test_model_registry.cpp
    tests/integration/test_session.cpp
    tests/integration/test_turn.cpp
)
target_include_directories(test_agent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/catch2
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_source_files_properties(
    vendor/catch2/catch_amalgamated.cpp
    PROPERTIES COMPILE_OPTIONS "-w"
               COMPILE_DEFINITIONS "CATCH_AMALGAMATED_CUSTOM_MAIN"
)
target_link_libraries(test_agent PRIVATE agent agent_store_sqlite)
if(AGENT_BUILD_HUB)
    target_sources(test_agent PRIVATE tests/unit/test_hub.cpp tests/unit/test_hub_http.cpp)
    target_link_libraries(test_agent PRIVATE agent_hub OpenSSL::SSL OpenSSL::Crypto)
    # test_hub_http.cpp runs a loopback server with the same httplib settings as agent_hub
    set_source_files_properties(tests/unit/test_hub_http.cpp tests/unit/test_cli_net.cpp PROPERTIES COMPILE_DEFINITIONS "CPPHTTPLIB_OPENSSL_SUPPORT;CPPHTTPLIB_NO_EXCEPTIONS")
    target_include_directories(test_agent SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib)
    if(WIN32)
        target_link_libraries(test_agent PRIVATE ws2_32 crypt32)
    endif()
endif()
if(AGENT_BUILD_CLI)
    target_sources(test_agent PRIVATE tests/unit/test_cli.cpp tests/unit/test_cli_status.cpp tests/unit/test_voice.cpp tests/unit/test_home.cpp tests/unit/test_secrets.cpp tests/unit/test_doctor.cpp tests/unit/test_mcp_config.cpp tests/unit/test_archive.cpp tests/unit/test_runtimes.cpp tests/unit/test_registries.cpp tests/unit/test_engine_release.cpp tests/unit/test_sandbox_windows.cpp tests/unit/test_path_gate.cpp tests/unit/test_cli_e2e.cpp tests/unit/test_cli_net.cpp)
    target_link_libraries(test_agent PRIVATE agent_cli_core)
    # The agent voice end-to-end test runs the real binary.
    target_compile_definitions(test_agent PRIVATE AGENT_TEST_CLI_PATH="$<TARGET_FILE:agent_cli_bin>")
    add_dependencies(test_agent agent_cli_bin)
endif()
target_compile_definitions(test_agent PRIVATE AGENT_TEST_MOCK_ENGINE_PATH="$<TARGET_FILE:agent-engine-mock>"
                                              AGENT_TEST_MOCK_ALL_ENGINE_PATH="$<TARGET_FILE:agent-engine-mock-all>"
                                              AGENT_TEST_MOCK_CONCURRENT_ENGINE_PATH="$<TARGET_FILE:agent-engine-mock-concurrent>"
                                              AGENT_TEST_ENGINE_HOST_PATH="$<TARGET_FILE:agent-engine-host>")
add_dependencies(test_agent agent-engine-mock agent-engine-mock-concurrent agent-engine-mock-all agent-engine-host)

# Tiny MCP stdio server the MCP client tests talk to.
add_executable(fake_mcp_server tests/helpers/fake_mcp_server.cpp)
target_link_libraries(fake_mcp_server PRIVATE agent)
target_include_directories(fake_mcp_server PRIVATE ${AGENT_SIMDJSON_INCLUDE_DIR})
target_compile_options(fake_mcp_server PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(fake_mcp_server PRIVATE ${SANITIZER_FLAGS})
target_compile_definitions(test_agent PRIVATE AGENT_TEST_FAKE_MCP_SERVER_PATH="$<TARGET_FILE:fake_mcp_server>"
                                              AGENT_TEST_EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples/03-graph-workflow/graphs")
add_dependencies(test_agent fake_mcp_server)
if(AGENT_ENGINE_LLAMA)
    target_compile_definitions(test_agent PRIVATE AGENT_TEST_LLAMA_ENGINE_PATH="$<TARGET_FILE:agent-engine-llama>")
    add_dependencies(test_agent agent-engine-llama)
endif()
if(AGENT_ENGINE_ONNX)
    target_compile_definitions(test_agent PRIVATE AGENT_TEST_ONNX_ENGINE_PATH="$<TARGET_FILE:agent-engine-onnx>")
    add_dependencies(test_agent agent-engine-onnx)
endif()
if(AGENT_ENGINE_SDCPP)
    target_compile_definitions(test_agent PRIVATE AGENT_TEST_SDCPP_ENGINE_PATH="$<TARGET_FILE:agent-engine-sdcpp>")
    add_dependencies(test_agent agent-engine-sdcpp)
endif()
if(AGENT_ENGINE_TRELLIS)
    target_sources(test_agent PRIVATE tests/unit/test_engine_trellis.cpp)
    target_compile_definitions(test_agent PRIVATE AGENT_TEST_TRELLIS_ENGINE_PATH="$<TARGET_FILE:agent-engine-trellis>")
    add_dependencies(test_agent agent-engine-trellis)
endif()
if(AGENT_ENGINE_AUDIOCPP)
    target_compile_definitions(test_agent PRIVATE AGENT_TEST_AUDIOCPP_ENGINE_PATH="$<TARGET_FILE:agent-engine-audiocpp>")
    add_dependencies(test_agent agent-engine-audiocpp)
endif()
target_compile_options(test_agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
if(MSVC)
    # Catch2's CHECK(...) stringifies its argument; the traditional MSVC preprocessor mangles raw string
    # literals that contain quotes (C2017 "illegal escape sequence"). The conforming one does not.
    target_compile_options(test_agent PRIVATE /Zc:preprocessor)
endif()
target_link_options(test_agent PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

include(${CMAKE_CURRENT_SOURCE_DIR}/vendor/catch2/extras/Catch.cmake)
catch_discover_tests(test_agent ADD_TAGS_AS_LABELS)

# Example programs and the code of docs/guides (smoke tests on the mock engine: ctest -L examples).
option(AGENT_BUILD_EXAMPLES "Build examples/ (guide code, image critique, RAG over docs) and their smoke tests" ON)
if(AGENT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
# Benchmark harness (bench/agent_bench, see docs/benchmarks.md).
option(AGENT_BUILD_BENCH "Build the agent_bench benchmark harness (bench/)" OFF)
if(AGENT_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# ----------------------------------------------------------------------------------------------
# Install / export: find_package(agent-cpp) gives agent-cpp::agent, agent-cpp::store_sqlite and
# (when built) agent-cpp::hub. The CLI goes to bin/, engine plugins to lib/agent/engines/ (see
# docs/packaging.md).
if(AGENT_INSTALL)
    set(AGENT_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/agent-cpp)
    set_target_properties(agent PROPERTIES EXPORT_NAME agent VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
    set_target_properties(agent_store_sqlite PROPERTIES EXPORT_NAME store_sqlite VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
    set_target_properties(agent_engine_abi PROPERTIES EXPORT_NAME engine_abi)
    set(_agent_export_targets agent agent_store_sqlite)
    if(AGENT_BUILD_HUB)
        set_target_properties(agent_hub PROPERTIES EXPORT_NAME hub VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
        list(APPEND _agent_export_targets agent_hub)
    endif()
    if(BUILD_SHARED_LIBS AND NOT APPLE)
        # agent_store_sqlite / agent_hub find libagent next to themselves.
        set_target_properties(${_agent_export_targets} PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
    install(TARGETS agent_engine_abi EXPORT agent-cppTargets)
    install(TARGETS ${_agent_export_targets}
        EXPORT agent-cppTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES
        include/agent-cpp/agent.hpp
        include/agent-cpp/engine_abi.h
        apps/store-sqlite/include/agent-cpp/store_sqlite.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/agent-cpp
    )
    if(AGENT_BUILD_HUB)
        install(FILES apps/hub/include/agent-cpp/hub.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/agent-cpp)
    endif()
    install(EXPORT agent-cppTargets NAMESPACE agent-cpp:: DESTINATION ${AGENT_CMAKE_INSTALL_DIR})
    install(FILES README.md LICENSE docs/cli.md DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/agent)

    set(AGENT_PACKAGE_HAS_HUB ${AGENT_BUILD_HUB})
    set(AGENT_PACKAGE_SHARED ${BUILD_SHARED_LIBS})
    configure_package_config_file(cmake/agent-cppConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/agent-cppConfig.cmake
        INSTALL_DESTINATION ${AGENT_CMAKE_INSTALL_DIR}
        PATH_VARS AGENT_INSTALL_ENGINEDIR CMAKE_INSTALL_BINDIR
    )
    # Pre-1.0: minor versions may break the API.
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/agent-cppConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/agent-cppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/agent-cppConfigVersion.cmake
        cmake/agent-cppEngine.cmake
        DESTINATION ${AGENT_CMAKE_INSTALL_DIR}
    )
endif()
