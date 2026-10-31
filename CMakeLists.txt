cmake_minimum_required(VERSION 3.21)
project("agent.cpp" C CXX)

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
        set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
    endif()
endif()

option(AGENT_ENABLE_COVERAGE "Instrument builds with gcov coverage flags" OFF)
set(COVERAGE_FLAGS "")
if(AGENT_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(COVERAGE_FLAGS --coverage)
    endif()
endif()

add_library(agent
    src/core/agent.cpp
    src/core/arena.cpp
    src/core/events.cpp
    src/core/executor/turn_executor.cpp
    src/core/executor/flow_executor.cpp
    src/core/executor/ambient_turn.cpp
    src/core/scheduler/scheduler.cpp
    src/core/data/data_source_registry.cpp
    src/core/skills/skills_registry.cpp
    src/core/providers/provider_ops.cpp
    src/core/providers/engine_provider.cpp
    src/core/engines/engine_registry.cpp
    src/core/providers/onnx_provider.cpp
    src/core/utils/utils.cpp
    src/core/memory/in_memory_store.cpp
    src/core/tools/tools_ops.cpp
    src/core/tools/filesystem_tools.cpp
    src/core/tools/terminal_tools.cpp
    src/core/tools/context_compressor.cpp
    src/hal/time.cpp
    src/hal/path.cpp
    src/hal/process.cpp
    src/hal/log_file.cpp
    src/hal/dylib.cpp
    vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_include_directories(agent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

set_source_files_properties(
    vendor/simdjson/singleheader/simdjson.cpp
    PROPERTIES COMPILE_OPTIONS "-w"
)

find_package(Threads REQUIRED)
target_link_libraries(agent PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

# Optional SQLite storage (KV store + sqlite-vec data source). Used by the CLI; library users opt in
# by linking agent_store_sqlite. libagent itself has no SQLite dependency.
add_library(agent_store_sqlite
    tools/store-sqlite/src/sqlite_store.cpp
    tools/store-sqlite/src/sqlite_vec_data_source.cpp
    vendor/sqlite/sqlite3.c
    vendor/sqlite-vec/sqlite-vec.c
)
target_include_directories(agent_store_sqlite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools/store-sqlite/include)
target_include_directories(agent_store_sqlite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(agent_store_sqlite SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite-vec
)
set_source_files_properties(
    vendor/sqlite/sqlite3.c
    vendor/sqlite-vec/sqlite-vec.c
    PROPERTIES COMPILE_OPTIONS "-w"
)
target_compile_definitions(agent_store_sqlite PRIVATE SQLITE_THREADSAFE=1 SQLITE_CORE SQLITE_VEC_STATIC)
target_link_libraries(agent_store_sqlite PUBLIC agent PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(agent_store_sqlite PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent_store_sqlite PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

# Optional Hugging Face model hub (downloads + models.json catalogue). Used by the CLI only;
# libagent has no network dependency. Needs OpenSSL for HTTPS and sha256.
option(AGENT_BUILD_HUB "Build the agent_hub model downloader (requires OpenSSL)" ON)
if(AGENT_BUILD_HUB)
find_package(OpenSSL REQUIRED)
add_library(agent_hub
    tools/hub/src/hub.cpp
    tools/hub/src/hub_http.cpp
)
target_include_directories(agent_hub PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools/hub/include)
target_include_directories(agent_hub SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib)
target_compile_definitions(agent_hub PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT CPPHTTPLIB_NO_EXCEPTIONS)
target_link_libraries(agent_hub PUBLIC agent PRIVATE OpenSSL::SSL OpenSSL::Crypto Threads::Threads)
if(WIN32)
    target_link_libraries(agent_hub PRIVATE ws2_32 crypt32)
endif()
target_compile_options(agent_hub PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent_hub PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
endif()

option(AGENT_ENABLE_ONNX "use ONNX Runtime for multimodal inference" OFF)
option(AGENT_ENGINE_LLAMA "Build the llama.cpp engine plugin (engines/llama)" ON)

if(AGENT_ENABLE_ONNX)
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        target_link_libraries(agent PRIVATE onnxruntime::onnxruntime)
    else()

        find_library(ONNXRUNTIME_LIB NAMES onnxruntime)
        if(NOT ONNXRUNTIME_LIB)
            message(FATAL_ERROR
                "ONNX Runtime library not found. "
                "Install libonnxruntime-dev or point ONNXRUNTIME_LIB to the .so/.a path.")
        endif()
        target_link_libraries(agent PRIVATE ${ONNXRUNTIME_LIB})
    endif()

    set(ONNX_VENDOR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/onnxruntime/include")
    if(EXISTS "${ONNX_VENDOR_INCLUDE}/onnxruntime/core/session/onnxruntime_c_api.h")
        target_include_directories(agent SYSTEM PRIVATE "${ONNX_VENDOR_INCLUDE}")
    endif()
    target_compile_definitions(agent PUBLIC AGENT_HAS_ONNX)
endif()

add_executable(basic examples/basic/basic.cpp)
target_link_libraries(basic PRIVATE agent)
target_compile_options(basic PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(basic PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

add_executable(agent_cli examples/agent_cli/agent_cli.cpp)
target_link_libraries(agent_cli PRIVATE agent agent_store_sqlite)
target_compile_options(agent_cli PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent_cli PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

if(AGENT_BUILD_HUB)
    add_executable(agent_pull examples/agent_pull/agent_pull.cpp)
    target_link_libraries(agent_pull PRIVATE agent_hub)
    target_compile_options(agent_pull PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
    target_link_options(agent_pull PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
endif()

add_subdirectory(engines/mock)
if(AGENT_ENGINE_LLAMA)
    add_subdirectory(engines/llama)
    target_compile_definitions(agent_cli PRIVATE AGENT_CLI_DEFAULT_ENGINE="$<TARGET_FILE:agent-engine-llama>")
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
    tests/unit/test_tools_ops.cpp
    tests/unit/test_events.cpp
    tests/unit/test_scheduler.cpp
    tests/unit/test_arena.cpp
    tests/unit/test_skills.cpp
    tests/unit/test_permissions.cpp
    tests/unit/test_engines.cpp
    tests/integration/test_session.cpp
    tests/integration/test_flow.cpp
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
    target_sources(test_agent PRIVATE tests/unit/test_hub.cpp)
    target_link_libraries(test_agent PRIVATE agent_hub)
endif()
target_compile_definitions(test_agent PRIVATE AGENT_TEST_MOCK_ENGINE_PATH="$<TARGET_FILE:agent-engine-mock>")
add_dependencies(test_agent agent-engine-mock)
if(AGENT_ENGINE_LLAMA)
    target_compile_definitions(test_agent PRIVATE AGENT_TEST_LLAMA_ENGINE_PATH="$<TARGET_FILE:agent-engine-llama>")
    add_dependencies(test_agent agent-engine-llama)
endif()
target_compile_options(test_agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(test_agent PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

include(${CMAKE_CURRENT_SOURCE_DIR}/vendor/catch2/extras/Catch.cmake)
catch_discover_tests(test_agent)
