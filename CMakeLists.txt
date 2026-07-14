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
    src/core/providers/llama_provider.cpp
    src/core/providers/openai_compatible_provider.cpp
    src/core/providers/onnx_provider.cpp
    src/core/utils/utils.cpp
    src/core/memory/memory_ops.cpp
    src/core/memory/sqlite_provider.cpp
    src/core/data/sqlite_vec_data_source.cpp
    src/core/tools/tools_ops.cpp
    src/core/tools/filesystem_tools.cpp
    src/core/tools/terminal_tools.cpp
    src/core/tools/context_compressor.cpp
    src/hal/time.cpp
    src/hal/path.cpp
    src/hal/process.cpp
    src/hal/log_file.cpp
    vendor/simdjson/singleheader/simdjson.cpp
    vendor/sqlite/sqlite3.c
    vendor/sqlite-vec/sqlite-vec.c
)
target_include_directories(agent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_include_directories(agent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(agent SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/sqlite-vec
)
target_compile_options(agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

set_source_files_properties(
    vendor/sqlite/sqlite3.c
    vendor/sqlite-vec/sqlite-vec.c
    vendor/simdjson/singleheader/simdjson.cpp
    PROPERTIES COMPILE_OPTIONS "-w"
)

target_compile_definitions(agent PRIVATE SQLITE_THREADSAFE=1 SQLITE_CORE SQLITE_VEC_STATIC)
find_package(Threads REQUIRED)
target_link_libraries(agent PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

option(AGENT_ENABLE_NETWORKING "use network capabilities to support HTTP/HTTPS" ON)
option(AGENT_ENABLE_LLAMACPP "use llama.cpp (local LLM provider)" ON)
option(AGENT_ENABLE_OPENAI "use OpenAI-compatible (LLM API Provider)" ON)
option(AGENT_ENABLE_ONNX "use ONNX Runtime for multimodal inference" OFF)

if(AGENT_ENABLE_NETWORKING)
    target_include_directories(agent SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib
    )
    target_compile_definitions(agent PUBLIC AGENT_HAS_NETWORKING)

    find_package(OpenSSL REQUIRED)
    target_link_libraries(agent PRIVATE
        OpenSSL::SSL
        OpenSSL::Crypto
    )
    target_compile_definitions(agent PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)

    if(AGENT_ENABLE_OPENAI)
        target_compile_definitions(agent PUBLIC AGENT_HAS_OPENAICOMPATIBLE)
    endif()
endif()

if(AGENT_ENABLE_LLAMACPP)
    add_subdirectory(vendor/llama.cpp)
    target_link_libraries(agent PRIVATE llama)
    target_compile_definitions(agent PUBLIC AGENT_HAS_LLAMACPP)
    set_target_properties(llama PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:llama,INTERFACE_INCLUDE_DIRECTORIES>")
    if(TARGET ggml)
        set_target_properties(ggml PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>")
    endif()
endif()

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
target_link_libraries(agent_cli PRIVATE agent)
target_compile_options(agent_cli PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(agent_cli PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

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
    tests/unit/test_scheduler_cron.cpp
    tests/unit/test_arena.cpp
    tests/unit/test_skills.cpp
    tests/unit/test_permissions.cpp
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
target_link_libraries(test_agent PRIVATE agent)
target_compile_options(test_agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
target_link_options(test_agent PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})

include(${CMAKE_CURRENT_SOURCE_DIR}/vendor/catch2/extras/Catch.cmake)
catch_discover_tests(test_agent)
