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

add_library(agent
    src/core/agent.cpp
    src/core/arena.cpp
    src/core/events.cpp
    src/core/executor/conversation_executor.cpp
    src/core/executor/fsm_executor.cpp
    src/core/providers/provider_ops.cpp
    src/core/providers/llama_provider.cpp
    src/core/providers/openai_compatible_provider.cpp
    src/core/providers/onnx_provider.cpp
    src/core/utils/utils.cpp
    src/core/memory/memory_ops.cpp
    src/core/memory/in_memory_provider.cpp
    src/core/memory/rocksdb_provider.cpp
    src/core/tools/tools_ops.cpp
    src/core/tools/basic_tools.cpp
    src/core/tools/filesystem_tools.cpp
    src/core/tools/terminal_tools.cpp
    src/core/tools/context_compressor.cpp
    vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_compile_options(agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(agent PRIVATE ${SANITIZER_FLAGS})

option(AGENT_ENABLE_NETWORKING "use network capabilities to support HTTP/HTTPS" ON)
option(AGENT_ENABLE_LLAMACPP "use llama.cpp (local LLM provider)" ON)
option(AGENT_ENABLE_OPENAI "use OpenAI-compatible (LLM API Provider)" ON)
option(AGENT_ENABLE_ROCKSDB "use RocksDB for LSM-Tree memory" ON)
option(AGENT_ENABLE_ONNX "use ONNX Runtime for multimodal inference" OFF)

if(AGENT_ENABLE_NETWORKING)
    target_include_directories(agent SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib
    )
    target_compile_definitions(agent PRIVATE AGENT_HAS_NETWORKING)

    find_package(OpenSSL REQUIRED)
    target_link_libraries(agent PRIVATE
        OpenSSL::SSL
        OpenSSL::Crypto
    )
    target_compile_definitions(agent PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)

    if(AGENT_ENABLE_OPENAI)
        target_compile_definitions(agent PRIVATE AGENT_HAS_OPENAICOMPATIBLE)
    endif()
endif()

if(AGENT_ENABLE_LLAMACPP)
    add_subdirectory(vendor/llama.cpp)
    target_link_libraries(agent PRIVATE llama)
    target_compile_definitions(agent PRIVATE AGENT_HAS_LLAMACPP)
    set_target_properties(llama PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:llama,INTERFACE_INCLUDE_DIRECTORIES>")
    if(TARGET ggml)
        set_target_properties(ggml PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>")
    endif()
endif()

if(AGENT_ENABLE_ONNX)
    # 1. Try CMake package (installed system-wide or via vcpkg/conan)
    find_package(onnxruntime QUIET)
    if(onnxruntime_FOUND)
        target_link_libraries(agent PRIVATE onnxruntime::onnxruntime)
    else()
        # 2. Fallback: bare library search (e.g. libonnxruntime.so installed to /usr/lib)
        find_library(ONNXRUNTIME_LIB NAMES onnxruntime)
        if(NOT ONNXRUNTIME_LIB)
            message(FATAL_ERROR
                "ONNX Runtime library not found. "
                "Install libonnxruntime-dev or point ONNXRUNTIME_LIB to the .so/.a path.")
        endif()
        target_link_libraries(agent PRIVATE ${ONNXRUNTIME_LIB})
    endif()
    # Use submodule headers if present (populated via sparse-checkout)
    set(ONNX_VENDOR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/onnxruntime/include")
    if(EXISTS "${ONNX_VENDOR_INCLUDE}/onnxruntime/core/session/onnxruntime_c_api.h")
        target_include_directories(agent SYSTEM PRIVATE "${ONNX_VENDOR_INCLUDE}")
    endif()
    target_compile_definitions(agent PRIVATE AGENT_HAS_ONNX)
endif()

if(AGENT_ENABLE_ROCKSDB)
    set(ROCKSDB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(WITH_TESTS OFF CACHE BOOL "" FORCE)
    set(WITH_BENCHMARK_TOOLS OFF CACHE BOOL "" FORCE)
    set(WITH_TOOLS OFF CACHE BOOL "" FORCE)
    set(WITH_CORE_TOOLS OFF CACHE BOOL "" FORCE)
    set(FAIL_ON_WARNINGS OFF CACHE BOOL "" FORCE)
    set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
    set(WITH_SNAPPY OFF CACHE BOOL "" FORCE)
    set(WITH_ZLIB OFF CACHE BOOL "" FORCE)
    set(WITH_LZ4 OFF CACHE BOOL "" FORCE)
    set(WITH_ZSTD OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)

    add_subdirectory(vendor/rocksdb)
    target_link_libraries(agent PRIVATE rocksdb)
    target_compile_definitions(agent PRIVATE AGENT_HAS_ROCKSDB)
    target_include_directories(agent SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/rocksdb/include
    )
endif()

add_executable(basic examples/basic/basic.cpp)
target_link_libraries(basic PRIVATE agent)
target_compile_options(basic PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(basic PRIVATE ${SANITIZER_FLAGS})

add_executable(agent_cli examples/agent_cli/agent_cli.cpp)
target_link_libraries(agent_cli PRIVATE agent)
target_compile_options(agent_cli PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(agent_cli PRIVATE ${SANITIZER_FLAGS})


enable_testing()
add_executable(test_agent
    tests/test_main.cpp
    tests/unit/test_providers.cpp
    tests/unit/test_memory.cpp
    tests/unit/test_utils.cpp
    tests/unit/test_onnx.cpp
    tests/integration/test_session.cpp
    tests/integration/test_conversation.cpp
)
target_link_libraries(test_agent PRIVATE agent)
target_compile_options(test_agent PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(test_agent PRIVATE ${SANITIZER_FLAGS})
add_test(NAME CoreTests COMMAND test_agent)