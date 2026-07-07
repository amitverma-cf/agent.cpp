cmake_minimum_required(VERSION 3.21)
project("agent.cpp" C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
                                                                                                                                    
set(AGENT_WARNINGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(AGENT_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror=return-type -Wno-missing-field-initializers)
endif()
                                                                                                                                   
option(AGENT_ENABLE_SANITIZERS "Enable Address and Undefined Behavior Sanitizers" OFF)                                                                                    
if(AGENT_ENABLE_SANITIZERS)                                                                                                                                               
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")                                                                                                                         
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)                                                                                         
        add_link_options(-fsanitize=address,undefined)                                                                                                                    
    endif()                                                                                                                                                               
endif() 

add_library(agent
    src/core/agent.cpp
    src/core/providers/provider_ops.cpp
    src/core/providers/llama_provider.cpp
    src/core/providers/openai_compatible_provider.cpp
    src/core/conversation.cpp
    src/core/utils/utils.cpp
    src/core/memory/memory_ops.cpp
    src/core/memory/in_memory_provider.cpp
    src/core/memory/rocksdb_provider.cpp
)
target_include_directories(agent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(agent SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann
)
target_compile_options(agent PRIVATE ${AGENT_WARNINGS})

option(AGENT_ENABLE_NETWORKING "use network capabilities to support HTTP/HTTPS" ON)
option(AGENT_ENABLE_LLAMACPP "use llama.cpp (local LLM provider)" ON)
option(AGENT_ENABLE_OPENAI "use OpenAI-compatible (LLM API Provider)" ON)
option(AGENT_ENABLE_ROCKSDB "use RocksDB for LSM-Tree memory" ON)

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
target_compile_options(basic PRIVATE ${AGENT_WARNINGS})

enable_testing()
add_executable(test_agent
    tests/test_main.cpp
    tests/unit/test_providers.cpp
    tests/unit/test_memory.cpp
    tests/unit/test_utils.cpp
    tests/integration/test_session.cpp
    tests/integration/test_conversation.cpp
)
target_link_libraries(test_agent PRIVATE agent)
target_compile_options(test_agent PRIVATE ${AGENT_WARNINGS})
add_test(NAME CoreTests COMMAND test_agent)