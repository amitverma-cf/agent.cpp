cmake_minimum_required(VERSION 3.21)
project("agent.cpp" C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(agent
    src/core/agent.cpp
    src/core/providers/llama_provider.cpp
    src/core/providers/openai_compatible_provider.cpp
)
target_include_directories(agent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann
)

option(AGENT_ENABLE_NETWORKING "use network capabilities to support HTTP/HTTPS" ON)
option(AGENT_ENABLE_LLAMACPP "use llama.cpp (local LLM provider)" ON)
option(AGENT_ENABLE_OPENAI "use OpenAI-compatible (LLM API Provider)" ON)

if(AGENT_ENABLE_NETWORKING)
    target_include_directories(agent PUBLIC
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
endif()

add_executable(basic examples/basic/basic.cpp)
target_link_libraries(basic PRIVATE agent)
