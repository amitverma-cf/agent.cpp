cmake_minimum_required(VERSION 3.21)
project("agent.cpp" C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(agent src/core/agent.cpp)
target_include_directories(agent PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nlohmann
)

option(AGENT_OPENSSL_HTTPS "use openssl to support HTTPS" ON)
if(AGENT_OPENSSL_HTTPS)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(agent PRIVATE 
        OpenSSL::SSL 
        OpenSSL::Crypto
    )
    target_compile_definitions(agent PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

add_executable(basic examples/basic/basic.cpp)

target_link_libraries(basic PRIVATE agent)
