# Out-of-tree consumer of an installed agent-cpp package. Not part of the main build:
#   cmake --install build --prefix /tmp/agent-prefix
#   cmake -S tests/consumer -B build-consumer -DCMAKE_PREFIX_PATH=/tmp/agent-prefix
#   cmake --build build-consumer && ./build-consumer/consumer
cmake_minimum_required(VERSION 3.21)
project(agent_cpp_consumer CXX)

find_package(agent-cpp 0.1 REQUIRED)
message(STATUS "agent-cpp ${agent-cpp_VERSION}, engine dir: ${agent-cpp_ENGINE_DIR}")

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE agent-cpp::agent agent-cpp::store_sqlite)
