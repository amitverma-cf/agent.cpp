# Standalone engine plugin project, built against an INSTALLED agent.cpp:
#
#   cmake --install <agent.cpp build> --prefix /opt/agent
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/agent
#   cmake --build build && ctest --test-dir build --output-on-failure
#   cmake --install build --prefix /opt/agent        # -> /opt/agent/lib/agent/engines/agent-engine-echo.so
#   /opt/agent/bin/agent engine add /opt/agent/lib/agent/engines/agent-engine-echo.so
#
# Rename the project, the target (keep the agent-engine-<name> pattern: the CLI derives the engine's
# file name from it) and the sources when you start your own engine.
cmake_minimum_required(VERSION 3.21)
project(agent_engine_echo VERSION 0.1.0 LANGUAGES C CXX)

# agent-cpp::engine_abi (the header-only <agent-cpp/engine_abi.h>), agent_add_engine(),
# agent_install_engine(), and agent-cpp::agent for the test harness.
find_package(agent-cpp 0.1 REQUIRED)

# A shared library named agent-engine-echo.so (no "lib" prefix) with hidden visibility, so only
# agent_engine_get_v1 is exported and the symbols of any library you link stay private.
agent_add_engine(agent-engine-echo src/echo_engine.c)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(agent-engine-echo PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(ECHO_ENGINE_BUILD_TESTS "Build the plugin test harness (links agent-cpp::agent)" ON)
if(ECHO_ENGINE_BUILD_TESTS)
    enable_testing()
    add_executable(echo_engine_test test/engine_test.cpp)
    target_compile_features(echo_engine_test PRIVATE cxx_std_20)
    target_link_libraries(echo_engine_test PRIVATE agent-cpp::agent)
    # agent-cpp_ENGINE_HOST: the installed agent-engine-host, for the process-isolation checks.
    add_test(NAME echo_engine COMMAND echo_engine_test $<TARGET_FILE:agent-engine-echo> ${agent-cpp_ENGINE_HOST})
endif()

# Installs into <prefix>/lib/agent/engines/, where the agent CLI looks. List any runtime libraries the
# plugin loads through an $ORIGIN RPATH after the target: agent_install_engine(agent-engine-echo libfoo.so.1)
agent_install_engine(agent-engine-echo)
