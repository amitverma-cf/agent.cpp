add_library(agent-engine-mock SHARED mock_engine.cpp)
target_include_directories(agent-engine-mock PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(agent-engine-mock PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(agent-engine-mock PRIVATE ${AGENT_WARNINGS})

# The same engine with AGENT_ENGINE_FLAG_CONCURRENT, for concurrency tests.
add_library(agent-engine-mock-concurrent SHARED mock_engine.cpp)
target_compile_definitions(agent-engine-mock-concurrent PRIVATE AGENT_MOCK_CONCURRENT)
target_include_directories(agent-engine-mock-concurrent PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(agent-engine-mock-concurrent PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(agent-engine-mock-concurrent PRIVATE ${AGENT_WARNINGS})

# Every capability with deterministic stand-in output, for model-free end-to-end tests.
add_library(agent-engine-mock-all SHARED mock_engine.cpp)
target_compile_definitions(agent-engine-mock-all PRIVATE AGENT_MOCK_ALL)
target_include_directories(agent-engine-mock-all PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(agent-engine-mock-all PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(agent-engine-mock-all PRIVATE ${AGENT_WARNINGS})

# For library users' tests: find_package(agent-cpp COMPONENTS testing) gives the target agent-cpp::testing, which defines
# AGENT_CPP_MOCK_ENGINE (a plugin that answers chat with canned text and calls tools on request) and AGENT_CPP_MOCK_ALL_ENGINE (every
# capability with deterministic stand-in output). Off by default so that CLI archives never carry it; it is installed outside
# lib/agent/engines, so the CLI does not list it as an engine.
option(AGENT_INSTALL_MOCK_ENGINE "Install the mock engine plugins for library users' tests (component: testing)" OFF)
if(AGENT_INSTALL_MOCK_ENGINE)
    install(TARGETS agent-engine-mock agent-engine-mock-all
        LIBRARY DESTINATION ${AGENT_INSTALL_TESTINGDIR}
        RUNTIME DESTINATION ${AGENT_INSTALL_TESTINGDIR}
        COMPONENT testing)
endif()
