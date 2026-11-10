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
