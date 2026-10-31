add_library(agent-engine-mock SHARED mock_engine.cpp)
target_include_directories(agent-engine-mock PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(agent-engine-mock PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(agent-engine-mock PRIVATE ${AGENT_WARNINGS})
