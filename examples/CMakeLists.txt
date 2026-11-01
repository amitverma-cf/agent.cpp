# Example programs (AGENT_BUILD_EXAMPLES). The guide programs under guides/ hold the code shown in
# docs/guides/*.md; the ones that run on the mock engine are registered as ctest tests (label
# "examples"), so the guides cannot drift from the API.

function(agent_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE agent)
    target_compile_options(${name} PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
    target_link_options(${name} PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
endfunction()

# ---- docs/guides -------------------------------------------------------------------------------
agent_example(guide_getting_started guides/getting_started_library.cpp)
agent_example(guide_tool guides/writing_a_tool.cpp)
agent_example(guide_graph guides/writing_a_graph.cpp)
target_link_libraries(guide_graph PRIVATE agent_store_sqlite)

# The engine from docs/guides/writing-an-engine.md, built exactly like the plugins in engines/.
add_library(agent-engine-reverse SHARED guides/reverse_engine.cpp)
target_include_directories(agent-engine-reverse PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(agent-engine-reverse PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(agent-engine-reverse PRIVATE ${AGENT_WARNINGS})
agent_example(guide_engine_host guides/writing_an_engine_host.cpp)
add_dependencies(guide_engine_host agent-engine-reverse)

# ---- examples ----------------------------------------------------------------------------------
agent_example(image_critique image_critique/image_critique.cpp)
agent_example(rag_docs rag_docs/rag_docs.cpp)

# ---- smoke tests on the mock engine (ctest -L examples) -----------------------------------------
set(_ws ${CMAKE_CURRENT_BINARY_DIR}/workspaces)
set(_host_env "AGENT_ENGINE_HOST=$<TARGET_FILE:agent-engine-host>")

add_test(NAME example.guide_getting_started
         COMMAND guide_getting_started $<TARGET_FILE:agent-engine-mock> mock-model "hello" ${_ws}/getting_started)
set_tests_properties(example.guide_getting_started PROPERTIES PASS_REGULAR_EXPRESSION "echo: hello")
add_test(NAME example.guide_engine COMMAND guide_engine_host $<TARGET_FILE:agent-engine-reverse> ${_ws}/engine)
add_test(NAME example.guide_tool COMMAND guide_tool $<TARGET_FILE:agent-engine-mock> $<TARGET_FILE:fake_mcp_server> ${_ws}/tool)
add_test(NAME example.guide_graph
         COMMAND guide_graph $<TARGET_FILE:agent-engine-mock> ${CMAKE_CURRENT_SOURCE_DIR}/graphs ${_ws}/graph)
add_test(NAME example.image_critique_unconfigured COMMAND image_critique)
set_tests_properties(example.image_critique_unconfigured PROPERTIES PASS_REGULAR_EXPRESSION "not configured")
add_test(NAME example.rag_docs_unconfigured COMMAND rag_docs)
set_tests_properties(example.rag_docs_unconfigured PROPERTIES PASS_REGULAR_EXPRESSION "not configured")
set_tests_properties(example.guide_getting_started example.guide_engine example.guide_tool example.guide_graph
                     example.image_critique_unconfigured example.rag_docs_unconfigured
                     PROPERTIES LABELS examples ENVIRONMENT "${_host_env}")
