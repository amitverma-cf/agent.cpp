# agent_bench (AGENT_BUILD_BENCH): tokens/s, time to first token / audio, peak RSS, in-process vs
# process isolation, single vs parallel graph branches. See docs/benchmarks.md.
add_executable(agent_bench agent_bench.cpp)
target_link_libraries(agent_bench PRIVATE agent)
target_compile_options(agent_bench PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(agent_bench PRIVATE ${SANITIZER_FLAGS})
# Next to agent-engine-host, so process isolation finds the host without --host.
set_target_properties(agent_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_dependencies(agent_bench agent-engine-host)
