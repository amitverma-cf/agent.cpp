# The example programs. They take no arguments: each one names its model files at the top of main() and needs the engine it uses.
#
# Build them against an installed agent.cpp (this is also how your own project would use the library):
#
#   cmake --install build --prefix ~/agent-cpp
#   cmake -S examples -B build-examples -DCMAKE_PREFIX_PATH=~/agent-cpp
#   cmake --build build-examples
#   examples/download-models.sh            # once: fetches the models they use into ./models
#   ./build-examples/hello
#
# Inside the main build (AGENT_BUILD_EXAMPLES) they are only compiled.

if(NOT TARGET agent)
    cmake_minimum_required(VERSION 3.21)
    project(agent_cpp_examples CXX)
    set(CMAKE_CXX_STANDARD 20)
    find_package(agent-cpp REQUIRED COMPONENTS graph rag)
    set(_agent_lib agent-cpp::agent)
    set(_agent_hub agent-cpp::hub)
else()
    set(_agent_lib agent)
    set(_agent_hub agent_hub)
endif()

function(agent_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE ${_agent_lib})
    if(TARGET agent)
        target_compile_options(${name} PRIVATE ${AGENT_WARNINGS} ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
        target_link_options(${name} PRIVATE ${SANITIZER_FLAGS} ${COVERAGE_FLAGS})
    endif()
endfunction()

agent_example(hello 01-hello-agent/hello.cpp)
agent_example(tools 02-tools/tools.cpp)
agent_example(graph 03-graph-workflow/graph.cpp)
agent_example(rag 05-rag-over-docs/rag.cpp)
agent_example(prompt_to_image 06-prompt-to-image/prompt_to_image.cpp)
agent_example(draw 09-image-generation/draw.cpp)
agent_example(speech 10-speech/speech.cpp)
if(TARGET ${_agent_hub} OR NOT TARGET agent)
    agent_example(pull 07-model-pull/pull.cpp)
    target_link_libraries(pull PRIVATE ${_agent_hub})
endif()

# 04: an engine plugin (reverse_engine.cpp) and a program that drives it in process and under process isolation.
add_library(agent-engine-reverse SHARED 04-custom-engine/reverse_engine.cpp)
if(TARGET agent)
    target_include_directories(agent-engine-reverse PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(agent-engine-reverse PRIVATE ${AGENT_WARNINGS})
else()
    target_link_libraries(agent-engine-reverse PRIVATE agent-cpp::engine_abi)
endif()
set_target_properties(agent-engine-reverse PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
agent_example(engine_host 04-custom-engine/host.cpp)
add_dependencies(engine_host agent-engine-reverse)
target_compile_definitions(engine_host PRIVATE REVERSE_ENGINE_PATH="$<TARGET_FILE:agent-engine-reverse>")
if(TARGET agent)
    add_dependencies(engine_host agent-engine-host)
    target_compile_definitions(engine_host PRIVATE ENGINE_HOST_PATH="$<TARGET_FILE:agent-engine-host>")
else()
    target_compile_definitions(engine_host PRIVATE ENGINE_HOST_PATH="${agent-cpp_ENGINE_HOST}")
endif()
