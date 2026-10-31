# agent-engine-llama: links llama.cpp statically into one self-contained plugin.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)

add_library(agent-engine-llama SHARED
    llama_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent-engine-llama PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_link_libraries(agent-engine-llama PRIVATE llama)
set_target_properties(agent-engine-llama PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-llama PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep the statically linked llama/ggml symbols private to this plugin.
    target_link_options(agent-engine-llama PRIVATE -Wl,--exclude-libs,ALL)
endif()
# Treat llama.cpp/ggml headers as system headers so our warning flags don't apply to them.
set_target_properties(llama PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:llama,INTERFACE_INCLUDE_DIRECTORIES>")
if(TARGET ggml)
    set_target_properties(ggml PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>")
endif()
