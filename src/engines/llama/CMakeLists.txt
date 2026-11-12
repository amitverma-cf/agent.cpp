# agent-engine-llama: links llama.cpp statically into one self-contained plugin.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# llama.cpp's own in-process engine (tools/server: the scheduler llama-server and llama-cli run on) is used for models whose GGUF
# carries an MTP head, because it already implements multi-token prediction, its checkpoints and its prompt cache. Nothing listens
# on a socket: the plugin calls its request handlers directly. AGENT_LLAMA_UPSTREAM=OFF leaves it out.
option(AGENT_LLAMA_UPSTREAM "Link llama.cpp's server engine (in process) for MTP models" ON)
if(AGENT_LLAMA_UPSTREAM)
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE) # the web page is not used: no download at configure time
else()
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
endif()
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)  # engines never download; the CLI hub does
# libmtmd (image / audio input) without the rest of tools/. Video goes through an ffmpeg
# subprocess in mtmd's helpers, which a plugin must not spawn, so it stays off.
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
# llama.cpp is a C++17 project (u8"" literals are passed to const char* APIs; under C++20 they are
# char8_t, which MSVC rejects). Build it as C++17 and keep the plugin itself at C++20.
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
set(CMAKE_CXX_STANDARD 20)
include(${PROJECT_SOURCE_DIR}/cmake/ggml_no_amx.cmake)

add_library(agent-engine-llama SHARED
    llama_engine.cpp
    upstream_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent-engine-llama PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_link_libraries(agent-engine-llama PRIVATE llama llama-common mtmd)
if(AGENT_LLAMA_UPSTREAM)
    target_link_libraries(agent-engine-llama PRIVATE llama-server-impl)
    target_include_directories(agent-engine-llama PRIVATE ${PROJECT_SOURCE_DIR}/vendor/llama.cpp/tools/server)
    target_compile_definitions(agent-engine-llama PRIVATE AGENT_LLAMA_UPSTREAM=1)
endif()
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
foreach(dep llama-common llama-common-base mtmd)
    if(TARGET ${dep})
        set_target_properties(${dep} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:${dep},INTERFACE_INCLUDE_DIRECTORIES>")
    endif()
endforeach()

agent_install_engine(agent-engine-llama)
