# agent-engine-llama: a plugin over llama.cpp, in one of two ways.
#
#   AGENT_LLAMA_PREBUILT=<folder>   (what a release does) the plugin is compiled against an upstream llama.cpp release that was unpacked
#                                   in <folder> and links its shared libraries (llama, llama-common, mtmd, ggml).
#                                   The release must be the one vendor/llama.cpp is at (its tag is in scripts/components.tsv), because the
#                                   plugin uses llama-common and compiles llama.cpp's scheduler sources, whose interfaces change with every commit. Only the
#                                   plugin is compiled: llama.cpp is not. `scripts/install.sh fetch llama <folder>` downloads the release.
#   (empty)                         llama.cpp is compiled from vendor/llama.cpp and linked statically into one self-contained plugin.
set(AGENT_LLAMA_PREBUILT "" CACHE PATH "Folder with an upstream llama.cpp release to link instead of compiling llama.cpp")
if(AGENT_LLAMA_PREBUILT)
    include(${PROJECT_SOURCE_DIR}/cmake/prebuilt_library.cmake)
    set(_ll ${PROJECT_SOURCE_DIR}/vendor/llama.cpp)
    # The engine for models with a multi-token-prediction head, a draft model or a native decision head is llama.cpp's own scheduler
    # (tools/server/server-context: llama-server and llama-cli run on it). It is compiled into the plugin from vendor/llama.cpp, the same
    # way on every system, against the release's libraries (llama, llama-common, mtmd): llama-server-impl exports it on Linux and macOS
    # but not on Windows. Only those sources are compiled, not llama.cpp itself, and nothing listens on a socket.
    set(_libs ggml-base ggml llama llama-common mtmd)
    foreach(_lib ${_libs})
        agent_prebuilt_library(${_lib} "${AGENT_LLAMA_PREBUILT}")
    endforeach()
    target_include_directories(llama SYSTEM INTERFACE ${_ll}/include ${_ll}/ggml/include)
    target_include_directories(llama-common SYSTEM INTERFACE ${_ll}/common ${_ll}/vendor)
    target_include_directories(mtmd SYSTEM INTERFACE ${_ll}/tools/mtmd)
    target_link_libraries(llama INTERFACE ggml ggml-base)
    set(_srv ${_ll}/tools/server)
    set(_server_sources
        ${_srv}/server-chat.cpp ${_srv}/server-task.cpp ${_srv}/server-queue.cpp ${_srv}/server-common.cpp
        ${_srv}/server-decision.cpp ${_srv}/server-context.cpp ${_srv}/server-realtime.cpp ${_srv}/server-stream.cpp
        ${_srv}/server-tools.cpp ${_srv}/server-mcp.cpp ${_srv}/server-schema.cpp)
    set(CMAKE_CXX_STANDARD 20)
    add_library(agent-engine-llama SHARED
        llama_engine.cpp
        upstream_engine.cpp
        ${_server_sources}
        ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
    )
    # llama.cpp is a C++17 project: its sources are compiled as such, without our warning flags.
    set_source_files_properties(${_server_sources} PROPERTIES COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/w;/bigobj,-w>" CXX_STANDARD 17)
    target_include_directories(agent-engine-llama PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
        ${_srv} ${_ll}/tools/mtmd ${_ll} ${_ll}/vendor ${_ll}/common)
    target_link_libraries(agent-engine-llama PRIVATE llama llama-common mtmd)
    target_compile_definitions(agent-engine-llama PRIVATE AGENT_LLAMA_PREBUILT=1 AGENT_LLAMA_UPSTREAM=1)
    set_target_properties(agent-engine-llama PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
    target_compile_options(agent-engine-llama PRIVATE ${AGENT_WARNINGS})
    if(UNIX AND NOT APPLE)
        # The libraries sit next to the plugin.
        set_target_properties(agent-engine-llama PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH ON)
        target_link_options(agent-engine-llama PRIVATE -Wl,--disable-new-dtags)
    endif()
    # What a test or a build tree needs next to the plugin: the release's libraries.
    if(WIN32)
        file(GLOB _rt "${AGENT_LLAMA_PREBUILT}/*.dll")
    else()
        file(GLOB _rt "${AGENT_LLAMA_PREBUILT}/*.so*" "${AGENT_LLAMA_PREBUILT}/*.dylib")
    endif()
    foreach(_f ${_rt})
        add_custom_command(TARGET agent-engine-llama POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_f}" "$<TARGET_FILE_DIR:agent-engine-llama>" VERBATIM)
    endforeach()
    agent_install_engine(agent-engine-llama)
    return()
endif()

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
