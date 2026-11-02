# agent-engine-whisper: links whisper.cpp (and the ggml it vendors) statically into one
# self-contained plugin.
#
# whisper.cpp ships its own ggml whose CMake targets (ggml, ggml-base, ggml-cpu) have the same names
# as llama.cpp's. Both cannot live in one CMake project, so whisper.cpp is built as a separate CMake
# project with ExternalProject and its static archives are linked in here (as engines/sdcpp does).
include(ExternalProject)

set(WHISPER_SOURCE_DIR ${PROJECT_SOURCE_DIR}/vendor/whisper.cpp)
set(WHISPER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/whisper.cpp)
set(WHISPER_LIB_DIR ${WHISPER_BINARY_DIR}/lib)

if(NOT EXISTS ${WHISPER_SOURCE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "AGENT_ENGINE_WHISPER=ON needs the whisper.cpp submodule:\n"
                        "  git submodule update --init --depth 1 vendor/whisper.cpp")
endif()

set(AGENT_WHISPER_BUILD_JOBS "" CACHE STRING "Parallel jobs for the nested whisper.cpp build (empty = build tool default)")
set(WHISPER_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT WHISPER_BUILD_TYPE)
    set(WHISPER_BUILD_TYPE Release)
endif()

find_package(OpenMP COMPONENTS C CXX QUIET)
set(WHISPER_OPENMP OFF)
if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    set(WHISPER_OPENMP ON)
endif()

set(WHISPER_LIBS
    ${WHISPER_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}whisper${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${WHISPER_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${WHISPER_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${WHISPER_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-base${CMAKE_STATIC_LIBRARY_SUFFIX}
)

get_property(WHISPER_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(WHISPER_BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target whisper)
set(WHISPER_CONFIG_ARGS -DCMAKE_BUILD_TYPE:STRING=${WHISPER_BUILD_TYPE})
if(WHISPER_MULTI_CONFIG)
    list(APPEND WHISPER_BUILD_COMMAND --config $<CONFIG>)
    set(WHISPER_CONFIG_ARGS)
    foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        list(APPEND WHISPER_CONFIG_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg}:PATH=${WHISPER_LIB_DIR})
    endforeach()
endif()
if(AGENT_WHISPER_BUILD_JOBS)
    list(APPEND WHISPER_BUILD_COMMAND --parallel ${AGENT_WHISPER_BUILD_JOBS})
endif()

ExternalProject_Add(whisper-build
    SOURCE_DIR ${WHISPER_SOURCE_DIR}
    BINARY_DIR ${WHISPER_BINARY_DIR}
    CMAKE_CACHE_ARGS
        ${WHISPER_CONFIG_ARGS}
        -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_C_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_CXX_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${WHISPER_LIB_DIR}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DWHISPER_BUILD_EXAMPLES:BOOL=OFF
        -DWHISPER_BUILD_TESTS:BOOL=OFF
        -DWHISPER_BUILD_SERVER:BOOL=OFF
        -DWHISPER_CURL:BOOL=OFF
        -DWHISPER_SDL2:BOOL=OFF
        -DGGML_OPENMP:BOOL=${WHISPER_OPENMP}
        -DGGML_BUILD_TESTS:BOOL=OFF
        -DGGML_BUILD_EXAMPLES:BOOL=OFF
    BUILD_COMMAND ${WHISPER_BUILD_COMMAND}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${WHISPER_LIBS}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL ON
)

add_library(agent-engine-whisper SHARED
    whisper_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
add_dependencies(agent-engine-whisper whisper-build)
target_include_directories(agent-engine-whisper PRIVATE ${PROJECT_SOURCE_DIR}/include)
# whisper.cpp / ggml / simdjson headers are system headers so our warning flags don't apply to them.
target_include_directories(agent-engine-whisper SYSTEM PRIVATE
    ${WHISPER_SOURCE_DIR}/include
    ${WHISPER_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
find_package(Threads REQUIRED)
target_link_libraries(agent-engine-whisper PRIVATE ${WHISPER_LIBS} Threads::Threads ${CMAKE_DL_LIBS})
if(WHISPER_OPENMP)
    target_link_libraries(agent-engine-whisper PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif()
set_target_properties(agent-engine-whisper PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-whisper PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep the statically linked whisper.cpp/ggml symbols private to this plugin, so its ggml never
    # clashes with another plugin's ggml in the same process. The version script also hides the
    # std:: template instantiations: only agent_engine_get_v1 is exported.
    target_link_options(agent-engine-whisper PRIVATE -Wl,--exclude-libs,ALL
                        -Wl,--version-script=${PROJECT_SOURCE_DIR}/engines/common/exports.map)
    set_property(TARGET agent-engine-whisper APPEND PROPERTY LINK_DEPENDS ${PROJECT_SOURCE_DIR}/engines/common/exports.map)
elseif(APPLE)
    target_link_options(agent-engine-whisper PRIVATE -Wl,-exported_symbol,_agent_engine_get_v1)
endif()

agent_install_engine(agent-engine-whisper)
