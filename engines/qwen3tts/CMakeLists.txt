# agent-engine-qwen3tts: links qwentts.cpp (a ggml port of Qwen3-TTS) and the ggml fork it vendors
# statically into one self-contained plugin.
#
# qwentts.cpp's ggml has the same CMake target names as llama.cpp's, so it is built as a separate
# CMake project with ExternalProject and its static archives are linked in here (as engines/sdcpp
# does). Only the qwen-core library target is built; the plugin uses its public C API (qwen.h).
include(ExternalProject)

set(QWENTTS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/vendor/qwentts.cpp)
set(QWENTTS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/qwentts.cpp)
set(QWENTTS_LIB_DIR ${QWENTTS_BINARY_DIR}/lib)

if(NOT EXISTS ${QWENTTS_SOURCE_DIR}/CMakeLists.txt OR NOT EXISTS ${QWENTTS_SOURCE_DIR}/ggml/CMakeLists.txt)
    message(FATAL_ERROR "AGENT_ENGINE_QWEN3TTS=ON needs the qwentts.cpp submodule and its ggml:\n"
                        "  git submodule update --init vendor/qwentts.cpp\n"
                        "  cd vendor/qwentts.cpp && git submodule update --init --depth 1 ggml")
endif()

set(AGENT_QWEN3TTS_BUILD_JOBS "" CACHE STRING "Parallel jobs for the nested qwentts.cpp build (empty = build tool default)")
set(QWENTTS_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT QWENTTS_BUILD_TYPE)
    set(QWENTTS_BUILD_TYPE Release)
endif()

find_package(OpenMP COMPONENTS C CXX QUIET)
set(QWENTTS_OPENMP OFF)
if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    set(QWENTTS_OPENMP ON)
endif()

set(QWENTTS_LIBS
    ${QWENTTS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}qwen-core${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${QWENTTS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${QWENTTS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${QWENTTS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-base${CMAKE_STATIC_LIBRARY_SUFFIX}
)

get_property(QWENTTS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(QWENTTS_BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target qwen-core)
set(QWENTTS_CONFIG_ARGS -DCMAKE_BUILD_TYPE:STRING=${QWENTTS_BUILD_TYPE})
if(QWENTTS_MULTI_CONFIG)
    list(APPEND QWENTTS_BUILD_COMMAND --config $<CONFIG>)
    set(QWENTTS_CONFIG_ARGS)
    foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        list(APPEND QWENTTS_CONFIG_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg}:PATH=${QWENTTS_LIB_DIR})
    endforeach()
endif()
if(AGENT_QWEN3TTS_BUILD_JOBS)
    list(APPEND QWENTTS_BUILD_COMMAND --parallel ${AGENT_QWEN3TTS_BUILD_JOBS})
endif()

ExternalProject_Add(qwentts-build
    SOURCE_DIR ${QWENTTS_SOURCE_DIR}
    BINARY_DIR ${QWENTTS_BINARY_DIR}
    CMAKE_CACHE_ARGS
        -DCMAKE_PROJECT_INCLUDE:FILEPATH=${PROJECT_SOURCE_DIR}/cmake/ggml_no_amx.cmake
        -DAGENT_GGML_AMX:BOOL=${AGENT_GGML_AMX}
        ${QWENTTS_CONFIG_ARGS}
        -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_C_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_CXX_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${QWENTTS_LIB_DIR}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DQWEN_SHARED:BOOL=OFF
        -DGGML_OPENMP:BOOL=${QWENTTS_OPENMP}
        -DGGML_BUILD_TESTS:BOOL=OFF
        -DGGML_BUILD_EXAMPLES:BOOL=OFF
    BUILD_COMMAND ${QWENTTS_BUILD_COMMAND}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${QWENTTS_LIBS}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL ON
)

add_library(agent-engine-qwen3tts SHARED
    qwen3tts_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
add_dependencies(agent-engine-qwen3tts qwentts-build)
target_include_directories(agent-engine-qwen3tts PRIVATE ${PROJECT_SOURCE_DIR}/include)
# qwen.h is plain C99. QWEN_STATIC selects the static-archive linkage of its QT_API macro.
target_include_directories(agent-engine-qwen3tts SYSTEM PRIVATE
    ${QWENTTS_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_compile_definitions(agent-engine-qwen3tts PRIVATE QWEN_STATIC)
find_package(Threads REQUIRED)
target_link_libraries(agent-engine-qwen3tts PRIVATE ${QWENTTS_LIBS} Threads::Threads ${CMAKE_DL_LIBS})
if(QWENTTS_OPENMP)
    target_link_libraries(agent-engine-qwen3tts PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif()
set_target_properties(agent-engine-qwen3tts PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-qwen3tts PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep the statically linked qwentts.cpp/ggml symbols (qwen.h marks its API default-visibility)
    # private to this plugin; export agent_engine_get_v1 only.
    target_link_options(agent-engine-qwen3tts PRIVATE -Wl,--exclude-libs,ALL
                        -Wl,--version-script=${PROJECT_SOURCE_DIR}/engines/common/exports.map)
    set_property(TARGET agent-engine-qwen3tts APPEND PROPERTY LINK_DEPENDS ${PROJECT_SOURCE_DIR}/engines/common/exports.map)
elseif(APPLE)
    target_link_options(agent-engine-qwen3tts PRIVATE -Wl,-exported_symbol,_agent_engine_get_v1)
endif()

agent_install_engine(agent-engine-qwen3tts)
