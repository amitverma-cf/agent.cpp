# agent-engine-parakeet: links parakeet.cpp (and the ggml it vendors) statically into one
# self-contained plugin.
#
# parakeet.cpp ships its own (patched) ggml whose CMake targets (ggml, ggml-base, ggml-cpu) have the
# same names as llama.cpp's. Both cannot live in one CMake project, so parakeet.cpp is built as a
# separate CMake project with ExternalProject and its static archives are linked in here (as
# engines/sdcpp does). parakeet.cpp's configure step applies its in-tree ggml patches
# (third_party/ggml-patches) to its ggml checkout; ced.cpp and voice-detect.cpp (sound events,
# speaker identification) are left out.
include(ExternalProject)

set(PARAKEET_SOURCE_DIR ${PROJECT_SOURCE_DIR}/vendor/parakeet.cpp)
set(PARAKEET_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/parakeet.cpp)
set(PARAKEET_LIB_DIR ${PARAKEET_BINARY_DIR}/lib)

if(NOT EXISTS ${PARAKEET_SOURCE_DIR}/CMakeLists.txt OR NOT EXISTS ${PARAKEET_SOURCE_DIR}/third_party/ggml/CMakeLists.txt)
    message(FATAL_ERROR "AGENT_ENGINE_PARAKEET=ON needs the parakeet.cpp submodule and its ggml:\n"
                        "  git submodule update --init --depth 1 vendor/parakeet.cpp\n"
                        "  cd vendor/parakeet.cpp && git submodule update --init --depth 1 third_party/ggml")
endif()

set(AGENT_PARAKEET_BUILD_JOBS "" CACHE STRING "Parallel jobs for the nested parakeet.cpp build (empty = build tool default)")
set(PARAKEET_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT PARAKEET_BUILD_TYPE)
    set(PARAKEET_BUILD_TYPE Release)
endif()

find_package(OpenMP COMPONENTS C CXX QUIET)
set(PARAKEET_OPENMP OFF)
if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    set(PARAKEET_OPENMP ON)
endif()

set(PARAKEET_LIBS
    ${PARAKEET_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}parakeet${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${PARAKEET_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${PARAKEET_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${PARAKEET_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-base${CMAKE_STATIC_LIBRARY_SUFFIX}
)

get_property(PARAKEET_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(PARAKEET_BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target parakeet)
set(PARAKEET_CONFIG_ARGS -DCMAKE_BUILD_TYPE:STRING=${PARAKEET_BUILD_TYPE})
if(PARAKEET_MULTI_CONFIG)
    list(APPEND PARAKEET_BUILD_COMMAND --config $<CONFIG>)
    set(PARAKEET_CONFIG_ARGS)
    foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        list(APPEND PARAKEET_CONFIG_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg}:PATH=${PARAKEET_LIB_DIR})
    endforeach()
endif()
if(AGENT_PARAKEET_BUILD_JOBS)
    list(APPEND PARAKEET_BUILD_COMMAND --parallel ${AGENT_PARAKEET_BUILD_JOBS})
endif()

ExternalProject_Add(parakeet-build
    SOURCE_DIR ${PARAKEET_SOURCE_DIR}
    BINARY_DIR ${PARAKEET_BINARY_DIR}
    CMAKE_CACHE_ARGS
        -DCMAKE_PROJECT_INCLUDE:FILEPATH=${PROJECT_SOURCE_DIR}/cmake/ggml_no_amx.cmake
        -DAGENT_GGML_AMX:BOOL=${AGENT_GGML_AMX}
        ${PARAKEET_CONFIG_ARGS}
        -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_C_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_CXX_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${PARAKEET_LIB_DIR}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DPARAKEET_BUILD_CLI:BOOL=OFF
        -DPARAKEET_BUILD_TESTS:BOOL=OFF
        -DPARAKEET_BUILD_SERVER:BOOL=OFF
        -DPARAKEET_SHARED:BOOL=OFF
        -DPARAKEET_WITH_CED:BOOL=OFF
        -DPARAKEET_WITH_VOICEDETECT:BOOL=OFF
        -DGGML_OPENMP:BOOL=${PARAKEET_OPENMP}
        -DGGML_BUILD_TESTS:BOOL=OFF
        -DGGML_BUILD_EXAMPLES:BOOL=OFF
    BUILD_COMMAND ${PARAKEET_BUILD_COMMAND}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${PARAKEET_LIBS}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL ON
)

# parakeet.cpp keeps dr_wav's implementation in an OBJECT library (dr_wav_impl) that its static
# archive does not contain, so the plugin compiles that one translation unit itself.
add_library(agent-engine-parakeet SHARED
    parakeet_engine.cpp
    ${PARAKEET_SOURCE_DIR}/src/dr_wav_impl.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
add_dependencies(agent-engine-parakeet parakeet-build)
target_include_directories(agent-engine-parakeet PRIVATE ${PROJECT_SOURCE_DIR}/include)
# parakeet.cpp / ggml / simdjson headers are system headers so our warning flags don't apply to them.
target_include_directories(agent-engine-parakeet SYSTEM PRIVATE
    ${PARAKEET_SOURCE_DIR}/include
    ${PARAKEET_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
find_package(Threads REQUIRED)
target_link_libraries(agent-engine-parakeet PRIVATE ${PARAKEET_LIBS} Threads::Threads ${CMAKE_DL_LIBS})
if(PARAKEET_OPENMP)
    target_link_libraries(agent-engine-parakeet PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif()
set_target_properties(agent-engine-parakeet PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
                            ${PARAKEET_SOURCE_DIR}/src/dr_wav_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-parakeet PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep the statically linked parakeet.cpp/ggml symbols private to this plugin, so its ggml never
    # clashes with another plugin's ggml in the same process; export agent_engine_get_v1 only.
    target_link_options(agent-engine-parakeet PRIVATE -Wl,--exclude-libs,ALL
                        -Wl,--version-script=${PROJECT_SOURCE_DIR}/engines/common/exports.map)
    set_property(TARGET agent-engine-parakeet APPEND PROPERTY LINK_DEPENDS ${PROJECT_SOURCE_DIR}/engines/common/exports.map)
elseif(APPLE)
    target_link_options(agent-engine-parakeet PRIVATE -Wl,-exported_symbol,_agent_engine_get_v1)
endif()

agent_install_engine(agent-engine-parakeet)
