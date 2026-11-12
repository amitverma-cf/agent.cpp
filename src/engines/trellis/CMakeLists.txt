# agent-engine-trellis: links trellis2.cpp (a C++/ggml TRELLIS.2 image->3D port with a flat C API)
# and the ggml it vendors statically into one self-contained plugin.
#
# trellis2.cpp bundles its own ggml (a fork) whose CMake targets (ggml, ggml-base, ggml-cpu) have the
# same names as llama.cpp's, so it is built as a separate CMake project with ExternalProject and its
# static archives are linked in here -- the same scheme as src/engines/sdcpp.
include(ExternalProject)

set(TRELLIS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/vendor/trellis2cpp)
set(TRELLIS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/trellis2cpp)
set(TRELLIS_LIB_DIR ${TRELLIS_BINARY_DIR}/lib)

if(NOT EXISTS ${TRELLIS_SOURCE_DIR}/trellis2_capi.h OR NOT EXISTS ${TRELLIS_SOURCE_DIR}/ggml/CMakeLists.txt)
    message(FATAL_ERROR "AGENT_ENGINE_TRELLIS=ON needs the trellis2cpp submodule and its ggml:\n"
                        "  git submodule update --init vendor/trellis2cpp\n"
                        "  git -C vendor/trellis2cpp submodule update --init --depth 1 ggml")
endif()

set(AGENT_TRELLIS_BUILD_JOBS "" CACHE STRING "Parallel jobs for the nested trellis2.cpp build (empty = build tool default)")
set(TRELLIS_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT TRELLIS_BUILD_TYPE)
    set(TRELLIS_BUILD_TYPE Release)
endif()

find_package(OpenMP COMPONENTS C CXX QUIET)
set(TRELLIS_OPENMP OFF)
if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    set(TRELLIS_OPENMP ON)
endif()

set(TRELLIS_LIBS
    ${TRELLIS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}trellis2${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${TRELLIS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${TRELLIS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${TRELLIS_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-base${CMAKE_STATIC_LIBRARY_SUFFIX}
)

set(TRELLIS_BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target trellis2)
if(AGENT_TRELLIS_BUILD_JOBS)
    list(APPEND TRELLIS_BUILD_COMMAND --parallel ${AGENT_TRELLIS_BUILD_JOBS})
endif()

ExternalProject_Add(trellis2cpp-build
    SOURCE_DIR ${TRELLIS_SOURCE_DIR}
    BINARY_DIR ${TRELLIS_BINARY_DIR}
    CMAKE_CACHE_ARGS
        -DCMAKE_PROJECT_INCLUDE:FILEPATH=${PROJECT_SOURCE_DIR}/cmake/ggml_no_amx.cmake
        -DAGENT_GGML_AMX:BOOL=${AGENT_GGML_AMX}
        -DCMAKE_BUILD_TYPE:STRING=${TRELLIS_BUILD_TYPE}
        -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_C_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_CXX_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${TRELLIS_LIB_DIR}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DTRELLIS2_BUILD_EXAMPLES:BOOL=OFF
        -DTRELLIS2_BUILD_TESTS:BOOL=OFF
        -DTRELLIS2_USE_EXTERNAL_GGML:BOOL=OFF
        # CGAL print remeshing is the only part of trellis2.cpp that uses exceptions; keep it out.
        -DTRELLIS2_CGAL:BOOL=OFF
        # CPU build everywhere (Metal would need extra archives + frameworks linked into the plugin).
        -DTRELLIS2_METAL:BOOL=OFF
        -DGGML_OPENMP:BOOL=${TRELLIS_OPENMP}
        -DGGML_BUILD_TESTS:BOOL=OFF
        -DGGML_BUILD_EXAMPLES:BOOL=OFF
    BUILD_COMMAND ${TRELLIS_BUILD_COMMAND}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${TRELLIS_LIBS}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL ON
)

add_library(agent-engine-trellis SHARED
    trellis_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
add_dependencies(agent-engine-trellis trellis2cpp-build)
target_include_directories(agent-engine-trellis PRIVATE ${PROJECT_SOURCE_DIR}/include)
# trellis2.cpp / simdjson headers are system headers so our warning flags don't apply to them.
target_include_directories(agent-engine-trellis SYSTEM PRIVATE
    ${TRELLIS_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
find_package(Threads REQUIRED)
target_link_libraries(agent-engine-trellis PRIVATE ${TRELLIS_LIBS} Threads::Threads ${CMAKE_DL_LIBS})
if(TRELLIS_OPENMP)
    target_link_libraries(agent-engine-trellis PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif()
set_target_properties(agent-engine-trellis PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-trellis PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep the statically linked trellis2/ggml symbols private to this plugin, so its ggml never
    # clashes with the llama or sdcpp plugin's ggml in the same process.
    # The version script also hides the std:: template instantiations (libstdc++ marks them
    # default-visibility), leaving agent_engine_get_v1 as the only dynamic symbol.
    target_link_options(agent-engine-trellis PRIVATE -Wl,--exclude-libs,ALL
                        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
    set_property(TARGET agent-engine-trellis APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
elseif(APPLE)
    target_link_options(agent-engine-trellis PRIVATE -Wl,-exported_symbol,_agent_engine_get_v1)
endif()
agent_install_engine(agent-engine-trellis)
