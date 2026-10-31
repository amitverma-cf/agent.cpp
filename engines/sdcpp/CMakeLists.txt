# agent-engine-sdcpp: links stable-diffusion.cpp (and the ggml it vendors) statically into one
# self-contained plugin.
#
# stable-diffusion.cpp ships its own patched ggml whose CMake targets (ggml, ggml-base, ggml-cpu)
# have the same names as llama.cpp's. Both cannot live in one CMake project, so sd.cpp is built as
# a separate CMake project with ExternalProject and its static archives are linked in here.
include(ExternalProject)

set(SDCPP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/vendor/stable-diffusion.cpp)
set(SDCPP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/stable-diffusion.cpp)
set(SDCPP_LIB_DIR ${SDCPP_BINARY_DIR}/lib)

if(NOT EXISTS ${SDCPP_SOURCE_DIR}/CMakeLists.txt OR NOT EXISTS ${SDCPP_SOURCE_DIR}/ggml/CMakeLists.txt)
    message(FATAL_ERROR "AGENT_ENGINE_SDCPP=ON needs the stable-diffusion.cpp submodule and its ggml:\n"
                        "  git submodule update --init vendor/stable-diffusion.cpp\n"
                        "  git -C vendor/stable-diffusion.cpp submodule update --init --depth 1 ggml")
endif()

set(AGENT_SDCPP_BUILD_JOBS "" CACHE STRING "Parallel jobs for the nested stable-diffusion.cpp build (empty = build tool default)")
set(SDCPP_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT SDCPP_BUILD_TYPE)
    set(SDCPP_BUILD_TYPE Release)
endif()

find_package(OpenMP COMPONENTS C CXX QUIET)
set(SDCPP_OPENMP OFF)
if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
    set(SDCPP_OPENMP ON)
endif()

set(SDCPP_LIBS
    ${SDCPP_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}stable-diffusion${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${SDCPP_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${SDCPP_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cpu${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${SDCPP_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-base${CMAKE_STATIC_LIBRARY_SUFFIX}
)

# Multi-config generators (Visual Studio, Xcode, Ninja Multi-Config) build the configuration the
# outer build is building; the per-config archive directories below keep the libraries in one place.
get_property(SDCPP_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(SDCPP_BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target stable-diffusion)
set(SDCPP_CONFIG_ARGS -DCMAKE_BUILD_TYPE:STRING=${SDCPP_BUILD_TYPE})
if(SDCPP_MULTI_CONFIG)
    list(APPEND SDCPP_BUILD_COMMAND --config $<CONFIG>)
    set(SDCPP_CONFIG_ARGS)
    foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        list(APPEND SDCPP_CONFIG_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg}:PATH=${SDCPP_LIB_DIR})
    endforeach()
endif()
if(AGENT_SDCPP_BUILD_JOBS)
    list(APPEND SDCPP_BUILD_COMMAND --parallel ${AGENT_SDCPP_BUILD_JOBS})
endif()

ExternalProject_Add(sdcpp-build
    SOURCE_DIR ${SDCPP_SOURCE_DIR}
    BINARY_DIR ${SDCPP_BINARY_DIR}
    CMAKE_CACHE_ARGS
        ${SDCPP_CONFIG_ARGS}
        -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_C_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_CXX_VISIBILITY_PRESET:STRING=hidden
        -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${SDCPP_LIB_DIR}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DSD_BUILD_EXAMPLES:BOOL=OFF
        -DSD_BUILD_SHARED_LIBS:BOOL=OFF
        -DSD_WEBP:BOOL=OFF
        -DSD_WEBM:BOOL=OFF
        -DGGML_OPENMP:BOOL=${SDCPP_OPENMP}
        -DGGML_BUILD_TESTS:BOOL=OFF
        -DGGML_BUILD_EXAMPLES:BOOL=OFF
    BUILD_COMMAND ${SDCPP_BUILD_COMMAND}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${SDCPP_LIBS}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL ON
)

add_library(agent-engine-sdcpp SHARED
    sdcpp_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
add_dependencies(agent-engine-sdcpp sdcpp-build)
target_include_directories(agent-engine-sdcpp PRIVATE ${PROJECT_SOURCE_DIR}/include)
# sd.cpp / stb / simdjson headers are system headers so our warning flags don't apply to them.
target_include_directories(agent-engine-sdcpp SYSTEM PRIVATE
    ${SDCPP_SOURCE_DIR}/include
    ${SDCPP_SOURCE_DIR}/thirdparty
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
find_package(Threads REQUIRED)
target_link_libraries(agent-engine-sdcpp PRIVATE ${SDCPP_LIBS} Threads::Threads ${CMAKE_DL_LIBS})
if(SDCPP_OPENMP)
    target_link_libraries(agent-engine-sdcpp PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif()
set_target_properties(agent-engine-sdcpp PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-sdcpp PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep the statically linked sd.cpp/ggml symbols private to this plugin, so its ggml never
    # clashes with the llama plugin's ggml in the same process.
    target_link_options(agent-engine-sdcpp PRIVATE -Wl,--exclude-libs,ALL)
elseif(APPLE)
    target_link_options(agent-engine-sdcpp PRIVATE -Wl,-exported_symbol,_agent_engine_get_v1)
endif()
