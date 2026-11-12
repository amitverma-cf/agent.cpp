# agent-engine-onnx: generic ONNX Runtime engine, linked against a prebuilt ONNX Runtime shared library.
#   -DONNXRUNTIME_ROOT=<dir>  an extracted official release (onnxruntime-<os>-<arch>-<ver>/: include/ + lib/)
#   otherwise find_package(onnxruntime) (system or vcpkg/conda install).
# The ORT shared library must ship next to the plugin: the plugin's RPATH is $ORIGIN, and the build
# copies the library into the plugin's output directory.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Prebuilt ONNX Runtime directory (contains include/ and lib/)")
# Without ONNXRUNTIME_ROOT the pinned upstream release is downloaded (sha256 checked); off: use a system install (find_package).
option(AGENT_ONNXRUNTIME_FETCH "Download the pinned upstream ONNX Runtime release when ONNXRUNTIME_ROOT is not set" ON)
if(NOT ONNXRUNTIME_ROOT AND AGENT_ONNXRUNTIME_FETCH)
    include(${PROJECT_SOURCE_DIR}/cmake/onnxruntime_release.cmake)
endif()

if(ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime" NO_DEFAULT_PATH)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "AGENT_ENGINE_ONNX: no onnxruntime_c_api.h / onnxruntime library under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
    endif()
    add_library(agent_onnxruntime INTERFACE)
    target_include_directories(agent_onnxruntime SYSTEM INTERFACE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(agent_onnxruntime INTERFACE "${ONNXRUNTIME_LIBRARY}")
    get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
else()
    find_package(onnxruntime CONFIG QUIET)
    if(NOT TARGET onnxruntime::onnxruntime)
        message(FATAL_ERROR "AGENT_ENGINE_ONNX: ONNX Runtime not found. Download a release from "
                            "https://github.com/microsoft/onnxruntime/releases and pass -DONNXRUNTIME_ROOT=<extracted dir>.")
    endif()
    add_library(agent_onnxruntime INTERFACE)
    target_link_libraries(agent_onnxruntime INTERFACE onnxruntime::onnxruntime)
    set(_ort_lib_dir "")
endif()

add_library(agent-engine-onnx SHARED
    onnx_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent-engine-onnx PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_link_libraries(agent-engine-onnx PRIVATE agent_onnxruntime)
set_target_properties(agent-engine-onnx PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    MACOSX_RPATH ON
)
if(APPLE)
    set_target_properties(agent-engine-onnx PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
endif()
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-onnx PRIVATE ${AGENT_WARNINGS})

# Mirror the shipping layout in the build tree: the ORT runtime library next to the plugin.
if(_ort_lib_dir)
    if(WIN32)
        file(GLOB _ort_runtime "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.dll")
    elseif(APPLE)
        file(GLOB _ort_runtime "${_ort_lib_dir}/libonnxruntime.*.dylib")
    else()
        # The SONAME file (libonnxruntime.so.1) and the providers bridge needed by non-CPU EPs.
        file(GLOB _ort_runtime "${_ort_lib_dir}/libonnxruntime.so.[0-9]" "${_ort_lib_dir}/libonnxruntime.so.[0-9][0-9]"
                               "${_ort_lib_dir}/libonnxruntime_providers_shared.so")
    endif()
    foreach(_f ${_ort_runtime})
        add_custom_command(TARGET agent-engine-onnx POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_f}" "$<TARGET_FILE_DIR:agent-engine-onnx>"
            VERBATIM)
    endforeach()
endif()

# Installed layout: the plugin and the bundled ORT libraries in lib/agent/engines/. With
# find_package(onnxruntime) the system ORT is used and nothing is bundled.
agent_install_engine(agent-engine-onnx ${_ort_runtime})
