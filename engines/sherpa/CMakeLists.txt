# agent-engine-sherpa: sherpa-onnx speech (TTS + ASR) through its C API.
#
# Built against the official prebuilt shared libraries from
# https://github.com/k2-fsa/sherpa-onnx/releases (sherpa-onnx-v<ver>-linux-x64-shared.tar.bz2):
#   cmake -DAGENT_ENGINE_SHERPA=ON -DSHERPA_ONNX_ROOT=/path/to/sherpa-onnx-v<ver>-linux-x64-shared
#
# The plugin's RPATH is $ORIGIN: libsherpa-onnx-c-api and libonnxruntime ship next to it. They
# are copied beside the plugin after each build so the build tree has the shipping layout.
set(SHERPA_ONNX_ROOT "" CACHE PATH "Extracted sherpa-onnx shared release (contains include/ and lib/)")
if(NOT SHERPA_ONNX_ROOT)
    message(FATAL_ERROR "AGENT_ENGINE_SHERPA=ON needs -DSHERPA_ONNX_ROOT=<extracted sherpa-onnx shared release>")
endif()

find_path(SHERPA_ONNX_INCLUDE_DIR sherpa-onnx/c-api/c-api.h PATHS "${SHERPA_ONNX_ROOT}/include" NO_DEFAULT_PATH)
find_library(SHERPA_ONNX_C_API_LIB NAMES sherpa-onnx-c-api PATHS "${SHERPA_ONNX_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT SHERPA_ONNX_INCLUDE_DIR OR NOT SHERPA_ONNX_C_API_LIB)
    message(FATAL_ERROR "sherpa-onnx C API not found under SHERPA_ONNX_ROOT=${SHERPA_ONNX_ROOT} "
                        "(expected include/sherpa-onnx/c-api/c-api.h and lib/libsherpa-onnx-c-api.*)")
endif()

add_library(agent-engine-sherpa SHARED
    sherpa_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent-engine-sherpa PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
target_include_directories(agent-engine-sherpa SYSTEM PRIVATE ${SHERPA_ONNX_INCLUDE_DIR})
target_link_libraries(agent-engine-sherpa PRIVATE ${SHERPA_ONNX_C_API_LIB})
set_target_properties(agent-engine-sherpa PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-sherpa PRIVATE ${AGENT_WARNINGS})

# Ship the sherpa-onnx / ONNX Runtime shared libraries next to the plugin.
file(GLOB SHERPA_ONNX_RUNTIME_LIBS
    "${SHERPA_ONNX_ROOT}/lib/libsherpa-onnx-c-api.*"
    "${SHERPA_ONNX_ROOT}/lib/libonnxruntime*"
    "${SHERPA_ONNX_ROOT}/lib/sherpa-onnx-c-api.dll"
    "${SHERPA_ONNX_ROOT}/lib/onnxruntime*.dll"
)
add_custom_command(TARGET agent-engine-sherpa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHERPA_ONNX_RUNTIME_LIBS} $<TARGET_FILE_DIR:agent-engine-sherpa>
    COMMENT "Copying sherpa-onnx runtime libraries next to agent-engine-sherpa"
    VERBATIM
)

# Installed layout: the plugin and the sherpa-onnx / ONNX Runtime libraries in lib/agent/engines/.
agent_install_engine(agent-engine-sherpa ${SHERPA_ONNX_RUNTIME_LIBS})
