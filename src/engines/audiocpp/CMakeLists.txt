# agent-engine-audiocpp: speech and audio models through audio.cpp's C ABI (include/audiocpp.h).
#
# audio.cpp ships its own ggml, so it is built as a separate CMake project with ExternalProject. It exports a shared
# library (libaudiocpp); the plugin links it and the installed layout puts both in lib/agent/engines/ ($ORIGIN rpath).
include(ExternalProject)

set(AUDIOCPP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/vendor/audio.cpp)
set(AUDIOCPP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/audio.cpp)
set(AUDIOCPP_LIB_DIR ${AUDIOCPP_BINARY_DIR}/bin) # audio.cpp puts its libraries in <build>/bin

if(NOT EXISTS ${AUDIOCPP_SOURCE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "AGENT_ENGINE_AUDIOCPP=ON needs the audio.cpp submodule:\n"
                        "  git submodule update --init --depth 1 vendor/audio.cpp")
endif()

# Model families compiled into libaudiocpp: "custom" builds only the listed ones (fast), "full" builds all 100+.
set(AGENT_AUDIOCPP_MODEL_SET "custom" CACHE STRING "audio.cpp model set: custom or full")
set(AGENT_AUDIOCPP_MODELS
    "pocket_tts,supertonic,parakeet_tdt,crisperwhisper,moonshine_asr,nemotron_3_diar,firered_vad"
    CACHE STRING "audio.cpp model families built when AGENT_AUDIOCPP_MODEL_SET=custom")
set(AGENT_AUDIOCPP_BUILD_JOBS "" CACHE STRING "Parallel jobs for the nested audio.cpp build (empty = build tool default)")

set(AUDIOCPP_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT AUDIOCPP_BUILD_TYPE)
    set(AUDIOCPP_BUILD_TYPE Release)
endif()

set(AUDIOCPP_LIB ${AUDIOCPP_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}audiocpp${CMAKE_SHARED_LIBRARY_SUFFIX})
set(AUDIOCPP_LINK_LIB ${AUDIOCPP_LIB})
if(WIN32)
    set(AUDIOCPP_LINK_LIB ${AUDIOCPP_LIB_DIR}/audiocpp.lib)
endif()

set(AUDIOCPP_BYPRODUCTS ${AUDIOCPP_LIB} ${AUDIOCPP_LINK_LIB})
list(REMOVE_DUPLICATES AUDIOCPP_BYPRODUCTS)

get_property(AUDIOCPP_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(AUDIOCPP_BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target audiocpp)
set(AUDIOCPP_CONFIG_ARGS -DCMAKE_BUILD_TYPE:STRING=${AUDIOCPP_BUILD_TYPE})
if(AUDIOCPP_MULTI_CONFIG)
    list(APPEND AUDIOCPP_BUILD_COMMAND --config $<CONFIG>)
    set(AUDIOCPP_CONFIG_ARGS)
    foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        list(APPEND AUDIOCPP_CONFIG_ARGS
            -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg}:PATH=${AUDIOCPP_LIB_DIR}
            -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg}:PATH=${AUDIOCPP_LIB_DIR}
            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg}:PATH=${AUDIOCPP_LIB_DIR})
    endforeach()
endif()
if(AGENT_AUDIOCPP_BUILD_JOBS)
    list(APPEND AUDIOCPP_BUILD_COMMAND --parallel ${AGENT_AUDIOCPP_BUILD_JOBS})
endif()

ExternalProject_Add(audiocpp-build
    SOURCE_DIR ${AUDIOCPP_SOURCE_DIR}
    BINARY_DIR ${AUDIOCPP_BINARY_DIR}
    CMAKE_CACHE_ARGS
        ${AUDIOCPP_CONFIG_ARGS}
        -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${AUDIOCPP_LIB_DIR}
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${AUDIOCPP_LIB_DIR}
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${AUDIOCPP_LIB_DIR}
        -DAUDIOCPP_BUILD_C_API:BOOL=ON
        -DAUDIOCPP_MODEL_SET:STRING=${AGENT_AUDIOCPP_MODEL_SET}
        -DAUDIOCPP_MODELS:STRING=${AGENT_AUDIOCPP_MODELS}
    BUILD_COMMAND ${AUDIOCPP_BUILD_COMMAND}
    BUILD_ALWAYS ON
    BUILD_BYPRODUCTS ${AUDIOCPP_BYPRODUCTS}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL ON
)

add_library(agent-engine-audiocpp SHARED
    audiocpp_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
add_dependencies(agent-engine-audiocpp audiocpp-build)
target_include_directories(agent-engine-audiocpp PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(agent-engine-audiocpp SYSTEM PRIVATE ${AUDIOCPP_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader)
target_compile_definitions(agent-engine-audiocpp PRIVATE AUDIOCPP_USE_DLL)
target_link_libraries(agent-engine-audiocpp PRIVATE ${AUDIOCPP_LINK_LIB} ${CMAKE_DL_LIBS})
set_target_properties(agent-engine-audiocpp PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-audiocpp PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(agent-engine-audiocpp PRIVATE -Wl,--version-script=${PROJECT_SOURCE_DIR}/src/engines/common/exports.map)
    set_property(TARGET agent-engine-audiocpp APPEND PROPERTY LINK_DEPENDS ${PROJECT_SOURCE_DIR}/src/engines/common/exports.map)
elseif(APPLE)
    target_link_options(agent-engine-audiocpp PRIVATE -Wl,-exported_symbol,_agent_engine_get_v1)
endif()

# libaudiocpp next to the plugin: in the build tree, and in the installed layout.
add_custom_command(TARGET agent-engine-audiocpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${AUDIOCPP_LIB_DIR} $<TARGET_FILE_DIR:agent-engine-audiocpp>
    COMMENT "Copying libaudiocpp next to agent-engine-audiocpp"
    VERBATIM
)
if(AGENT_INSTALL)
    install(TARGETS agent-engine-audiocpp LIBRARY DESTINATION ${AGENT_INSTALL_ENGINEDIR} RUNTIME DESTINATION ${AGENT_INSTALL_ENGINEDIR})
    # The library's file names (soname symlinks) only exist after the build, so they are looked up at install time.
    install(CODE "
        file(GLOB _audiocpp_libs \"${AUDIOCPP_LIB_DIR}/*audiocpp*${CMAKE_SHARED_LIBRARY_SUFFIX}*\")
        file(INSTALL \${_audiocpp_libs} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${AGENT_INSTALL_ENGINEDIR}\" FOLLOW_SYMLINK_CHAIN)
    ")
endif()
