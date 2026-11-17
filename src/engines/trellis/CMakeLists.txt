# agent-engine-trellis: TRELLIS.2 image to 3D through trellis.cpp (github.com/pwilkin/trellis.cpp).
#
# trellis.cpp publishes its builds as programs (trellis-cli) and no library, so the plugin is a small file that runs trellis-cli:
# nothing of trellis.cpp is compiled here. The installer (`agent engine pull trellis`) puts the release's trellis-cli and its
# libraries next to the plugin; the plugin finds them in its own folder (or at options.cli / $AGENT_TRELLIS_CLI).
add_library(agent-engine-trellis SHARED
    trellis_engine.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent-engine-trellis PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(agent-engine-trellis SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader)
target_link_libraries(agent-engine-trellis PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(agent-engine-trellis PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-trellis PRIVATE ${AGENT_WARNINGS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The version script keeps agent_engine_get_v1 as the only dynamic symbol (libstdc++ marks template instantiations default-visibility).
    target_link_options(agent-engine-trellis PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
    set_property(TARGET agent-engine-trellis APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
elseif(APPLE)
    target_link_options(agent-engine-trellis PRIVATE -Wl,-exported_symbol,_agent_engine_get_v1)
endif()
agent_install_engine(agent-engine-trellis)
