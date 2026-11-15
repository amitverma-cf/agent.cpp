# agent-engine-api: remote OpenAI-compatible chat through a bun child process (no libraries, no network code in C++).
# bridge.js is embedded into the plugin at configure time and written next to the other runtime files when the engine starts.
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/bridge.js _bridge_hex HEX)
string(LENGTH "${_bridge_hex}" _bridge_hex_len)
math(EXPR _bridge_size "${_bridge_hex_len} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bridge_bytes "${_bridge_hex}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bridge_embedded.cpp
    "// Generated from bridge.js by CMake.\nextern \"C\" {\nextern const unsigned char agent_api_bridge_js[] = {${_bridge_bytes}0};\nextern const unsigned long agent_api_bridge_js_size = ${_bridge_size};\n}\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bridge.js)

add_library(agent-engine-api SHARED
    api_engine.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/bridge_embedded.cpp
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp
)
target_include_directories(agent-engine-api PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader
)
set_target_properties(agent-engine-api PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_source_files_properties(${PROJECT_SOURCE_DIR}/vendor/simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_OPTIONS "-w")
target_compile_options(agent-engine-api PRIVATE ${AGENT_WARNINGS})
agent_install_engine(agent-engine-api)
