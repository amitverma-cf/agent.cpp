# Overlay port of the agent.cpp library: the core (agent-cpp::agent), the SQLite store (agent-cpp::store_sqlite) and, with the "hub"
# feature, the downloader (agent-cpp::hub). Engines, the CLI, the tests and the examples are not built. REF is the commit the port
# builds; bump REF and both SHA512 values together (the simdjson commit is the one vendor/simdjson points at in that agent.cpp commit).
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO amitverma-cf/agent.cpp
    REF 63f86d79e0a19fcdef6ba368087444fb25f074de
    SHA512 7f0570156f91d2573f7e5be8c9ea087715bd4952b4cbb461ae3fd69d76746105cf016dcc287ac86e37277f327f27df2e2640ec82fb488a69640c13ac5c40084b
    HEAD_REF main
)

# vendor/simdjson is a git submodule, which a GitHub archive does not contain.
vcpkg_from_github(
    OUT_SOURCE_PATH SIMDJSON_SOURCE_PATH
    REPO simdjson/simdjson
    REF 9b33047a878264250c5361f865d0b2da86217d14
    SHA512 0e91c68a3e24fe510e35594918401180643fe9ef7d45659e1ce2b68f40dfa605ce35569441762b29b22aa957192316834c7be7419b59d75705993d8bee317b36
)
file(REMOVE_RECURSE "${SOURCE_PATH}/vendor/simdjson")
file(COPY "${SIMDJSON_SOURCE_PATH}/" DESTINATION "${SOURCE_PATH}/vendor/simdjson")

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        hub AGENT_BUILD_HUB
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DAGENT_INSTALL=ON
        -DAGENT_BUILD_CLI=OFF
        -DAGENT_BUILD_TESTS=OFF
        -DAGENT_BUILD_EXAMPLES=OFF
        -DAGENT_ENGINE_LLAMA=OFF
        -DAGENT_ENGINE_ONNX=OFF
        -DAGENT_ENGINE_SDCPP=OFF
        -DAGENT_ENGINE_AUDIOCPP=OFF
        -DAGENT_ENGINE_TRELLIS=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME agent-cpp CONFIG_PATH lib/cmake/agent-cpp)

# The engine host is a program: it goes to tools/, not bin/.
if(EXISTS "${CURRENT_PACKAGES_DIR}/bin/agent-engine-host${VCPKG_TARGET_EXECUTABLE_SUFFIX}")
    vcpkg_copy_tools(TOOL_NAMES agent-engine-host AUTO_CLEAN)
endif()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
    "${CURRENT_PACKAGES_DIR}/share/doc"
)
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
