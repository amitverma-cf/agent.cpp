# Overlay port of the agent.cpp library: the core (agent-cpp::agent), the SQLite store (agent-cpp::store_sqlite) and, with the "hub"
# feature, the downloader (agent-cpp::hub). Engines, the CLI, the tests and the examples are not built. REF is the commit the port
# builds. Bump it with scripts/update-vcpkg-port.sh [COMMIT]: it writes REF and both SHA512 values (the simdjson commit is the one
# vendor/simdjson points at in that agent.cpp commit).
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO amitverma-cf/agent.cpp
    REF 19186a09318a0389610f337ca45a0c2c4f33941f
    SHA512 4d126593c4029fc222d378e956c7f8c6e644fe59ff725903be9726e73148a5a2dd70c2283218aa4675176a62c54a9ed7bb15d6af20d888862770cc4386de18db
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
