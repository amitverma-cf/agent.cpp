# Pinned ONNX Runtime release, fetched from the upstream GitHub release (nothing is built from source).
#
# Sets ONNXRUNTIME_ROOT to the extracted release (include/ + lib/) for this platform. Leaves it unset, with a status message,
# when upstream publishes no archive for the platform (build ONNX Runtime yourself and pass -DONNXRUNTIME_ROOT=...).
#
# Updating: change AGENT_ORT_VERSION and the hashes below to a newer release. GitHub shows each asset's sha256 digest on the
# release page (and in the API: assets[].digest). `scripts/update-deps.sh ort` prints them.
include_guard(GLOBAL)
include(FetchContent)

set(AGENT_ORT_VERSION "1.31.1")
set(_ort_base "https://github.com/microsoft/onnxruntime/releases/download/v${AGENT_ORT_VERSION}")

set(_ort_asset "")
set(_ort_sha256 "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_ort_asset "onnxruntime-linux-x64-${AGENT_ORT_VERSION}.tgz")
    set(_ort_sha256 "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_ort_asset "onnxruntime-linux-aarch64-${AGENT_ORT_VERSION}.tgz")
    set(_ort_sha256 "5317505e2da99be65f8e2775ba3c2afd6d2961fd78aba1bd8f56159f48820c78")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(_ort_asset "onnxruntime-osx-arm64-${AGENT_ORT_VERSION}.tgz")
    set(_ort_sha256 "2e1344a1926ee50eb25b38153cb6992c3567515aef0e677a1a520a1798896d6f")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$")
    set(_ort_asset "onnxruntime-win-x64-${AGENT_ORT_VERSION}.zip")
    set(_ort_sha256 "5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64)$")
    set(_ort_asset "onnxruntime-win-arm64-${AGENT_ORT_VERSION}.zip")
    set(_ort_sha256 "817f4129073f84f81cde3ff92b1a4e06d007691e7208deed08a8b6d6df143f7f")
endif()

if(_ort_asset)
    FetchContent_Declare(agent_onnxruntime_release
        URL "${_ort_base}/${_ort_asset}"
        URL_HASH "SHA256=${_ort_sha256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(agent_onnxruntime_release)
    set(ONNXRUNTIME_ROOT "${agent_onnxruntime_release_SOURCE_DIR}" CACHE PATH "Prebuilt ONNX Runtime directory (contains include/ and lib/)" FORCE)
    message(STATUS "ONNX Runtime ${AGENT_ORT_VERSION}: ${_ort_asset} (upstream release, sha256 checked)")
else()
    message(STATUS "ONNX Runtime: upstream publishes no release archive for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}; pass -DONNXRUNTIME_ROOT=<dir>")
endif()
