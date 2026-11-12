#include "engine_release.hpp"

#include <catch_amalgamated.hpp>

using namespace agent::cli;

TEST_CASE("engine release: asset names and URLs", "[engine_release]") {
    CHECK(engine_asset_name("llama", "linux", "x86_64") == "agent-engine-llama-linux-x86_64.so");
    CHECK(engine_asset_name("llama", "macos", "aarch64") == "agent-engine-llama-macos-aarch64.dylib");
    CHECK(engine_asset_name("sdcpp", "windows", "x86_64") == "agent-engine-sdcpp-windows-x86_64.dll");
    CHECK(release_manifest_url("o/r", "latest") == "https://github.com/o/r/releases/latest/download/manifest.json");
    CHECK(release_asset_url("o/r", "v0.2.0", "a.so") == "https://github.com/o/r/releases/download/v0.2.0/a.so");
}

TEST_CASE("engine release: reading the manifest", "[engine_release]") {
    const std::string a(64, 'a'), b(64, 'b');
    const std::string json = R"({"version":"v0.2.0","repository":"o/r","files":[)"
                             "{\"name\":\"agent-engine-llama-linux-x86_64.so\",\"sha256\":\"" +
                             a +
                             "\",\"size\":1234},"
                             "{\"name\":\"bad-hash.so\",\"sha256\":\"short\",\"size\":1},"
                             "{\"name\":\"agent-cpp-v0.2.0-windows-x86_64.zip\",\"sha256\":\"" +
                             b + "\"}]}";
    auto m = parse_release_manifest(json);
    REQUIRE(m.ok);
    CHECK(m.value.version == "v0.2.0");
    CHECK(m.value.repository == "o/r");
    REQUIRE(m.value.files.size() == 2); // the entry with a malformed hash is dropped
    CHECK(m.value.files[0].size == 1234);
    CHECK(m.value.files[1].sha256 == b);
    CHECK_FALSE(parse_release_manifest("[]").ok);
    CHECK_FALSE(parse_release_manifest("nope").ok);
    CHECK_FALSE(pull_engine("x", "../evil", "o/r", "latest", {}).ok);
}
