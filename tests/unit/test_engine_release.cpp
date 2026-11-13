#include "engine_release.hpp"
#include "release_verify.hpp"

#include <catch_amalgamated.hpp>
#include <openssl/evp.h>

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

TEST_CASE("update: the manifest signature", "[update][signature]") {
    using namespace agent::cli;
    // A key pair made here: sign a manifest the way scripts/sign-release.sh does, then check what verify_ed25519 accepts.
    EVP_PKEY_CTX *kctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    REQUIRE(kctx);
    REQUIRE(EVP_PKEY_keygen_init(kctx) == 1);
    EVP_PKEY *pkey = nullptr;
    REQUIRE(EVP_PKEY_keygen(kctx, &pkey) == 1);
    EVP_PKEY_CTX_free(kctx);
    unsigned char raw[32];
    size_t raw_len = sizeof(raw);
    REQUIRE(EVP_PKEY_get_raw_public_key(pkey, raw, &raw_len) == 1);
    const std::string manifest = "{\"version\":\"v1.0.0\",\"files\":[]}";
    EVP_MD_CTX *mctx = EVP_MD_CTX_new();
    REQUIRE(EVP_DigestSignInit(mctx, nullptr, nullptr, nullptr, pkey) == 1);
    unsigned char sig[64];
    size_t sig_len = sizeof(sig);
    REQUIRE(EVP_DigestSign(mctx, sig, &sig_len, reinterpret_cast<const unsigned char *>(manifest.data()), manifest.size()) == 1);
    EVP_MD_CTX_free(mctx);
    EVP_PKEY_free(pkey);
    const auto b64 = [](const unsigned char *data, size_t n) {
        static const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        for (size_t i = 0; i < n; i += 3) {
            const unsigned v = (data[i] << 16) | (i + 1 < n ? data[i + 1] << 8 : 0) | (i + 2 < n ? data[i + 2] : 0);
            out += alphabet[(v >> 18) & 63];
            out += alphabet[(v >> 12) & 63];
            out += i + 1 < n ? alphabet[(v >> 6) & 63] : '=';
            out += i + 2 < n ? alphabet[v & 63] : '=';
        }
        return out;
    };
    const std::string public_key = b64(raw, raw_len), signature = b64(sig, sig_len);

    CHECK(verify_ed25519(manifest, signature, public_key));
    CHECK_FALSE(verify_ed25519(manifest + " ", signature, public_key));  // the manifest was changed
    CHECK_FALSE(verify_ed25519(manifest, signature, kReleasePublicKey)); // another key
    CHECK_FALSE(verify_ed25519(manifest, "AAAA", public_key));           // not a signature
    CHECK_FALSE(verify_ed25519(manifest, "!!!!", public_key));
    std::string bytes;
    CHECK(base64_to_bytes("aGVsbG8=", bytes));
    CHECK(bytes == "hello");
    CHECK(base64_to_bytes("aGVs\nbG8", bytes));
    CHECK(bytes == "hello");
}

TEST_CASE("update: versions and the CLI archive name", "[update]") {
    using namespace agent::cli;
    CHECK(version_is_newer("v0.2.0", "0.1.9"));
    CHECK(version_is_newer("0.10.0", "0.9.0"));
    CHECK(version_is_newer("1.0.0", "0.9.9"));
    CHECK_FALSE(version_is_newer("0.2.0", "v0.2.0"));
    CHECK_FALSE(version_is_newer("0.1.0", "0.2.0"));
    CHECK_FALSE(version_is_newer("build-12", "0.1.0")); // not a dotted number: never newer
    CHECK_FALSE(version_is_newer("0.2.0", "dev"));
    CHECK(cli_archive_name("v0.2.0", "linux", "x86_64") == "agent-cpp-v0.2.0-linux-x86_64.tar.gz");
    CHECK(cli_archive_name("v0.2.0", "windows", "aarch64") == "agent-cpp-v0.2.0-windows-aarch64.zip");
}
