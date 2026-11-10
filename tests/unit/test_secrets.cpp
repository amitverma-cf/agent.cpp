#include "secrets.hpp"

#include <catch_amalgamated.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>

namespace fs = std::filesystem;
using namespace agent::cli;

namespace {

fs::path temp_dir(const char *name) {
    const fs::path p = fs::temp_directory_path() / (std::string("agent-secrets-test-") + name);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p, ec);
    return p;
}

// An in-memory store: lets the access and placeholder rules be tested without touching the OS credential store.
struct MemoryStore {
    std::map<std::string, std::string> values;
};
bool mem_available(void *) { return true; }
agent::Result<void> mem_set(void *s, std::string_view n, std::string_view v) {
    static_cast<MemoryStore *>(s)->values[std::string(n)] = std::string(v);
    return agent::ok();
}
agent::Result<std::string> mem_get(void *s, std::string_view n) {
    auto &m = static_cast<MemoryStore *>(s)->values;
    auto it = m.find(std::string(n));
    if (it == m.end()) return agent::fail<std::string>(agent::ErrorCode::KeyNotFound, "missing");
    return agent::ok(it->second);
}
agent::Result<void> mem_remove(void *s, std::string_view n) {
    static_cast<MemoryStore *>(s)->values.erase(std::string(n));
    return agent::ok();
}

SecretBackend memory_backend(MemoryStore &store) { return SecretBackend{"memory", &store, mem_available, mem_set, mem_get, mem_remove}; }

} // namespace

TEST_CASE("secrets: names are restricted to safe characters", "[secrets]") {
    CHECK(valid_secret_name("github"));
    CHECK(valid_secret_name("GITHUB_TOKEN-2.prod"));
    CHECK_FALSE(valid_secret_name(""));
    CHECK_FALSE(valid_secret_name("a b"));
    CHECK_FALSE(valid_secret_name("a;rm -rf"));
    CHECK_FALSE(valid_secret_name("a/b"));
    CHECK_FALSE(valid_secret_name(std::string(65, 'x')));
}

TEST_CASE("secrets: the encrypted file keeps values, survives a restart, and rejects a wrong passphrase", "[secrets]") {
    const fs::path dir = temp_dir("file");
    const std::string path = (dir / "secrets.enc").string();
#if defined(_WIN32)
    _putenv_s("AGENT_SECRETS_PASSPHRASE", "correct horse");
#else
    setenv("AGENT_SECRETS_PASSPHRASE", "correct horse", 1);
#endif
    {
        const SecretBackend b = encrypted_file_backend(path);
        REQUIRE(b.set(b.state, "github", "ghp_supersecretvalue").ok);
        REQUIRE(b.set(b.state, "openai", "sk-another").ok);
        auto got = b.get(b.state, "github");
        REQUIRE(got.ok);
        CHECK(got.value == "ghp_supersecretvalue");
        REQUIRE(b.remove(b.state, "openai").ok);
        CHECK_FALSE(b.get(b.state, "openai").ok);
    }
    // The values are not readable in the file.
    {
        std::ifstream in(path, std::ios::binary);
        std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(raw.find("ghp_supersecretvalue") == std::string::npos);
        CHECK(raw.rfind("AGSEC1", 0) == 0);
    }
#if defined(_WIN32)
    _putenv_s("AGENT_SECRETS_PASSPHRASE", "wrong");
#else
    setenv("AGENT_SECRETS_PASSPHRASE", "wrong", 1);
#endif
    // A backend for the same file keeps the passphrase it first saw, so use a new path copy to force a re-read.
    const std::string copy = (dir / "copy.enc").string();
    fs::copy_file(path, copy);
    const SecretBackend wrong = encrypted_file_backend(copy);
    auto denied = wrong.get(wrong.state, "github");
    CHECK_FALSE(denied.ok);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("secrets: the index stores names and bindings only", "[secrets]") {
    const fs::path dir = temp_dir("index");
    const std::string path = (dir / "secrets.json").string();
    CHECK(load_secret_index(path).value.empty()); // no file yet

    std::vector<SecretEntry> index = {{"github", {{"mcp:github", "abc"}, {"tool:search", "def"}}}, {"plain", {}}};
    REQUIRE(save_secret_index(path, index).ok);
    auto back = load_secret_index(path);
    REQUIRE(back.ok);
    REQUIRE(back.value.size() == 2);
    CHECK(back.value[0].name == "github");
    REQUIRE(back.value[0].bindings.size() == 2);
    CHECK(back.value[0].bindings[1].consumer == "tool:search");
    CHECK(back.value[1].bindings.empty());
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("secrets: a secret reaches only the consumer it is bound to, and only while that consumer is unchanged", "[secrets]") {
    const std::string id = consumer_identity({"npx", "-y", "@modelcontextprotocol/server-github"});
    const std::string other = consumer_identity({"npx", "-y", "evil-package"});
    CHECK(id != other);
    CHECK(consumer_identity({"a", "bc"}) != consumer_identity({"ab", "c"})); // parts cannot be shifted across a boundary

    const std::vector<SecretEntry> index = {{"github", {{"mcp:github", id}}}};
    CHECK(check_secret_access(index, "github", "mcp:github", id) == SecretAccess::Allowed);
    CHECK(check_secret_access(index, "github", "mcp:other", id) == SecretAccess::NotBound);          // another server
    CHECK(check_secret_access(index, "github", "mcp:github", other) == SecretAccess::IdentityChanged); // same name, different command
    CHECK(check_secret_access(index, "missing", "mcp:github", id) == SecretAccess::UnknownSecret);
}

TEST_CASE("secrets: placeholders are expanded for an allowed consumer and refused otherwise", "[secrets]") {
    MemoryStore store;
    store.values["github"] = "ghp_value";
    store.values["other"] = "zzz";
    const SecretBackend backend = memory_backend(store);
    const std::vector<SecretEntry> index = {{"github", {{"mcp:github", "id1"}}}, {"other", {}}};

    auto ok = resolve_secret_placeholders("Bearer ${secret:github}!", backend, index, "mcp:github", "id1");
    REQUIRE(ok.ok);
    CHECK(ok.value == "Bearer ghp_value!");

    CHECK(resolve_secret_placeholders("no placeholders", backend, index, "mcp:github", "id1").value == "no placeholders");

    auto not_bound = resolve_secret_placeholders("${secret:other}", backend, index, "mcp:github", "id1"); // exists, bound to nobody
    CHECK_FALSE(not_bound.ok);
    CHECK(not_bound.value.empty());
    auto changed = resolve_secret_placeholders("${secret:github}", backend, index, "mcp:github", "different");
    CHECK_FALSE(changed.ok);
    CHECK(changed.error.message.find("approve it again") != std::string::npos);
    CHECK_FALSE(resolve_secret_placeholders("${secret:nope}", backend, index, "mcp:github", "id1").ok);
    CHECK_FALSE(resolve_secret_placeholders("${secret:github", backend, index, "mcp:github", "id1").ok);
}

TEST_CASE("secrets: known values are scrubbed from output", "[secrets]") {
    CHECK(redact_secrets("token is ghp_value and again ghp_value.", {"ghp_value"}) == "token is *** and again ***.");
    CHECK(redact_secrets("nothing here", {"ghp_value"}) == "nothing here");
    CHECK(redact_secrets("abc", {"ab"}) == "abc"); // too short to mask without damaging ordinary text
}

#if defined(_WIN32) || defined(__APPLE__)
TEST_CASE("secrets: the OS credential store round-trips a value", "[secrets][keychain]") {
    // Touches the real store of whoever runs the tests, so it is opt-in.
    if (!std::getenv("AGENT_TEST_OS_KEYCHAIN")) SKIP("set AGENT_TEST_OS_KEYCHAIN=1 to test the OS credential store");
    const SecretBackend &b = default_secret_backend((fs::temp_directory_path() / "agent-secrets-os").string());
    const std::string name = "agent-cpp-test-secret";
    REQUIRE(b.set(b.state, name, "value-123").ok);
    auto got = b.get(b.state, name);
    REQUIRE(got.ok);
    CHECK(got.value == "value-123");
    REQUIRE(b.remove(b.state, name).ok);
    CHECK_FALSE(b.get(b.state, name).ok);
}
#endif
