#include "mcp_config.hpp"
#include "secrets.hpp"

#include <catch_amalgamated.hpp>

#include <filesystem>
#include <fstream>

using namespace agent::cli;
namespace fs = std::filesystem;

namespace {

struct Mem {
    std::string value = "ghp_value";
};
bool mem_available(void *) { return true; }
agent::Result<void> mem_set(void *, std::string_view, std::string_view) { return agent::ok(); }
agent::Result<std::string> mem_get(void *s, std::string_view) { return agent::ok(static_cast<Mem *>(s)->value); }
agent::Result<void> mem_remove(void *, std::string_view) { return agent::ok(); }

} // namespace

TEST_CASE("mcp config: the common mcpServers shape is read, with keys left as placeholders", "[mcpcfg]") {
    auto f = parse_mcp_config(R"({"mcpServers":{
        "github":{"command":"npx","args":["-y","@modelcontextprotocol/server-github"],"env":{"GITHUB_TOKEN":"${secret:github}","LOG":"1"},"timeout_ms":9000},
        "bad name":{"command":"x"},
        "nocmd":{"args":[]},
        "locked":{"command":"srv","sandbox":{"network":"deny","read":["/data"],"write":["/out"]}}}})",
                              "test.json");
    REQUIRE(f.ok);
    REQUIRE(f.value.servers.size() == 2);
    const auto &gh = f.value.servers[0];
    CHECK(gh.name == "github");
    CHECK(gh.command == "npx");
    REQUIRE(gh.args.size() == 2);
    CHECK(gh.env.at("GITHUB_TOKEN") == "${secret:github}"); // not resolved here
    CHECK(gh.timeout_ms == 9000);
    CHECK(f.value.warnings.size() == 2); // the bad name and the missing command
    REQUIRE(f.value.own_sandbox.size() == 1);
    CHECK(f.value.own_sandbox[0] == "locked");
    const auto &locked = f.value.servers[1];
    CHECK(locked.sandbox.enabled);
    CHECK(locked.sandbox.network == agent::SandboxNetwork::Deny);
    CHECK(locked.sandbox.read_write_paths == std::vector<std::string>{"/out"});
}

TEST_CASE("mcp config: input that is not an mcpServers file is an error", "[mcpcfg]") {
    CHECK_FALSE(parse_mcp_config("not json", "x").ok);
    CHECK_FALSE(parse_mcp_config(R"({"other":1})", "x").ok);
    auto empty = parse_mcp_config(R"({"mcpServers":{}})", "x");
    REQUIRE(empty.ok);
    CHECK(empty.value.servers.empty());
}

TEST_CASE("mcp config: every server gets its own data folder and a scoped sandbox", "[mcpcfg]") {
    agent::McpServerConfig s;
    s.name = "github";
    s.command = "srv";
    apply_mcp_defaults(s, "/h/plugins/github/data", "/h/runtimes", "/h/cache", false);
    CHECK(s.working_directory == "/h/plugins/github/data");
    CHECK(s.sandbox.enabled);
    CHECK(s.sandbox.network == agent::SandboxNetwork::Allow);
    CHECK(s.sandbox.read_write_paths == std::vector<std::string>{"/h/plugins/github/data", "/h/cache"}); // not the whole home

    agent::McpServerConfig own;
    own.name = "x";
    own.command = "srv";
    own.sandbox.enabled = false;
    apply_mcp_defaults(own, "/h/plugins/x/data", "/h/runtimes", "/h/cache", true);
    CHECK_FALSE(own.sandbox.enabled); // an explicit sandbox object is respected
    CHECK(own.working_directory == "/h/plugins/x/data");
}

TEST_CASE("mcp config: a project server replaces a global one of the same name", "[mcpcfg]") {
    const fs::path root = fs::temp_directory_path() / "agent-mcpcfg-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    Context ctx;
    ctx.has_home = true;
    ctx.home = home_layout((root / "home").string());
    REQUIRE(create_home_layout(ctx.home).ok);
    auto project = init_project((root / "proj").string());
    REQUIRE(project.ok);
    ctx.project_root = project.value.root;
    {
        std::ofstream(fs::path(ctx.home.mcp) / "mcp.json") << R"({"mcpServers":{"a":{"command":"global-a"},"b":{"command":"global-b"}}})";
        std::ofstream(project.value.mcp_config) << R"({"mcpServers":{"a":{"command":"project-a"}}})";
    }

    std::vector<std::string> warnings;
    const auto servers = collect_mcp_servers(ctx, warnings);
    REQUIRE(servers.size() == 2);
    CHECK(warnings.empty());
    for (const auto &s : servers) {
        if (s.name == "a") CHECK(s.command == "project-a");
        if (s.name == "b") CHECK(s.command == "global-b");
        CHECK(fs::is_directory(s.working_directory)); // the data folder exists
    }
    fs::remove_all(root, ec);
}

TEST_CASE("mcp secrets: the CLI resolver pins a key to the server's command line", "[mcpcfg][secrets]") {
    const std::vector<std::string> parts = {"npx", "-y", "server-github"};
    std::vector<std::string> inserted;
    Mem mem;
    SecretContext ctx;
    ctx.backend = SecretBackend{"memory", &mem, mem_available, mem_set, mem_get, mem_remove};
    ctx.index = {{"github", {{"mcp:github", consumer_identity(parts)}}}};

    const agent::SecretRequest ok_who{.consumer = "mcp:github", .identity_parts = parts};
    auto good = cli_resolve_secrets("Bearer ${secret:github}", ok_who, inserted, &ctx);
    REQUIRE(good.ok);
    CHECK(good.value == "Bearer ghp_value");
    REQUIRE(inserted.size() == 1);
    CHECK(inserted[0] == "ghp_value");

    // The same server name running another command is refused, and so is another server.
    const std::vector<std::string> changed = {"npx", "-y", "evil"};
    const agent::SecretRequest changed_who{.consumer = "mcp:github", .identity_parts = changed};
    CHECK_FALSE(cli_resolve_secrets("${secret:github}", changed_who, inserted, &ctx).ok);
    const agent::SecretRequest other_who{.consumer = "mcp:other", .identity_parts = parts};
    CHECK_FALSE(cli_resolve_secrets("${secret:github}", other_who, inserted, &ctx).ok);
}
