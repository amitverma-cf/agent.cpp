#include "runtimes.hpp"

#include <catch_amalgamated.hpp>

#include <filesystem>
#include <fstream>

using namespace agent::cli;
namespace fs = std::filesystem;

TEST_CASE("runtimes: release assets for each platform", "[runtimes]") {
    auto bun = runtime_asset("bun", "latest", "linux", "x86_64");
    REQUIRE(bun.ok);
    CHECK(bun.value.url == "https://github.com/oven-sh/bun/releases/latest/download/bun-linux-x64.zip");
    CHECK(bun.value.checksum_url == "https://github.com/oven-sh/bun/releases/latest/download/SHASUMS256.txt");
    CHECK(runtime_asset("bun", "latest", "macos", "aarch64").value.archive_name == "bun-darwin-aarch64.zip");
    CHECK(runtime_asset("bun", "1.2.3", "windows", "x86_64").value.url == "https://github.com/oven-sh/bun/releases/download/bun-v1.2.3/bun-windows-x64.zip");
    CHECK_FALSE(runtime_asset("bun", "latest", "windows", "aarch64").ok);

    auto uv = runtime_asset("uv", "latest", "linux", "aarch64");
    REQUIRE(uv.ok);
    CHECK(uv.value.archive_name == "uv-aarch64-unknown-linux-gnu.tar.gz");
    CHECK(uv.value.checksum_url == uv.value.url + ".sha256");
    CHECK(runtime_asset("uv", "0.5.1", "windows", "x86_64").value.url == "https://github.com/astral-sh/uv/releases/download/0.5.1/uv-x86_64-pc-windows-msvc.zip");
    CHECK(runtime_asset("uv", "", "macos", "x86_64").value.archive_name == "uv-x86_64-apple-darwin.tar.gz");

    CHECK_FALSE(runtime_asset("node", "latest", "linux", "x86_64").ok);
    CHECK_FALSE(runtime_asset("uv", "latest", "plan9", "x86_64").ok);
    CHECK_FALSE(runtime_asset("uv", "latest", "linux", "riscv").ok);
}

TEST_CASE("runtimes: reading a checksum file", "[runtimes]") {
    const std::string a(64, 'a'), b(64, 'b');
    const std::string sums = a + "  bun-linux-x64.zip\n" + b + " *bun-darwin-x64.zip\n";
    CHECK(find_sha256(sums, "bun-linux-x64.zip") == a);
    CHECK(find_sha256(sums, "bun-darwin-x64.zip") == b);
    CHECK(find_sha256(sums, "bun-windows-x64.zip").empty());
    CHECK(find_sha256(b + " *uv-x86_64-unknown-linux-gnu.tar.gz\r\n", "uv-x86_64-unknown-linux-gnu.tar.gz") == b);
    CHECK(find_sha256(a + "\n", "anything.tar.gz") == a); // a file that holds just the hash
    CHECK(find_sha256("not a checksum file", "x.zip").empty());
}

TEST_CASE("runtimes: npx / uvx run on the private runtimes, caches stay in the home", "[runtimes]") {
    const fs::path root = fs::temp_directory_path() / "agent_test_runtimes";
    std::error_code ec;
    fs::remove_all(root, ec);
    HomeLayout home = home_layout(root.string());
    REQUIRE(create_home_layout(home).ok);

    agent::McpServerConfig npx;
    npx.name = "github";
    npx.command = "npx";
    npx.args = {"-y", "@modelcontextprotocol/server-github", "--flag"};

    // Not installed yet: the command is left alone and the note says what to run.
    agent::McpServerConfig untouched = npx;
    auto missing = rewrite_launcher(untouched, home);
    CHECK_FALSE(missing.rewritten);
    CHECK(missing.note.find("agent runtime install bun") != std::string::npos);
    CHECK(untouched.command == "npx");

    // Pretend both runtimes are installed.
    for (const char *name : {"bun", "uv"}) {
        fs::create_directories(fs::path(home.runtimes) / name);
#ifdef _WIN32
        std::ofstream(fs::path(home.runtimes) / name / (std::string(name) + ".exe")) << "x";
#else
        std::ofstream(fs::path(home.runtimes) / name / name) << "x";
#endif
    }

    auto r = rewrite_launcher(npx, home);
    CHECK(r.rewritten);
    CHECK(fs::path(npx.command).stem().string() == "bun");
    REQUIRE(npx.args.size() == 3);
    CHECK(npx.args[0] == "x");
    CHECK(npx.args[1] == "@modelcontextprotocol/server-github"); // -y dropped
    CHECK(npx.args[2] == "--flag");
    CHECK(npx.env.at("BUN_INSTALL_CACHE_DIR").starts_with(home.cache));
    CHECK(npx.env.at("UV_CACHE_DIR").starts_with(home.cache));

    agent::McpServerConfig uvx;
    uvx.name = "fetch";
    uvx.command = "uvx";
    uvx.args = {"mcp-server-fetch"};
    uvx.env["UV_CACHE_DIR"] = "/mine"; // an explicit value wins
    CHECK(rewrite_launcher(uvx, home).rewritten);
    CHECK(fs::path(uvx.command).stem().string() == "uv");
    CHECK(uvx.args == std::vector<std::string>{"tool", "run", "mcp-server-fetch"});
    CHECK(uvx.env.at("UV_CACHE_DIR") == "/mine");
    CHECK(uvx.env.count("UV_TOOL_DIR") == 1);

    agent::McpServerConfig other;
    other.name = "x";
    other.command = "python3";
    CHECK_FALSE(rewrite_launcher(other, home).rewritten);
    CHECK(other.command == "python3");

    CHECK(remove_runtime(home, "bun").ok);
    CHECK_FALSE(remove_runtime(home, "bun").ok);
    CHECK_FALSE(remove_runtime(home, "node").ok);
    fs::remove_all(root, ec);
}
