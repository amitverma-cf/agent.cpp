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
    CHECK(runtime_asset("bun", "1.2.3", "windows", "x86_64").value.url ==
          "https://github.com/oven-sh/bun/releases/download/bun-v1.2.3/bun-windows-x64.zip");
    CHECK_FALSE(runtime_asset("bun", "latest", "windows", "aarch64").ok);

    auto uv = runtime_asset("uv", "latest", "linux", "aarch64");
    REQUIRE(uv.ok);
    CHECK(uv.value.archive_name == "uv-aarch64-unknown-linux-gnu.tar.gz");
    CHECK(uv.value.checksum_url == uv.value.url + ".sha256");
    CHECK(runtime_asset("uv", "0.5.1", "windows", "x86_64").value.url ==
          "https://github.com/astral-sh/uv/releases/download/0.5.1/uv-x86_64-pc-windows-msvc.zip");
    CHECK(runtime_asset("uv", "", "macos", "x86_64").value.archive_name == "uv-x86_64-apple-darwin.tar.gz");

    CHECK_FALSE(runtime_asset("deno", "latest", "linux", "x86_64").ok);
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
    agent::McpServerConfig npm;
    npm.name = "n";
    npm.command = "npm";
    npm.args = {"i", "left-pad"};
    npm.working_directory = "/srv/own";
    CHECK(rewrite_launcher(npm, home).rewritten);
    CHECK(npm.args == std::vector<std::string>{"install", "left-pad"});
    CHECK(fs::path(npm.env.at("BUN_INSTALL_CACHE_DIR")).generic_string().starts_with("/srv/own/cache"));
    CHECK(npm.env.at("HOME").find("cache") != std::string::npos); // HOME and temp are contained too
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

TEST_CASE("runtimes: node (with pnpm) and nub assets", "[runtimes]") {
    const std::string sums = std::string(64, 'a') + "  node-v24.22.2-darwin-arm64.tar.gz\n" + std::string(64, 'b') +
                             "  node-v24.22.2-linux-x64.tar.gz\n" + std::string(64, 'c') + "  node-v24.22.2-win-x64.zip\n" +
                             std::string(64, 'd') + "  node-v24.22.2-linux-x64.tar.xz\n";
    CHECK(find_node_archive(sums, "linux", "x86_64") == "node-v24.22.2-linux-x64.tar.gz");
    CHECK(find_node_archive(sums, "windows", "x86_64") == "node-v24.22.2-win-x64.zip");
    CHECK(find_node_archive(sums, "macos", "aarch64") == "node-v24.22.2-darwin-arm64.tar.gz");
    CHECK(find_node_archive(sums, "linux", "aarch64").empty());

    auto node = runtime_asset("node", "latest", "linux", "x86_64");
    REQUIRE(node.ok);
    CHECK(node.value.checksum_url == "https://nodejs.org/dist/latest-v24.x/SHASUMS256.txt");
    CHECK(runtime_asset("node", "v22.1.0", "linux", "x86_64").value.url == "https://nodejs.org/dist/v22.1.0/");
    CHECK(runtime_asset("nub", "latest", "linux", "x86_64").value.archive_name == "nub-linux-x64.tar.gz");
    CHECK(runtime_asset("nub", "latest", "windows", "x86_64").value.archive_name == "nub-win32-x64.zip");
    CHECK(runtime_asset("nub", "0.10.0", "macos", "aarch64").value.url ==
          "https://github.com/nubjs/nub/releases/download/v0.10.0/nub-darwin-arm64.tar.gz");
    CHECK(runtimes_for_js("node") == std::vector<std::string>{"node", "pnpm"});
    CHECK(runtimes_for_js("nub") == std::vector<std::string>{"nub"});
    CHECK(runtimes_for_js("bun") == std::vector<std::string>{"bun"});
    CHECK(is_runtime_name("pnpm"));
    CHECK_FALSE(is_runtime_name("npm"));
}

TEST_CASE("runtimes: the JavaScript choice decides what npx / npm / node become", "[runtimes]") {
    const fs::path root = fs::temp_directory_path() / "agent_test_runtimes_js";
    std::error_code ec;
    fs::remove_all(root, ec);
    HomeLayout home = home_layout(root.string());
    REQUIRE(create_home_layout(home).ok);
    auto touch = [&](const std::string &rel) {
        fs::create_directories((fs::path(home.runtimes) / rel).parent_path());
        std::ofstream(fs::path(home.runtimes) / rel) << "x";
    };
    auto server = [&](const char *command, std::vector<std::string> args) {
        agent::McpServerConfig s;
        s.name = "s";
        s.command = command;
        s.args = std::move(args);
        s.working_directory = (root / "data").string();
        return s;
    };

    // Nothing installed: left alone, with the command to run.
    auto missing = server("npx", {"-y", "pkg"});
    auto note = rewrite_launcher(missing, home, "node");
    CHECK_FALSE(note.rewritten);
    CHECK(note.note.find("agent runtime install node") != std::string::npos);

    touch("node/" + std::string(
#ifdef _WIN32
                        "node.exe"
#else
                        "node"
#endif
                        ));
    touch("pnpm/bin/pnpm.cjs");
    touch("nub/" + std::string(
#ifdef _WIN32
                       "nub.exe"
#else
                       "nub"
#endif
                       ));

    auto npx = server("npx", {"-y", "pkg", "--flag"});
    REQUIRE(rewrite_launcher(npx, home, "node").rewritten);
    CHECK(fs::path(npx.command).stem().string() == "node");
    REQUIRE(npx.args.size() == 4);
    CHECK(fs::path(npx.args[0]).filename().string() == "pnpm.cjs");
    CHECK(npx.args[1] == "dlx");
    CHECK(npx.args[2] == "pkg");                                                 // -y dropped
    CHECK(npx.env.at("npm_config_store_dir").find("data") != std::string::npos); // inside the server's folder
    CHECK_FALSE(rewrite_launcher(npx, home, "node").rewritten);                  // a second pass changes nothing
    CHECK(npx.args.size() == 4);

    auto npm = server("npm", {"i", "left-pad"});
    REQUIRE(rewrite_launcher(npm, home, "node").rewritten);
    CHECK(fs::path(npm.command).stem().string() == "node");
    CHECK(npm.args.size() == 3);
    CHECK(npm.args[1] == "install");

    auto plain_node = server("node", {"server.js"});
    REQUIRE(rewrite_launcher(plain_node, home, "node").rewritten);
    CHECK(plain_node.args == std::vector<std::string>{"server.js"});

    auto nub_npx = server("npx", {"-y", "pkg"});
    REQUIRE(rewrite_launcher(nub_npx, home, "nub").rewritten);
    CHECK(fs::path(nub_npx.command).stem().string() == "nub");
    CHECK(nub_npx.args == std::vector<std::string>{"dlx", "pkg"});

    auto bun_npm = server("npm", {"exec", "pkg"});
    REQUIRE(rewrite_launcher(bun_npm, home, "bun").rewritten == false); // bun is not installed here
    fs::remove_all(root, ec);
}
