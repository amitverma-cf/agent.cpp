#include "runtimes.hpp"

#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>

using namespace agent::cli;
namespace fs = std::filesystem;

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

    fs::remove_all(root, ec);
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
