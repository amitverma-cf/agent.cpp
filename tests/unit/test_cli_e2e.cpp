// The `agent` command line program run as a real process (AGENT_TEST_CLI_PATH) against the mock engine, in a private home folder.
// Covers the command dispatcher, the small commands and the chat / task / run paths end to end.
#if defined(AGENT_TEST_CLI_PATH) && defined(AGENT_TEST_MOCK_ENGINE_PATH)

#include <catch_amalgamated.hpp>

#include <cstdio>
#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

struct CliResult {
    int code = -1;
    std::string out; // stdout and stderr together
};

#ifdef _WIN32
#define AGENT_POPEN _popen
#define AGENT_PCLOSE _pclose
#else
#define AGENT_POPEN popen
#define AGENT_PCLOSE pclose
#endif

void set_env(const char *name, const std::string &value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

std::string quote(const std::string &s) { return "\"" + s + "\""; }

struct Home {
    fs::path root = fs::temp_directory_path() / "agent_test_cli_e2e";
    fs::path workspace = root / "ws";
    fs::path model = root / "model.gguf"; // the mock engine ignores the file's content; the CLI wants it to exist
    fs::path graph = root / "graph.json";
    Home() {
        std::error_code ec;
        fs::remove_all(root, ec);
        fs::create_directories(workspace);
        std::ofstream(model) << "x";
        std::ofstream(graph) << R"({"name":"g","engine":"mock","entry":"a","nodes":[{"name":"a","kind":"agent","system_prompt":"s","input_key":"request","output_key":"answer"}],"edges":[]})";
        set_env("AGENT_HOME", (root / "home").string());
        set_env("AGENT_CONFIG_DIR", "");
    }
    ~Home() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    // `agent <args>`, with stdin from a file when given.
    CliResult run(const std::string &args, const std::string &stdin_text = {}) const {
        std::string cmd;
        if (!stdin_text.empty()) {
            const fs::path in = root / "stdin.txt";
            std::ofstream(in) << stdin_text;
            cmd = quote(AGENT_TEST_CLI_PATH) + " " + args + " < " + quote(in.string()) + " 2>&1";
        } else {
            cmd = quote(AGENT_TEST_CLI_PATH) + " " + args + " 2>&1";
        }
#ifdef _WIN32
        cmd = "\"" + cmd + "\"";
#endif
        CliResult r;
        FILE *p = AGENT_POPEN(cmd.c_str(), "r");
        if (!p) return r;
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), p)) > 0) r.out.append(buf, n);
        const int status = AGENT_PCLOSE(p);
#ifdef _WIN32
        r.code = status;
#else
        r.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
        return r;
    }
    std::string engine_args() const {
        return "-e " + quote(AGENT_TEST_MOCK_ENGINE_PATH) + " -m " + quote(model.string()) + " --inprocess --workspace " + quote(workspace.string());
    }
};

bool has(const CliResult &r, const std::string &text) { return r.out.find(text) != std::string::npos; }

} // namespace

TEST_CASE("cli e2e: version, help and bad commands", "[cli_e2e]") {
    Home h;
    auto v = h.run("version");
    CHECK(v.code == 0);
    CHECK(has(v, "agent "));
    auto bad = h.run("no-such-command");
    CHECK(bad.code == 2);
    CHECK(has(bad, "unknown command"));
    CHECK(has(bad, "usage: agent"));
    auto missing = h.run("model rm");
    CHECK(missing.code != 0);
}

TEST_CASE("cli e2e: the listing commands work in an empty home", "[cli_e2e]") {
    Home h;
    for (const char *cmd : {"doctor", "engine ls", "model ls", "sandbox ls", "mcp ls", "runtime ls", "skill ls"}) {
        auto r = h.run(cmd);
        INFO(cmd << ": " << r.out);
        CHECK(r.code == 0);
        if (std::string(cmd) != "skill ls") CHECK_FALSE(r.out.empty()); // an empty skill list prints nothing
    }
}

TEST_CASE("cli e2e: engine add, list and remove", "[cli_e2e]") {
    Home h;
    auto add = h.run("engine add " + quote(AGENT_TEST_MOCK_ENGINE_PATH));
    INFO(add.out);
    CHECK(add.code == 0);
    auto ls = h.run("engine ls");
    CHECK(has(ls, "mock"));
    CHECK(has(ls, "text_gen"));
    auto rm = h.run("engine rm mock");
    CHECK(rm.code == 0);
    CHECK(has(h.run("engine ls"), "no engines"));
    CHECK(h.run("engine rm mock").code != 0); // already gone
    CHECK(h.run("engine add " + quote((h.root / "not-a-plugin.so").string())).code != 0);
}

TEST_CASE("cli e2e: sandbox grants are stored, listed and revoked", "[cli_e2e]") {
    Home h;
    const std::string dir = quote(h.workspace.string());
    auto grant = h.run("sandbox grant " + dir + " --write --for run_command");
    INFO(grant.out);
    CHECK(grant.code == 0);
    auto ls = h.run("sandbox ls");
    CHECK(has(ls, "write"));
    CHECK(has(ls, "run_command"));
    CHECK(h.run("sandbox revoke " + dir).code == 0);
    CHECK(has(h.run("sandbox ls"), "nothing granted"));
    CHECK(h.run("sandbox grant").code != 0); // a path is required
}

TEST_CASE("cli e2e: task runs one engine call", "[cli_e2e]") {
    Home h;
    auto r = h.run("task text_gen " + h.engine_args() + " --text hello");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(has(r, "task: text_gen"));
    CHECK(has(r, "echo: hello"));
    CHECK(has(r, "usage:"));
    auto isolated = h.run("task text_gen -e " + quote(AGENT_TEST_MOCK_ENGINE_PATH) + " -m " + quote(h.model.string()) + " --text again --workspace " +
                          quote(h.workspace.string())); // out of process, through the engine host
    INFO(isolated.out);
    CHECK(has(isolated, "echo: again"));
    CHECK(h.run("task not_a_task " + h.engine_args()).code != 0);
    CHECK(h.run("task text_gen -e " + quote(AGENT_TEST_MOCK_ENGINE_PATH) + " -m " + quote((h.root / "missing.gguf").string())).code != 0);
}

TEST_CASE("cli e2e: chat answers, saves the conversation and resumes it", "[cli_e2e]") {
    Home h;
    auto r = h.run("chat " + h.engine_args(), "hi there\n/help\n/exit\n");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(has(r, "echo: hi there"));
    CHECK(has(r, "saved as chat-"));
    auto list = h.run("chat --list " + h.engine_args());
    CHECK(has(list, "chat-"));
    const auto at = r.out.find("--resume ");
    REQUIRE(at != std::string::npos);
    const std::string id = r.out.substr(at + 9, r.out.find_first_of(" \r\n)", at + 9) - (at + 9));
    auto again = h.run("chat --resume " + id + " " + h.engine_args(), "second message\n/exit\n");
    INFO(again.out);
    CHECK(has(again, "echo: second message"));
}

TEST_CASE("cli e2e: run executes a graph file", "[cli_e2e]") {
    Home h;
    auto r = h.run("run " + quote(h.graph.string()) + " " + h.engine_args() + " --set request=ping");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(has(r, "status: completed"));
    CHECK(has(r, "answer = echo: ping"));
    CHECK(h.run("run " + quote((h.root / "nope.json").string()) + " " + h.engine_args()).code != 0);
}

TEST_CASE("cli e2e: init creates the project folders", "[cli_e2e]") {
    Home h;
    const fs::path proj = h.root / "proj";
    auto r = h.run("init " + quote(proj.string()));
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(fs::exists(proj / ".agents"));
    CHECK(fs::exists(proj / ".agents" / "memory"));
    CHECK(h.run("init " + quote(proj.string())).code == 0); // running it again is harmless
}

#endif
