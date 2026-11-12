// Shared by the command line end to end tests: runs the agent binary in a private home and captures its output.
#pragma once
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

namespace cli_e2e {

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

inline void set_env(const char *name, const std::string &value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

inline std::string quote(const std::string &s) { return "\"" + s + "\""; }

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
        std::ofstream(graph)
            << R"({"name":"g","engine":"mock","entry":"a","nodes":[{"name":"a","kind":"agent","system_prompt":"s","input_key":"request","output_key":"answer"}],"edges":[]})";
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
        return "-e " + quote(AGENT_TEST_MOCK_ENGINE_PATH) + " -m " + quote(model.string()) + " --inprocess --workspace " +
               quote(workspace.string());
    }
};

inline bool has(const CliResult &r, const std::string &text) { return r.out.find(text) != std::string::npos; }

} // namespace cli_e2e

#endif
