// The Windows sandbox: a child runs at Low integrity and can write only in the folders it was allowed (see src/hal/sandbox.cpp).
#ifdef _WIN32

#include "hal/child_process.hpp"
#include "hal/sandbox.hpp"

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

std::string native(std::string p) {
    for (char &c : p)
        if (c == '/') c = '\\';
    return p;
}

// Runs cmd.exe /c <command> under `policy`; returns the exit code, or -1 when it could not be started.
int run_cmd(const std::string &command, const agent::SandboxPolicy &policy, std::string *spawn_error = nullptr, std::string *output = nullptr) {
    agent::hal::SpawnOptions opts;
    opts.program = "cmd.exe";
    opts.args = {"/c", command};
    opts.sandbox = policy;
    auto child = agent::hal::spawn_child(opts);
    if (!child.ok) {
        if (spawn_error) *spawn_error = child.error.message;
        return -1;
    }
    child.value->close_stdin();
    std::string local;
    std::string &sink = output ? *output : local;
    while (child.value->read(agent::hal::ChildStream::Stdout, sink, 10000) == agent::hal::PipeRead::Data) {}
    while (child.value->read(agent::hal::ChildStream::Stderr, sink, 10000) == agent::hal::PipeRead::Data) {}
    return child.value->terminate(10000);
}

} // namespace

TEST_CASE("Windows: a sandboxed child writes only in its allowed folder", "[sandbox][windows]") {
    const fs::path dir = fs::temp_directory_path() / "agent_test_win_sandbox";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "rw");
    fs::create_directories(dir / "outside");

    agent::SandboxPolicy policy;
    policy.enabled = true;
    policy.filesystem = true;
    policy.windows_low_integrity = true;
    policy.read_write_paths = {(dir / "rw").string()};

    std::string error;
    CHECK(run_cmd("echo inside> " + native((dir / "rw" / "ok.txt").string()), policy, &error) == 0);
    INFO(error);
    CHECK(fs::exists(dir / "rw" / "ok.txt"));

    // Anything else the user owns refuses the write.
    CHECK(run_cmd("echo nope> " + native((dir / "outside" / "bad.txt").string()), policy) != 0);
    CHECK_FALSE(fs::exists(dir / "outside" / "bad.txt"));

    // Reading is not restricted on Windows, and what the child wrote in its folder is readable from outside.
    CHECK(run_cmd("type " + native((dir / "rw" / "ok.txt").string()) + " >nul", policy) == 0);

    // Without the sandbox the same write works: the refusal above came from the confinement, not from the path.
    agent::SandboxPolicy off;
    CHECK(run_cmd("echo free> " + native((dir / "outside" / "free.txt").string()), off) == 0);
    CHECK(fs::exists(dir / "outside" / "free.txt"));

    fs::remove_all(dir, ec);
}

#endif
