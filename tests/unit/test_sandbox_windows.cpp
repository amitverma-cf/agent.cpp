// The Windows sandbox: a child runs at Low integrity and can write only in the folders it was allowed (see src/hal/sandbox.cpp).
#ifdef _WIN32

#include "hal/child_process.hpp"
#include "hal/sandbox.hpp"

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

#define WIN32_LEAN_AND_MEAN
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <windows.h>

namespace fs = std::filesystem;

namespace {

std::string native(std::string p) {
    for (char &c : p)
        if (c == '/') c = '\\';
    return p;
}

// Runs cmd.exe /c <command> under `policy`; returns the exit code, or -1 when it could not be started.
int run_cmd(const std::string &command, const agent::SandboxPolicy &policy, std::string *spawn_error = nullptr,
            std::string *output = nullptr, const std::string &cwd = {}) {
    agent::hal::SpawnOptions opts;
    opts.program = "cmd.exe";
    opts.args = {"/c", command};
    opts.sandbox = policy;
    opts.working_directory = cwd;
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

    // The child reads its own folder, not other folders of the user (the AppContainer has no access to them).
    CHECK(run_cmd("type " + native((dir / "rw" / "ok.txt").string()) + " >nul", policy) == 0);
    { std::ofstream(dir / "outside" / "secret.txt") << "secret"; }
    CHECK(run_cmd("type " + native((dir / "outside" / "secret.txt").string()) + " >nul", policy) != 0);
    agent::SandboxPolicy granted = policy;
    granted.read_only_paths = {(dir / "outside").string()};
    {
        std::string out;
        int rc = run_cmd("type " + native((dir / "outside" / "secret.txt").string()), granted, nullptr, &out);
        INFO(out);
        CHECK(rc == 0);
    }

    // Without the sandbox the same write works: the refusal above came from the confinement, not from the path.
    agent::SandboxPolicy off;
    CHECK(run_cmd("echo free> " + native((dir / "outside" / "free.txt").string()), off) == 0);
    CHECK(fs::exists(dir / "outside" / "free.txt"));

    fs::remove_all(dir, ec);
}

// Needs the internet, so it runs only when AGENT_TEST_NETWORK is set.
TEST_CASE("Windows: the sandbox keeps the network when allowed and cuts it when denied", "[sandbox][windows][network]") {
    if (!std::getenv("AGENT_TEST_NETWORK")) SKIP("set AGENT_TEST_NETWORK=1 to run (needs the internet)");
    const fs::path dir = fs::temp_directory_path() / "agent_test_win_sandbox_net";
    std::error_code ec;
    fs::create_directories(dir);
    agent::SandboxPolicy policy;
    policy.enabled = true;
    policy.filesystem = true;
    policy.windows_low_integrity = true;
    policy.read_write_paths = {dir.string()};
    const std::string fetch = "curl.exe -s -m 8 -o nul https://example.com";
    policy.network = agent::SandboxNetwork::Allow;
    {
        std::string out;
        int rc = run_cmd("curl.exe -sS -m 8 -o nul https://example.com 2>&1", policy, nullptr, &out, dir.string());
        INFO(out);
        CHECK(rc == 0);
    }
    policy.network = agent::SandboxNetwork::Deny;
    CHECK(run_cmd(fetch, policy, nullptr, nullptr, dir.string()) != 0);
    fs::remove_all(dir, ec);
}

// A child gets only its three stream handles: an inheritable pipe end the parent holds must not reach it. If it did, the pipe would keep
// a writer alive and the reader would never see the end of the stream while the child runs.
TEST_CASE("Windows: a spawned child does not inherit unrelated inheritable handles", "[sandbox][windows]") {
    for (bool confined : {false, true}) {
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        HANDLE r = nullptr, w = nullptr;
        REQUIRE(CreatePipe(&r, &w, &sa, 0));
        SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);

        agent::SandboxPolicy policy;
        policy.enabled = confined; // false: the plain spawn (like an engine host); true: the Low integrity / AppContainer path
        policy.filesystem = confined;
        policy.windows_low_integrity = confined;
        fs::path dir = fs::temp_directory_path() / "agent_test_win_handles";
        std::error_code ec;
        fs::create_directories(dir, ec);
        policy.read_write_paths = {dir.string()};

        agent::hal::SpawnOptions opts;
        opts.program = "cmd.exe";
        opts.args = {"/c", "for /l %i in (1,1,300000000) do @rem"}; // keeps running for a few seconds
        opts.sandbox = policy;
        opts.working_directory = dir.string();
        auto child = agent::hal::spawn_child(opts);
        REQUIRE(child.ok);
        CloseHandle(w); // the parent's own copy: no writer may be left if the child did not inherit it

        bool broken = false;
        for (int i = 0; i < 100 && !broken; ++i) {
            DWORD avail = 0;
            if (!PeekNamedPipe(r, nullptr, 0, nullptr, &avail, nullptr)) broken = GetLastError() == ERROR_BROKEN_PIPE;
            else Sleep(20);
        }
        CHECK(broken);
        child.value->terminate(1000);
        CloseHandle(r);
        fs::remove_all(dir, ec);
    }
}
#endif
