// OS-level sandbox (hal/sandbox.hpp, docs/security-sandbox.md): Landlock filesystem allow-lists,
// the seccomp network filter, rlimits, and the three integration points (run_command, the engine
// host, MCP servers). Linux only; Landlock / seccomp cases SKIP when the kernel lacks them.

#include "hal/child_process.hpp"
#include "hal/sandbox.hpp"

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;

const std::string kRoot = "/tmp/agent_test_sandbox";

void write_file(const fs::path &p, const std::string &content) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << content;
}

std::string fresh_dir(const std::string &name) {
    const std::string dir = kRoot + "/" + name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir;
}

struct RunResult {
    bool spawned = false;
    std::string spawn_error;
    int exit_code = -1;
    std::string out; // stdout + stderr
    std::vector<std::string> warnings;
};

RunResult run(std::vector<std::string> argv, const agent::SandboxPolicy &policy) {
    agent::hal::SpawnOptions opts;
    opts.program = argv.front();
    opts.args.assign(argv.begin() + 1, argv.end());
    opts.sandbox = policy;
    RunResult r;
    auto child = agent::hal::spawn_child(opts);
    if (!child.ok) {
        r.spawn_error = child.error.message;
        return r;
    }
    r.spawned = true;
    r.warnings = child.value->sandbox_warnings();
    child.value->close_stdin();
    while (child.value->read(agent::hal::ChildStream::Stdout, r.out, 10000) == agent::hal::PipeRead::Data) {}
    while (child.value->read(agent::hal::ChildStream::Stderr, r.out, 10000) == agent::hal::PipeRead::Data) {}
    r.exit_code = child.value->terminate(10000);
    return r;
}

bool landlock() { return agent::hal::sandbox_support().landlock_abi > 0; }
bool seccomp() { return agent::hal::sandbox_support().seccomp; }

bool readable(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return f.good() && f.peek() != std::ifstream::traits_type::eof();
}

const agent::Tool *find_tool(const agent::Session &session, std::string_view name) {
    for (const auto &t : session.config.tools)
        if (t.name == name) return &t;
    return nullptr;
}

} // namespace

TEST_CASE("sandbox support on this kernel", "[sandbox]") {
    const auto s = agent::hal::sandbox_support();
    // Informational: shows up in the test log so a run documents what was actually exercised.
    WARN("Landlock ABI: " << s.landlock_abi << " (0 = unavailable), seccomp: " << (s.seccomp ? "yes" : "no"));
    SUCCEED();
}

TEST_CASE("a disabled policy changes nothing", "[sandbox]") {
    auto prepared = agent::hal::prepare_sandbox(agent::SandboxPolicy{});
    REQUIRE(prepared.ok);
    CHECK_FALSE(prepared.value.active);
    CHECK(prepared.value.landlock_fd < 0);
    CHECK(prepared.value.seccomp.empty());
    CHECK(agent::hal::apply_prepared_sandbox(prepared.value) == 0);
}

TEST_CASE("Landlock: a sandboxed child reads only the allow-list", "[sandbox][landlock]") {
    if (!landlock()) SKIP("Landlock is not available on this kernel");
    const std::string dir = fresh_dir("fs");
    write_file(dir + "/allowed/data.txt", "visible");
    write_file(dir + "/secret/secret.txt", "top secret");

    agent::SandboxPolicy p{.enabled = true, .allow_tmp = false};
    p.read_only_paths = {dir + "/allowed"};

    auto ok = run({"cat", dir + "/allowed/data.txt"}, p);
    REQUIRE(ok.spawned);
    INFO(ok.out);
    CHECK(ok.exit_code == 0);
    CHECK(ok.out == "visible");
    CHECK(ok.warnings.empty());

    auto denied = run({"cat", dir + "/secret/secret.txt"}, p);
    REQUIRE(denied.spawned);
    INFO(denied.out);
    CHECK(denied.exit_code != 0);
    CHECK(denied.out.find("top secret") == std::string::npos);
    CHECK(denied.out.find("Permission denied") != std::string::npos);

    // Listing a directory outside the allow-list fails too.
    auto ls = run({"ls", dir + "/secret"}, p);
    REQUIRE(ls.spawned);
    CHECK(ls.exit_code != 0);
    CHECK(ls.out.find("secret.txt") == std::string::npos);

    // /etc/shadow is not part of the system allow-list (this check needs a user who can read it).
    if (readable("/etc/shadow")) {
        auto shadow = run({"cat", "/etc/shadow"}, p);
        REQUIRE(shadow.spawned);
        CHECK(shadow.exit_code != 0);
        CHECK(shadow.out.find("root:") == std::string::npos);
    }
}

TEST_CASE("Landlock: writes are confined to the read-write paths", "[sandbox][landlock]") {
    if (!landlock()) SKIP("Landlock is not available on this kernel");
    const std::string dir = fresh_dir("write");
    fs::create_directories(dir + "/rw");
    fs::create_directories(dir + "/ro");
    fs::create_directories(dir + "/outside");

    agent::SandboxPolicy p{.enabled = true, .allow_tmp = false};
    p.read_only_paths = {dir + "/ro"};
    p.read_write_paths = {dir + "/rw"};

    auto w = run({"sh", "-c", "echo hi > " + dir + "/rw/out.txt && cat " + dir + "/rw/out.txt"}, p);
    REQUIRE(w.spawned);
    INFO(w.out);
    CHECK(w.exit_code == 0);
    CHECK(w.out == "hi\n");

    for (const char *sub : {"/ro/out.txt", "/outside/out.txt"}) {
        auto denied = run({"sh", "-c", "echo hi > " + dir + sub}, p);
        REQUIRE(denied.spawned);
        INFO(sub << ": " << denied.out);
        CHECK(denied.exit_code != 0);
        CHECK_FALSE(fs::exists(dir + sub));
    }
    // /dev/null stays writable (shell redirections).
    auto devnull = run({"sh", "-c", "echo hi > /dev/null"}, p);
    REQUIRE(devnull.spawned);
    CHECK(devnull.exit_code == 0);
}

TEST_CASE("seccomp: the network-deny filter blocks inet sockets but not AF_UNIX", "[sandbox][seccomp]") {
    if (!seccomp()) SKIP("seccomp is not available on this kernel/arch");
    agent::SandboxPolicy p{.enabled = true, .filesystem = false, .network = agent::SandboxNetwork::Deny};
    auto prepared = agent::hal::prepare_sandbox(p);
    REQUIRE(prepared.ok);
    REQUIRE_FALSE(prepared.value.seccomp.empty());
    prepared.value.new_session = false;

    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        if (agent::hal::apply_prepared_sandbox(prepared.value) != 0) ::_exit(10);
        int code = 0;
        errno = 0;
        if (::socket(AF_INET, SOCK_STREAM, 0) >= 0 || errno != EACCES) code |= 1;
        errno = 0;
        if (::socket(AF_INET6, SOCK_DGRAM, 0) >= 0 || errno != EACCES) code |= 2;
        if (::socket(AF_UNIX, SOCK_STREAM, 0) < 0) code |= 4;
        int sv[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) code |= 8;
        ::_exit(code);
    }
    int status = 0;
    REQUIRE(::waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);

    // The parent is unaffected.
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    if (fd >= 0) ::close(fd);
}

TEST_CASE("RLIMIT_AS makes a large allocation fail", "[sandbox][rlimit]") {
    // awk is POSIX and not built with ASan, so this also holds in a sanitizer build of the tests.
    // Doubles a string up to 128 MiB (peak ~384 MiB with the concatenation temporaries).
    const std::vector<std::string> big = {"awk", "BEGIN { s = \"x\"; while (length(s) < 100000000) s = s s; print length(s) }"};
    auto unlimited = run(big, agent::SandboxPolicy{.enabled = true, .filesystem = false});
    REQUIRE(unlimited.spawned);
    if (unlimited.exit_code != 0) SKIP("awk cannot allocate 384 MiB here even without a limit");
    CHECK(unlimited.out.find("134217728") != std::string::npos);

    agent::SandboxPolicy limited{.enabled = true, .filesystem = false, .memory_limit_bytes = 64ull << 20};
    auto small = run({"awk", "BEGIN { print \"fine\" }"}, limited);
    REQUIRE(small.spawned);
    CHECK(small.exit_code == 0);
    CHECK(small.out == "fine\n");

    auto r = run(big, limited);
    REQUIRE(r.spawned);
    INFO(r.out);
    CHECK(r.exit_code != 0);
    CHECK(r.out.find("134217728") == std::string::npos);
}

TEST_CASE("a sandboxed child gets its own session and no core dumps", "[sandbox]") {
    // Prints the core limit, the shell's pid and the session id of a process the shell starts.
    auto r = run({"sh", "-c", "ulimit -c; echo $$; awk '{ print $6 }' /proc/self/stat"},
                 agent::SandboxPolicy{.enabled = true, .filesystem = false});
    REQUIRE(r.spawned);
    INFO(r.out);
    CHECK(r.exit_code == 0);
    std::istringstream lines(r.out);
    std::string core, pid, sid;
    std::getline(lines, core);
    std::getline(lines, pid);
    std::getline(lines, sid);
    CHECK(core == "0");
    CHECK(!pid.empty());
    CHECK(pid == sid); // the sandboxed shell leads its own session
    CHECK(sid != std::to_string(::getsid(0)));
}

TEST_CASE("run_command under the default tool sandbox stays in the workspace", "[sandbox][landlock][terminal_tools]") {
    if (!landlock()) SKIP("Landlock is not available on this kernel");
    const std::string root = fresh_dir("run_command");
    const std::string ws = root + "/ws";
    write_file(ws + "/inside.txt", "inside");
    write_file(root + "/secret/secret.txt", "top secret");

    agent::Config cfg{.provider = agent::AiProvider::Mock, .workspace_dir = ws};
    // The secret lives under /tmp, which run_command may use by default; take that away for the test.
    cfg.tool_sandbox.allow_tmp = false;
    auto session_res = agent::init(cfg);
    REQUIRE(session_res.ok);
    agent::Session session = std::move(session_res.value);
    const agent::Tool *tool = find_tool(session, "run_command");
    REQUIRE(tool);
    auto call = [&](const std::string &cmd) {
        std::string args = "{\"command\":\"" + cmd + "\"}";
        auto r = tool->callback(args, tool->user_data);
        REQUIRE(r.ok);
        return r.value;
    };

    std::string inside = call("cat inside.txt && echo made > new.txt && cat new.txt");
    INFO(inside);
    CHECK(inside.find("\"exit_code\":0") != std::string::npos);
    CHECK(inside.find("inside") != std::string::npos);
    CHECK(inside.find("made") != std::string::npos);
    CHECK(inside.find("sandbox_warning") == std::string::npos);

    const std::vector<std::string> escapes = {"cat " + root + "/secret/secret.txt", "cat ../secret/secret.txt",
                                              "ln -s " + root + "/secret/secret.txt link.txt && cat link.txt", "cat /etc/shadow"};
    for (const std::string &cmd : escapes) {
        std::string out = call(cmd);
        INFO(cmd << " -> " << out);
        CHECK(out.find("\"exit_code\":0") == std::string::npos);
        CHECK(out.find("top secret") == std::string::npos);
        CHECK(out.find("root:") == std::string::npos);
    }
    std::string escape = call("echo x > " + root + "/secret/planted.txt");
    CHECK(escape.find("\"exit_code\":0") == std::string::npos);
    CHECK_FALSE(fs::exists(root + "/secret/planted.txt"));

    // Config::sandbox_filesystem = false lifts the filesystem part (but keeps the rest).
    cfg.sandbox_filesystem = false;
    auto open_res = agent::init(cfg);
    REQUIRE(open_res.ok);
    agent::Session open_session = std::move(open_res.value);
    const agent::Tool *open_tool = find_tool(open_session, "run_command");
    std::string args = "{\"command\":\"cat " + root + "/secret/secret.txt\"}";
    auto out = open_tool->callback(args, open_tool->user_data);
    REQUIRE(out.ok);
    CHECK(out.value.find("top secret") != std::string::npos);
}

TEST_CASE("run_command with network denied cannot open inet sockets", "[sandbox][seccomp][terminal_tools]") {
    if (!seccomp()) SKIP("seccomp is not available on this kernel/arch");
    const std::string ws = fresh_dir("run_command_net");
    agent::Config cfg{.provider = agent::AiProvider::Mock, .workspace_dir = ws};
    cfg.tool_sandbox.network = agent::SandboxNetwork::Deny;
    auto session_res = agent::init(cfg);
    REQUIRE(session_res.ok);
    agent::Session session = std::move(session_res.value);
    const agent::Tool *tool = find_tool(session, "run_command");
    // bash's /dev/tcp needs socket(AF_INET); the filter answers EACCES before any connect.
    auto r = tool->callback(R"({"command":"bash -c 'exec 3<>/dev/tcp/127.0.0.1/9' 2>&1; echo rc=$?"})", tool->user_data);
    REQUIRE(r.ok);
    INFO(r.value);
    CHECK(r.value.find("rc=0") == std::string::npos);
    CHECK(r.value.find("ermission denied") != std::string::npos);
}

TEST_CASE("the engine host runs the mock engine under its default sandbox", "[sandbox][engine_host]") {
    const std::string root = fresh_dir("engine");
    const std::string ws = root + "/ws";
    write_file(root + "/models/model.bin", "weights");
    write_file(root + "/secret/secret.txt", "top secret");

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_host_path = AGENT_TEST_ENGINE_HOST_PATH;
    cfg.model = root + "/models/model.bin";
    cfg.workspace_dir = ws;
    REQUIRE(cfg.engine_sandbox.enabled); // the default
    auto session_res = agent::init(cfg);
    INFO(session_res.error.message);
    REQUIRE(session_res.ok);
    agent::Session session = std::move(session_res.value);

    auto turn = [&](const std::string &prompt) {
        agent::FlowMemory memory;
        auto r = agent::run_turn(session, memory, prompt);
        INFO(r.error.message);
        REQUIRE(r.ok);
        return r.value;
    };

    CHECK(turn("hello") == "echo: hello");
    CHECK(turn("read:" + cfg.model) == "read ok: 7 bytes");
    CHECK(turn("write:" + ws + "/artifacts/out.txt") == "write ok");
    CHECK(turn("draw").starts_with("echo: draw"));

    if (seccomp()) CHECK(turn("net").starts_with("net denied"));
    if (landlock()) {
        CHECK(turn("read:/etc/shadow").starts_with("read denied"));
        CHECK(turn("read:" + root + "/secret/secret.txt").starts_with("read denied"));
        CHECK(turn("write:" + ws + "/planted.txt").starts_with("write denied"));
        CHECK(turn("write:/tmp/agent_test_sandbox_planted.txt").starts_with("write denied"));
        CHECK_FALSE(fs::exists(ws + "/planted.txt"));
    } else {
        WARN("Landlock unavailable: engine host filesystem checks skipped");
    }

    // engine_sandbox.enabled = false restores the old, unrestricted host.
    cfg.engine_sandbox.enabled = false;
    auto open_res = agent::init(cfg);
    REQUIRE(open_res.ok);
    agent::FlowMemory memory;
    auto r = agent::run_turn(open_res.value, memory, "read:" + root + "/secret/secret.txt");
    REQUIRE(r.ok);
    CHECK(r.value == "read ok: 10 bytes");
}

TEST_CASE("an opted-in MCP server still works inside the sandbox", "[sandbox][mcp]") {
    const std::string ws = fresh_dir("mcp");
    agent::Config cfg;
    cfg.workspace_dir = ws;
    cfg.auto_default_tools = false;
    auto session_res = agent::init(cfg);
    REQUIRE(session_res.ok);
    agent::Session session = std::move(session_res.value);

    agent::McpServerConfig mcp;
    mcp.name = "boxed";
    mcp.command = AGENT_TEST_FAKE_MCP_SERVER_PATH;
    mcp.timeout_ms = 5000;
    mcp.sandbox = agent::SandboxPolicy{.enabled = true, .allow_tmp = false, .network = agent::SandboxNetwork::Deny};
    auto conn = agent::connect_mcp_server(session, mcp);
    INFO(conn.error.message);
    REQUIRE(conn.ok);
    auto echo = agent::execute_tool(session, "boxed.echo", R"({"text":"inside"})");
    REQUIRE(echo.ok);
    CHECK(echo.value == "echo: inside");
    agent::disconnect_mcp_servers(session);
}

#endif // __linux__
