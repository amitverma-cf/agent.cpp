// The permission gate for paths outside the workspace: grants, the host's gate, and what each answer remembers.
#include "agent/tools/path_gate.hpp"
#include "hal/sandbox.hpp"

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct GateLog {
    int asked = 0;
    bool last_write = false;
    std::string last_consumer;
    agent::PathGateDecision answer = agent::PathGateDecision::Deny;
};

agent::PathGateDecision gate(const agent::PathAccessRequest &request, void *user) {
    auto *log = static_cast<GateLog *>(user);
    ++log->asked;
    log->last_write = request.write;
    log->last_consumer = std::string(request.consumer);
    return log->answer;
}

struct Fixture {
    fs::path root = fs::temp_directory_path() / "agent_test_path_gate";
    fs::path workspace = root / "ws";
    fs::path outside = root / "outside";
    Fixture() {
        std::error_code ec;
        fs::remove_all(root, ec);
        fs::create_directories(workspace);
        fs::create_directories(outside);
        std::ofstream(outside / "secret.txt") << "outside data";
    }
    ~Fixture() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    agent::Session open(agent::Config cfg) const {
        cfg.workspace_dir = workspace.string();
        cfg.provider = agent::AiProvider::Mock;
        auto s = agent::init(cfg);
        REQUIRE(s.ok);
        return std::move(s.value);
    }
    std::string read_args() const { return "{\"path\":\"" + generic(outside / "secret.txt") + "\"}"; }
    std::string write_args() const { return "{\"path\":\"" + generic(outside / "new.txt") + "\",\"content\":\"hi\"}"; }
    static std::string generic(const fs::path &p) { return p.generic_string(); }
};

} // namespace

TEST_CASE("path gate: without a grant or a gate a path outside the workspace is refused", "[pathgate]") {
    Fixture f;
    agent::Session session = f.open({});
    auto r = agent::execute_tool(session, "read_file", f.read_args());
    CHECK_FALSE(r.ok);
    CHECK(r.error.message.find("agent sandbox grant") != std::string::npos);
}

TEST_CASE("path gate: a read grant allows reading but not writing", "[pathgate]") {
    Fixture f;
    agent::Config cfg;
    cfg.path_grants = {agent::PathGrant{.path = f.outside.string(), .write = false, .consumer = "*"}};
    agent::Session session = f.open(cfg);
    auto r = agent::execute_tool(session, "read_file", f.read_args());
    REQUIRE(r.ok);
    CHECK(r.value.find("outside data") != std::string::npos);
    CHECK_FALSE(agent::execute_tool(session, "write_file", f.write_args()).ok);
    CHECK_FALSE(fs::exists(f.outside / "new.txt"));
}

TEST_CASE("path gate: a write grant allows writing, only for the named consumer", "[pathgate]") {
    Fixture f;
    agent::Config cfg;
    cfg.path_grants = {agent::PathGrant{.path = f.outside.string(), .write = true, .consumer = "run_command"}};
    agent::Session other = f.open(cfg);
    CHECK_FALSE(agent::execute_tool(other, "write_file", f.write_args()).ok); // the grant is for run_command, not the file tools

    cfg.path_grants[0].consumer = "files";
    agent::Session files = f.open(cfg);
    CHECK(agent::execute_tool(files, "write_file", f.write_args()).ok);
    CHECK(fs::exists(f.outside / "new.txt"));
}

TEST_CASE("path gate: the gate is asked, and AllowOnce / AllowAlways / Deny mean what they say", "[pathgate]") {
    Fixture f;
    GateLog log;
    agent::Config cfg;
    cfg.path_gate = gate;
    cfg.path_gate_user_data = &log;
    agent::Session session = f.open(cfg);

    log.answer = agent::PathGateDecision::Deny;
    CHECK_FALSE(agent::execute_tool(session, "read_file", f.read_args()).ok);
    CHECK(log.asked == 1);
    CHECK(log.last_consumer == "files");
    CHECK_FALSE(log.last_write);

    log.answer = agent::PathGateDecision::AllowOnce;
    CHECK(agent::execute_tool(session, "read_file", f.read_args()).ok);
    CHECK(agent::execute_tool(session, "read_file", f.read_args()).ok);
    CHECK(log.asked == 3); // once means it is asked again every time

    log.answer = agent::PathGateDecision::AllowAlways;
    CHECK(agent::execute_tool(session, "read_file", f.read_args()).ok);
    CHECK(log.asked == 4);
    CHECK(agent::execute_tool(session, "read_file", f.read_args()).ok);
    CHECK(log.asked == 4); // remembered for the session

    // The read grant does not stretch to writing: that is a new question, flagged as a write.
    log.answer = agent::PathGateDecision::Deny;
    CHECK_FALSE(agent::execute_tool(session, "write_file", f.write_args()).ok);
    CHECK(log.asked == 5);
    CHECK(log.last_write);
}

TEST_CASE("path gate: command lines are searched for paths outside the workspace", "[pathgate]") {
    using agent::tools::command_looks_like_write;
    using agent::tools::external_paths_in_command;
    const fs::path ws = fs::temp_directory_path() / "agent_test_path_gate_cmd";
    std::error_code ec;
    fs::create_directories(ws, ec);

#ifdef _WIN32
    const std::string other = "C:\\Users\\someone\\docs\\a.txt";
    const std::string inside = (ws / "x.txt").string();
#else
    const std::string other =
        (ws.parent_path() / "agent_gate_elsewhere" / "a.txt").string(); // a sibling: outside the workspace, not a system folder
    const std::string inside = (ws / "x.txt").string();
#endif
    auto found = external_paths_in_command("cat " + other + " && echo hi > " + inside, ws);
    REQUIRE(found.size() == 1);
    CHECK(found[0].filename() == "a.txt");

    CHECK(external_paths_in_command("echo hello world", ws).empty());
    CHECK(external_paths_in_command("grep -r foo .", ws).empty());
    CHECK(command_looks_like_write("echo x > file"));
    CHECK(command_looks_like_write("cp a b"));
    CHECK_FALSE(command_looks_like_write("cat file | wc -l"));
    fs::remove_all(ws, ec);
}

TEST_CASE("path gate: run_command writes outside the workspace only with a grant", "[pathgate][sandbox][terminal_tools]") {
#if !defined(_WIN32)
    if (agent::hal::sandbox_support().landlock_abi <= 0) SKIP("Landlock is not available on this kernel");
#endif
    Fixture f;
    auto json_path = [](const fs::path &p) {
        std::string out;
        for (char c : p.string()) {
            if (c == '\\') out += "\\\\";
            else out += c;
        }
        return out;
    };
    const fs::path target = f.outside / "rc.txt";
    const std::string args = "{\"command\":\"echo hi > " + json_path(target) + "\"}";

    agent::Config cfg;
    cfg.tool_sandbox.enabled = true;
    cfg.tool_sandbox.allow_tmp = false; // the fixture lives under the temp directory, which the sandbox otherwise leaves writable
    cfg.sandbox_filesystem = true;
    {
        agent::Session session = f.open(cfg);
        auto first = agent::execute_tool(session, "run_command", args);
        INFO(first.ok << " " << first.value << " " << first.error.message);
        CHECK_FALSE(fs::exists(target)); // refused: no grant and no gate
    }
    cfg.path_grants = {agent::PathGrant{.path = f.outside.string(), .write = true, .consumer = "run_command"}};
    {
        agent::Session session = f.open(cfg);
        auto r = agent::execute_tool(session, "run_command", args);
        REQUIRE(r.ok);
        CHECK(fs::exists(target)); // granted: the command ran in the sandbox with that folder writable
    }
}
