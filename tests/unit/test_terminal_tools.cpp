#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#ifndef _WIN32
#include <csignal>
#include <sys/types.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

std::string make_workspace(const char *name) {
    std::string dir = "/tmp/agent_test_term_" + std::string(name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

const agent::Tool *find_tool(const agent::Session &session, std::string_view name) {
    for (const auto &t : session.config.tools)
        if (t.name == name) return &t;
    return nullptr;
}

long long elapsed_ms(Clock::time_point since) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - since).count();
}

[[maybe_unused]] std::string read_text_file(const std::string &path) {
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Calls request_cancel(session) from another thread once `marker` exists in the workspace (the
// command is up and running) plus a short delay, or after 5 s regardless.
std::thread cancel_when_started(agent::Session &session, const std::string &marker) {
    return std::thread([&session, marker] {
        auto t0 = Clock::now();
        while (elapsed_ms(t0) < 5000 && !std::filesystem::exists(marker)) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        agent::request_cancel(session);
    });
}

#ifndef _WIN32
// kill(pid, 0) alone cannot tell a dead process from a zombie nobody reaped (PID 1 of a container).
bool process_alive(long pid) {
    if (kill(static_cast<pid_t>(pid), 0) != 0) return false;
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (std::getline(stat, line)) {
        auto close_paren = line.rfind(')');
        if (close_paren != std::string::npos && close_paren + 2 < line.size() && line[close_paren + 2] == 'Z') return false;
    }
    return true;
}

bool wait_until_dead(long pid, int max_ms) {
    auto t0 = Clock::now();
    while (process_alive(pid)) {
        if (elapsed_ms(t0) > max_ms) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}
#endif

} // namespace

TEST_CASE("run_command executes and captures output + exit code", "[terminal_tools]") {
    std::string ws = make_workspace("basic");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
    REQUIRE(run_tool != nullptr);

    auto res = run_tool->callback(R"({"command":"echo hello"})", run_tool->user_data);
    REQUIRE(res.ok);
    REQUIRE(res.value.find("\"exit_code\":0") != std::string::npos);
    REQUIRE(res.value.find("hello") != std::string::npos);
}

TEST_CASE("run_command reports a nonzero exit code", "[terminal_tools]") {
    std::string ws = make_workspace("exitcode");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
    auto res = run_tool->callback(R"({"command":"exit 3"})", run_tool->user_data);
    REQUIRE(res.ok);
    REQUIRE(res.value.find("\"exit_code\":3") != std::string::npos);
}

TEST_CASE("run_command enforces a timeout", "[terminal_tools]") {
    std::string ws = make_workspace("timeout");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
#ifdef _WIN32
    auto res = run_tool->callback(R"({"command":"for /l %i in (1,1,2000000000) do @rem","timeout_seconds":1})", run_tool->user_data);
#else
    auto res = run_tool->callback(R"({"command":"sleep 10","timeout_seconds":1})", run_tool->user_data);
#endif
    REQUIRE(res.ok);
    INFO(res.value);
    REQUIRE(res.value.find("\"timed_out\":true") != std::string::npos);
}

TEST_CASE("run_command requires a 'command' field", "[terminal_tools]") {
    std::string ws = make_workspace("missing_cmd");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *run_tool = find_tool(session, "run_command");
    auto res = run_tool->callback(R"({})", run_tool->user_data);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("run_command stops on request_cancel and returns the partial output", "[terminal_tools][cancel]") {
    std::string ws = make_workspace("cancel");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    const auto *run_tool = find_tool(session, "run_command");
    REQUIRE(run_tool != nullptr);

#ifdef _WIN32
    const char *args = R"({"command":"echo partial & echo x > started.txt & for /l %i in (1,1,2000000000) do @rem","timeout_seconds":0})";
#else
    const char *args = R"({"command":"echo partial; echo x > started.txt; sleep 30","timeout_seconds":0})";
#endif
    std::thread canceller = cancel_when_started(session, ws + "/started.txt");
    auto t0 = Clock::now();
    auto res = run_tool->callback(args, run_tool->user_data);
    long long took = elapsed_ms(t0);
    canceller.join();
    agent::clear_cancel(session);

    REQUIRE(res.ok);
    CHECK(took < 2500);
    CHECK(res.value.find("\"cancelled\":true") != std::string::npos);
    CHECK(res.value.find("\"timed_out\":false") != std::string::npos);
    CHECK(res.value.find("partial") != std::string::npos);
}

TEST_CASE("run_command with the cancel flag already set does not start", "[terminal_tools][cancel]") {
    std::string ws = make_workspace("cancel_preset");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    const auto *run_tool = find_tool(session, "run_command");

    agent::request_cancel(session);
    auto res = run_tool->callback(R"({"command":"echo hi > should_not_exist.txt"})", run_tool->user_data);
    agent::clear_cancel(session);
    REQUIRE_FALSE(res.ok);
    CHECK(res.error.code == agent::ErrorCode::Cancelled);
    CHECK_FALSE(std::filesystem::exists(ws + "/should_not_exist.txt"));

    // With the flag cleared again the tool works as before.
    auto again = run_tool->callback(R"({"command":"echo hello"})", run_tool->user_data);
    REQUIRE(again.ok);
    CHECK(again.value.find("\"cancelled\":false") != std::string::npos);
}

#ifndef _WIN32
TEST_CASE("run_command cancel kills grandchildren too", "[terminal_tools][cancel]") {
    std::string ws = make_workspace("cancel_tree");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    const auto *run_tool = find_tool(session, "run_command");

    // A background grandchild (its pid is written down) and a nested shell with its own children.
    const char *args =
        R"({"command":"sleep 30 & echo $! > bg.pid; sh -c 'echo $$ > sh.pid; sleep 30 & echo $! > fg.pid; sleep 30'","timeout_seconds":0})";
    std::thread canceller = cancel_when_started(session, ws + "/fg.pid");
    auto t0 = Clock::now();
    auto res = run_tool->callback(args, run_tool->user_data);
    long long took = elapsed_ms(t0);
    canceller.join();
    agent::clear_cancel(session);

    REQUIRE(res.ok);
    CHECK(took < 2500);
    CHECK(res.value.find("\"cancelled\":true") != std::string::npos);
    for (const char *name : {"bg.pid", "sh.pid", "fg.pid"}) {
        long pid = std::atol(read_text_file(ws + "/" + name).c_str());
        INFO(name << " = " << pid);
        REQUIRE(pid > 1);
        CHECK(wait_until_dead(pid, 1000));
    }
}

TEST_CASE("run_command cancel escalates to SIGKILL for a command that ignores SIGTERM", "[terminal_tools][cancel]") {
    std::string ws = make_workspace("cancel_sigkill");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    const auto *run_tool = find_tool(session, "run_command");

    const char *args = R"({"command":"trap '' TERM; echo $$ > stubborn.pid; while :; do sleep 1; done","timeout_seconds":0})";
    std::thread canceller = cancel_when_started(session, ws + "/stubborn.pid");
    auto t0 = Clock::now();
    auto res = run_tool->callback(args, run_tool->user_data);
    long long took = elapsed_ms(t0);
    canceller.join();
    agent::clear_cancel(session);

    REQUIRE(res.ok);
    CHECK(took < 3000);
    CHECK(res.value.find("\"cancelled\":true") != std::string::npos);
    long pid = std::atol(read_text_file(ws + "/stubborn.pid").c_str());
    REQUIRE(pid > 1);
    CHECK(wait_until_dead(pid, 1000));
}

TEST_CASE("run_command timeout kills grandchildren too", "[terminal_tools][cancel]") {
    std::string ws = make_workspace("timeout_tree");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    const auto *run_tool = find_tool(session, "run_command");

    auto t0 = Clock::now();
    auto res = run_tool->callback(R"({"command":"sleep 30 & echo $! > bg.pid; sleep 30","timeout_seconds":1})", run_tool->user_data);
    long long took = elapsed_ms(t0);
    REQUIRE(res.ok);
    CHECK(took < 3500);
    CHECK(res.value.find("\"timed_out\":true") != std::string::npos);
    CHECK(res.value.find("\"cancelled\":false") != std::string::npos);
    long pid = std::atol(read_text_file(ws + "/bg.pid").c_str());
    REQUIRE(pid > 1);
    CHECK(wait_until_dead(pid, 1000));
}
#endif
