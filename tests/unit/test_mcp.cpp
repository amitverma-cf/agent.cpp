#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

const char *kWorkspace = "/tmp/agent_test_mcp_ws";

struct LogCapture {
    std::mutex mutex;
    std::vector<std::string> lines;
    bool contains(std::string_view needle) {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &l : lines)
            if (l.find(needle) != std::string::npos) return true;
        return false;
    }
};

void capture_log(agent::LogLevel, std::string_view message, void *ud) {
    auto *cap = static_cast<LogCapture *>(ud);
    std::lock_guard<std::mutex> lock(cap->mutex);
    cap->lines.emplace_back(message);
}

agent::Session make_session(LogCapture *logs = nullptr) {
    agent::Config cfg;
    cfg.workspace_dir = kWorkspace;
    cfg.auto_default_tools = false;
    if (logs) {
        cfg.logger = capture_log;
        cfg.logger_user_data = logs;
    }
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    return std::move(res.value);
}

agent::McpServerConfig fake(std::string name = "fake", std::string mode = {}) {
    agent::McpServerConfig c;
    c.name = std::move(name);
    c.command = AGENT_TEST_FAKE_MCP_SERVER_PATH;
    if (!mode.empty()) c.args.push_back(std::move(mode));
    c.timeout_ms = 5000;
    return c;
}

agent::PermissionDecision deny_fail_tool(std::string_view tool, std::string_view, void *) {
    return tool == "fake.fail" ? agent::PermissionDecision::Deny : agent::PermissionDecision::Allow;
}

} // namespace

TEST_CASE("connect_mcp_server lists tools across pages and calls them", "[mcp]") {
    LogCapture logs;
    auto session = make_session(&logs);
    auto conn = agent::connect_mcp_server(session, fake());
    REQUIRE(conn.ok);
    REQUIRE(conn.value.size() == 2);
    CHECK(conn.value[0].name == "fake.echo");
    CHECK(conn.value[0].description == "Echo text back");
    CHECK(conn.value[0].parameter_schema == R"({"type":"object","properties":{"text":{"type":"string"}}})");
    CHECK(conn.value[1].name == "fake.fail");
    CHECK(session.tool_by_name.count("fake.echo") == 1);
    CHECK(session.tools_system_prompt.find("fake.echo") != std::string::npos);
    CHECK(agent::list_mcp_servers(session) == std::vector<std::string>{"fake"});

    // Arguments with whitespace/newlines are minified onto one line before sending.
    auto echo = agent::execute_tool(session, "fake.echo", "{\n  \"text\": \"hi there\"\n}");
    REQUIRE(echo.ok);
    CHECK(echo.value == "echo: hi there");

    auto empty_args = agent::execute_tool(session, "fake.echo", "");
    REQUIRE(empty_args.ok);
    CHECK(empty_args.value == "echo: ");

    auto failed = agent::execute_tool(session, "fake.fail", "{}");
    REQUIRE_FALSE(failed.ok);
    CHECK(failed.error.message == "boom");

    auto bad_args = agent::execute_tool(session, "fake.echo", "[1,2]");
    REQUIRE_FALSE(bad_args.ok);
    CHECK(bad_args.error.code == agent::ErrorCode::ParseError);
    auto not_json = agent::execute_tool(session, "fake.echo", "{oops");
    REQUIRE_FALSE(not_json.ok);
    CHECK(not_json.error.code == agent::ErrorCode::ParseError);

    // The server's stderr and log notifications reach the Session logger.
    for (int i = 0; i < 100 && !logs.contains("fake: started"); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(logs.contains("[mcp:fake] fake: started"));
    CHECK(logs.contains("working"));

    agent::disconnect_mcp_servers(session);
    CHECK(session.tool_by_name.count("fake.echo") == 0);
    CHECK(agent::list_mcp_servers(session).empty());
    auto after = agent::execute_tool(session, "fake.echo", "{}");
    CHECK_FALSE(after.ok);
}

TEST_CASE("MCP tools go through the permission gate and survive Session moves", "[mcp]") {
    agent::Config cfg;
    cfg.workspace_dir = kWorkspace;
    cfg.auto_default_tools = false;
    cfg.permission_check = deny_fail_tool;
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session first = std::move(res.value);
    REQUIRE(agent::connect_mcp_server(first, fake()).ok);

    agent::Session moved = std::move(first);
    auto echo = agent::execute_tool(moved, "fake.echo", R"({"text":"moved"})");
    REQUIRE(echo.ok);
    CHECK(echo.value == "echo: moved");
    auto denied = agent::execute_tool(moved, "fake.fail", "{}");
    REQUIRE_FALSE(denied.ok);
    CHECK(denied.error.code == agent::ErrorCode::PermissionDenied);
    // No explicit disconnect: destroying the Session stops the server.
}

TEST_CASE("MCP env, image content and duplicate names", "[mcp]") {
    auto session = make_session();
    auto cfg = fake();
    cfg.env["FAKE_MCP_VAR"] = "42";
    REQUIRE(agent::connect_mcp_server(session, cfg).ok);

    auto echo = agent::execute_tool(session, "fake.echo", R"({"text":"pic","image":true})");
    REQUIRE(echo.ok);
    CHECK(echo.value.starts_with("echo: pic env=42\n[image saved to "));
    auto start = echo.value.find("saved to ") + 9;
    auto end = echo.value.find(" (image/png");
    REQUIRE(end != std::string::npos);
    std::string path = echo.value.substr(start, end - start);
    CHECK(path.find("/artifacts/") != std::string::npos);
    CHECK(path.ends_with(".png"));
    std::ifstream f(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(bytes == "hello");
    std::filesystem::remove(path);

    auto dup = agent::connect_mcp_server(session, cfg);
    REQUIRE_FALSE(dup.ok);
    CHECK(dup.error.code == agent::ErrorCode::InvalidConfig);
    CHECK(dup.error.message.find("already connected") != std::string::npos);

    auto second = agent::connect_mcp_server(session, fake("other"));
    REQUIRE(second.ok);
    CHECK(agent::list_mcp_servers(session) == std::vector<std::string>{"fake", "other"});
    REQUIRE(agent::disconnect_mcp_server(session, "fake").ok);
    CHECK(session.tool_by_name.count("fake.echo") == 0);
    CHECK(session.tool_by_name.count("other.echo") == 1);
    CHECK(agent::execute_tool(session, "other.echo", R"({"text":"x"})").ok);
    CHECK(agent::disconnect_mcp_server(session, "fake").error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("MCP tolerates servers without initialize and non-JSON banners", "[mcp]") {
    auto session = make_session();
    auto no_init = agent::connect_mcp_server(session, fake("noinit", "--no-init"));
    REQUIRE(no_init.ok);
    CHECK(no_init.value.size() == 2);
    auto banner = agent::connect_mcp_server(session, fake("banner", "--banner"));
    REQUIRE(banner.ok);
    CHECK(agent::execute_tool(session, "banner.echo", R"({"text":"b"})").value == "echo: b");
}

TEST_CASE("MCP config validation and spawn failures", "[mcp]") {
    auto session = make_session();
    auto bad_name = fake("bad.name");
    CHECK(agent::connect_mcp_server(session, bad_name).error.code == agent::ErrorCode::InvalidConfig);
    auto no_cmd = fake();
    no_cmd.command.clear();
    CHECK(agent::connect_mcp_server(session, no_cmd).error.code == agent::ErrorCode::InvalidConfig);

    auto missing = fake();
    missing.command = "/nonexistent/mcp-server-nope";
    auto r = agent::connect_mcp_server(session, missing);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == agent::ErrorCode::ProviderInitFailed);

    auto not_on_path = fake();
    not_on_path.command = "agent-cpp-no-such-mcp-server";
    CHECK_FALSE(agent::connect_mcp_server(session, not_on_path).ok);

    auto exits = agent::connect_mcp_server(session, fake("exits", "--exit-at-start"));
    REQUIRE_FALSE(exits.ok);
    CHECK(exits.error.code == agent::ErrorCode::ProviderInitFailed);
    CHECK(exits.error.message.find("exit code 2") != std::string::npos);
    CHECK(agent::list_mcp_servers(session).empty());
}

TEST_CASE("MCP reports a server that exits mid-call", "[mcp]") {
    auto session = make_session();
    REQUIRE(agent::connect_mcp_server(session, fake("dies", "--exit-on-call")).ok);
    auto r = agent::execute_tool(session, "dies.echo", "{}");
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == agent::ErrorCode::NetworkError);
    CHECK(r.error.message.find("exit code 3") != std::string::npos);
    CHECK(r.error.message.find("dying on purpose") != std::string::npos);
    // Later calls fail fast instead of writing to a dead pipe.
    auto again = agent::execute_tool(session, "dies.echo", "{}");
    REQUIRE_FALSE(again.ok);
    CHECK(again.error.message.find("not running") != std::string::npos);
    REQUIRE(agent::disconnect_mcp_server(session, "dies").ok);
}

TEST_CASE("MCP reports invalid JSON from the server", "[mcp]") {
    auto session = make_session();
    REQUIRE(agent::connect_mcp_server(session, fake("junk", "--bad-json")).ok);
    auto r = agent::execute_tool(session, "junk.echo", "{}");
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == agent::ErrorCode::ParseError);
    CHECK(r.error.message.find("invalid JSON") != std::string::npos);
}

TEST_CASE("MCP request timeout keeps the connection usable", "[mcp]") {
    auto session = make_session();
    auto cfg = fake("slow", "--sleep-tool");
    cfg.timeout_ms = 300;
    auto conn = agent::connect_mcp_server(session, cfg);
    REQUIRE(conn.ok);
    REQUIRE(conn.value.size() == 3);

    auto t0 = std::chrono::steady_clock::now();
    auto r = agent::execute_tool(session, "slow.sleep", R"({"ms":800})");
    auto elapsed = std::chrono::steady_clock::now() - t0;
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == agent::ErrorCode::NetworkError);
    CHECK(r.error.message.find("timed out") != std::string::npos);
    CHECK(elapsed < std::chrono::milliseconds(700));

    // The late reply to the timed-out request is skipped; the next call gets its own answer.
    auto next_call = agent::execute_tool(session, "slow.sleep", R"({"ms":0})");
    // This one may itself time out while the server finishes the 800 ms sleep; retry until it answers.
    for (int i = 0; i < 5 && !next_call.ok; ++i) next_call = agent::execute_tool(session, "slow.sleep", R"({"ms":0})");
    REQUIRE(next_call.ok);
    CHECK(next_call.value == "slept");
}

TEST_CASE("MCP serialises concurrent calls on one server", "[mcp]") {
    auto session = make_session();
    REQUIRE(agent::connect_mcp_server(session, fake()).ok);
    std::vector<std::thread> threads;
    std::vector<std::string> results(8);
    for (size_t i = 0; i < results.size(); ++i) {
        threads.emplace_back([&session, &results, i]() {
            auto r = agent::execute_tool(session, "fake.echo", "{\"text\":\"" + std::to_string(i) + "\"}");
            results[i] = r.ok ? r.value : "error: " + r.error.message;
        });
    }
    for (auto &t : threads) t.join();
    for (size_t i = 0; i < results.size(); ++i) CHECK(results[i] == "echo: " + std::to_string(i));
}
