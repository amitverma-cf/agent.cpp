// An MCP server reached over Streamable HTTP: a loopback server stands in for it, the transport is the hub's HTTP client.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/hub.hpp>
#include <atomic>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <httplib.h>
#include <thread>

namespace {

agent::Result<agent::HttpReply> hub_post(std::string_view url, const std::vector<std::pair<std::string, std::string>> &headers,
                                         std::string_view body, int timeout_ms, void *) {
    agent::hub::HttpOptions http;
    auto r = agent::hub::http_post(url, headers, body, http, std::max(1, timeout_ms / 1000));
    if (!r.ok) return agent::fail<agent::HttpReply>(r.error.code, r.error.message);
    return agent::ok(agent::HttpReply{r.value.status, r.value.content_type, r.value.session_id, r.value.body});
}

std::string id_of(const std::string &body) {
    const size_t at = body.find("\"id\":");
    if (at == std::string::npos) return "null";
    size_t end = at + 5;
    while (end < body.size() && body[end] != ',' && body[end] != '}') ++end;
    return body.substr(at + 5, end - at - 5);
}

bool has(const std::string &body, const char *method) {
    return body.find(std::string("\"method\":\"") + method + "\"") != std::string::npos;
}

} // namespace

TEST_CASE("mcp: a server reached over Streamable HTTP", "[mcp][http]") {
    std::atomic<int> without_session{0};
    std::atomic<int> with_session{0};
    std::atomic<int> unauthorized{0};
    httplib::Server svr;
    svr.Post("/mcp", [&](const httplib::Request &req, httplib::Response &res) {
        if (req.get_header_value("Authorization") != "Bearer sek") {
            ++unauthorized;
            res.status = 401;
            return;
        }
        const std::string &b = req.body;
        const std::string id = id_of(b);
        auto json = [&](const std::string &result) {
            res.set_content("{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":" + result + "}", "application/json");
        };
        if (has(b, "initialize")) {
            res.set_header("Mcp-Session-Id", "session-1");
            json("{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{},\"resources\":{},\"prompts\":{}},\"serverInfo\":{"
                 "\"name\":\"fake\",\"version\":\"1\"}}");
            return;
        }
        if (has(b, "server/discover")) { // the probe of a newer-revision client: this server predates it
            res.status = 400;
            return;
        }
        (req.get_header_value("Mcp-Session-Id") == "session-1" ? with_session : without_session)++;
        if (has(b, "notifications/initialized")) {
            res.status = 202;
        } else if (has(b, "tools/list")) {
            json("{\"tools\":[{\"name\":\"echo\",\"description\":\"echoes\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"text\":{"
                 "\"type\":\"string\"}}}}]}");
        } else if (has(b, "tools/call")) {
            // The answer comes as a stream of events, as many servers send it.
            res.set_content("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":" + id +
                                ",\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"echo: hi\"}]}}\n\n",
                            "text/event-stream");
        } else if (has(b, "resources/list")) {
            json("{\"resources\":[{\"uri\":\"file:///notes\",\"name\":\"notes\"}]}");
        } else if (has(b, "resources/read")) {
            json("{\"contents\":[{\"uri\":\"file:///notes\",\"text\":\"the budget is 42\"}]}");
        } else if (has(b, "prompts/list")) {
            json("{\"prompts\":[{\"name\":\"review\"}]}");
        } else {
            res.status = 400;
        }
    });
    const int port = svr.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread server([&] { svr.listen_after_bind(); });

    const auto ws = std::filesystem::temp_directory_path() / "agent_mcp_http_ws";
    std::filesystem::remove_all(ws);
    std::filesystem::create_directories(ws);
    agent::Config cfg;
    cfg.workspace_dir = ws.string();
    cfg.http_post = hub_post;
    auto session = agent::init(cfg);
    REQUIRE(session.ok);

    SECTION("a wrong token is refused") {
        agent::McpServerConfig mc;
        mc.name = "remote";
        mc.url = "http://127.0.0.1:" + std::to_string(port) + "/mcp";
        mc.headers["Authorization"] = "Bearer wrong";
        auto r = agent::connect_mcp_server(session.value, mc);
        CHECK_FALSE(r.ok);
        CHECK(unauthorized.load() >= 1);
    }

    SECTION("tools, resources and prompts") {
        agent::McpServerConfig mc;
        mc.name = "remote";
        mc.url = "http://127.0.0.1:" + std::to_string(port) + "/mcp";
        mc.headers["Authorization"] = "Bearer sek";
        auto r = agent::connect_mcp_server(session.value, mc);
        INFO(r.error.message);
        REQUIRE(r.ok);
        std::vector<std::string> names;
        for (const auto &t : r.value) names.emplace_back(t.name);
        CHECK(std::find(names.begin(), names.end(), "remote.echo") != names.end());
        CHECK(std::find(names.begin(), names.end(), "remote.list_resources") != names.end());
        CHECK(std::find(names.begin(), names.end(), "remote.read_resource") != names.end());
        CHECK(std::find(names.begin(), names.end(), "remote.list_prompts") != names.end());

        auto echo = agent::execute_tool(session.value, "remote.echo", "{\"text\":\"hi\"}");
        INFO(echo.error.message);
        REQUIRE(echo.ok);
        CHECK(echo.value.find("echo: hi") != std::string::npos); // read from the event stream

        auto listed = agent::execute_tool(session.value, "remote.list_resources", "{}");
        REQUIRE(listed.ok);
        CHECK(listed.value.find("file:///notes") != std::string::npos);
        auto read = agent::execute_tool(session.value, "remote.read_resource", "{\"uri\":\"file:///notes\"}");
        REQUIRE(read.ok);
        CHECK(read.value.find("the budget is 42") != std::string::npos);
        auto prompts = agent::execute_tool(session.value, "remote.list_prompts", "{}");
        REQUIRE(prompts.ok);
        CHECK(prompts.value.find("review") != std::string::npos);

        // Every request after the handshake carried the session the server handed out.
        CHECK(with_session.load() >= 5);
        CHECK(without_session.load() == 0);
        agent::disconnect_mcp_server(session.value, "remote");
    }

    SECTION("a url without Config::http_post is an error") {
        agent::Config bare;
        bare.workspace_dir = ws.string();
        auto other = agent::init(bare);
        REQUIRE(other.ok);
        agent::McpServerConfig mc;
        mc.name = "remote";
        mc.url = "http://127.0.0.1:1/mcp";
        auto r = agent::connect_mcp_server(other.value, mc);
        CHECK_FALSE(r.ok);
        CHECK(r.error.message.find("http_post") != std::string::npos);
    }

    svr.stop();
    server.join();
}

TEST_CASE("mcp: a 2026-07-28 server needs no handshake", "[mcp][http]") {
    std::atomic<int> handshakes{0};
    std::atomic<int> with_meta{0};
    std::atomic<int> with_headers{0};
    httplib::Server svr;
    svr.Post("/mcp", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string &b = req.body;
        const std::string id = id_of(b);
        auto json = [&](const std::string &result) {
            res.set_content("{\"jsonrpc\":\"2.0\",\"id\":" + id + ",\"result\":" + result + "}", "application/json");
        };
        if (has(b, "initialize") || has(b, "notifications/initialized")) {
            ++handshakes;
            res.status = 400;
            return;
        }
        if (has(b, "server/discover")) {
            json("{\"resultType\":\"complete\",\"supportedVersions\":[\"2026-07-28\"],\"capabilities\":{\"tools\":{}},\"_meta\":{"
                 "\"io.modelcontextprotocol/serverInfo\":{\"name\":\"modern\",\"version\":\"2\"}}}");
            return;
        }
        if (b.find("io.modelcontextprotocol/protocolVersion") != std::string::npos) ++with_meta;
        if (req.get_header_value("Mcp-Method") == (has(b, "tools/call") ? "tools/call" : "tools/list") &&
            req.get_header_value("MCP-Protocol-Version") == "2026-07-28")
            ++with_headers;
        if (has(b, "tools/list"))
            json("{\"resultType\":\"complete\",\"ttlMs\":0,\"cacheScope\":\"private\",\"tools\":[{\"name\":\"echo\",\"description\":\"e\","
                 "\"inputSchema\":{\"type\":\"object\"}}]}");
        else if (has(b, "tools/call") && req.get_header_value("Mcp-Name") == "echo")
            json("{\"resultType\":\"complete\",\"content\":[{\"type\":\"text\",\"text\":\"modern ok\"}]}");
        else res.status = 400;
    });
    const int port = svr.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread server([&] { svr.listen_after_bind(); });

    const auto ws = std::filesystem::temp_directory_path() / "agent_mcp_modern_ws";
    std::filesystem::remove_all(ws);
    std::filesystem::create_directories(ws);
    agent::Config cfg;
    cfg.workspace_dir = ws.string();
    cfg.http_post = hub_post;
    auto session = agent::init(cfg);
    REQUIRE(session.ok);
    agent::McpServerConfig mc;
    mc.name = "m";
    mc.url = "http://127.0.0.1:" + std::to_string(port) + "/mcp";
    auto r = agent::connect_mcp_server(session.value, mc);
    INFO(r.error.message);
    REQUIRE(r.ok);
    auto out = agent::execute_tool(session.value, "m.echo", "{}");
    INFO(out.error.message);
    REQUIRE(out.ok);
    CHECK(out.value.find("modern ok") != std::string::npos);
    CHECK(handshakes.load() == 0);
    CHECK(with_meta.load() >= 2);
    CHECK(with_headers.load() >= 2);
    agent::disconnect_mcp_server(session.value, "m");
    svr.stop();
    server.join();
}
