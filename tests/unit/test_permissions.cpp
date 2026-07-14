#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <string>

namespace {

int g_call_count = 0;

agent::Tool make_counting_tool(const char *name) {
    agent::Tool t;
    t.name = name;
    t.description = "test tool";
    t.parameter_schema = "{}";
    t.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        g_call_count++;
        return agent::ok(std::string("ran"));
    };
    return t;
}

agent::PermissionDecision always_deny(std::string_view, std::string_view, void *) { return agent::PermissionDecision::Deny; }

agent::PermissionDecision always_ask(std::string_view, std::string_view, void *) { return agent::PermissionDecision::Ask; }

struct PromptCall {
    std::string tool_name;
    std::string arguments;
    bool answer = false;
};

bool recording_prompt(std::string_view tool_name, std::string_view arguments, void *user_data) {
    auto *call = static_cast<PromptCall *>(user_data);
    call->tool_name = std::string(tool_name);
    call->arguments = std::string(arguments);
    return call->answer;
}

} // namespace

TEST_CASE("permission_check = nullptr allows everything (backward compatible default)", "[permissions]") {
    g_call_count = 0;
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    session.config.tools.push_back(make_counting_tool("counter"));
    agent::rebuild_tool_index(session);

    auto res = agent::execute_tool(session, "counter", "{}");
    REQUIRE(res.ok);
    REQUIRE(g_call_count == 1);
}

TEST_CASE("Deny blocks dispatch without ever invoking the tool callback", "[permissions]") {
    g_call_count = 0;
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws", .permission_check = always_deny});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    session.config.tools.push_back(make_counting_tool("counter"));
    agent::rebuild_tool_index(session);

    auto res = agent::execute_tool(session, "counter", "{}");
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::PermissionDenied);
    REQUIRE(g_call_count == 0);
}

TEST_CASE("Ask with no permission_prompt is treated as Deny", "[permissions]") {
    g_call_count = 0;
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws", .permission_check = always_ask});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    session.config.tools.push_back(make_counting_tool("counter"));
    agent::rebuild_tool_index(session);

    auto res = agent::execute_tool(session, "counter", "{}");
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::PermissionDenied);
    REQUIRE(g_call_count == 0);
}

TEST_CASE("Ask invokes permission_prompt with the tool name and arguments, honoring the answer", "[permissions]") {
    g_call_count = 0;
    PromptCall call;
    call.answer = true;

    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = "/tmp/agent_test_ws",
                                       .permission_check = always_ask,
                                       .permission_prompt = recording_prompt,
                                       .permission_user_data = &call});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    session.config.tools.push_back(make_counting_tool("counter"));
    agent::rebuild_tool_index(session);

    auto res = agent::execute_tool(session, "counter", R"({"x":1})");
    REQUIRE(res.ok);
    REQUIRE(g_call_count == 1);
    REQUIRE(call.tool_name == "counter");
    REQUIRE(call.arguments == R"({"x":1})");

    call.answer = false;
    auto res2 = agent::execute_tool(session, "counter", "{}");
    REQUIRE_FALSE(res2.ok);
    REQUIRE(res2.error.code == agent::ErrorCode::PermissionDenied);
    REQUIRE(g_call_count == 1); // unchanged -- second call was declined
}

TEST_CASE("default_permission_policy asks before risky tools and allows the rest", "[permissions]") {
    REQUIRE(agent::tools::default_permission_policy("run_command", "", nullptr) == agent::PermissionDecision::Ask);
    REQUIRE(agent::tools::default_permission_policy("delete_path", "", nullptr) == agent::PermissionDecision::Ask);
    REQUIRE(agent::tools::default_permission_policy("move_path", "", nullptr) == agent::PermissionDecision::Ask);
    REQUIRE(agent::tools::default_permission_policy("read_file", "", nullptr) == agent::PermissionDecision::Allow);
}
