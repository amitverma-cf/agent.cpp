#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

int g_log_count = 0;
void test_logger(agent::LogLevel, std::string_view, void *) { g_log_count++; }

} // namespace

TEST_CASE("Logger callback fires during inference", "[utils][logging]") {
    g_log_count = 0;
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws", .logger = test_logger});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::MessageView user_msg;
    user_msg.role = "user";
    user_msg.content = "test";
    user_msg.tool_calls = {};
    user_msg.tool_call_id = "";

    agent::InferRequest request{.messages = std::span<const agent::MessageView>(&user_msg, 1),
                                .stream = true,
                                .on_token = [](std::string_view, void *) {},
                                .token_user_data = nullptr};

    auto stream_res = agent::infer(session, request);
    REQUIRE(stream_res.ok);
    REQUIRE(g_log_count > 0);
}

TEST_CASE("File logging writes timestamped entries to workspace_dir/logs", "[utils][logging]") {
    const std::string ws = "/tmp/agent_test_filelog";
    std::error_code ec;
    std::filesystem::remove_all(ws, ec);

    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws, .enable_file_logging = true});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    agent::FlowMemory conv;
    conv.id = "filelog_chat";
    REQUIRE(agent::run_turn(session, conv, "hello").ok);

    std::filesystem::path logs_dir = std::filesystem::path(ws) / "logs";
    REQUIRE(std::filesystem::is_directory(logs_dir));

    std::filesystem::path log_file;
    for (const auto &entry : std::filesystem::directory_iterator(logs_dir)) {
        if (entry.path().extension() == ".log") log_file = entry.path();
    }
    REQUIRE_FALSE(log_file.empty());

    std::ifstream f(log_file);
    std::stringstream contents;
    contents << f.rdbuf();
    std::string text = contents.str();

    REQUIRE(text.find("Session initialised.") != std::string::npos);
    REQUIRE(text.find("[INFO]") != std::string::npos);
}
