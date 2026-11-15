// Config::tool_result_limit: a long tool result reaches the history shortened, with the whole text kept in .agents/results/.

#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

namespace fs = std::filesystem;

agent::Result<std::string> long_output(std::string_view, void *) {
    std::string text;
    for (int i = 0; i < 400; ++i) text += "line " + std::to_string(i) + " of a long listing\n";
    return agent::ok(std::move(text));
}

agent::Config limited_config(const std::string &name, size_t limit) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.workspace_dir = (fs::temp_directory_path() / name).string();
    cfg.tool_result_limit = limit;
    cfg.tools.push_back(agent::Tool{.name = "long_output", .description = "a long listing", .callback = long_output});
    fs::remove_all(cfg.workspace_dir);
    return cfg;
}

const agent::Message *last_tool_message(const agent::Conversation &memory) {
    for (size_t i = memory.history.size(); i-- > 0;)
        if (memory.history[i].role == "tool") return &memory.history[i];
    return nullptr;
}

} // namespace

TEST_CASE("tool result limit: a long result is shortened and the whole text is kept", "[tool_limit]") {
    auto cfg = limited_config("agent_test_tool_limit_on", 1000);
    auto s = agent::init(cfg);
    REQUIRE(s.ok);
    agent::Conversation memory;
    memory.id = "limit_on";
    REQUIRE(agent::run_turn(s.value, memory, "call:long_output {}").ok);
    const agent::Message *tool = last_tool_message(memory);
    REQUIRE(tool);
    CHECK(tool->content.size() < 1600);
    CHECK(tool->content.find("line 0 ") == 0);
    CHECK(tool->content.find("line 399 of a long listing") != std::string::npos);
    CHECK(tool->content.find("left out of") != std::string::npos);
    CHECK(tool->content.find(".agents/results/long_output-") != std::string::npos);

    size_t files = 0;
    for (const auto &entry : fs::directory_iterator(fs::path(cfg.workspace_dir) / ".agents" / "results")) {
        ++files;
        std::ifstream in(entry.path(), std::ios::binary);
        const std::string whole((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(whole.find("line 200 of a long listing") != std::string::npos);
    }
    CHECK(files == 1);
}

TEST_CASE("tool result limit: off by default, and short results pass through", "[tool_limit]") {
    {
        auto s = agent::init(limited_config("agent_test_tool_limit_off", 0));
        REQUIRE(s.ok);
        agent::Conversation memory;
        memory.id = "limit_off";
        REQUIRE(agent::run_turn(s.value, memory, "call:long_output {}").ok);
        const agent::Message *tool = last_tool_message(memory);
        REQUIRE(tool);
        CHECK(tool->content.find("left out of") == std::string::npos);
        CHECK(tool->content.size() > 8000);
    }
    {
        auto s = agent::init(limited_config("agent_test_tool_limit_big", 1u << 20));
        REQUIRE(s.ok);
        agent::Conversation memory;
        memory.id = "limit_big";
        REQUIRE(agent::run_turn(s.value, memory, "call:long_output {}").ok);
        const agent::Message *tool = last_tool_message(memory);
        REQUIRE(tool);
        CHECK(tool->content.find("left out of") == std::string::npos);
    }
}
