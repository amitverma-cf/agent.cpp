#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <string>

namespace {

const char *kWorkspace = "/tmp/agent_test_engines_ws";

agent::Config engine_config(std::string model = "mock.bin") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.model = std::move(model);
    cfg.workspace_dir = kWorkspace;
    return cfg;
}

int g_ping_calls = 0;

agent::Tool make_ping_tool() {
    agent::Tool t;
    t.name = "ping";
    t.description = "test tool";
    t.parameter_schema = "{}";
    t.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        ++g_ping_calls;
        return agent::ok(std::string("pong"));
    };
    return t;
}

} // namespace

TEST_CASE("load_engine_plugin loads the mock engine and is idempotent", "[engines]") {
    auto first = agent::load_engine_plugin(AGENT_TEST_MOCK_ENGINE_PATH);
    REQUIRE(first.ok);
    REQUIRE(first.value == "mock");

    const agent_engine_v1 *engine = agent::find_engine("mock");
    REQUIRE(engine != nullptr);
    REQUIRE(engine->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE((engine->caps & AGENT_CAP_TEXT_GEN) != 0);

    auto second = agent::load_engine_plugin(AGENT_TEST_MOCK_ENGINE_PATH);
    REQUIRE(second.ok);
    REQUIRE(agent::find_engine("mock") == engine);
}

TEST_CASE("load_engine_plugin reports a missing library", "[engines]") {
    auto res = agent::load_engine_plugin("/nonexistent/agent-engine-nope.so");
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ProviderInitFailed);
}

TEST_CASE("register_engine rejects an ABI mismatch and incomplete tables", "[engines]") {
    static const agent_engine_v1 wrong_abi = {AGENT_ENGINE_ABI_VERSION + 1, "wrong_abi", "0", 0, 0, nullptr, nullptr, nullptr,
                                              nullptr,                      nullptr,     nullptr, nullptr};
    auto r1 = agent::register_engine(&wrong_abi);
    REQUIRE_FALSE(r1.ok);
    REQUIRE(r1.error.code == agent::ErrorCode::UnsupportedProvider);

    static const agent_engine_v1 incomplete = {AGENT_ENGINE_ABI_VERSION, "incomplete", "0", 0, 0, nullptr, nullptr, nullptr,
                                               nullptr,                  nullptr,      nullptr, nullptr};
    auto r2 = agent::register_engine(&incomplete);
    REQUIRE_FALSE(r2.ok);
    REQUIRE(r2.error.code == agent::ErrorCode::InvalidConfig);
    REQUIRE(agent::find_engine("incomplete") == nullptr);
}

TEST_CASE("AiProvider::Engine runs a turn through the plugin", "[engines]") {
    auto res = agent::init(engine_config());
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "engine_turn";
    auto reply = agent::run_turn(session, memory, "hello");
    REQUIRE(reply.ok);
    REQUIRE(reply.value == "echo: hello");
}

TEST_CASE("AiProvider::Engine forwards streamed text chunks", "[engines]") {
    auto res = agent::init(engine_config());
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    std::string streamed;
    agent::FlowMemory memory;
    memory.id = "engine_stream";
    auto reply = agent::run_turn(
        session, memory, "stream me", true, [](std::string_view tok, void *ud) { *static_cast<std::string *>(ud) += tok; }, &streamed);
    REQUIRE(reply.ok);
    REQUIRE(streamed == "echo: stream me");
}

TEST_CASE("AiProvider::Engine tool calls go through execute_tool", "[engines]") {
    auto cfg = engine_config();
    cfg.auto_default_tools = false;
    cfg.tools = {make_ping_tool()};
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    g_ping_calls = 0;
    agent::FlowMemory memory;
    memory.id = "engine_tools";
    auto reply = agent::run_turn(session, memory, "call:ping");
    REQUIRE(reply.ok);
    REQUIRE(g_ping_calls == 1);
    REQUIRE(reply.value == "tool done");
}

TEST_CASE("AiProvider::Engine surfaces create() failures", "[engines]") {
    auto res = agent::init(engine_config("fail"));
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
    REQUIRE(res.error.message.find("refusing model") != std::string::npos);
}

TEST_CASE("AiProvider::Engine requires a known engine", "[engines]") {
    auto cfg = engine_config();
    cfg.engine = "no_such_engine";
    auto res = agent::init(cfg);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::UnsupportedProvider);

    cfg.engine.clear();
    auto res2 = agent::init(cfg);
    REQUIRE_FALSE(res2.ok);
    REQUIRE(res2.error.code == agent::ErrorCode::InvalidConfig);
}
