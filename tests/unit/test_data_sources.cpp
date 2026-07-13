#include <catch_amalgamated.hpp>

#include <agent-cpp/agent.hpp>
#include <string>

namespace {

const char *kWorkspace = "/tmp/agent_test_workspace";

struct FakeStore {
    std::string canned_response;
};

agent::Result<std::string> fake_query(void *state, std::string_view query, void *) {
    auto *store = static_cast<FakeStore *>(state);
    if (query.empty())
        return agent::fail<std::string>(agent::ErrorCode::InvalidConfig, "empty query");
    return agent::ok(store->canned_response);
}

} // namespace

TEST_CASE("register_data_source and query_data_source", "[data_sources]") {
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock,
                     .workspace_dir = kWorkspace,
                     .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto store = std::make_shared<FakeStore>();
    store->canned_response = "installation steps: run cmake -B build";

    auto reg_res = agent::register_data_source(
        session, agent::DataSource{.name = "docs", .state = store, .query = fake_query});
    REQUIRE(reg_res.ok);

    auto q_res = agent::query_data_source(session, "docs", "how to install");
    REQUIRE(q_res.ok);
    REQUIRE(q_res.value == "installation steps: run cmake -B build");

    auto missing_res = agent::query_data_source(session, "nonexistent", "x");
    REQUIRE_FALSE(missing_res.ok);
    REQUIRE(missing_res.error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("bind_data_source_tool wraps a DataSource as a Tool", "[data_sources][tools]") {
    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock,
                     .workspace_dir = kWorkspace,
                     .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto store = std::make_shared<FakeStore>();
    store->canned_response = "42";

    auto source = std::make_shared<agent::DataSource>(
        agent::DataSource{.name = "calc_store", .state = store, .query = fake_query});

    agent::Tool tool =
        agent::bind_data_source_tool(session, source, "test data source tool", R"({"type":"object"})");
    REQUIRE(tool.name == "calc_store");
    REQUIRE(tool.callback != nullptr);

    auto exec_res = tool.callback("what is the answer", tool.user_data);
    REQUIRE(exec_res.ok);
    REQUIRE(exec_res.value == "42");
}
