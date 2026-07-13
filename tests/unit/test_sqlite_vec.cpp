#include <catch_amalgamated.hpp>

#include <agent-cpp/agent.hpp>
#include <filesystem>
#include <string>
#include <vector>

TEST_CASE("sqlite-vec DataSource finds nearest neighbor", "[sqlite_vec]") {
    const std::string db_path = "/tmp/agent_test_ws/test_sqlite_vec.sqlite3";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);

    auto source_result = agent::make_sqlite_vec_data_source("docs", db_path, 3);
    REQUIRE(source_result.ok);
    auto source = source_result.value;

    auto ins1 = agent::sqlite_vec_insert(*source, "near x-axis", std::vector<float>{1.0f, 0.0f, 0.0f});
    REQUIRE(ins1.ok);
    auto ins2 = agent::sqlite_vec_insert(*source, "near y-axis", std::vector<float>{0.0f, 1.0f, 0.0f});
    REQUIRE(ins2.ok);
    auto ins3 = agent::sqlite_vec_insert(*source, "near z-axis", std::vector<float>{0.0f, 0.0f, 1.0f});
    REQUIRE(ins3.ok);

    auto query_res =
        source->query(source->state.get(), R"({"vector": [0.9, 0.1, 0.0], "top_k": 2})", source->user_data);
    REQUIRE(query_res.ok);
    REQUIRE(query_res.value.find("near x-axis") != std::string::npos);
    REQUIRE(query_res.value.find("distance") != std::string::npos);

    std::filesystem::remove(db_path, ec);
}
