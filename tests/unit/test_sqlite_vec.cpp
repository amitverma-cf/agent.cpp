#include <agent-cpp/agent.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <catch_amalgamated.hpp>
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

    auto query_res = source->query(source->state.get(), R"({"vector": [0.9, 0.1, 0.0], "top_k": 2})", source->user_data);
    REQUIRE(query_res.ok);
    REQUIRE(query_res.value.find("near x-axis") != std::string::npos);
    REQUIRE(query_res.value.find("distance") != std::string::npos);

    std::filesystem::remove(db_path, ec);
}

TEST_CASE("sqlite vector store: hybrid search finds a rare word that the embedding misses", "[sqlite_vec][hybrid]") {
    const std::string db_path = "/tmp/agent_test_ws/test_hybrid.sqlite3";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);

    agent::RagIndex index;
    index.store = agent::store::sqlite_vector_store();
    index.config.name = "kb";
    index.config.path = db_path;
    index.config.dimensions = 3;
    REQUIRE(agent::open_rag_index(index).ok);
    auto *ops = index.store;
    REQUIRE(ops->query_hybrid != nullptr);

    // Three entries whose embeddings are all far from the query vector; only one holds the word "zx81".
    REQUIRE(ops->upsert(index.state.get(), "a#0", "The garden needs water twice a week.", std::vector<float>{0.0f, 1.0f, 0.0f}, "{}").ok);
    REQUIRE(ops->upsert(index.state.get(), "b#0", "Order number zx81 shipped on Tuesday.", std::vector<float>{0.0f, 0.0f, 1.0f}, "{}").ok);
    REQUIRE(ops->upsert(index.state.get(), "c#0", "Invoices are paid within thirty days.", std::vector<float>{0.0f, 0.7f, 0.7f}, "{}").ok);
    const std::vector<float> query{1.0f, 0.0f, 0.0f}; // equally unlike all three

    auto fused = ops->query_hybrid(index.state.get(), "where is zx81?", query, 1);
    REQUIRE(fused.ok);
    REQUIRE(fused.value.size() == 1);
    CHECK(fused.value[0].id == "b#0"); // the full-text side lifts it to the top
    CHECK(fused.value[0].score > 0.0f);

    // A text that matches nothing falls back to the vector ranking; punctuation cannot break the query.
    auto none = ops->query_hybrid(index.state.get(), "\"unbalanced AND (quote -", query, 3);
    REQUIRE(none.ok);
    CHECK(none.value.size() == 3);

    // Edits and deletes reach the full-text index.
    REQUIRE(ops->erase(index.state.get(), "b#0").ok);
    auto after = ops->query_hybrid(index.state.get(), "zx81", query, 3);
    REQUIRE(after.ok);
    for (const auto &h : after.value) CHECK(h.id != "b#0");
    REQUIRE(ops->upsert(index.state.get(), "a#0", "The zx81 manual is in the drawer.", std::vector<float>{0.0f, 1.0f, 0.0f}, "{}").ok);
    auto updated = ops->query_hybrid(index.state.get(), "zx81", query, 1);
    REQUIRE(updated.ok);
    REQUIRE(updated.value.size() == 1);
    CHECK(updated.value[0].id == "a#0");

    // The flag is on by default.
    CHECK(index.hybrid); // on by default; stores without query_hybrid just use the vector ranking
    index.state.reset();
    std::filesystem::remove(db_path, ec);
}
