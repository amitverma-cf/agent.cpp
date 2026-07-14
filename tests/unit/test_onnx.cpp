#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

TEST_CASE("ONNX missing model path returns error", "[onnx]") {
    auto result = agent::init({.provider = agent::AiProvider::OnnxRuntime,
                               .workspace_dir = "/tmp/agent_test_ws",
                               .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE_FALSE(result.ok);
#ifdef AGENT_HAS_ONNX
    REQUIRE(result.error.code == agent::ErrorCode::InvalidConfig);
#else
    REQUIRE(result.error.code == agent::ErrorCode::UnsupportedProvider);
#endif
}

TEST_CASE("ONNX bad model path returns ModelLoadFailed", "[onnx]") {
#ifdef AGENT_HAS_ONNX
    auto result = agent::init({.provider = agent::AiProvider::OnnxRuntime,
                               .model = "/nonexistent/model.onnx",
                               .workspace_dir = "/tmp/agent_test_ws",
                               .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE_FALSE(result.ok);
    REQUIRE(result.error.code == agent::ErrorCode::ModelLoadFailed);
#else
    SUCCEED("AGENT_ENABLE_ONNX=OFF; nothing to test");
#endif
}

TEST_CASE("ONNX tensor types compose correctly", "[onnx][tensors]") {
    agent::TensorData td;
    td.name = "input";
    td.dtype = agent::DType::Float32;
    td.shape = {1, 3, 224, 224};
    td.data.resize(1 * 3 * 224 * 224 * 4, 0);

    REQUIRE(td.shape.size() == 4);
    REQUIRE(td.data.size() == 1 * 3 * 224 * 224 * 4);

    agent::Message msg;
    msg.role = "user";
    msg.tensors.push_back(std::move(td));
    REQUIRE(msg.tensors.size() == 1);
    REQUIRE(msg.tensors[0].name == "input");
}
