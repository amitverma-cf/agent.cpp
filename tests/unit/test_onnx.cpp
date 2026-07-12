#include "../test_assert.hpp"

#include <agent-cpp/agent.hpp>
#include <iostream>

void test_onnx_missing_model_path() {
    auto result = agent::init(
        {.provider = agent::AiProvider::OnnxRuntime, .memory = {.provider = agent::MemoryProvider::InMemory}});
    TEST_ASSERT(!result.ok);
#ifdef AGENT_HAS_ONNX
    TEST_ASSERT(result.error.code == agent::ErrorCode::InvalidConfig);
#else
    TEST_ASSERT(result.error.code == agent::ErrorCode::UnsupportedProvider);
#endif
    std::cout << "  [PASS] Unit: ONNX missing model path returns error\n";
}

void test_onnx_bad_model_path() {
#ifdef AGENT_HAS_ONNX
    auto result = agent::init({.provider = agent::AiProvider::OnnxRuntime,
                               .model = "/nonexistent/model.onnx",
                               .memory = {.provider = agent::MemoryProvider::InMemory}});
    TEST_ASSERT(!result.ok);
    TEST_ASSERT(result.error.code == agent::ErrorCode::ModelLoadFailed);
    std::cout << "  [PASS] Unit: ONNX bad model path returns ModelLoadFailed\n";
#else
    std::cout << "  [SKIP] Unit: ONNX bad model path (AGENT_ENABLE_ONNX=OFF)\n";
#endif
}

void test_onnx_tensor_types() {
    agent::TensorData td;
    td.name = "input";
    td.dtype = agent::DType::Float32;
    td.shape = {1, 3, 224, 224};
    td.data.resize(1 * 3 * 224 * 224 * 4, 0);

    TEST_ASSERT(td.shape.size() == 4);
    TEST_ASSERT(td.data.size() == 1 * 3 * 224 * 224 * 4);

    agent::Message msg;
    msg.role = "user";
    msg.tensors.push_back(std::move(td));
    TEST_ASSERT(msg.tensors.size() == 1);
    TEST_ASSERT(msg.tensors[0].name == "input");

    std::cout << "  [PASS] Unit: ONNX tensor types compose correctly\n";
}
