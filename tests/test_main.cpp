#include <agent-cpp/agent.hpp>
#include <iostream>

void test_provider_mock_generation();
void test_provider_mock_streaming();
void test_tool_configuration();
void test_memory_in_memory();
void test_memory_rocksdb();
void test_utils_logging();
void test_session_lifecycle();
void test_conversation_flow();
void test_conversation_sliding_window();
void test_conversation_fsm();
void test_onnx_missing_model_path();
void test_onnx_bad_model_path();
void test_onnx_tensor_types();

int main() {
    agent::init_backend();

    std::cout << "Running test suite...\n";

    test_provider_mock_generation();
    test_provider_mock_streaming();
    test_tool_configuration();
    test_memory_in_memory();
    test_memory_rocksdb();
    test_utils_logging();
    test_session_lifecycle();
    test_conversation_flow();
    test_conversation_sliding_window();
    test_conversation_fsm();
    test_onnx_missing_model_path();
    test_onnx_bad_model_path();
    test_onnx_tensor_types();

    std::cout << "All tests passed successfully!\n";
    agent::free_backend();
    return 0;
}
