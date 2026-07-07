#include <agent-cpp/agent.hpp>
#include <iostream>
#include <exception>

void test_provider_mock_generation();
void test_provider_mock_streaming();
void test_memory_in_memory();
void test_utils_logging();
void test_session_lifecycle();
void test_conversation_flow();
void test_conversation_sliding_window();
void test_memory_rocksdb();

int main() {
    agent::init_backend();

    std::cout << "Running test suite...\n";
    try {
        test_provider_mock_generation();
        test_provider_mock_streaming();
        test_memory_in_memory();
        test_memory_rocksdb();
        test_utils_logging();
        test_session_lifecycle();
        test_conversation_flow();
        test_conversation_sliding_window();

        std::cout << "All tests passed successfully!\n";
        agent::free_backend();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test suite failed with exception: " << e.what() << "\n";
        agent::free_backend();
        return 1;
    }
}
