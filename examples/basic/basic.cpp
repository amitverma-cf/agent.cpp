#include <agent-cpp/agent.h>
#include <iostream>

int main() {
    agent::Session session = agent::init({.provider = agent::Provider::Mock});
    std::string result = agent::run(session, "hello");
    std::cout << result << '\n';

    return 0;
}