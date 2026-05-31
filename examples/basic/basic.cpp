#include <agent-cpp/agent.h>
#include <iostream>

int main() {
    auto session = agent::init();
    auto result = agent::run(session, "hello");
    std::cout << result << '\n';

    return 0;
}