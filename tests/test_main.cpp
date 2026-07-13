#define CATCH_CONFIG_RUNNER
#include <catch_amalgamated.hpp>

#include <agent-cpp/agent.hpp>

int main(int argc, char *argv[]) {
    agent::init_backend();
    int result = Catch::Session().run(argc, argv);
    agent::free_backend();
    return result;
}
