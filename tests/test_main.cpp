#define CATCH_CONFIG_RUNNER
#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

int main(int argc, char *argv[]) {
    agent::init_backend();
    int result = Catch::Session().run(argc, argv);
    agent::free_backend();
    return result;
}
