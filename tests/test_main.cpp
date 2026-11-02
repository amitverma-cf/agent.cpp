#define CATCH_CONFIG_RUNNER
#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

#if defined(__SANITIZE_ADDRESS__)
// The audio device probe (miniaudio -> libjack static initialisers) leaks a few bytes inside the
// system JACK library; it is not ours and not fixable here.
extern "C" const char *__lsan_default_suppressions() { return "leak:ma_context_init__jack\n"; }
#endif

int main(int argc, char *argv[]) {
    agent::init_backend();
    int result = Catch::Session().run(argc, argv);
    agent::free_backend();
    return result;
}
