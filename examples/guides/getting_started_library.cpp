// docs/guides/getting-started-library.md: a first program against the installed library.
//
//   guide_getting_started <engine plugin or name> <model> [prompt] [workspace]
//   guide_getting_started <prefix>/lib/agent/engines/agent-engine-llama.so qwen2.5-0.5b-instruct-q4_k_m.gguf
//
// The engine runs out of process (agent-engine-host) inside the OS sandbox.

#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <string>

namespace {

void print_token(std::string_view token, void *) {
    std::fwrite(token.data(), 1, token.size(), stdout);
    std::fflush(stdout);
}

void print_log(agent::LogLevel level, std::string_view message, void *) {
    if (level >= agent::LogLevel::Warning) std::fprintf(stderr, "[agent] %.*s\n", static_cast<int>(message.size()), message.data());
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <engine plugin or name> <model> [prompt] [workspace]\n", argv[0]);
        return 2;
    }

    agent::AgentRuntime runtime; // process-wide backend init/teardown

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = argv[1]; // a plugin path (loaded on demand) or the name of an already loaded engine
    cfg.model = argv[2];
    cfg.workspace_dir = argc > 4 ? argv[4] : ".workspace";
    cfg.context_window = 4096;
    cfg.max_tokens = 256;
    cfg.logger = print_log;

    // Run the engine in an agent-engine-host child: a crash or abort() in the engine fails the call with
    // ErrorCode::EngineCrashed and the next call restarts the host; this process survives.
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_timeout_ms = 5 * 60 * 1000; // watchdog per request

    // The host's OS sandbox (Config::engine_sandbox) is on by default: plugin dir + model read-only,
    // workspace read-only, workspace/artifacts read-write, no network, no /tmp. Add a memory cap:
    cfg.engine_sandbox.memory_limit_bytes = 8ull << 30; // RLIMIT_AS of the host

    // run_command (a default tool) runs in its own sandbox (Config::tool_sandbox, on by default): the
    // workspace read-write, system dirs read-only. The network is allowed by default; turn it off:
    cfg.tool_sandbox.network = agent::SandboxNetwork::Deny;

    auto sess = agent::init(cfg);
    if (!sess.ok) {
        std::fprintf(stderr, "init failed: %s\n", sess.error.message.c_str());
        return 1;
    }
    agent::Session session = std::move(sess.value);

    agent::FlowMemory conversation;
    conversation.id = "getting-started";
    const char *prompt = argc > 3 ? argv[3] : "Say hello in one short sentence.";

    auto reply = agent::run_turn(session, conversation, prompt, /*stream=*/true, print_token);
    std::printf("\n");
    if (!reply.ok) {
        std::fprintf(stderr, "turn failed (%d): %s\n", static_cast<int>(reply.error.code), reply.error.message.c_str());
        return 1;
    }

    const agent::SessionStats stats = agent::get_stats(session);
    std::printf("turns=%d prompt_tokens=%d completion_tokens=%d\n", stats.total_turns, stats.total_prompt_tokens,
                stats.total_completion_tokens);
    return 0;
}
