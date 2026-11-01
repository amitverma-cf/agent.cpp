// Test harness for the engine plugin: loads it through the installed libagent exactly as the agent CLI
// does, then runs requests in process and in an agent-engine-host child (OS sandbox on).
//
//   echo_engine_test <path/to/agent-engine-echo.so> [agent-engine-host]
//
// Exits non-zero on the first failure. Adapt the expected replies when you change the engine.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const char *what, const std::string &detail = {}) {
    std::printf("[%s] %s%s%s\n", ok ? "ok" : "FAIL", what, detail.empty() ? "" : ": ", detail.c_str());
    if (!ok) ++g_failures;
}

void collect(std::string_view token, void *user_data) { static_cast<std::string *>(user_data)->append(token); }

void run_checks(const std::string &plugin, const std::string &host, agent::EngineIsolation isolation, const char *label) {
    std::printf("-- %s\n", label);
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = plugin;
    cfg.workspace_dir = (std::filesystem::temp_directory_path() / "agent-engine-template-test").string();
    cfg.auto_default_tools = false;
    cfg.engine_isolation = isolation;
    cfg.engine_host_path = host; // empty: $AGENT_ENGINE_HOST, next to this executable, ../bin, PATH

    auto sess = agent::init(cfg);
    check(sess.ok, "init", sess.error.message);
    if (!sess.ok) return;
    agent::Session session = std::move(sess.value);

    agent::FlowMemory memory;
    std::string streamed;
    auto reply = agent::run_turn(session, memory, "hello engine", true, collect, &streamed);
    check(reply.ok && reply.value == "echo: hello engine", "run_turn", reply.ok ? reply.value : reply.error.message);
    check(streamed == "echo: hello engine", "streamed chunks", streamed);

    agent::MessageView msg{.role = "user", .content = "x"};
    auto refused = agent::run_task(session, AGENT_CAP_TTS, {.messages = {&msg, 1}});
    check(!refused.ok, "undeclared task refused", refused.error.message);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path/to/agent-engine-echo.so> [agent-engine-host]\n", argv[0]);
        return 2;
    }
    const std::string plugin = argv[1];
    const std::string host = argc > 2 ? argv[2] : "";
    agent::AgentRuntime runtime;

    auto name = agent::load_engine_plugin(plugin); // checks abi_version and struct_size
    check(name.ok, "load_engine_plugin", name.ok ? name.value : name.error.message);
    if (!name.ok) return 1;
    const agent_engine_v1 *engine = agent::find_engine(name.value);
    check(engine && engine->abi_version == AGENT_ENGINE_ABI_VERSION && engine->struct_size >= AGENT_ENGINE_V2_SIZE, "ABI fields");
    check(engine && (engine->caps & AGENT_CAP_TEXT_GEN), "declares AGENT_CAP_TEXT_GEN");

    run_checks(plugin, host, agent::EngineIsolation::InProcess, "in process");
    run_checks(plugin, host, agent::EngineIsolation::Process, "process isolation");

    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
