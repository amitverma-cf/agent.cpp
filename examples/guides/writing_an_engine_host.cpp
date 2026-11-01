// docs/guides/writing-an-engine.md: drives agent-engine-reverse (reverse_engine.cpp) through libagent,
// in process and under process isolation (agent-engine-host, OS sandbox on).
//
//   guide_engine_host <path/to/agent-engine-reverse.so> [workspace]
//
// Exits non-zero on the first failed check.

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

void collect_token(std::string_view token, void *user_data) { static_cast<std::string *>(user_data)->append(token); }

struct ProgressLog {
    uint64_t last_step = 0, total = 0;
    int32_t *cancel = nullptr; // set to cancel the job once cancel_at is reached
    uint64_t cancel_at = 0;
};

void on_progress(uint64_t step, uint64_t total, void *user_data) {
    auto *log = static_cast<ProgressLog *>(user_data);
    log->last_step = step;
    log->total = total;
    if (log->cancel && step >= log->cancel_at) *log->cancel = 1;
}

void exercise(const std::string &plugin, const std::string &workspace, agent::EngineIsolation isolation, const char *label) {
    std::printf("-- %s\n", label);
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = plugin;
    cfg.engine_options = R"({"step_delay_ms": 1})"; // reaches create() as config_json "options"
    cfg.workspace_dir = workspace;
    cfg.engine_isolation = isolation;
    cfg.auto_default_tools = false;

    auto sess = agent::init(cfg);
    check(sess.ok, "init", sess.error.message);
    if (!sess.ok) return;
    agent::Session session = std::move(sess.value);

    // 1. TEXT_GEN through run_turn, streamed: every on_chunk(AGENT_CHUNK_TEXT) reaches the token callback.
    agent::FlowMemory memory;
    std::string streamed;
    auto reply = agent::run_turn(session, memory, "hello world", true, collect_token, &streamed);
    check(reply.ok && reply.value == "dlrow olleh", "run_turn reverses the text", reply.ok ? reply.value : reply.error.message);
    check(streamed == "dlrow olleh", "streamed chunks add up to the reply", streamed);

    // 2. IMAGE_GEN through infer: a FILE part comes back as a MediaView, progress is reported.
    agent::MessageView msg{.role = "user", .content = "a gradient"};
    ProgressLog progress;
    agent::InferRequest req{.messages = {&msg, 1}, .on_progress = on_progress, .progress_user_data = &progress};
    auto image = agent::run_task(session, AGENT_CAP_IMAGE_GEN, req);
    const bool got_file = image.ok && image.value.message.media.size() == 1;
    check(got_file, "IMAGE_GEN returns one FILE part", image.ok ? std::string(image.value.message.content) : image.error.message);
    if (got_file) {
        const std::string path(image.value.message.media[0].path);
        check(std::filesystem::exists(path), "the image is on disk", path);
        check(image.value.message.media[0].mime == "image/x-portable-pixmap", "with its mime type");
    }
    check(progress.last_step == 32 && progress.total == 32, "progress reached 32/32",
          std::to_string(progress.last_step) + "/" + std::to_string(progress.total));

    // 3. Cancel: the progress callback raises the flag at step 8; the engine stops with AGENT_ERR_CANCELLED.
    int32_t cancel = 0;
    ProgressLog cancelling{.cancel = &cancel, .cancel_at = 8};
    agent::InferRequest creq{.messages = {&msg, 1}, .on_progress = on_progress, .progress_user_data = &cancelling, .cancel = &cancel};
    auto cancelled = agent::run_task(session, AGENT_CAP_IMAGE_GEN, creq);
    check(!cancelled.ok && cancelled.error.code == agent::ErrorCode::Cancelled, "cancel stops the job",
          cancelled.ok ? "completed" : cancelled.error.message);

    // 4. A task the engine does not declare is refused by the core before it reaches the plugin.
    auto tts = agent::run_task(session, AGENT_CAP_TTS, req);
    check(!tts.ok && tts.error.code == agent::ErrorCode::UnsupportedProvider, "undeclared task refused", tts.error.message);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path/to/agent-engine-reverse.so> [workspace]\n", argv[0]);
        return 2;
    }
    const std::string plugin = argv[1];
    const std::string workspace = argc > 2 ? argv[2] : (std::filesystem::temp_directory_path() / "agent-guide-engine").string();

    agent::AgentRuntime runtime;

    // load_engine_plugin dlopens the library, checks abi_version / struct_size and registers it by name.
    auto name = agent::load_engine_plugin(plugin);
    check(name.ok && name.value == "reverse", "load_engine_plugin", name.ok ? name.value : name.error.message);
    if (!name.ok) return 1;
    const agent_engine_v1 *engine = agent::find_engine("reverse");
    check(engine && (engine->caps & AGENT_CAP_IMAGE_GEN), "find_engine + caps");

    exercise(plugin, workspace, agent::EngineIsolation::InProcess, "in process");
    // The same plugin in an agent-engine-host child, under Config::engine_sandbox (default: plugin dir and
    // model read-only, workspace read-only, workspace/artifacts read-write, no network, no /tmp).
    exercise(plugin, workspace, agent::EngineIsolation::Process, "process isolation");

    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
