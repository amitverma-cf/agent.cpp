#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <cstdlib>
#include <string>

namespace {

const char *kWorkspace = "/tmp/agent_test_engines_ws";

agent::Config engine_config(std::string model = "mock.bin") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.model = std::move(model);
    cfg.workspace_dir = kWorkspace;
    return cfg;
}

int g_ping_calls = 0;

agent::Tool make_ping_tool() {
    agent::Tool t;
    t.name = "ping";
    t.description = "test tool";
    t.parameter_schema = "{}";
    t.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        ++g_ping_calls;
        return agent::ok(std::string("pong"));
    };
    return t;
}

} // namespace

TEST_CASE("load_engine_plugin loads the mock engine and is idempotent", "[engines]") {
    auto first = agent::load_engine_plugin(AGENT_TEST_MOCK_ENGINE_PATH);
    REQUIRE(first.ok);
    REQUIRE(first.value == "mock");

    const agent_engine_v1 *engine = agent::find_engine("mock");
    REQUIRE(engine != nullptr);
    REQUIRE(engine->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE((engine->caps & AGENT_CAP_TEXT_GEN) != 0);

    auto second = agent::load_engine_plugin(AGENT_TEST_MOCK_ENGINE_PATH);
    REQUIRE(second.ok);
    REQUIRE(agent::find_engine("mock") == engine);
}

TEST_CASE("load_engine_plugin reports a missing library", "[engines]") {
    auto res = agent::load_engine_plugin("/nonexistent/agent-engine-nope.so");
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ProviderInitFailed);
}

TEST_CASE("register_engine rejects an ABI mismatch and incomplete tables", "[engines]") {
    static const agent_engine_v1 wrong_abi = {AGENT_ENGINE_ABI_VERSION + 1, "wrong_abi", "0", 0, 0, nullptr, nullptr, nullptr,
                                              nullptr,                      nullptr,     nullptr, nullptr};
    auto r1 = agent::register_engine(&wrong_abi);
    REQUIRE_FALSE(r1.ok);
    REQUIRE(r1.error.code == agent::ErrorCode::UnsupportedProvider);

    static const agent_engine_v1 incomplete = {AGENT_ENGINE_ABI_VERSION, "incomplete", "0", 0, 0, nullptr, nullptr, nullptr,
                                               nullptr,                  nullptr,      nullptr, nullptr};
    auto r2 = agent::register_engine(&incomplete);
    REQUIRE_FALSE(r2.ok);
    REQUIRE(r2.error.code == agent::ErrorCode::InvalidConfig);
    REQUIRE(agent::find_engine("incomplete") == nullptr);
}

TEST_CASE("AiProvider::Engine runs a turn through the plugin", "[engines]") {
    auto res = agent::init(engine_config());
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "engine_turn";
    auto reply = agent::run_turn(session, memory, "hello");
    REQUIRE(reply.ok);
    REQUIRE(reply.value == "echo: hello");
}

TEST_CASE("AiProvider::Engine forwards streamed text chunks", "[engines]") {
    auto res = agent::init(engine_config());
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    std::string streamed;
    agent::FlowMemory memory;
    memory.id = "engine_stream";
    auto reply = agent::run_turn(
        session, memory, "stream me", true, [](std::string_view tok, void *ud) { *static_cast<std::string *>(ud) += tok; }, &streamed);
    REQUIRE(reply.ok);
    REQUIRE(streamed == "echo: stream me");
}

TEST_CASE("AiProvider::Engine tool calls go through execute_tool", "[engines]") {
    auto cfg = engine_config();
    cfg.auto_default_tools = false;
    cfg.tools = {make_ping_tool()};
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    g_ping_calls = 0;
    agent::FlowMemory memory;
    memory.id = "engine_tools";
    auto reply = agent::run_turn(session, memory, "call:ping");
    REQUIRE(reply.ok);
    REQUIRE(g_ping_calls == 1);
    REQUIRE(reply.value == "tool done");
}

TEST_CASE("AiProvider::Engine surfaces create() failures", "[engines]") {
    auto res = agent::init(engine_config("fail"));
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
    REQUIRE(res.error.message.find("refusing model") != std::string::npos);
}

TEST_CASE("AiProvider::Engine requires a known engine", "[engines]") {
    auto cfg = engine_config();
    cfg.engine = "no_such_engine";
    auto res = agent::init(cfg);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::UnsupportedProvider);

    cfg.engine.clear();
    auto res2 = agent::init(cfg);
    REQUIRE_FALSE(res2.ok);
    REQUIRE(res2.error.code == agent::ErrorCode::InvalidConfig);
}

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH

namespace {

agent::Config llama_config(std::string model) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_LLAMA_ENGINE_PATH;
    cfg.model = std::move(model);
    cfg.workspace_dir = kWorkspace;
    cfg.context_window = 512;
    cfg.max_tokens = 16;
    cfg.temperature = 0.0f;
    cfg.auto_default_tools = false; // keep the prompt inside the tiny test context
    return cfg;
}

// Set AGENT_TEST_LLAMA_MODEL to a GGUF file to run the real-model tests, e.g.
// https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
const char *test_model() { return std::getenv("AGENT_TEST_LLAMA_MODEL"); }

} // namespace

TEST_CASE("llama engine plugin loads and declares its capabilities", "[engines][llama]") {
    auto res = agent::load_engine_plugin(AGENT_TEST_LLAMA_ENGINE_PATH);
    REQUIRE(res.ok);
    REQUIRE(res.value == "llama");
    const agent_engine_v1 *engine = agent::find_engine("llama");
    REQUIRE(engine != nullptr);
    REQUIRE((engine->caps & AGENT_CAP_TEXT_GEN) != 0);
    REQUIRE((engine->flags & AGENT_ENGINE_FLAG_CONCURRENT) != 0);
}

TEST_CASE("llama engine reports a missing model file", "[engines][llama]") {
    auto res = agent::init(llama_config("/nonexistent/model.gguf"));
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
}

TEST_CASE("llama engine generates text with a real model", "[engines][llama][model]") {
    if (!test_model()) SKIP("AGENT_TEST_LLAMA_MODEL not set");
    auto res = agent::init(llama_config(test_model()));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    std::string streamed;
    agent::FlowMemory memory;
    memory.id = "llama_real";
    auto reply = agent::run_turn(
        session, memory, "Once upon a time", true, [](std::string_view tok, void *ud) { *static_cast<std::string *>(ud) += tok; }, &streamed);
    REQUIRE(reply.ok);
    REQUIRE_FALSE(streamed.empty());
    REQUIRE(memory.stats.completion_tokens > 0);
    REQUIRE(agent::clear_provider_kv_cache(session).ok);
}

TEST_CASE("llama engine runs parallel subagents through continuous batching", "[engines][llama][model]") {
    if (!test_model()) SKIP("AGENT_TEST_LLAMA_MODEL not set");
    auto cfg = llama_config(test_model());
    cfg.max_parallel_subagents = 3;
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    static const agent::FlowState kStates[] = {
        {.state_id = "go",
         .on_transition = [](agent::Session &s, agent::FlowMemory &m, void *) -> agent::Result<std::string_view> {
             auto r = agent::run_turn(s, m, "Once upon a time");
             if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
             return agent::ok(std::string_view("exit"));
         }},
    };
    agent::AgentScheduler scheduler;
    for (int i = 0; i < 3; ++i) {
        agent::Flow flow;
        flow.states = kStates;
        flow.memory.id = "batch_" + std::to_string(i);
        REQUIRE(scheduler.spawn("w" + std::to_string(i), std::move(flow)).ok);
    }
    REQUIRE(scheduler.run_until_done(session).ok);
    for (const auto &r : scheduler.results()) {
        INFO(r.id << ": " << r.error.message);
        REQUIRE(r.ok);
    }
}

#endif

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
TEST_CASE("llama engine rejects a prompt longer than the context instead of aborting", "[engines][llama][model]") {
    if (!test_model()) SKIP("AGENT_TEST_LLAMA_MODEL not set");
    auto cfg = llama_config(test_model());
    cfg.context_window = 64;
    cfg.max_tokens = 8;
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "llama_overflow";
    std::string huge(4000, 'a');
    for (size_t i = 0; i < huge.size(); i += 2) huge[i] = ' ';
    auto reply = agent::run_turn(session, memory, huge);
    REQUIRE_FALSE(reply.ok);
}
#endif

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
TEST_CASE("llama engine prefills prompts longer than n_batch in chunks", "[engines][llama][model]") {
    if (!test_model()) SKIP("AGENT_TEST_LLAMA_MODEL not set");
    auto cfg = llama_config(test_model());
    cfg.context_window = 8192; // n_batch defaults to 2048, so a ~3000-token prompt needs two decodes
    cfg.max_tokens = 4;
    cfg.max_parallel_subagents = 1;
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "llama_chunked";
    std::string prompt;
    for (int i = 0; i < 3000; ++i) prompt += "a ";
    auto reply = agent::run_turn(session, memory, prompt);
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    REQUIRE(memory.stats.prompt_tokens > 2048);
}
#endif

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
// Set AGENT_TEST_LLAMA_TOOL_MODEL to an instruct GGUF with a tool-calling chat template, e.g.
// https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF (qwen2.5-0.5b-instruct-q4_k_m.gguf)
TEST_CASE("llama engine completes a native tool-call round trip", "[engines][llama][model]") {
    const char *model = std::getenv("AGENT_TEST_LLAMA_TOOL_MODEL");
    if (!model) SKIP("AGENT_TEST_LLAMA_TOOL_MODEL not set");
    auto cfg = llama_config(model);
    cfg.context_window = 2048;
    cfg.max_tokens = 128;
    cfg.tools = {make_ping_tool()};
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    g_ping_calls = 0;
    agent::FlowMemory memory;
    memory.id = "llama_tools";
    auto reply = agent::run_turn(session, memory, "Call the ping tool now.");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    REQUIRE(g_ping_calls >= 1);
}
#endif

TEST_CASE("Media on the user message reaches the engine as FILE parts", "[engines][media]") {
    auto res = agent::init(engine_config());
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    const agent::Media media[] = {{"/tmp/cat.png", "image/png"}, {"/tmp/hello.wav", "audio/wav"}};
    agent::FlowMemory memory;
    memory.id = "engine_media_in";
    auto reply = agent::run_turn_with_media(session, memory, "describe", media);
    REQUIRE(reply.ok);
    REQUIRE(reply.value == "echo: describe\nmedia: image/png /tmp/cat.png\nmedia: audio/wav /tmp/hello.wav");
    REQUIRE(memory.history.front().media.size() == 2);
}

TEST_CASE("Engine FILE outputs become media on the assistant message and persist", "[engines][media]") {
    auto res = agent::init(engine_config());
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::FlowMemory memory;
    memory.id = "engine_media_out";
    auto reply = agent::run_turn(session, memory, "draw");
    REQUIRE(reply.ok);
    REQUIRE(reply.value == "echo: draw");
    const agent::Message &out = memory.history.back();
    REQUIRE(out.media.size() == 1);
    REQUIRE(out.media[0].mime == "text/plain");
    REQUIRE(out.media[0].path.find("artifacts/mock_drawing.txt") != std::string::npos);

    agent::FlowMemory reloaded;
    reloaded.id = "engine_media_out";
    REQUIRE(agent::load_flow_memory(session, reloaded).ok);
    REQUIRE(reloaded.history.back().media.size() == 1);
    REQUIRE(reloaded.history.back().media[0].path == out.media[0].path);
}

TEST_CASE("Progress reaches Config::on_progress and request_cancel stops the engine", "[engines][cancel]") {
    struct Progress {
        uint64_t last = 0, total = 0;
        agent::Session *session = nullptr;
        uint64_t cancel_at = 0;
    } progress;

    auto cfg = engine_config();
    cfg.on_progress = [](uint64_t step, uint64_t total, void *ud) {
        auto *p = static_cast<Progress *>(ud);
        p->last = step;
        p->total = total;
        if (p->cancel_at && step == p->cancel_at) agent::request_cancel(*p->session);
    };
    cfg.progress_user_data = &progress;
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    progress.session = &session;

    agent::FlowMemory memory;
    memory.id = "engine_progress";
    auto done = agent::run_turn(session, memory, "slow");
    REQUIRE(done.ok);
    REQUIRE(progress.last == 100);
    REQUIRE(progress.total == 100);

    progress.cancel_at = 10;
    auto cancelled = agent::run_turn(session, memory, "slow");
    REQUIRE_FALSE(cancelled.ok);
    REQUIRE(cancelled.error.code == agent::ErrorCode::Cancelled);
    REQUIRE(progress.last == 10);

    // Still cancelled until cleared.
    auto blocked = agent::run_turn(session, memory, "hello");
    REQUIRE(blocked.error.code == agent::ErrorCode::Cancelled);
    agent::clear_cancel(session);
    progress.cancel_at = 0;
    REQUIRE(agent::run_turn(session, memory, "hello").ok);
}
