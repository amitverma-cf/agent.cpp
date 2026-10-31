#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
namespace {

// 64x64 solid-color 24-bit BMP (stb_image reads BMP, so no PNG encoder is needed here).
// channel: 0 = blue, 1 = green, 2 = red (BMP pixels are BGR).
std::string solid_bmp(uint32_t channel) {
    const uint32_t w = 64, h = 64, row = w * 3, data = row * h;
    std::string bmp(54 + data, '\0');
    auto put32 = [&](size_t at, uint32_t v) {
        for (size_t i = 0; i < 4; ++i) bmp[at + i] = static_cast<char>((v >> (8 * i)) & 0xff);
    };
    bmp[0] = 'B';
    bmp[1] = 'M';
    put32(2, 54 + data);
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, h);
    bmp[26] = 1;  // planes
    bmp[28] = 24; // bits per pixel
    put32(34, data);
    for (uint32_t i = 0; i < w * h; ++i) bmp[54 + i * 3 + channel] = static_cast<char>(0xff);
    return bmp;
}

agent_str abi_str(std::string_view s) { return agent_str{s.data(), s.size()}; }

struct LlamaAbi {
    const agent_engine_v1 *engine = nullptr;
    void *inst = nullptr;
    ~LlamaAbi() {
        if (inst) engine->destroy(inst);
    }
};

// Loads the plugin and creates an instance through the C ABI (no core message model involved).
bool create_llama(LlamaAbi &abi, const std::string &config) {
    if (!agent::load_engine_plugin(AGENT_TEST_LLAMA_ENGINE_PATH).ok) return false;
    abi.engine = agent::find_engine("llama");
    if (!abi.engine) return false;
    char err[512] = {0};
    int32_t rc = abi.engine->create(abi_str(config), &abi.inst, err, sizeof(err));
    if (rc != AGENT_OK) WARN(err);
    return rc == AGENT_OK;
}

int32_t run_with_media(LlamaAbi &abi, const agent_part &media, std::string_view prompt, std::string &text) {
    agent_part parts[2] = {};
    parts[0] = media;
    parts[1].kind = AGENT_PART_TEXT;
    parts[1].text = abi_str(prompt);
    agent_message msg{};
    msg.role = abi_str("user");
    msg.parts = parts;
    msg.n_parts = 2;
    agent_request req{};
    req.task = AGENT_CAP_TEXT_GEN;
    req.messages = &msg;
    req.n_messages = 1;
    req.params_json = abi_str(R"({"max_tokens":24,"temperature":0})");
    agent_response resp{};
    int32_t rc = abi.engine->run(abi.inst, &req, &resp);
    if (rc == AGENT_OK) {
        for (size_t i = 0; i < resp.message.n_parts; ++i)
            if (resp.message.parts[i].kind == AGENT_PART_TEXT) text.append(resp.message.parts[i].text.data, resp.message.parts[i].text.size);
        abi.engine->free_response(abi.inst, &resp);
    }
    return rc;
}

std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

} // namespace

TEST_CASE("llama engine declares image and audio input", "[engines][llama]") {
    REQUIRE(agent::load_engine_plugin(AGENT_TEST_LLAMA_ENGINE_PATH).ok);
    const agent_engine_v1 *engine = agent::find_engine("llama");
    REQUIRE(engine != nullptr);
    REQUIRE((engine->caps & AGENT_CAP_VISION_IN) != 0);
    REQUIRE((engine->caps & AGENT_CAP_AUDIO_IN) != 0);
}

TEST_CASE("llama engine rejects media parts without an mmproj", "[engines][llama][model]") {
    if (!test_model()) SKIP("AGENT_TEST_LLAMA_MODEL not set");
    LlamaAbi abi;
    REQUIRE(create_llama(abi, std::string(R"({"model":")") + test_model() + R"(","context_window":256})"));

    const std::string bmp = solid_bmp(2);
    agent_part media{};
    media.kind = AGENT_PART_BYTES;
    media.mime = abi_str("image/bmp");
    media.data = bmp.data();
    media.size = bmp.size();
    std::string text;
    REQUIRE(run_with_media(abi, media, "What color is this?", text) == AGENT_ERR_UNSUPPORTED);
    REQUIRE(std::string(abi.engine->last_error(abi.inst)).find("mmproj") != std::string::npos);
}

// Set AGENT_TEST_LLAMA_VISION_MODEL and AGENT_TEST_LLAMA_VISION_MMPROJ to a vision GGUF and its
// projector, e.g. https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF
// (SmolVLM-256M-Instruct-Q8_0.gguf + mmproj-SmolVLM-256M-Instruct-Q8_0.gguf).
TEST_CASE("llama engine reads an image through libmtmd", "[engines][llama][model]") {
    const char *model = std::getenv("AGENT_TEST_LLAMA_VISION_MODEL");
    const char *mmproj = std::getenv("AGENT_TEST_LLAMA_VISION_MMPROJ");
    if (!model || !mmproj) SKIP("AGENT_TEST_LLAMA_VISION_MODEL / AGENT_TEST_LLAMA_VISION_MMPROJ not set");
    LlamaAbi abi;
    std::string config = "{\"model\":\"";
    config += model;
    config += "\",\"context_window\":2048,\"options\":{\"mmproj\":\"";
    config += mmproj;
    config += "\"}}";
    REQUIRE(create_llama(abi, config));

    const std::string bmp = solid_bmp(2);
    const std::string path = "/tmp/agent_test_red.bmp";
    FILE *f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    REQUIRE(std::fwrite(bmp.data(), 1, bmp.size(), f) == bmp.size());
    std::fclose(f);
    const char *question = "What color is this image? Answer in one word.";

    SECTION("file part") {
        agent_part media{};
        media.kind = AGENT_PART_FILE;
        media.path = abi_str(path);
        media.mime = abi_str("image/bmp");
        std::string text;
        int32_t rc = run_with_media(abi, media, question, text);
        INFO(abi.engine->last_error(abi.inst));
        REQUIRE(rc == AGENT_OK);
        INFO("reply: " << text);
        REQUIRE_FALSE(text.empty());
        CHECK(lower(text).find("red") != std::string::npos);
    }
    SECTION("bytes part") {
        const std::string blue = solid_bmp(0); // a second color, so the answer can't be a fixed guess
        agent_part media{};
        media.kind = AGENT_PART_BYTES;
        media.mime = abi_str("image/bmp");
        media.data = blue.data();
        media.size = blue.size();
        std::string text;
        int32_t rc = run_with_media(abi, media, question, text);
        INFO(abi.engine->last_error(abi.inst));
        REQUIRE(rc == AGENT_OK);
        INFO("reply: " << text);
        REQUIRE_FALSE(text.empty());
        CHECK(lower(text).find("blue") != std::string::npos);
    }
    SECTION("unreadable file") {
        agent_part media{};
        media.kind = AGENT_PART_FILE;
        media.path = abi_str("/nonexistent/image.png");
        media.mime = abi_str("image/png");
        std::string text;
        REQUIRE(run_with_media(abi, media, "What is this?", text) == AGENT_ERR_FILESYSTEM);
    }
}
#endif

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
TEST_CASE("run_turn_with_media sends an image through the core to the llama engine", "[engines][llama][model][media]") {
    const char *model = std::getenv("AGENT_TEST_LLAMA_VISION_MODEL");
    const char *mmproj = std::getenv("AGENT_TEST_LLAMA_VISION_MMPROJ");
    if (!model || !mmproj) SKIP("AGENT_TEST_LLAMA_VISION_MODEL / AGENT_TEST_LLAMA_VISION_MMPROJ not set");

    const std::string image_path = std::string(kWorkspace) + "/red.bmp";
    {
        std::FILE *f = std::fopen(image_path.c_str(), "wb");
        REQUIRE(f != nullptr);
        const std::string bmp = solid_bmp(2);
        std::fwrite(bmp.data(), 1, bmp.size(), f);
        std::fclose(f);
    }

    auto cfg = llama_config(model);
    cfg.context_window = 4096;
    cfg.max_tokens = 16;
    cfg.engine_options = std::string("{\"mmproj\":\"") + mmproj + "\"}";
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    const agent::Media media[] = {{image_path, "image/bmp"}};
    agent::FlowMemory memory;
    memory.id = "llama_media_e2e";
    auto reply = agent::run_turn_with_media(session, memory, "What color is this image? Answer with one word.", media);
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    std::string lower = reply.value;
    for (char &c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    REQUIRE(lower.find("red") != std::string::npos);
}
#endif

TEST_CASE("load_engine_manifest loads plugins listed in engines.json", "[engines][manifest]") {
    const std::filesystem::path dir = std::filesystem::path(kWorkspace) / "manifest";
    std::filesystem::create_directories(dir);
    // A relative path resolves against the manifest's directory.
    const auto relative = std::filesystem::relative(AGENT_TEST_MOCK_ENGINE_PATH, dir);
    {
        std::ofstream f(dir / "engines.json");
        f << "{\"engines\": [{\"path\": \"" << relative.generic_string() << "\"}]}";
    }
    auto res = agent::load_engine_manifest((dir / "engines.json").string());
    INFO(res.error.message);
    REQUIRE(res.ok);
    REQUIRE(res.value == std::vector<std::string>{"mock"});

    {
        std::ofstream f(dir / "bad.json");
        f << "{\"engines\": [{\"path\": \"missing-engine.so\"}]}";
    }
    auto bad = agent::load_engine_manifest((dir / "bad.json").string());
    REQUIRE_FALSE(bad.ok);
    REQUIRE(bad.error.code == agent::ErrorCode::ProviderInitFailed);

    auto absent = agent::load_engine_manifest((dir / "nope.json").string());
    REQUIRE(absent.error.code == agent::ErrorCode::FilesystemError);
}
