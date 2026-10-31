// Out-of-process engines (EngineIsolation::Process): the engine_ipc codec, the mock engine driven
// through agent-engine-host, crash / hang / cancel recovery, and the restart limit.

#include "core/engines/engine_ipc.hpp"
#include "core/providers/provider_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace ipc = agent::engine_ipc;

const char *kWorkspace = "/tmp/agent_test_engine_host_ws";

agent::Config host_config(std::string model = "mock.bin") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    cfg.engine_isolation = agent::EngineIsolation::Process;
    cfg.engine_host_path = AGENT_TEST_ENGINE_HOST_PATH;
    cfg.model = std::move(model);
    cfg.workspace_dir = kWorkspace;
    return cfg;
}

agent::Session open_session(const agent::Config &cfg) {
    auto res = agent::init(cfg);
    INFO(res.error.message);
    REQUIRE(res.ok);
    return std::move(res.value);
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

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

} // namespace

// ------------------------------------------------------------------ codec

TEST_CASE("engine_ipc round-trips requests and responses", "[engine_host][ipc]") {
    const int64_t shape[] = {2, 3};
    const float values[] = {1, 2, 3, 4, 5, 6};
    const uint8_t png[] = {0x89, 'P', 'N', 'G'};
    agent_part parts[4] = {};
    parts[0].kind = AGENT_PART_TEXT;
    parts[0].text = {"hello", 5};
    parts[1].kind = AGENT_PART_FILE;
    parts[1].path = {"/tmp/a.png", 10};
    parts[1].mime = {"image/png", 9};
    parts[2].kind = AGENT_PART_BYTES;
    parts[2].mime = {"image/png", 9};
    parts[2].data = png;
    parts[2].size = sizeof(png);
    parts[3].kind = AGENT_PART_TENSOR;
    parts[3].name = {"x", 1};
    parts[3].dtype = AGENT_DTYPE_F32;
    parts[3].shape = shape;
    parts[3].ndim = 2;
    parts[3].data = values;
    parts[3].size = sizeof(values);
    const agent_tool_call call{{"c1", 2}, {"ping", 4}, {"{}", 2}};
    agent_message msgs[2] = {};
    msgs[0].role = {"user", 4};
    msgs[0].parts = parts;
    msgs[0].n_parts = 4;
    msgs[1].role = {"assistant", 9};
    msgs[1].tool_calls = &call;
    msgs[1].n_tool_calls = 1;
    agent_request req{};
    req.task = AGENT_CAP_TEXT_GEN;
    req.messages = msgs;
    req.n_messages = 2;
    req.tools_json = {"[]", 2};
    req.params_json = {"{\"max_tokens\":4}", 16};
    req.output_dir = {"/tmp/out", 8};

    std::string frame;
    ipc::Writer w(frame);
    w.begin(ipc::Msg::Run);
    w.request(req, ipc::kRunStream);
    REQUIRE(w.finish());
    const ipc::FrameHeader h = ipc::read_header(frame.data());
    REQUIRE(h.type == ipc::Msg::Run);
    REQUIRE(h.size == frame.size() - ipc::kHeaderSize);

    agent::Arena arena(ipc::decode_arena_bound(h.size));
    ipc::Reader r(frame.data() + ipc::kHeaderSize, h.size);
    agent_request got{};
    uint32_t flags = 0;
    REQUIRE(r.request(got, flags, arena));
    REQUIRE(r.at_end());
    REQUIRE(flags == ipc::kRunStream);
    REQUIRE(got.task == AGENT_CAP_TEXT_GEN);
    REQUIRE(got.n_messages == 2);
    REQUIRE(sv(got.messages[0].role) == "user");
    REQUIRE(got.messages[0].n_parts == 4);
    REQUIRE(sv(got.messages[0].parts[0].text) == "hello");
    REQUIRE(sv(got.messages[0].parts[1].path) == "/tmp/a.png");
    REQUIRE(got.messages[0].parts[2].size == sizeof(png));
    REQUIRE(std::memcmp(got.messages[0].parts[2].data, png, sizeof(png)) == 0);
    const agent_part &t = got.messages[0].parts[3];
    REQUIRE(sv(t.name) == "x");
    REQUIRE(t.ndim == 2);
    REQUIRE(t.shape[0] == 2);
    REQUIRE(t.shape[1] == 3);
    REQUIRE(t.size == sizeof(values));
    REQUIRE(std::memcmp(t.data, values, sizeof(values)) == 0);
    REQUIRE(got.messages[1].n_tool_calls == 1);
    REQUIRE(sv(got.messages[1].tool_calls[0].name) == "ping");
    REQUIRE(sv(got.params_json) == "{\"max_tokens\":4}");
    REQUIRE(sv(got.output_dir) == "/tmp/out");

    // Every truncation of the payload is rejected without reading past it.
    for (size_t n = 0; n < h.size; ++n) {
        arena.reset();
        ipc::Reader cut(frame.data() + ipc::kHeaderSize, n);
        agent_request bad{};
        REQUIRE_FALSE(cut.request(bad, flags, arena));
    }

    agent_response resp{};
    resp.message = msgs[0];
    resp.usage = {7, 3};
    std::string out;
    ipc::Writer rw(out);
    rw.begin(ipc::Msg::Response);
    rw.response(resp);
    REQUIRE(rw.finish());
    arena.reset();
    ipc::Reader rr(out.data() + ipc::kHeaderSize, out.size() - ipc::kHeaderSize);
    agent_response back{};
    REQUIRE(rr.response(back, arena));
    REQUIRE(back.usage.prompt_tokens == 7);
    REQUIRE(back.usage.completion_tokens == 3);
    REQUIRE(back.message.n_parts == 4);
}

TEST_CASE("engine_ipc rejects counts the frame cannot hold", "[engine_host][ipc]") {
    std::string frame;
    ipc::Writer w(frame);
    w.begin(ipc::Msg::Run);
    w.u32(AGENT_CAP_TEXT_GEN);
    w.u32(0);
    w.u32(0x7fffffff); // n_messages
    REQUIRE(w.finish());
    agent::Arena arena(4096);
    ipc::Reader r(frame.data() + ipc::kHeaderSize, frame.size() - ipc::kHeaderSize);
    agent_request req{};
    uint32_t flags = 0;
    REQUIRE_FALSE(r.request(req, flags, arena));
    REQUIRE_FALSE(r.arena_exhausted());
}

// ------------------------------------------------------------------ the mock engine through the host

TEST_CASE("Process isolation runs a turn in agent-engine-host", "[engine_host]") {
    agent::Session session = open_session(host_config());
    REQUIRE(agent::providers::engine_host_starts(session) == 1);

    agent::FlowMemory memory;
    memory.id = "host_turn";
    auto reply = agent::run_turn(session, memory, "hello");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    REQUIRE(reply.value == "echo: hello");
    auto again = agent::run_turn(session, memory, "again");
    REQUIRE(again.ok);
    REQUIRE(again.value == "echo: again");
    REQUIRE(agent::providers::engine_host_starts(session) == 1);
}

TEST_CASE("Process isolation forwards streamed chunks", "[engine_host]") {
    agent::Session session = open_session(host_config());
    std::string streamed;
    agent::FlowMemory memory;
    memory.id = "host_stream";
    auto reply = agent::run_turn(
        session, memory, "stream:200", true, [](std::string_view tok, void *ud) { *static_cast<std::string *>(ud) += tok; }, &streamed);
    REQUIRE(reply.ok);
    REQUIRE(streamed == std::string(200, 'x') + "echo: stream:200");
}

TEST_CASE("Process isolation completes a tool-call round trip", "[engine_host]") {
    auto cfg = host_config();
    cfg.auto_default_tools = false;
    cfg.tools = {make_ping_tool()};
    agent::Session session = open_session(cfg);

    g_ping_calls = 0;
    agent::FlowMemory memory;
    memory.id = "host_tools";
    auto reply = agent::run_turn(session, memory, "call:ping");
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    REQUIRE(g_ping_calls == 1);
    REQUIRE(reply.value == "tool done");
}

TEST_CASE("Process isolation carries media in and out", "[engine_host][media]") {
    agent::Session session = open_session(host_config());
    const agent::Media media[] = {{"/tmp/cat.png", "image/png"}, {"/tmp/hello.wav", "audio/wav"}};
    agent::FlowMemory memory;
    memory.id = "host_media";
    auto reply = agent::run_turn_with_media(session, memory, "describe", media);
    REQUIRE(reply.ok);
    REQUIRE(reply.value == "echo: describe\nmedia: image/png /tmp/cat.png\nmedia: audio/wav /tmp/hello.wav");

    auto drawn = agent::run_turn(session, memory, "draw");
    REQUIRE(drawn.ok);
    const agent::Message &out = memory.history.back();
    REQUIRE(out.media.size() == 1);
    REQUIRE(out.media[0].mime == "text/plain");
    REQUIRE(out.media[0].path.find("artifacts/mock_drawing.txt") != std::string::npos);
    REQUIRE(std::filesystem::exists(out.media[0].path)); // written by the host process
}

TEST_CASE("Process isolation carries tensors both ways", "[engine_host][media]") {
    agent::Session session = open_session(host_config());
    const int64_t shape[] = {4};
    const float values[] = {0.5f, -1.0f, 2.0f, 3.25f};
    const agent::TensorView tensors[] = {{.name = "input",
                                          .dtype = agent::DType::Float32,
                                          .shape = shape,
                                          .data = {reinterpret_cast<const uint8_t *>(values), sizeof(values)}}};
    agent::MessageView msg{};
    msg.role = "user";
    msg.content = "tensors";
    msg.tensors = tensors;
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    auto res = agent::infer(session, req);
    INFO(res.error.message);
    REQUIRE(res.ok);
    REQUIRE(res.value.message.content == "echo: tensors");
    REQUIRE(res.value.output_tensors.size() == 1);
    const agent::TensorView &t = res.value.output_tensors[0];
    REQUIRE(t.name == "input");
    REQUIRE(t.dtype == agent::DType::Float32);
    REQUIRE(t.shape.size() == 1);
    REQUIRE(t.shape[0] == 4);
    REQUIRE(t.data.size() == sizeof(values));
    REQUIRE(std::memcmp(t.data.data(), values, sizeof(values)) == 0);
}

TEST_CASE("Process isolation reports progress and cancels across the process boundary", "[engine_host][cancel]") {
    struct Progress {
        uint64_t last = 0, total = 0;
        agent::Session *session = nullptr;
        uint64_t cancel_at = 0;
    } progress;

    auto cfg = host_config();
    cfg.on_progress = [](uint64_t step, uint64_t total, void *ud) {
        auto *p = static_cast<Progress *>(ud);
        p->last = step;
        p->total = total;
        if (p->cancel_at && step == p->cancel_at) agent::request_cancel(*p->session);
    };
    cfg.progress_user_data = &progress;
    agent::Session session = open_session(cfg);
    progress.session = &session;

    agent::FlowMemory memory;
    memory.id = "host_progress";
    auto done = agent::run_turn(session, memory, "slow");
    REQUIRE(done.ok);
    REQUIRE(progress.last == 100);
    REQUIRE(progress.total == 100);

    progress.cancel_at = 10;
    auto cancelled = agent::run_turn(session, memory, "slow");
    REQUIRE_FALSE(cancelled.ok);
    REQUIRE(cancelled.error.code == agent::ErrorCode::Cancelled);
    // A few progress frames may already be in flight when the cancel arrives.
    REQUIRE(progress.last >= 10);
    REQUIRE(progress.last < 100);
    // The engine stopped cooperatively: the host was not killed.
    REQUIRE(agent::providers::engine_host_starts(session) == 1);

    agent::clear_cancel(session);
    progress.cancel_at = 0;
    REQUIRE(agent::run_turn(session, memory, "hello").ok);
}

// ------------------------------------------------------------------ crashes

TEST_CASE("A crashing engine fails the turn, and the next turn restarts the host", "[engine_host][crash]") {
    struct Case {
        const char *prompt;
        const char *expect; // substring of the error message
    };
    const Case cases[] = {
        {"crash:segv", "died during run"},
        {"crash:abort", "died during run"},
        {"crash:exit", "exited with status 3"},
    };
    for (const Case &c : cases) {
        DYNAMIC_SECTION(c.prompt) {
            agent::Session session = open_session(host_config());
            agent::FlowMemory memory;
            memory.id = std::string("host_") + c.prompt;

            auto crashed = agent::run_turn(session, memory, c.prompt);
            REQUIRE_FALSE(crashed.ok);
            INFO(crashed.error.message);
            REQUIRE(crashed.error.code == agent::ErrorCode::EngineCrashed);
            REQUIRE(crashed.error.message.find(c.expect) != std::string::npos);

            auto next = agent::run_turn(session, memory, "still here?");
            INFO(next.error.message);
            REQUIRE(next.ok);
            REQUIRE(next.value == "echo: still here?");
            REQUIRE(agent::providers::engine_host_starts(session) == 2);
        }
    }
}

TEST_CASE("The watchdog kills a hung engine", "[engine_host][crash]") {
    auto cfg = host_config();
    cfg.engine_timeout_ms = 300;
    agent::Session session = open_session(cfg);
    agent::FlowMemory memory;
    memory.id = "host_hang";

    const auto t0 = std::chrono::steady_clock::now();
    auto hung = agent::run_turn(session, memory, "hang");
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    REQUIRE_FALSE(hung.ok);
    INFO(hung.error.message);
    REQUIRE(hung.error.code == agent::ErrorCode::EngineCrashed);
    REQUIRE(hung.error.message.find("watchdog") != std::string::npos);
    REQUIRE(elapsed < std::chrono::seconds(5));

    auto next = agent::run_turn(session, memory, "hello");
    REQUIRE(next.ok);
    REQUIRE(next.value == "echo: hello");
}

TEST_CASE("A cancel the engine ignores kills the host after the grace period", "[engine_host][cancel]") {
    auto cfg = host_config();
    cfg.engine_cancel_grace_ms = 200;
    cfg.engine_max_restarts = 0; // a kill after cancel does not count as a crash restart
    agent::Session session = open_session(cfg);
    agent::FlowMemory memory;
    memory.id = "host_cancel_hang";

    std::thread canceller([&session] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        agent::request_cancel(session);
    });
    auto hung = agent::run_turn(session, memory, "hang");
    canceller.join();
    REQUIRE_FALSE(hung.ok);
    INFO(hung.error.message);
    REQUIRE(hung.error.code == agent::ErrorCode::Cancelled);

    agent::clear_cancel(session);
    auto next = agent::run_turn(session, memory, "hello");
    INFO(next.error.message);
    REQUIRE(next.ok);
    REQUIRE(agent::providers::engine_host_starts(session) == 2);
}

TEST_CASE("Host restarts are limited per minute", "[engine_host][crash]") {
    auto cfg = host_config();
    cfg.engine_max_restarts = 1;
    agent::Session session = open_session(cfg);
    agent::FlowMemory memory;
    memory.id = "host_restart_limit";

    REQUIRE(agent::run_turn(session, memory, "crash:exit").error.code == agent::ErrorCode::EngineCrashed);
    // First restart is allowed; the engine crashes again.
    REQUIRE(agent::run_turn(session, memory, "crash:exit").error.code == agent::ErrorCode::EngineCrashed);
    auto refused = agent::run_turn(session, memory, "hello");
    REQUIRE_FALSE(refused.ok);
    INFO(refused.error.message);
    REQUIRE(refused.error.code == agent::ErrorCode::EngineCrashed);
    REQUIRE(refused.error.message.find("restarts in the last minute") != std::string::npos);
    REQUIRE(agent::providers::engine_host_starts(session) == 2);
}

// ------------------------------------------------------------------ setup errors and selection

TEST_CASE("Process isolation reports startup failures", "[engine_host]") {
    auto res = agent::init(host_config("fail"));
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
    REQUIRE(res.error.message.find("refusing model") != std::string::npos);

    auto cfg = host_config();
    cfg.engine_host_path = "/nonexistent/agent-engine-host";
    auto no_host = agent::init(cfg);
    REQUIRE_FALSE(no_host.ok);
    REQUIRE(no_host.error.code == agent::ErrorCode::ProviderInitFailed);

    cfg = host_config();
    cfg.engine = "/nonexistent/agent-engine-nope.so";
    auto no_plugin = agent::init(cfg);
    REQUIRE_FALSE(no_plugin.ok);
    INFO(no_plugin.error.message);
    REQUIRE(no_plugin.error.code == agent::ErrorCode::ProviderInitFailed);
    REQUIRE(no_plugin.error.message.find("agent-engine-nope") != std::string::npos);
}

TEST_CASE("engines.json \"isolation\" selects process isolation by default", "[engine_host]") {
    std::filesystem::create_directories(kWorkspace);
    const std::string manifest = std::string(kWorkspace) + "/engines.json";
    {
        std::ofstream f(manifest);
        f << R"({"engines": [{"name": "mock-isolated", "path": ")" << AGENT_TEST_MOCK_ENGINE_PATH << R"(", "isolation": "process"}]})";
    }
    auto names = agent::load_engine_manifest(manifest);
    INFO(names.error.message);
    REQUIRE(names.ok);
    REQUIRE(names.value == std::vector<std::string>{"mock-isolated"});

    agent::Config cfg = host_config();
    cfg.engine = "mock-isolated";
    cfg.engine_isolation = agent::EngineIsolation::Default;
    agent::Session isolated = open_session(cfg);
    REQUIRE(agent::providers::engine_host_starts(isolated) == 1);
    agent::FlowMemory memory;
    memory.id = "host_manifest";
    REQUIRE(agent::run_turn(isolated, memory, "hi").value == "echo: hi");

    // An explicit InProcess overrides the manifest.
    cfg.engine_isolation = agent::EngineIsolation::InProcess;
    agent::Session local = open_session(cfg);
    REQUIRE(agent::providers::engine_host_starts(local) == -1);

    {
        std::ofstream f(manifest);
        f << R"({"engines": [{"path": ")" << AGENT_TEST_MOCK_ENGINE_PATH << R"(", "isolation": "sideways"}]})";
    }
    auto bad = agent::load_engine_manifest(manifest);
    REQUIRE_FALSE(bad.ok);
    REQUIRE(bad.error.code == agent::ErrorCode::ParseError);
}

TEST_CASE("Process isolation needs a plugin file", "[engine_host]") {
    static const agent_engine_v1 *linked = [] {
        static agent_engine_v1 e = {AGENT_ENGINE_ABI_VERSION,
                                    "linked_in",
                                    "0",
                                    AGENT_CAP_TEXT_GEN,
                                    0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr};
        e.create = [](agent_str, void **out, char *, size_t) -> int32_t {
            static int dummy;
            *out = &dummy;
            return AGENT_OK;
        };
        e.destroy = [](void *) {};
        e.run = [](void *, const agent_request *, agent_response *) -> int32_t { return AGENT_ERR_UNSUPPORTED; };
        e.free_response = [](void *, agent_response *) {};
        e.last_error = [](void *) -> const char * { return ""; };
        return &e;
    }();
    REQUIRE(agent::register_engine(linked).ok);
    auto cfg = host_config();
    cfg.engine = "linked_in";
    auto res = agent::init(cfg);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::InvalidConfig);
}

// ------------------------------------------------------------------ overhead

// Not run by default: `test_agent "[.bench]" -s` prints the per-turn cost of process isolation.
TEST_CASE("Process isolation per-turn overhead", "[.bench][engine_host]") {
    constexpr int kTurns = 300;
    auto measure = [&](agent::EngineIsolation isolation, bool stream) {
        auto cfg = host_config();
        cfg.engine_isolation = isolation;
        cfg.auto_default_tools = false;
        agent::Session session = open_session(cfg);
        agent::FlowMemory memory;
        memory.id = "bench";
        size_t chunks = 0;
        auto on_token = [](std::string_view, void *ud) { ++*static_cast<size_t *>(ud); };
        for (int i = 0; i < 10; ++i) REQUIRE(agent::run_turn(session, memory, "warm up", stream, on_token, &chunks).ok);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kTurns; ++i) {
            memory.history.clear();
            REQUIRE(agent::run_turn(session, memory, stream ? "stream:64" : "hello", stream, on_token, &chunks).ok);
        }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        return static_cast<double>(us) / kTurns;
    };
    const double local = measure(agent::EngineIsolation::InProcess, false);
    const double remote = measure(agent::EngineIsolation::Process, false);
    const double local_stream = measure(agent::EngineIsolation::InProcess, true);
    const double remote_stream = measure(agent::EngineIsolation::Process, true);
    std::printf("per turn (mock engine, %d turns): in-process %.1f us, process %.1f us (+%.1f us)\n", kTurns, local, remote,
                remote - local);
    std::printf("per turn streaming 64+1 chunks:          in-process %.1f us, process %.1f us (+%.1f us)\n", local_stream, remote_stream,
                remote_stream - local_stream);
}

// ------------------------------------------------------------------ real engine

#if defined(AGENT_TEST_LLAMA_ENGINE_PATH)
#define AGENT_HOST_LLAMA_DEFAULT AGENT_TEST_LLAMA_ENGINE_PATH
#else
#define AGENT_HOST_LLAMA_DEFAULT nullptr
#endif

// Runs with AGENT_ENGINE_LLAMA=ON (or AGENT_TEST_LLAMA_PLUGIN=<agent-engine-llama.so>) and
// AGENT_TEST_LLAMA_TOOL_MODEL=<instruct gguf>.
TEST_CASE("The llama engine runs a tool-call turn through the host", "[engine_host][llama][model]") {
    const char *plugin = std::getenv("AGENT_TEST_LLAMA_PLUGIN");
    if (!plugin) plugin = AGENT_HOST_LLAMA_DEFAULT;
    const char *model = std::getenv("AGENT_TEST_LLAMA_TOOL_MODEL");
    if (!plugin || !model) SKIP("needs the llama plugin and AGENT_TEST_LLAMA_TOOL_MODEL");

    auto cfg = host_config(model);
    cfg.engine = plugin;
    cfg.context_window = 2048;
    cfg.max_tokens = 128;
    cfg.temperature = 0.0f;
    cfg.auto_default_tools = false;
    cfg.tools = {make_ping_tool()};
    agent::Session session = open_session(cfg);

    g_ping_calls = 0;
    std::string streamed;
    agent::FlowMemory memory;
    memory.id = "host_llama";
    auto reply = agent::run_turn(
        session, memory, "Call the ping tool now.", true, [](std::string_view tok, void *ud) { *static_cast<std::string *>(ud) += tok; },
        &streamed);
    INFO(reply.error.message);
    REQUIRE(reply.ok);
    REQUIRE(g_ping_calls >= 1);
    REQUIRE_FALSE(streamed.empty());
}
