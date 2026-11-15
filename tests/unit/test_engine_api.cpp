// The api engine (src/engines/api) against a scripted stand-in for its bun bridge (tests/helpers/fake_api_bridge.cpp): no network, no bun.
#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string slash(const fs::path &p) { return p.generic_string(); }

struct Dir {
    fs::path path;
    explicit Dir(const std::string &name) {
        path = fs::temp_directory_path() / ("agent_test_api_" + name);
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directories(path, ec);
    }
    ~Dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string file(const std::string &n) const { return slash(path / n); }
};

std::string read_all(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// engine_options for the fake bridge; `bridge_args` are extra arguments for it (already JSON strings, comma separated).
std::string options(const Dir &dir, const std::string &bridge_args = {}, const std::string &extra = {}) {
    std::string o =
        "{\"base_url\":\"https://example.invalid/v1\",\"model\":\"vendor/model-1\",\"bridge_cmd\":\"" AGENT_TEST_FAKE_API_BRIDGE_PATH
        "\",\"bridge_args\":[\"--state\",\"" +
        dir.file("state") + "\"" + (bridge_args.empty() ? "" : "," + bridge_args) +
        "],\"retry_base_ms\":10,\"retry_max_ms\":40,\"max_tries\":4" + (extra.empty() ? "" : "," + extra) + "}";
    return o;
}

agent::Config config(const std::string &engine_options, const Dir &dir) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_API_ENGINE_PATH;
    cfg.model = "remote";
    cfg.engine_options = engine_options;
    cfg.workspace_dir = slash(dir.path / "ws");
    cfg.auto_default_tools = false;
    return cfg;
}

int g_ping_calls = 0;

agent::Tool ping_tool() {
    agent::Tool t;
    t.name = "ping";
    t.description = "test tool";
    t.parameter_schema = "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"}}}";
    t.callback = [](std::string_view, void *) -> agent::Result<std::string> {
        ++g_ping_calls;
        return agent::ok(std::string("pong"));
    };
    return t;
}

void collect(std::string_view tok, void *ud) { static_cast<std::string *>(ud)->append(tok); }

} // namespace

TEST_CASE("api engine: a plain answer, streamed tokens and usage", "[api]") {
    Dir dir("plain");
    auto res = agent::init(config(options(dir, "\"--log\",\"" + dir.file("log") + "\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    const agent_engine_v1 *engine = agent::find_engine("api");
    REQUIRE(engine != nullptr);
    CHECK((engine->caps & AGENT_CAP_TEXT_GEN) != 0);
    CHECK((engine->caps & AGENT_CAP_TOOL_CALL) != 0);
    CHECK((engine->flags & AGENT_ENGINE_FLAG_NATIVE_TOOLS) != 0);

    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "hello \"world\"");
    REQUIRE(reply.ok);
    CHECK(reply.value == "echo: hello \"world\"");
    CHECK(memory.stats.last.prompt_tokens == 7);
    CHECK(memory.stats.last.completion_tokens == 3);
    CHECK(memory.stats.last.retries == 0);

    std::string streamed;
    auto second = agent::run_turn(session, memory, "again", true, collect, &streamed);
    REQUIRE(second.ok);
    CHECK(streamed == "echo: again");
    CHECK(second.value == "echo: again");

    // The request the bridge got: model, key variable, whole conversation, OpenAI roles.
    const std::string log = read_all(dir.file("log"));
    CHECK(log.find("\"model\":\"vendor/model-1\"") != std::string::npos);
    CHECK(log.find("\"key_env\":\"NUMES_API_KEY\"") != std::string::npos);
    CHECK(log.find("\"role\":\"assistant\"") != std::string::npos);
    CHECK(log.find("hello \\\"world\\\"") != std::string::npos);
}

TEST_CASE("api engine: a tool call round trip", "[api]") {
    Dir dir("tools");
    auto cfg = config(options(dir, "\"--log\",\"" + dir.file("log") + "\""), dir);
    cfg.tools = {ping_tool()};
    auto res = agent::init(cfg);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    g_ping_calls = 0;
    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "call:ping");
    REQUIRE(reply.ok);
    CHECK(g_ping_calls == 1);
    CHECK(reply.value == "tool done");

    const std::string log = read_all(dir.file("log"));
    CHECK(log.find("\"tools\":[{\"name\":\"ping\"") != std::string::npos);
    // The second request carries the assistant tool call and the tool result with its id.
    CHECK(log.find("\"tool_call_id\":\"call_1\"") != std::string::npos);
    CHECK(log.find("\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\"") != std::string::npos);
}

TEST_CASE("api engine: retries a retryable error and reports how many", "[api][retry]") {
    Dir dir("retry");
    auto res = agent::init(config(options(dir, "\"--fail\",\"2\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "ping");
    REQUIRE(reply.ok);
    CHECK(reply.value == "echo: ping");
    CHECK(memory.stats.last.retries == 2);
    // The failed attempts left no trace in the conversation.
    CHECK(memory.history.size() == 2);
}

TEST_CASE("api engine: an empty first reply is retried", "[api][retry]") {
    Dir dir("empty");
    auto res = agent::init(config(options(dir, "\"--bad-first\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "ok?");
    REQUIRE(reply.ok);
    CHECK(reply.value == "echo: ok?");
    CHECK(memory.stats.last.retries == 1);
}

TEST_CASE("api engine: gives up after max_tries and says how often it tried", "[api][retry]") {
    Dir dir("giveup");
    auto res = agent::init(config(options(dir, "\"--fail\",\"99\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "ping");
    REQUIRE_FALSE(reply.ok);
    CHECK(reply.error.message.find("HTTP 503") != std::string::npos);
    CHECK(reply.error.message.find("after 4 tries") != std::string::npos);
}

TEST_CASE("api engine: a non-retryable error is reported at once", "[api][retry]") {
    Dir dir("fatal");
    auto res = agent::init(config(options(dir, "\"--fatal\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "ping");
    REQUIRE_FALSE(reply.ok);
    CHECK(reply.error.message.find("HTTP 401") != std::string::npos);
    CHECK(reply.error.message.find("tries") == std::string::npos);
    CHECK(std::stoi(read_all(dir.file("state"))) == 1);
}

TEST_CASE("api engine: a retry after partial streaming does not repeat text", "[api][retry]") {
    Dir dir("partial");
    auto res = agent::init(config(options(dir, "\"--partial\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    std::string streamed;
    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "go", true, collect, &streamed);
    REQUIRE(reply.ok);
    CHECK(reply.value == "Hello world");
    CHECK(streamed == "Hello world");
    CHECK(memory.stats.last.retries == 1);
}

TEST_CASE("api engine: a bridge that dies is started again", "[api][retry]") {
    Dir dir("die");
    auto res = agent::init(config(options(dir, "\"--die-first\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    agent::Conversation memory;
    auto reply = agent::run_turn(session, memory, "still there?");
    REQUIRE(reply.ok);
    CHECK(reply.value == "echo: still there?");
    CHECK(memory.stats.last.retries == 1);
}

TEST_CASE("api engine: request_cancel stops a waiting request", "[api][cancel]") {
    Dir dir("cancel");
    auto res = agent::init(config(options(dir, "\"--wait-cancel\""), dir));
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        agent::request_cancel(session);
    });
    agent::Conversation memory;
    const auto start = std::chrono::steady_clock::now();
    auto reply = agent::run_turn(session, memory, "wait");
    canceller.join();
    REQUIRE_FALSE(reply.ok);
    CHECK(reply.error.code == agent::ErrorCode::Cancelled);
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
}

TEST_CASE("api engine: refuses to start when Local only is on, and checks its options", "[api]") {
    Dir dir("refuse");
    auto local = agent::init(config(options(dir, {}, "\"local_only\":true"), dir));
    REQUIRE_FALSE(local.ok);
    CHECK(local.error.message.find("Local only") != std::string::npos);

    auto no_model = agent::init(config("{\"base_url\":\"https://example.invalid/v1\"}", dir));
    REQUIRE_FALSE(no_model.ok);
    CHECK(no_model.error.message.find("model") != std::string::npos);

    auto bad_url = agent::init(config("{\"base_url\":\"example.invalid\",\"model\":\"m\"}", dir));
    REQUIRE_FALSE(bad_url.ok);
    CHECK(bad_url.error.message.find("base_url") != std::string::npos);

    auto no_runtime =
        agent::init(config("{\"base_url\":\"https://example.invalid/v1\",\"model\":\"m\",\"bun\":\"/nonexistent/bun\"}", dir));
    REQUIRE_FALSE(no_runtime.ok);
    CHECK(no_runtime.error.message.find("bun") != std::string::npos);
}
