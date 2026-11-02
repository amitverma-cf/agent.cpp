// decide() / decide_score() / decide_yes_no() and InferRequest::params_json: plumbing against a fake
// engine, then the llama engine's AGENT_CAP_DECIDE with a real instruct model.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

const char *kWorkspace = "/tmp/agent_test_decide_ws";

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

// ------------------------------------------------------------------ fake decide engine
// Records the last request's params_json and texts; returns "probs" over n options where n is the
// number of "options" entries (choice), the scale size (score) or 2 (yesno): option i gets weight i+1,
// so the last option is the most likely.

struct FakeDecide {
    std::string params;
    std::string roles;
    std::vector<std::string> texts;
    size_t n_media = 0;
    std::vector<float> probs;
    int64_t shape = 0;
    agent_part part{};
};

FakeDecide *g_fake = nullptr;

int32_t fake_create(agent_str, void **out, char *, size_t) {
    g_fake = new FakeDecide();
    *out = g_fake;
    return AGENT_OK;
}
void fake_destroy(void *inst) {
    if (g_fake == inst) g_fake = nullptr;
    delete static_cast<FakeDecide *>(inst);
}

size_t count(const std::string &s, std::string_view needle) {
    size_t n = 0;
    for (size_t at = s.find(needle); at != std::string::npos; at = s.find(needle, at + 1)) ++n;
    return n;
}

int32_t fake_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *f = static_cast<FakeDecide *>(instance);
    if (req->task != AGENT_CAP_DECIDE) return AGENT_ERR_UNSUPPORTED;
    f->params = std::string(sv(req->params_json));
    f->roles.clear();
    f->texts.clear();
    f->n_media = 0;
    for (size_t i = 0; i < req->n_messages; ++i) {
        f->roles += std::string(sv(req->messages[i].role)) + ";";
        for (size_t p = 0; p < req->messages[i].n_parts; ++p) {
            const agent_part &part = req->messages[i].parts[p];
            if (part.kind == AGENT_PART_TEXT) f->texts.emplace_back(sv(part.text));
            if (part.kind == AGENT_PART_FILE) ++f->n_media;
        }
    }
    size_t n = 2;
    if (f->params.find("\"choice\"") != std::string::npos) {
        const size_t open = f->params.find("\"options\":[");
        n = count(f->params.substr(open, f->params.find(']', open) - open), "\",\"") + 1;
    } else if (f->params.find("\"score\"") != std::string::npos) {
        const size_t open = f->params.find("\"scale\":[") + 9;
        const int lo = std::atoi(f->params.c_str() + open);
        const int hi = std::atoi(f->params.c_str() + f->params.find(',', open) + 1);
        n = static_cast<size_t>(hi - lo + 1);
    }
    f->probs.assign(n, 0.0f);
    const float total = static_cast<float>(n * (n + 1) / 2);
    for (size_t i = 0; i < n; ++i) f->probs[i] = static_cast<float>(i + 1) / total;
    f->shape = static_cast<int64_t>(n);
    f->part = agent_part{};
    f->part.kind = AGENT_PART_TENSOR;
    f->part.name = agent_str{"probs", 5};
    f->part.dtype = AGENT_DTYPE_F32;
    f->part.shape = &f->shape;
    f->part.ndim = 1;
    f->part.data = f->probs.data();
    f->part.size = n * sizeof(float);
    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = &f->part;
    resp->message.n_parts = 1;
    return AGENT_OK;
}

void fake_free_response(void *, agent_response *resp) { agent_response_reset(resp); }
const char *fake_last_error(void *) { return ""; }

// clang-format off
const agent_engine_v1 kFakeDecideEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "decide-fake",
    "1.0",
    AGENT_CAP_DECIDE,
    0,
    fake_create,
    fake_destroy,
    fake_run,
    fake_free_response,
    fake_last_error,
    nullptr,
    nullptr,
};
// clang-format on

agent::Session fake_session() {
    REQUIRE(agent::register_engine(&kFakeDecideEngine).ok);
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = "decide-fake";
    cfg.model = "fake";
    cfg.workspace_dir = kWorkspace;
    auto s = agent::init(cfg);
    REQUIRE(s.ok);
    return std::move(s.value);
}

} // namespace

TEST_CASE("decide needs an engine with AGENT_CAP_DECIDE", "[decide]") {
    agent::Config cfg;
    cfg.workspace_dir = kWorkspace;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH; // TEXT_GEN only
    cfg.model = "mock.bin";
    auto s = agent::init(cfg);
    REQUIRE(s.ok);
    const std::string_view options[] = {"a", "b"};
    auto d = agent::decide(s.value, "q?", options);
    REQUIRE_FALSE(d.ok);
    REQUIRE(d.error.code == agent::ErrorCode::UnsupportedProvider);
    REQUIRE_FALSE(agent::decide_yes_no(s.value, "q?").ok);
}

TEST_CASE("decide sends the options in params_json and returns the engine's distribution", "[decide]") {
    agent::Session session = fake_session();
    const std::string_view options[] = {"Berlin", "Paris \"City of Light\"", "Rome"};
    auto d = agent::decide(session, "Capital of France?", options);
    REQUIRE(d.ok);
    REQUIRE(g_fake != nullptr);
    // The core's own keys come first, the decide parameters after them, in one object.
    CHECK(g_fake->params.starts_with("{\"max_tokens\":"));
    CHECK(g_fake->params.find(R"("mode":"choice","options":["Berlin","Paris \"City of Light\"","Rome"]})") != std::string::npos);
    REQUIRE(g_fake->texts.size() == 1);
    CHECK(g_fake->texts[0] == "Capital of France?");
    CHECK(g_fake->roles == "user;");
    REQUIRE(d.value.options.size() == 3);
    CHECK(d.value.options[1] == "Paris \"City of Light\"");
    REQUIRE(d.value.probs.size() == 3);
    CHECK(d.value.choice == 2);
    CHECK(d.value.confidence == Catch::Approx(0.5f));

    SECTION("validation") {
        const std::string_view one[] = {"only"};
        CHECK(agent::decide(session, "q?", one).error.code == agent::ErrorCode::InvalidConfig);
        const std::string_view empty[] = {"a", ""};
        CHECK(agent::decide(session, "q?", empty).error.code == agent::ErrorCode::InvalidConfig);
        CHECK(agent::decide(session, "", options).error.code == agent::ErrorCode::InvalidConfig);
        CHECK(agent::decide_score(session, "q?", 5, 5).error.code == agent::ErrorCode::InvalidConfig);
        CHECK(agent::decide_score(session, "q?", 0, 101).error.code == agent::ErrorCode::InvalidConfig);
    }
}

TEST_CASE("decide_score computes the expected value; decide_yes_no returns P(yes)", "[decide]") {
    agent::Session session = fake_session();
    agent::DecideOptions opts;
    opts.system = "You are a strict grader.";
    opts.instructions = false;
    const agent::Media image{"/tmp/x.png", "image/png"};
    opts.media = std::span<const agent::Media>(&image, 1);
    auto d = agent::decide_score(session, "How good is this?", 1, 4, opts);
    REQUIRE(d.ok);
    CHECK(g_fake->params.find(R"("mode":"score","scale":[1,4],"instructions":false})") != std::string::npos);
    CHECK(g_fake->roles == "system;user;");
    CHECK(g_fake->n_media == 1);
    REQUIRE(d.value.options == std::vector<std::string>{"1", "2", "3", "4"});
    // probs 0.1, 0.2, 0.3, 0.4 -> 0.1 + 0.4 + 0.9 + 1.6
    CHECK(d.value.score == Catch::Approx(3.0f));

    auto yes = agent::decide_yes_no(session, "Is it?");
    REQUIRE(yes.ok);
    CHECK(g_fake->params.find(R"("mode":"yesno"})") != std::string::npos);
    CHECK(yes.value == Catch::Approx(1.0f / 3.0f));
}

TEST_CASE("InferRequest::params_json must be a JSON object", "[decide][engines]") {
    agent::Session session = fake_session();
    agent::MessageView msg{.role = "user", .content = "q"};
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    req.params_json = "[1,2]";
    auto r = agent::run_task(session, AGENT_CAP_DECIDE, req);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == agent::ErrorCode::InvalidConfig);
    req.params_json = "  { }  "; // an empty object adds nothing
    REQUIRE(agent::run_task(session, AGENT_CAP_DECIDE, req).ok);
    CHECK(g_fake->params.ends_with("false}"));
}

// ------------------------------------------------------------------ llama engine, real model

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
namespace {

// Set AGENT_TEST_LLAMA_TOOL_MODEL to an instruct GGUF, e.g. Qwen2.5-0.5B-Instruct Q4_K_M.
const char *decide_model() { return std::getenv("AGENT_TEST_LLAMA_TOOL_MODEL"); }

agent::Config llama_decide_config(const char *model) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_LLAMA_ENGINE_PATH;
    cfg.model = model;
    cfg.workspace_dir = kWorkspace;
    cfg.context_window = 1024;
    cfg.auto_default_tools = false;
    return cfg;
}

} // namespace

TEST_CASE("llama decide: yes/no, choice and score from log-probabilities", "[decide][llama][model]") {
    if (!decide_model()) SKIP("AGENT_TEST_LLAMA_TOOL_MODEL not set");
    auto s = agent::init(llama_decide_config(decide_model()));
    REQUIRE(s.ok);
    agent::Session &session = s.value;

    auto yes = agent::decide_yes_no(session, "Is Paris the capital of France?");
    REQUIRE(yes.ok);
    INFO("P(yes | Paris) = " << yes.value);
    CHECK(yes.value > 0.5f);
    auto no = agent::decide_yes_no(session, "Is Madrid the capital of France?");
    REQUIRE(no.ok);
    INFO("P(yes | Madrid) = " << no.value);
    CHECK(no.value < 0.5f);
    CHECK(yes.value > no.value);

    const std::string_view capitals[] = {"Berlin", "Paris", "Madrid", "Rome"};
    auto choice = agent::decide(session, "What is the capital of France?", capitals);
    REQUIRE(choice.ok);
    float sum = 0.0f;
    for (float p : choice.value.probs) sum += p;
    CHECK(sum == Catch::Approx(1.0f).margin(1e-4));
    INFO("P(Paris) = " << choice.value.probs[1]);
    CHECK(choice.value.choice == 1);

    // Multi-token options and one option that is a prefix of another.
    const std::string_view animals[] = {"a cat", "a cathedral", "a fish"};
    auto multi = agent::decide(session, "Which of these lives in water?", animals);
    REQUIRE(multi.ok);
    CHECK(multi.value.choice == 2);

    auto low = agent::decide_score(session, "On a scale of 1 to 5, how hot is ice?", 1, 5);
    REQUIRE(low.ok);
    auto high = agent::decide_score(session, "On a scale of 1 to 5, how hot is the surface of the sun?", 1, 5);
    REQUIRE(high.ok);
    INFO("ice " << low.value.score << ", sun " << high.value.score);
    CHECK(high.value.score > low.value.score);
    CHECK(low.value.score >= 1.0f);
    CHECK(high.value.score <= 5.0f);

    // The raw engine response also carries the JSON summary.
    agent::MessageView msg{.role = "user", .content = "Is water wet?"};
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    req.params_json = R"({"mode":"yesno"})";
    auto raw = agent::run_task(session, AGENT_CAP_DECIDE, req);
    REQUIRE(raw.ok);
    CHECK(raw.value.message.content.find("\"mode\":\"yesno\"") != std::string_view::npos);
    CHECK(raw.value.message.content.find("\"yes\":") != std::string_view::npos);
    REQUIRE(raw.value.output_tensors.size() == 1);
    CHECK(raw.value.output_tensors[0].name == "probs");

    req.params_json = R"({"mode":"choice","options":["only one"]})";
    auto bad = agent::run_task(session, AGENT_CAP_DECIDE, req);
    REQUIRE_FALSE(bad.ok);
    CHECK(bad.error.message.find("options") != std::string::npos); // engine errors arrive as DecodeFailed
}
#endif
