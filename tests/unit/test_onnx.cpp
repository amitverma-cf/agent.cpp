#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

TEST_CASE("ONNX tensor types compose correctly", "[onnx][tensors]") {
    agent::TensorData td;
    td.name = "input";
    td.dtype = agent::DType::Float32;
    td.shape = {1, 3, 224, 224};
    td.data.resize(1 * 3 * 224 * 224 * 4, 0);

    REQUIRE(td.shape.size() == 4);
    REQUIRE(td.data.size() == 1 * 3 * 224 * 224 * 4);

    agent::Message msg;
    msg.role = "user";
    msg.tensors.push_back(std::move(td));
    REQUIRE(msg.tensors.size() == 1);
    REQUIRE(msg.tensors[0].name == "input");
}

#ifdef AGENT_TEST_ONNX_ENGINE_PATH

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// Hand-encoded ONNX ModelProto (IR 8, opset 13): sum = Add(a, b), all float32 [2, 2].
constexpr unsigned char kAddModel[] = {
    0x08, 0x08, 0x12, 0x09, 0x61, 0x67, 0x65, 0x6e, 0x74, 0x2e, 0x63, 0x70, 0x70, 0x3a, 0x63, 0x0a, 0x15, 0x0a, 0x01, 0x61,
    0x0a, 0x01, 0x62, 0x12, 0x03, 0x73, 0x75, 0x6d, 0x1a, 0x03, 0x61, 0x64, 0x64, 0x22, 0x03, 0x41, 0x64, 0x64, 0x12, 0x09,
    0x61, 0x64, 0x64, 0x5f, 0x67, 0x72, 0x61, 0x70, 0x68, 0x5a, 0x13, 0x0a, 0x01, 0x61, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01,
    0x12, 0x08, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x5a, 0x13, 0x0a, 0x01, 0x62, 0x12, 0x0e, 0x0a, 0x0c, 0x08,
    0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x62, 0x15, 0x0a, 0x03, 0x73, 0x75, 0x6d, 0x12, 0x0e,
    0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d,
};

const char *kWorkspace = "/tmp/agent_test_onnx_ws";

std::string add_model_path() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "agent_test_onnx";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "add.onnx").string();
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(kAddModel), sizeof(kAddModel));
    return path;
}

const agent_engine_v1 *onnx_engine() {
    auto loaded = agent::load_engine_plugin(AGENT_TEST_ONNX_ENGINE_PATH);
    REQUIRE(loaded.ok);
    REQUIRE(loaded.value == "onnx");
    const agent_engine_v1 *engine = agent::find_engine("onnx");
    REQUIRE(engine != nullptr);
    return engine;
}

agent_str str(const std::string &s) { return agent_str{s.data(), s.size()}; }

void *create_or_fail(const agent_engine_v1 *engine, const std::string &config) {
    void *inst = nullptr;
    char err[512] = {0};
    const int32_t rc = engine->create(str(config), &inst, err, sizeof(err));
    INFO(err);
    REQUIRE(rc == AGENT_OK);
    REQUIRE(inst != nullptr);
    return inst;
}

constexpr int64_t kShape[] = {2, 2};

agent_part tensor_part(const char *name, const float *data) {
    agent_part p{};
    p.kind = AGENT_PART_TENSOR;
    p.name = agent_str{name, std::strlen(name)};
    p.dtype = AGENT_DTYPE_F32;
    p.shape = kShape;
    p.ndim = 2;
    p.data = data;
    p.size = 4 * sizeof(float);
    return p;
}

} // namespace

TEST_CASE("onnx engine plugin loads and reports GENERIC caps", "[onnx][engines]") {
    const agent_engine_v1 *engine = onnx_engine();
    REQUIRE(engine->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE(engine->caps == AGENT_CAP_GENERIC);
    REQUIRE((engine->flags & AGENT_ENGINE_FLAG_CONCURRENT) != 0);
}

TEST_CASE("onnx engine create() validates its config", "[onnx][engines]") {
    const agent_engine_v1 *engine = onnx_engine();
    const std::string model = add_model_path();
    void *inst = nullptr;
    char err[512] = {0};

    REQUIRE(engine->create(str(std::string("{}")), &inst, err, sizeof(err)) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(std::string(err).find("model") != std::string::npos);

    REQUIRE(engine->create(str(std::string(R"({"model":"/nonexistent/model.onnx"})")), &inst, err, sizeof(err)) ==
            AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);

    REQUIRE(engine->create(str(R"({"model":")" + model + R"(","options":{"execution_provider":"tpu"}})"), &inst, err, sizeof(err)) ==
            AGENT_ERR_INVALID_CONFIG);
    REQUIRE(engine->create(str(R"({"model":")" + model + R"(","options":{"graph_optimization":"max"}})"), &inst, err, sizeof(err)) ==
            AGENT_ERR_INVALID_CONFIG);

    // The official CPU release has no CUDA EP: create() must fail cleanly, not crash.
    err[0] = 0;
    const int32_t rc =
        engine->create(str(R"({"model":")" + model + R"(","options":{"execution_provider":"cuda"}})"), &inst, err, sizeof(err));
    if (rc == AGENT_OK) {
        engine->destroy(inst);
    } else {
        REQUIRE(rc == AGENT_ERR_UNSUPPORTED);
        REQUIRE(std::string(err).find("cuda") != std::string::npos);
    }
}

TEST_CASE("onnx engine adds two [2,2] tensors through the C ABI", "[onnx][engines]") {
    const agent_engine_v1 *engine = onnx_engine();
    void *inst = create_or_fail(engine, R"({"model":")" + add_model_path() +
                                            R"(","options":{"threads":1,"graph_optimization":"basic","execution_provider":"cpu"}})");

    const float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    // a and b arrive on different messages; the engine matches inputs by name.
    const agent_part parts_a[] = {tensor_part("a", a)};
    const agent_part parts_b[] = {tensor_part("b", b)};
    const agent_message msgs[] = {
        agent_message{.role = {"user", 4}, .parts = parts_a, .n_parts = 1},
        agent_message{.role = {"user", 4}, .parts = parts_b, .n_parts = 1},
    };
    agent_request req{};
    req.task = AGENT_CAP_GENERIC;
    req.messages = msgs;
    req.n_messages = 2;

    agent_response resp{};
    REQUIRE(engine->run(inst, &req, &resp) == AGENT_OK);
    REQUIRE(resp.message.n_parts == 1);
    const agent_part &out = resp.message.parts[0];
    REQUIRE(out.kind == AGENT_PART_TENSOR);
    REQUIRE(std::string(out.name.data, out.name.size) == "sum");
    REQUIRE(out.dtype == AGENT_DTYPE_F32);
    REQUIRE(out.ndim == 2);
    REQUIRE(out.shape[0] == 2);
    REQUIRE(out.shape[1] == 2);
    REQUIRE(out.size == 4 * sizeof(float));
    const auto *sum = static_cast<const float *>(out.data);
    REQUIRE(sum[0] == 11.0f);
    REQUIRE(sum[1] == 22.0f);
    REQUIRE(sum[2] == 33.0f);
    REQUIRE(sum[3] == 44.0f);
    engine->free_response(inst, &resp);
    REQUIRE(resp.message.parts == nullptr);

    SECTION("missing input") {
        req.n_messages = 1;
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_INVALID_CONFIG);
        REQUIRE(std::string(engine->last_error(inst)).find("'b'") != std::string::npos);
    }
    SECTION("wrong task") {
        req.task = AGENT_CAP_TEXT_GEN;
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_UNSUPPORTED);
    }
    SECTION("wrong dtype") {
        agent_part bad_b = tensor_part("b", b);
        bad_b.dtype = AGENT_DTYPE_I32;
        const agent_message bad[] = {msgs[0], agent_message{.role = {"user", 4}, .parts = &bad_b, .n_parts = 1}};
        req.messages = bad;
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_INVALID_CONFIG);
    }
    SECTION("concurrent runs keep per-call outputs") {
        std::vector<std::thread> threads;
        std::vector<int> failures(4, 0);
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&, t] {
                for (int iter = 0; iter < 50; ++iter) {
                    const float x = static_cast<float>(t * 1000 + iter);
                    const float ta[4] = {x, x, x, x};
                    const float tb[4] = {1, 2, 3, 4};
                    const agent_part tp[] = {tensor_part("a", ta), tensor_part("b", tb)};
                    const agent_message tm{.role = {"user", 4}, .parts = tp, .n_parts = 2};
                    agent_request tr{};
                    tr.task = AGENT_CAP_GENERIC;
                    tr.messages = &tm;
                    tr.n_messages = 1;
                    agent_response r{};
                    if (engine->run(inst, &tr, &r) != AGENT_OK) {
                        ++failures[static_cast<size_t>(t)];
                        continue;
                    }
                    const auto *s = static_cast<const float *>(r.message.parts[0].data);
                    for (int k = 0; k < 4; ++k)
                        if (s[k] != x + static_cast<float>(k + 1)) ++failures[static_cast<size_t>(t)];
                    engine->free_response(inst, &r);
                }
            });
        }
        for (auto &th : threads) th.join();
        for (int f : failures) REQUIRE(f == 0);
    }
    engine->destroy(inst);
}

TEST_CASE("onnx engine runs through agent::init with AiProvider::Engine", "[onnx][engines]") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_ONNX_ENGINE_PATH;
    cfg.model = add_model_path();
    cfg.engine_options = R"({"threads":1})";
    cfg.workspace_dir = kWorkspace;
    auto res = agent::init(cfg);
    INFO(res.error.message);
    REQUIRE(res.ok);
    agent::Session session = std::move(res.value);

    const float a[4] = {0.5f, -1.0f, 2.0f, 8.0f};
    const float b[4] = {0.5f, 1.0f, -2.0f, 2.0f};
    const agent::TensorView inputs[] = {
        {.name = "a", .dtype = agent::DType::Float32, .shape = kShape, .data = {reinterpret_cast<const uint8_t *>(a), sizeof(a)}},
        {.name = "b", .dtype = agent::DType::Float32, .shape = kShape, .data = {reinterpret_cast<const uint8_t *>(b), sizeof(b)}},
    };
    agent::MessageView msg{};
    msg.role = "user";
    msg.tensors = inputs;
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);

    auto out = agent::infer(session, req);
    INFO(out.error.message);
    REQUIRE(out.ok);
    REQUIRE(out.value.output_tensors.size() == 1);
    const agent::TensorView &sum = out.value.output_tensors[0];
    REQUIRE(sum.name == "sum");
    REQUIRE(sum.dtype == agent::DType::Float32);
    REQUIRE(sum.shape.size() == 2);
    REQUIRE(sum.data.size() == sizeof(a));
    float got[4];
    std::memcpy(got, sum.data.data(), sizeof(got));
    REQUIRE(got[0] == 1.0f);
    REQUIRE(got[1] == 0.0f);
    REQUIRE(got[2] == 0.0f);
    REQUIRE(got[3] == 10.0f);

    // Text-only input has no tensors: the engine reports the missing input.
    agent::MessageView text{};
    text.role = "user";
    text.content = "hello";
    req.messages = std::span<const agent::MessageView>(&text, 1);
    auto bad = agent::infer(session, req);
    REQUIRE_FALSE(bad.ok);
    REQUIRE(bad.error.message.find("missing required input") != std::string::npos);
}

TEST_CASE("onnx engine reports a bad model path through agent::init", "[onnx][engines]") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_ONNX_ENGINE_PATH;
    cfg.model = "/nonexistent/model.onnx";
    cfg.workspace_dir = kWorkspace;
    auto res = agent::init(cfg);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
}

#endif // AGENT_TEST_ONNX_ENGINE_PATH
