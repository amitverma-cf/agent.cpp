#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#ifdef AGENT_TEST_TRELLIS_ENGINE_PATH

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

const char *kTrellisOutputDir = "/tmp/agent_test_trellis_out";

const agent_engine_v1 *trellis_engine() {
    auto res = agent::load_engine_plugin(AGENT_TEST_TRELLIS_ENGINE_PATH);
    if (!res.ok) return nullptr;
    return agent::find_engine("trellis");
}

agent_str str(const std::string &s) { return agent_str{s.data(), s.size()}; }

int32_t create(const agent_engine_v1 *engine, const std::string &config, void **inst, std::string &error) {
    char err[512] = {0};
    int32_t rc = engine->create(str(config), inst, err, sizeof(err));
    error = err;
    return rc;
}

void write_file(const std::string &path, const std::string &content) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fwrite(content.data(), 1, content.size(), f);
    std::fclose(f);
}

void put_u16(std::string &s, uint32_t v) {
    s.push_back(static_cast<char>(v & 0xff));
    s.push_back(static_cast<char>((v >> 8) & 0xff));
}
void put_u32(std::string &s, uint32_t v) {
    put_u16(s, v & 0xffff);
    put_u16(s, v >> 16);
}

// A size x size 24-bit BMP: a shaded orange sphere on a white background (trellis2.cpp's automatic
// background mode turns the border-connected white into alpha).
std::string sphere_bmp(int size) {
    const uint32_t row = static_cast<uint32_t>((size * 3 + 3) & ~3);
    const uint32_t pixels = row * static_cast<uint32_t>(size);
    std::string s = "BM";
    put_u32(s, 54 + pixels);
    put_u32(s, 0);
    put_u32(s, 54);
    put_u32(s, 40);
    put_u32(s, static_cast<uint32_t>(size));
    put_u32(s, static_cast<uint32_t>(size));
    put_u16(s, 1);
    put_u16(s, 24);
    put_u32(s, 0);
    put_u32(s, pixels);
    put_u32(s, 2835);
    put_u32(s, 2835);
    put_u32(s, 0);
    put_u32(s, 0);
    const float c = (static_cast<float>(size) - 1.0f) * 0.5f, r = static_cast<float>(size) * 0.35f;
    for (int y = 0; y < size; ++y) {
        std::string line;
        for (int x = 0; x < size; ++x) {
            const float dx = (static_cast<float>(x) - c) / r, dy = (static_cast<float>(y) - c) / r;
            const float d2 = dx * dx + dy * dy;
            unsigned char rgb[3] = {255, 255, 255};
            if (d2 <= 1.0f) {
                // Lambert shading with the light at the upper left (BMP rows run bottom-up).
                const float nz = std::sqrt(1.0f - d2);
                const float l = std::fmax(0.15f, (-dx * 0.4f + dy * 0.4f + nz * 0.82f));
                rgb[0] = static_cast<unsigned char>(230.0f * l);
                rgb[1] = static_cast<unsigned char>(120.0f * l);
                rgb[2] = static_cast<unsigned char>(30.0f * l);
            }
            line.push_back(static_cast<char>(rgb[2])); // BGR
            line.push_back(static_cast<char>(rgb[1]));
            line.push_back(static_cast<char>(rgb[0]));
        }
        line.resize(row, '\0');
        s += line;
    }
    return s;
}

bool has_glb_magic(const std::string &path) {
    unsigned char buf[4] = {0};
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const size_t n = std::fread(buf, 1, sizeof(buf), f);
    std::fclose(f);
    return n == 4 && std::memcmp(buf, "glTF", 4) == 0;
}

// Point AGENT_TEST_TRELLIS_MODEL_DIR at a directory with the TRELLIS.2 GGUFs to run the real-model
// test. The coarse path needs only dino_f16.gguf (LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF),
// ss_flow_f16.gguf (LocalAI-io/TRELLIS.2-4B-GGUF) and ss_dec_f16.gguf
// (LocalAI-io/TRELLIS-image-large-GGUF), ~3.4 GB in total. AGENT_TEST_TRELLIS_STEPS sets the flow
// steps (default 2, to keep the CPU run short).
const char *trellis_model_dir() { return std::getenv("AGENT_TEST_TRELLIS_MODEL_DIR"); }

} // namespace

TEST_CASE("trellis engine plugin loads and declares mesh generation", "[engines][trellis]") {
    const agent_engine_v1 *engine = trellis_engine();
    REQUIRE(engine != nullptr);
    REQUIRE(std::string(engine->name) == "trellis");
    REQUIRE(engine->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE(engine->caps == AGENT_CAP_MESH_GEN);
    REQUIRE(engine->flags == 0);
    REQUIRE(engine->count_tokens == nullptr);
}

TEST_CASE("trellis engine fails cleanly without usable model files", "[engines][trellis]") {
    const agent_engine_v1 *engine = trellis_engine();
    REQUIRE(engine != nullptr);
    void *inst = nullptr;
    std::string error;

    REQUIRE(create(engine, R"({"options":{}})", &inst, error) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(inst == nullptr);
    REQUIRE(error.find("model") != std::string::npos);

    REQUIRE(create(engine, "not json", &inst, error) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(inst == nullptr);

    REQUIRE(create(engine, R"({"model":"/nonexistent/trellis"})", &inst, error) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);
    REQUIRE(error.find("/nonexistent/trellis") != std::string::npos);

    const std::filesystem::path dir = "/tmp/agent_test_trellis_models";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    // An empty directory: the first required file is reported.
    REQUIRE(create(engine, R"({"model":")" + dir.string() + "\"}", &inst, error) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);
    REQUIRE(error.find("dino_f16.gguf") != std::string::npos);

    // An explicit option pointing at a missing file.
    REQUIRE(create(engine, R"({"model":")" + dir.string() + R"(","options":{"dino":"missing.gguf"}})", &inst, error) ==
            AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(error.find("missing.gguf") != std::string::npos);

    // Bad generation defaults are rejected before anything is loaded.
    REQUIRE(create(engine, R"({"model":")" + dir.string() + R"(","options":{"pipeline":"2048"}})", &inst, error) ==
            AGENT_ERR_INVALID_CONFIG);
    REQUIRE(error.find("pipeline") != std::string::npos);

    // Files that exist but are not GGUFs: trellis2.cpp's loader must fail, not crash.
    for (const char *name : {"dino_f16.gguf", "ss_flow_f16.gguf", "ss_dec_f16.gguf"}) write_file((dir / name).string(), "this is not a gguf");
    REQUIRE(create(engine, R"({"model":")" + dir.string() + "\"}", &inst, error) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);
    REQUIRE(error.find("dino") != std::string::npos);
    std::filesystem::remove_all(dir);
}

TEST_CASE("trellis engine via AiProvider::Engine reports a missing model", "[engines][trellis]") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_TRELLIS_ENGINE_PATH;
    cfg.model = "/nonexistent/trellis";
    cfg.workspace_dir = "/tmp/agent_test_engines_ws";
    auto res = agent::init(cfg);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
}

TEST_CASE("trellis engine generates a GLB with real models", "[engines][trellis][model]") {
    if (!trellis_model_dir()) SKIP("AGENT_TEST_TRELLIS_MODEL_DIR not set");
    const agent_engine_v1 *engine = trellis_engine();
    REQUIRE(engine != nullptr);

    const char *steps_env = std::getenv("AGENT_TEST_TRELLIS_STEPS");
    const std::string steps = steps_env ? steps_env : "2";
    const std::string config = std::string(R"({"model":")") + trellis_model_dir() +
                               R"(","options":{"threads":4,"pipeline":"coarse","seed":42,"steps":)" + steps + "}}";
    void *inst = nullptr;
    std::string error;
    REQUIRE(create(engine, config, &inst, error) == AGENT_OK);
    REQUIRE(inst != nullptr);

    std::filesystem::remove_all(kTrellisOutputDir);
    std::filesystem::create_directories(kTrellisOutputDir);
    const std::string image_path = std::string(kTrellisOutputDir) + "/sphere.bmp";
    write_file(image_path, sphere_bmp(256));

    const std::string prompt = "make this a 3D model";
    agent_part parts[2] = {};
    parts[0].kind = AGENT_PART_TEXT;
    parts[0].text = str(prompt);
    parts[1].kind = AGENT_PART_FILE;
    parts[1].path = str(image_path);
    parts[1].mime = agent_str{"image/bmp", 9};
    agent_message msg{};
    msg.role = agent_str{"user", 4};
    msg.parts = parts;
    msg.n_parts = 2;

    const std::string out_dir = kTrellisOutputDir;
    struct ProgressLog {
        uint64_t calls = 0;
    } progress;
    volatile int32_t cancel = 0;
    agent_request req{};
    req.task = AGENT_CAP_MESH_GEN;
    req.messages = &msg;
    req.n_messages = 1;
    req.output_dir = str(out_dir);
    req.on_progress = [](void *ud, uint64_t, uint64_t) { ++static_cast<ProgressLog *>(ud)->calls; };
    req.user_data = &progress;
    req.cancel = &cancel;

    SECTION("image to GLB") {
        agent_response resp{};
        const int32_t rc = engine->run(inst, &req, &resp);
        INFO(engine->last_error(inst));
        REQUIRE(rc == AGENT_OK);
        REQUIRE(resp.message.n_parts == 2);
        const agent_part &file = resp.message.parts[0];
        REQUIRE(file.kind == AGENT_PART_FILE);
        REQUIRE(std::string(file.mime.data, file.mime.size) == "model/gltf-binary");
        const std::string path(file.path.data, file.path.size);
        REQUIRE(path.rfind(out_dir + "/trellis-", 0) == 0);
        REQUIRE(path.size() > 4);
        REQUIRE(path.substr(path.size() - 4) == ".glb");
        REQUIRE(has_glb_magic(path));
        REQUIRE(std::filesystem::file_size(path) > 10 * 1024);
        const agent_part &text = resp.message.parts[1];
        REQUIRE(text.kind == AGENT_PART_TEXT);
        REQUIRE(std::string(text.text.data, text.text.size).find(path) != std::string::npos);
        REQUIRE(progress.calls > 0);
        engine->free_response(inst, &resp);
    }

    SECTION("a request without an image fails cleanly") {
        msg.n_parts = 1; // text only
        agent_response resp{};
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_INVALID_CONFIG);
        REQUIRE(std::string(engine->last_error(inst)).find("image") != std::string::npos);
    }

    SECTION("bad params_json is rejected") {
        const std::string params = R"({"pipeline":"huge"})";
        req.params_json = str(params);
        agent_response resp{};
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_INVALID_CONFIG);
    }

    SECTION("a pre-set cancel flag stops the run") {
        cancel = 1;
        agent_response resp{};
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_CANCELLED);
    }

    SECTION("non-mesh tasks are rejected") {
        req.task = AGENT_CAP_TEXT_GEN;
        agent_response resp{};
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_UNSUPPORTED);
    }

    engine->destroy(inst);
}

#endif // AGENT_TEST_TRELLIS_ENGINE_PATH
