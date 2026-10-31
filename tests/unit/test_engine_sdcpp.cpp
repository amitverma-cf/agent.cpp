#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#ifdef AGENT_TEST_SDCPP_ENGINE_PATH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace {

const char *kSdOutputDir = "/tmp/agent_test_sdcpp_out";

const agent_engine_v1 *sdcpp_engine() {
    auto res = agent::load_engine_plugin(AGENT_TEST_SDCPP_ENGINE_PATH);
    if (!res.ok) return nullptr;
    return agent::find_engine("sdcpp");
}

agent_str str(const std::string &s) { return agent_str{s.data(), s.size()}; }

int32_t create(const agent_engine_v1 *engine, const std::string &config, void **inst, std::string &error) {
    char err[512] = {0};
    int32_t rc = engine->create(str(config), inst, err, sizeof(err));
    error = err;
    return rc;
}

bool has_png_signature(const std::string &path) {
    static const unsigned char kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    unsigned char buf[8] = {0};
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const size_t n = std::fread(buf, 1, sizeof(buf), f);
    std::fclose(f);
    return n == sizeof(buf) && std::memcmp(buf, kSig, sizeof(buf)) == 0;
}

struct ProgressLog {
    uint64_t calls = 0;
    uint64_t last_total = 0;
};

// Set AGENT_TEST_SDCPP_MODEL to a small single-file SD model to run the real-model test, e.g. the
// one-step SDXS-512 GGUF (683 MB):
// https://huggingface.co/concedo/sdxs-512-tinySDdistilled-GGUF/resolve/main/sdxs-512-tinySDdistilled_Q8_0.gguf
const char *sdcpp_test_model() { return std::getenv("AGENT_TEST_SDCPP_MODEL"); }

} // namespace

TEST_CASE("sdcpp engine plugin loads and declares image generation", "[engines][sdcpp]") {
    const agent_engine_v1 *engine = sdcpp_engine();
    REQUIRE(engine != nullptr);
    REQUIRE(std::string(engine->name) == "sdcpp");
    REQUIRE(engine->abi_version == AGENT_ENGINE_ABI_VERSION);
    REQUIRE((engine->caps & AGENT_CAP_IMAGE_GEN) != 0);
    REQUIRE((engine->caps & AGENT_CAP_TEXT_GEN) == 0);
    REQUIRE(engine->flags == 0);
}

TEST_CASE("sdcpp engine fails cleanly without a usable model", "[engines][sdcpp]") {
    const agent_engine_v1 *engine = sdcpp_engine();
    REQUIRE(engine != nullptr);
    void *inst = nullptr;
    std::string error;

    REQUIRE(create(engine, R"({"options":{}})", &inst, error) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(inst == nullptr);
    REQUIRE(error.find("model") != std::string::npos);

    REQUIRE(create(engine, R"({"model":"/nonexistent/sd.gguf"})", &inst, error) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);
    REQUIRE(error.find("/nonexistent/sd.gguf") != std::string::npos);

    REQUIRE(create(engine, "not json", &inst, error) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(inst == nullptr);

    const std::string junk = "/tmp/agent_test_sdcpp_junk.gguf";
    std::FILE *f = std::fopen(junk.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fputs("this is not a model", f);
    std::fclose(f);
    REQUIRE(create(engine, R"({"model":")" + junk + "\"}", &inst, error) == AGENT_ERR_MODEL_LOAD_FAILED);
    REQUIRE(inst == nullptr);
    std::filesystem::remove(junk);
}

TEST_CASE("sdcpp engine via AiProvider::Engine reports a missing model", "[engines][sdcpp]") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_SDCPP_ENGINE_PATH;
    cfg.model = "/nonexistent/sd.gguf";
    cfg.workspace_dir = "/tmp/agent_test_engines_ws";
    auto res = agent::init(cfg);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
}

TEST_CASE("sdcpp engine generates a PNG with a real model", "[engines][sdcpp][model]") {
    if (!sdcpp_test_model()) SKIP("AGENT_TEST_SDCPP_MODEL not set");
    const agent_engine_v1 *engine = sdcpp_engine();
    REQUIRE(engine != nullptr);

    const std::string config = std::string(R"({"model":")") + sdcpp_test_model() +
                               R"(","options":{"threads":2,"width":64,"height":64,"steps":1,"cfg_scale":1.0,"seed":42}})";
    void *inst = nullptr;
    std::string error;
    REQUIRE(create(engine, config, &inst, error) == AGENT_OK);
    REQUIRE(inst != nullptr);

    std::filesystem::remove_all(kSdOutputDir);
    std::filesystem::create_directories(kSdOutputDir);

    const std::string prompt = "a red apple on a wooden table";
    agent_part text_part{};
    text_part.kind = AGENT_PART_TEXT;
    text_part.text = str(prompt);
    agent_message msg{};
    msg.role = agent_str{"user", 4};
    msg.parts = &text_part;
    msg.n_parts = 1;

    const std::string out_dir = kSdOutputDir;
    const std::string params = R"({"width":128,"height":128})";
    ProgressLog progress;
    volatile int32_t cancel = 0;
    agent_request req{};
    req.task = AGENT_CAP_IMAGE_GEN;
    req.messages = &msg;
    req.n_messages = 1;
    req.params_json = str(params);
    req.output_dir = str(out_dir);
    req.on_progress = [](void *ud, uint64_t, uint64_t total) {
        auto *p = static_cast<ProgressLog *>(ud);
        ++p->calls;
        p->last_total = total;
    };
    req.user_data = &progress;
    req.cancel = &cancel;

    SECTION("txt2img writes a PNG and reports progress") {
        agent_response resp{};
        const int32_t rc = engine->run(inst, &req, &resp);
        INFO(engine->last_error(inst));
        REQUIRE(rc == AGENT_OK);
        REQUIRE(resp.message.n_parts == 2);
        const agent_part &file = resp.message.parts[0];
        REQUIRE(file.kind == AGENT_PART_FILE);
        REQUIRE(std::string(file.mime.data, file.mime.size) == "image/png");
        const std::string path(file.path.data, file.path.size);
        REQUIRE(path.rfind(out_dir, 0) == 0);
        REQUIRE(std::filesystem::file_size(path) > 8);
        REQUIRE(has_png_signature(path));
        const agent_part &text = resp.message.parts[1];
        REQUIRE(text.kind == AGENT_PART_TEXT);
        REQUIRE(std::string(text.text.data, text.text.size) == "generated image: " + path);
        REQUIRE(progress.calls > 0);
        engine->free_response(inst, &resp);

        SECTION("img2img uses the generated PNG as the init image") {
            agent_part parts[2] = {text_part, agent_part{}};
            parts[1].kind = AGENT_PART_FILE;
            parts[1].path = str(path);
            parts[1].mime = agent_str{"image/png", 9};
            msg.parts = parts;
            msg.n_parts = 2;
            req.task = AGENT_CAP_IMAGE_EDIT;
            const std::string edit_params = R"({"strength":0.5,"steps":2})";
            req.params_json = str(edit_params);
            agent_response edit{};
            const int32_t erc = engine->run(inst, &req, &edit);
            INFO(engine->last_error(inst));
            REQUIRE(erc == AGENT_OK);
            const std::string edited(edit.message.parts[0].path.data, edit.message.parts[0].path.size);
            REQUIRE(edited != path);
            REQUIRE(has_png_signature(edited));
            // img2img keeps the input image's 128x128 size, not the 64x64 create() default.
            std::FILE *f = std::fopen(edited.c_str(), "rb");
            REQUIRE(f != nullptr);
            unsigned char ihdr[24] = {0};
            const size_t got = std::fread(ihdr, 1, sizeof(ihdr), f);
            std::fclose(f);
            REQUIRE(got == sizeof(ihdr));
            REQUIRE((ihdr[16] << 24 | ihdr[17] << 16 | ihdr[18] << 8 | ihdr[19]) == 128);
            REQUIRE((ihdr[20] << 24 | ihdr[21] << 16 | ihdr[22] << 8 | ihdr[23]) == 128);
            engine->free_response(inst, &edit);
        }
    }

    SECTION("a pre-set cancel flag stops the run") {
        cancel = 1;
        agent_response resp{};
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_CANCELLED);
    }

    SECTION("setting cancel from on_progress stops sampling mid-run") {
        const std::string slow = R"({"steps":4})";
        req.params_json = str(slow);
        struct CancelCtx {
            volatile int32_t *flag;
            uint64_t calls;
        } ctx{&cancel, 0};
        req.on_progress = [](void *ud, uint64_t step, uint64_t) {
            auto *c = static_cast<CancelCtx *>(ud);
            ++c->calls;
            if (step >= 1) *c->flag = 1;
        };
        req.user_data = &ctx;
        agent_response resp{};
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_CANCELLED);
        REQUIRE(ctx.calls > 0);
    }

    SECTION("non-image tasks are rejected") {
        req.task = AGENT_CAP_TEXT_GEN;
        agent_response resp{};
        REQUIRE(engine->run(inst, &req, &resp) == AGENT_ERR_UNSUPPORTED);
    }

    engine->destroy(inst);
}

#endif // AGENT_TEST_SDCPP_ENGINE_PATH
