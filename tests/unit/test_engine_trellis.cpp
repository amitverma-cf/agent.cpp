#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#ifdef AGENT_TEST_TRELLIS_ENGINE_PATH

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

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

// JSON text of a path (backslashes doubled).
std::string js(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

void write_file(const std::string &path, const std::string &content) {
    std::ofstream f(path, std::ios::binary);
    REQUIRE(f.good());
    f << content;
}

std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool has_glb_magic(const std::string &path) { return read_file(path).rfind("glTF", 0) == 0; }

struct Workdir {
    fs::path root;
    Workdir() {
        root = fs::temp_directory_path() / ("agent_test_trellis_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root / "models");
        fs::create_directories(root / "out");
    }
    ~Workdir() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    std::string models() const { return (root / "models").string(); }
    std::string out() const { return (root / "out").string(); }
};

#ifdef AGENT_TEST_FAKE_TRELLIS_CLI
const std::string kFakeCli = AGENT_TEST_FAKE_TRELLIS_CLI;

void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

std::string config_with_fake(const Workdir &w, const std::string &options = "") {
    return R"({"model":")" + js(w.models()) + R"(","options":{"cli":")" + js(kFakeCli) + "\"" + (options.empty() ? "" : "," + options) + "}}";
}

// One request: a text part and an image part (a FILE, or BYTES when `bytes` is set).
struct Request {
    std::string prompt = "make this a 3D model";
    std::string image_path;
    std::string bytes = "fake image bytes";
    bool as_bytes = false;
    std::string output_dir;
    std::string params;
    volatile int32_t cancel = 0;
    agent_part parts[2] = {};
    agent_message msg{};
    agent_request req{.struct_size = sizeof(agent_request)};
    uint64_t progress_calls = 0;

    void build() {
        parts[0].kind = AGENT_PART_TEXT;
        parts[0].text = str(prompt);
        if (as_bytes) {
            parts[1].kind = AGENT_PART_BYTES;
            parts[1].data = bytes.data();
            parts[1].size = bytes.size();
        } else {
            parts[1].kind = AGENT_PART_FILE;
            parts[1].path = str(image_path);
        }
        parts[1].mime = agent_str{"image/png", 9};
        msg.role = agent_str{"user", 4};
        msg.parts = parts;
        msg.n_parts = 2;
        req.task = AGENT_CAP_MESH_GEN;
        req.messages = &msg;
        req.n_messages = 1;
        req.output_dir = str(output_dir);
        if (!params.empty()) req.params_json = str(params);
        req.on_progress = [](void *ud, uint64_t, uint64_t) { ++*static_cast<uint64_t *>(ud); };
        req.user_data = &progress_calls;
        req.cancel = &cancel;
    }
};
#endif

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

TEST_CASE("trellis engine rejects a bad configuration", "[engines][trellis]") {
    const agent_engine_v1 *engine = trellis_engine();
    REQUIRE(engine != nullptr);
    Workdir w;
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

    // Bad generation defaults are rejected before anything else.
    REQUIRE(create(engine, R"({"model":")" + js(w.models()) + R"(","options":{"resolution":2048}})", &inst, error) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(error.find("resolution") != std::string::npos);
    REQUIRE(create(engine, R"({"model":")" + js(w.models()) + R"(","options":{"background":"purple"}})", &inst, error) == AGENT_ERR_INVALID_CONFIG);
    REQUIRE(error.find("background") != std::string::npos);

    // A program that is not there.
    REQUIRE(create(engine, R"({"model":")" + js(w.models()) + R"(","options":{"cli":"/nonexistent/trellis-cli"}})", &inst, error) == AGENT_ERR_INIT_FAILED);
    REQUIRE(inst == nullptr);
    REQUIRE(error.find("trellis-cli") != std::string::npos);
}

TEST_CASE("trellis engine via AiProvider::Engine reports a missing model", "[engines][trellis]") {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_TRELLIS_ENGINE_PATH;
    cfg.model = "/nonexistent/trellis";
    Workdir w;
    cfg.workspace_dir = (w.root / "ws").string();
    auto res = agent::init(cfg);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::ModelLoadFailed);
}

#ifdef AGENT_TEST_FAKE_TRELLIS_CLI

TEST_CASE("trellis engine runs the program and returns its GLB", "[engines][trellis][process]") {
    const agent_engine_v1 *engine = trellis_engine();
    REQUIRE(engine != nullptr);
    Workdir w;
    void *inst = nullptr;
    std::string error;
    REQUIRE(create(engine, config_with_fake(w, R"("resolution":512,"seed":42,"threads":3)"), &inst, error) == AGENT_OK);
    REQUIRE(inst != nullptr);

    const std::string image = (w.root / "in.png").string();
    write_file(image, "png bytes");
    Request r;
    r.image_path = image;
    r.output_dir = w.out();
    r.build();

    SECTION("a file image becomes a GLB, with the settings on the command line") {
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_OK);
        INFO(engine->last_error(inst));
        REQUIRE(resp.message.n_parts == 2);
        const agent_part &file = resp.message.parts[0];
        REQUIRE(file.kind == AGENT_PART_FILE);
        REQUIRE(std::string(file.mime.data, file.mime.size) == "model/gltf-binary");
        const std::string path(file.path.data, file.path.size);
        REQUIRE(path.rfind(w.out(), 0) == 0);
        REQUIRE(path.substr(path.size() - 4) == ".glb");
        REQUIRE(has_glb_magic(path));
        REQUIRE(fs::file_size(path) > 10 * 1024);
        const std::string args = read_file(path + ".args");
        for (const char *want : {"--image\n", image.c_str(), "--models\n", "--res\n512\n", "--seed\n42\n", "--threads\n3\n", "--gpu\n0\n"})
            REQUIRE(args.find(want) != std::string::npos);
        REQUIRE(args.find("--no-texture") == std::string::npos);
        REQUIRE(r.progress_calls > 0);
        const agent_part &text = resp.message.parts[1];
        REQUIRE(std::string(text.text.data, text.text.size).find(path) != std::string::npos);
        engine->free_response(inst, &resp);
    }

    SECTION("params_json overrides the defaults for one call") {
        r.params = R"({"resolution":1536,"texture":false,"xatlas":false,"faces":50000,"guidance":5.5,"background":"birefnet"})";
        r.build();
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_OK);
        const std::string path(resp.message.parts[0].path.data, resp.message.parts[0].path.size);
        const std::string args = read_file(path + ".args");
        for (const char *want : {"--res\n1536\n", "--no-texture", "--box-uv", "--faces\n50000\n", "--gss\n5.5\n", "--bg-removal\nbirefnet\n"})
            REQUIRE(args.find(want) != std::string::npos);
        engine->free_response(inst, &resp);
    }

    SECTION("image bytes are written for the run and removed afterwards") {
        r.as_bytes = true;
        r.build();
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_OK);
        size_t files = 0, glbs = 0;
        for (const auto &e : fs::directory_iterator(w.out())) {
            ++files;
            if (e.path().extension() == ".glb") ++glbs;
        }
        REQUIRE(glbs == 1);
        REQUIRE(files == 2); // the GLB and the recorded arguments, no input copy
        engine->free_response(inst, &resp);
    }

    SECTION("a failing program is reported with what it printed") {
        set_env("FAKE_TRELLIS_FAIL", "1");
        agent_response resp{.struct_size = sizeof(agent_response)};
        const int32_t rc = engine->run(inst, &r.req, &resp);
        set_env("FAKE_TRELLIS_FAIL", nullptr);
        REQUIRE(rc == AGENT_ERR_DECODE_FAILED);
        const std::string why = engine->last_error(inst);
        REQUIRE(why.find("boom") != std::string::npos);
        REQUIRE(why.find("exit 3") != std::string::npos);
    }

    SECTION("cancelling while the program runs ends it") {
        set_env("FAKE_TRELLIS_SLEEP_MS", "20000");
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(600));
            r.cancel = 1;
        });
        const auto start = std::chrono::steady_clock::now();
        agent_response resp{.struct_size = sizeof(agent_response)};
        const int32_t rc = engine->run(inst, &r.req, &resp);
        canceller.join();
        set_env("FAKE_TRELLIS_SLEEP_MS", nullptr);
        REQUIRE(rc == AGENT_ERR_CANCELLED);
        REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::seconds(10));
        size_t glbs = 0;
        for (const auto &e : fs::directory_iterator(w.out()))
            if (e.path().extension() == ".glb") ++glbs;
        REQUIRE(glbs == 0);
    }

    SECTION("a pre-set cancel flag stops the run before anything starts") {
        r.cancel = 1;
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_ERR_CANCELLED);
    }

    SECTION("a request without an image fails cleanly") {
        r.msg.n_parts = 1; // text only
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_ERR_INVALID_CONFIG);
        REQUIRE(std::string(engine->last_error(inst)).find("image") != std::string::npos);
    }

    SECTION("an image file that is not there fails cleanly") {
        r.image_path = (w.root / "missing.png").string();
        r.build();
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_ERR_FILESYSTEM);
    }

    SECTION("bad params_json is rejected") {
        r.params = R"({"resolution":4096})";
        r.build();
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_ERR_INVALID_CONFIG);
        r.params = "[1]";
        r.build();
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_ERR_PARSE);
    }

    SECTION("non-mesh tasks are rejected") {
        r.req.task = AGENT_CAP_TEXT_GEN;
        agent_response resp{.struct_size = sizeof(agent_response)};
        REQUIRE(engine->run(inst, &r.req, &resp) == AGENT_ERR_UNSUPPORTED);
    }

    engine->destroy(inst);
}

#endif // AGENT_TEST_FAKE_TRELLIS_CLI

// Point AGENT_TEST_TRELLIS_MODEL_DIR at the TRELLIS.2 GGUF files (ilintar/trellis2-gguf) and AGENT_TEST_TRELLIS_CLI at a trellis-cli
// release program to run the real model. AGENT_TEST_TRELLIS_STEPS sets the flow steps (default 2, to keep the run short).
TEST_CASE("trellis engine generates a GLB with the real program and models", "[engines][trellis][model]") {
    const char *dir = std::getenv("AGENT_TEST_TRELLIS_MODEL_DIR");
    const char *cli = std::getenv("AGENT_TEST_TRELLIS_CLI");
    if (!dir || !cli) SKIP("AGENT_TEST_TRELLIS_MODEL_DIR and AGENT_TEST_TRELLIS_CLI not set");
    const agent_engine_v1 *engine = trellis_engine();
    REQUIRE(engine != nullptr);
    const char *steps_env = std::getenv("AGENT_TEST_TRELLIS_STEPS");
    const std::string steps = steps_env ? steps_env : "2";
    const std::string config = R"({"model":")" + js(dir) + R"(","options":{"cli":")" + js(cli) + R"(","resolution":512,"seed":42,"steps":)" + steps + "}}";
    void *inst = nullptr;
    std::string error;
    REQUIRE(create(engine, config, &inst, error) == AGENT_OK);

    Workdir w;
    const char *image_env = std::getenv("AGENT_TEST_TRELLIS_IMAGE");
    REQUIRE(image_env != nullptr);
    const std::string prompt = "make this a 3D model", image = image_env, out = w.out();
    agent_part parts[2] = {};
    parts[0].kind = AGENT_PART_TEXT;
    parts[0].text = str(prompt);
    parts[1].kind = AGENT_PART_FILE;
    parts[1].path = str(image);
    parts[1].mime = agent_str{"image/png", 9};
    agent_message msg{};
    msg.role = agent_str{"user", 4};
    msg.parts = parts;
    msg.n_parts = 2;
    agent_request req{.struct_size = sizeof(agent_request)};
    req.task = AGENT_CAP_MESH_GEN;
    req.messages = &msg;
    req.n_messages = 1;
    req.output_dir = str(out);
    agent_response resp{.struct_size = sizeof(agent_response)};
    REQUIRE(engine->run(inst, &req, &resp) == AGENT_OK);
    INFO(engine->last_error(inst));
    const std::string path(resp.message.parts[0].path.data, resp.message.parts[0].path.size);
    REQUIRE(has_glb_magic(path));
    REQUIRE(fs::file_size(path) > 10 * 1024);
    engine->free_response(inst, &resp);
    engine->destroy(inst);
}

#endif // AGENT_TEST_TRELLIS_ENGINE_PATH
