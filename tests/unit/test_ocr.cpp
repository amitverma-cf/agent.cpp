// agent::ocr() and the llama engine's AGENT_CAP_OCR: plumbing against a fake engine, then real
// OCR-capable VLMs on an image with text rendered by a tiny built-in 5x7 bitmap font.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <catch_amalgamated.hpp>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

const char *kWorkspace = "/tmp/agent_test_ocr_ws";

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

// ------------------------------------------------------------------ text image

struct Glyph {
    char c;
    const char *rows[7];
};
// 5x7 glyphs, '#' = ink. Only the characters the tests draw.
constexpr Glyph kFont[] = {
    {'A', {".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"}},
    {'C', {".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."}},
    {'D', {"####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."}},
    {'E', {"#####", "#....", "#....", "####.", "#....", "#....", "#####"}},
    {'G', {".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"}},
    {'H', {"#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"}},
    {'I', {".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."}},
    {'L', {"#....", "#....", "#....", "#....", "#....", "#....", "#####"}},
    {'N', {"#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"}},
    {'O', {".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."}},
    {'P', {"####.", "#...#", "#...#", "####.", "#....", "#....", "#...."}},
    {'R', {"####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"}},
    {'S', {".####", "#....", "#....", ".###.", "....#", "....#", "####."}},
    {'T', {"#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."}},
    {'W', {"#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"}},
    {'0', {".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."}},
    {'2', {".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"}},
    {'6', {"..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."}},
    {' ', {".....", ".....", ".....", ".....", ".....", ".....", "....."}},
};

const Glyph *glyph(char c) {
    for (const Glyph &g : kFont)
        if (g.c == c) return &g;
    return nullptr;
}

// Black text on white, one line per entry, as a 24-bit BMP (scale = pixels per font dot).
std::string text_bmp(const std::vector<std::string> &lines, int scale) {
    size_t cols = 0;
    for (const auto &l : lines) cols = std::max(cols, l.size());
    const int margin = 4 * scale, cell_w = 6 * scale, cell_h = 10 * scale;
    const int w = 2 * margin + static_cast<int>(cols) * cell_w, h = 2 * margin + static_cast<int>(lines.size()) * cell_h;
    std::vector<unsigned char> px(static_cast<size_t>(w * h), 255);
    for (size_t li = 0; li < lines.size(); ++li)
        for (size_t ci = 0; ci < lines[li].size(); ++ci) {
            const Glyph *g = glyph(lines[li][ci]);
            REQUIRE(g != nullptr);
            for (int r = 0; r < 7; ++r)
                for (int c = 0; c < 5; ++c) {
                    if (g->rows[r][c] != '#') continue;
                    for (int dy = 0; dy < scale; ++dy)
                        for (int dx = 0; dx < scale; ++dx) {
                            const int x = margin + static_cast<int>(ci) * cell_w + c * scale + dx;
                            const int y = margin + static_cast<int>(li) * cell_h + r * scale + dy;
                            px[static_cast<size_t>(y * w + x)] = 0;
                        }
                }
        }
    const int row = (w * 3 + 3) & ~3;
    const uint32_t data = static_cast<uint32_t>(row * h), file = 54 + data;
    std::string bmp(54, '\0');
    auto put32 = [&](size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) bmp[at + static_cast<size_t>(i)] = static_cast<char>(v >> (8 * i));
    };
    bmp[0] = 'B';
    bmp[1] = 'M';
    put32(2, file);
    put32(10, 54);
    put32(14, 40);
    put32(18, static_cast<uint32_t>(w));
    put32(22, static_cast<uint32_t>(h));
    bmp[26] = 1;
    bmp[28] = 24;
    put32(34, data);
    for (int y = h - 1; y >= 0; --y) { // bottom-up
        std::string line(static_cast<size_t>(row), '\0');
        for (int x = 0; x < w; ++x) line[static_cast<size_t>(x * 3)] = line[static_cast<size_t>(x * 3 + 1)] =
            line[static_cast<size_t>(x * 3 + 2)] = static_cast<char>(px[static_cast<size_t>(y * w + x)]);
        bmp += line;
    }
    return bmp;
}

std::string write_text_image(const std::vector<std::string> &lines, const char *path, int scale = 4) {
    const std::string bmp = text_bmp(lines, scale);
    FILE *f = std::fopen(path, "wb");
    REQUIRE(f != nullptr);
    REQUIRE(std::fwrite(bmp.data(), 1, bmp.size(), f) == bmp.size());
    std::fclose(f);
    return path;
}

std::string upper(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

// ------------------------------------------------------------------ fake OCR engine

struct FakeOcr {
    std::string params;
    std::vector<std::string> paths;
    std::string reply;
    agent_part part{};
};
FakeOcr *g_ocr = nullptr;

int32_t fake_create(agent_str, void **out, char *, size_t) {
    g_ocr = new FakeOcr();
    *out = g_ocr;
    return AGENT_OK;
}
void fake_destroy(void *inst) {
    if (g_ocr == inst) g_ocr = nullptr;
    delete static_cast<FakeOcr *>(inst);
}
int32_t fake_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *f = static_cast<FakeOcr *>(instance);
    if (req->task != AGENT_CAP_OCR) return AGENT_ERR_UNSUPPORTED;
    f->params = std::string(sv(req->params_json));
    f->paths.clear();
    for (size_t i = 0; i < req->n_messages; ++i)
        for (size_t p = 0; p < req->messages[i].n_parts; ++p)
            if (req->messages[i].parts[p].kind == AGENT_PART_FILE) f->paths.emplace_back(sv(req->messages[i].parts[p].path));
    f->reply = "TEXT FROM " + std::to_string(f->paths.size()) + " IMAGE";
    f->part = agent_part{};
    f->part.kind = AGENT_PART_TEXT;
    f->part.text = agent_str{f->reply.data(), f->reply.size()};
    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = &f->part;
    resp->message.n_parts = 1;
    return AGENT_OK;
}
void fake_free_response(void *, agent_response *resp) { agent_response_reset(resp); }
const char *fake_last_error(void *) { return ""; }

// clang-format off
const agent_engine_v1 kFakeOcrEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "ocr-fake",
    "1.0",
    AGENT_CAP_OCR,
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

} // namespace

TEST_CASE("ocr sends the image and the prompt override to an OCR engine", "[ocr]") {
    REQUIRE(agent::register_engine(&kFakeOcrEngine).ok);
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = "ocr-fake";
    cfg.model = "fake";
    cfg.workspace_dir = kWorkspace;
    auto s = agent::init(cfg);
    REQUIRE(s.ok);

    auto text = agent::ocr(s.value, agent::Media{"/tmp/page.png", "image/png"});
    REQUIRE(text.ok);
    CHECK(text.value == "TEXT FROM 1 IMAGE");
    REQUIRE(g_ocr->paths.size() == 1);
    CHECK(g_ocr->paths[0] == "/tmp/page.png");
    CHECK(g_ocr->params.find("\"prompt\"") == std::string::npos);

    REQUIRE(agent::ocr(s.value, agent::Media{"/tmp/t.png", "image/png"}, "Table \"Recognition\":").ok);
    CHECK(g_ocr->params.find(R"("prompt":"Table \"Recognition\":")") != std::string::npos);

    CHECK(agent::ocr(s.value, agent::Media{}).error.code == agent::ErrorCode::InvalidConfig);

    agent::Config mock;
    mock.workspace_dir = kWorkspace;
    auto m = agent::init(mock);
    REQUIRE(m.ok);
    CHECK(agent::ocr(m.value, agent::Media{"/tmp/page.png", "image/png"}).error.code == agent::ErrorCode::UnsupportedProvider);
}

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
namespace {

agent::Config llama_ocr_config(const char *model, const char *mmproj) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_LLAMA_ENGINE_PATH;
    cfg.model = model;
    cfg.workspace_dir = kWorkspace;
    cfg.context_window = 4096;
    cfg.max_tokens = 64;
    cfg.temperature = 0.0f;
    cfg.auto_default_tools = false;
    cfg.engine_options = std::string(R"({"mmproj":")") + mmproj + "\"}";
    return cfg;
}

} // namespace

// Set AGENT_TEST_LLAMA_OCR_MODEL / AGENT_TEST_LLAMA_OCR_MMPROJ to an OCR VLM, e.g. GLM-OCR
// (https://huggingface.co/ggml-org/GLM-OCR-GGUF: GLM-OCR-Q8_0.gguf + mmproj-GLM-OCR-Q8_0.gguf, MIT).
TEST_CASE("llama OCR reads rendered text with an OCR model", "[ocr][llama][model]") {
    const char *model = std::getenv("AGENT_TEST_LLAMA_OCR_MODEL");
    const char *mmproj = std::getenv("AGENT_TEST_LLAMA_OCR_MMPROJ");
    if (!model || !mmproj) SKIP("AGENT_TEST_LLAMA_OCR_MODEL / AGENT_TEST_LLAMA_OCR_MMPROJ not set");
    auto s = agent::init(llama_ocr_config(model, mmproj));
    REQUIRE(s.ok);
    const std::string path = write_text_image({"HELLO WORLD", "OCR TEST 2026"}, "/tmp/agent_test_ocr_text.bmp");
    auto text = agent::ocr(s.value, agent::Media{path, "image/bmp"});
    INFO(text.error.message);
    REQUIRE(text.ok);
    INFO("ocr: " << text.value);
    const std::string up = upper(text.value);
    CHECK(up.find("HELLO") != std::string::npos);
    CHECK(up.find("WORLD") != std::string::npos);
    CHECK(up.find("2026") != std::string::npos);
}

// The generic prompt on a general VLM (SmolVLM-256M): a weak OCR, so only the plumbing and one easy
// word are checked.
TEST_CASE("llama OCR with a general VLM and the generic prompt", "[ocr][llama][model]") {
    const char *model = std::getenv("AGENT_TEST_LLAMA_VISION_MODEL");
    const char *mmproj = std::getenv("AGENT_TEST_LLAMA_VISION_MMPROJ");
    if (!model || !mmproj) SKIP("AGENT_TEST_LLAMA_VISION_MODEL / AGENT_TEST_LLAMA_VISION_MMPROJ not set");
    auto s = agent::init(llama_ocr_config(model, mmproj));
    REQUIRE(s.ok);
    const std::string path = write_text_image({"HELLO"}, "/tmp/agent_test_ocr_hello.bmp", 8);
    auto text = agent::ocr(s.value, agent::Media{path, "image/bmp"});
    INFO(text.error.message);
    REQUIRE(text.ok);
    INFO("ocr: " << text.value);
    REQUIRE_FALSE(text.value.empty());
    CHECK(upper(text.value).find("HELLO") != std::string::npos);

    // A text-only request is rejected: OCR needs an image.
    agent::MessageView msg{.role = "user", .content = "no image here"};
    agent::InferRequest req;
    req.messages = std::span<const agent::MessageView>(&msg, 1);
    auto r = agent::run_task(s.value, AGENT_CAP_OCR, req);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.message.find("image") != std::string::npos);
}
#endif
