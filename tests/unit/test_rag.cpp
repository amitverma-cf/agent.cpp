// RAG: in-memory / sqlite-vec vector stores, chunker, embed / rerank plumbing through a fake
// in-process engine, the search_knowledge tool, and env-gated real-model tests:
//   AGENT_TEST_EMBED_MODEL   embedding GGUF, e.g. ggml-org/bge-small-en-v1.5-Q8_0-GGUF
//   AGENT_TEST_RERANK_MODEL  reranker GGUF, e.g. gpustack/jina-reranker-v1-tiny-en-GGUF

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <agent-cpp/store_sqlite.hpp>
#include <algorithm>
#include <catch_amalgamated.hpp>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

const char *kWorkspace = "/tmp/agent_test_rag_ws";

float dot(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// ------------------------------------------------------------------ fake embed / rerank engine
// Embedding: hashed bag of lowercase words (64 dims). Rerank: number of distinct query words found in
// the document.

constexpr size_t kFakeDims = 64;

std::vector<std::string> words(std::string_view text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!cur.empty()) out.push_back(std::move(cur)), cur.clear();
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

size_t word_bucket(const std::string &w) {
    uint32_t h = 2166136261u;
    for (char c : w) h = (h ^ static_cast<uint8_t>(c)) * 16777619u;
    return h % kFakeDims;
}

struct FakeInstance {
    std::vector<float> floats;
    int64_t shape = 0;
    std::vector<agent_part> parts;
    std::string error;
    int runs = 0;
};

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

int32_t fake_create(agent_str, void **out, char *, size_t) {
    *out = new FakeInstance();
    return AGENT_OK;
}
void fake_destroy(void *inst) { delete static_cast<FakeInstance *>(inst); }

int32_t fake_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<FakeInstance *>(instance);
    ++inst->runs;
    std::vector<std::string_view> texts;
    for (size_t i = 0; i < req->n_messages; ++i)
        for (size_t p = 0; p < req->messages[i].n_parts; ++p)
            if (req->messages[i].parts[p].kind == AGENT_PART_TEXT) texts.push_back(sv(req->messages[i].parts[p].text));

    inst->floats.clear();
    size_t rows = 0;
    const char *name = nullptr;
    if (req->task == AGENT_CAP_EMBED) {
        for (std::string_view t : texts) {
            std::vector<float> v(kFakeDims, 0.0f);
            for (const auto &w : words(t)) v[word_bucket(w)] += 1.0f;
            v[kFakeDims - 1] += 0.01f; // never the zero vector
            inst->floats.insert(inst->floats.end(), v.begin(), v.end());
        }
        inst->shape = static_cast<int64_t>(kFakeDims);
        rows = texts.size();
        name = "embedding";
    } else if (req->task == AGENT_CAP_RERANK) {
        if (texts.size() < 2) {
            inst->error = "fake: need a query and documents";
            return AGENT_ERR_INVALID_CONFIG;
        }
        const auto q = words(texts[0]);
        for (size_t d = 1; d < texts.size(); ++d) {
            float score = 0.0f;
            const auto dw = words(texts[d]);
            for (const auto &w : q) score += std::find(dw.begin(), dw.end(), w) != dw.end() ? 1.0f : 0.0f;
            inst->floats.push_back(score);
        }
        inst->shape = static_cast<int64_t>(texts.size() - 1);
        rows = 1;
        name = "scores";
    } else {
        inst->error = "fake: unsupported task";
        return AGENT_ERR_UNSUPPORTED;
    }
    inst->parts.assign(rows, agent_part{});
    for (size_t r = 0; r < rows; ++r) {
        agent_part &p = inst->parts[r];
        p.kind = AGENT_PART_TENSOR;
        p.name = agent_str{name, std::char_traits<char>::length(name)};
        p.dtype = AGENT_DTYPE_F32;
        p.shape = &inst->shape;
        p.ndim = 1;
        p.data = inst->floats.data() + r * static_cast<size_t>(inst->shape);
        p.size = static_cast<size_t>(inst->shape) * sizeof(float);
    }
    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = inst->parts.data();
    resp->message.n_parts = inst->parts.size();
    return AGENT_OK;
}

void fake_free_response(void *, agent_response *resp) { agent_response_reset(resp); }
const char *fake_last_error(void *inst) { return static_cast<FakeInstance *>(inst)->error.c_str(); }

const agent_engine_v1 kFakeEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "rag-fake",
    "1.0",
    AGENT_CAP_EMBED | AGENT_CAP_RERANK,
    0,
    fake_create,
    fake_destroy,
    fake_run,
    fake_free_response,
    fake_last_error,
    nullptr,
    nullptr,
};

agent::Session fake_session() {
    REQUIRE(agent::register_engine(&kFakeEngine).ok);
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = "rag-fake";
    cfg.model = "fake";
    cfg.workspace_dir = kWorkspace;
    auto s = agent::init(cfg);
    REQUIRE(s.ok);
    return std::move(s.value);
}

const char *kDocs[][2] = {
    {"paris", "Paris is the capital of France. The Eiffel Tower stands in Paris and draws millions of visitors."},
    {"plants", "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar and oxygen."},
    {"rust", "Rust is a systems programming language focused on memory safety without a garbage collector."},
    {"reef", "The Great Barrier Reef off Australia is the largest coral reef system in the world."},
};

} // namespace

// ------------------------------------------------------------------ in-memory vector store

TEST_CASE("in-memory vector store: upsert, top-k, replace, erase, clear", "[rag]") {
    const agent::VectorStoreOps *ops = agent::in_memory_vector_store();
    REQUIRE_FALSE(ops->open(agent::VectorStoreConfig{}).ok); // dimensions required
    auto opened = ops->open(agent::VectorStoreConfig{.dimensions = 3});
    REQUIRE(opened.ok);
    void *st = opened.value.get();

    REQUIRE(ops->upsert(st, "x", "x axis", std::vector<float>{2.0f, 0.0f, 0.0f}, "{}").ok);
    REQUIRE(ops->upsert(st, "y", "y axis", std::vector<float>{0.0f, 1.0f, 0.0f}, "{}").ok);
    REQUIRE(ops->upsert(st, "z", "z axis", std::vector<float>{0.0f, 0.0f, 1.0f}, R"({"k":1})").ok);
    REQUIRE_FALSE(ops->upsert(st, "bad", "wrong dims", std::vector<float>{1.0f, 0.0f}, "{}").ok);
    REQUIRE_FALSE(ops->upsert(st, "nan", "nan", std::vector<float>{NAN, 0.0f, 0.0f}, "{}").ok);

    auto hits = ops->query_topk(st, std::vector<float>{0.9f, 0.1f, 0.0f}, 2);
    REQUIRE(hits.ok);
    REQUIRE(hits.value.size() == 2);
    REQUIRE(hits.value[0].id == "x");
    REQUIRE(hits.value[1].id == "y");
    REQUIRE(hits.value[0].score == Catch::Approx(0.9f / std::sqrt(0.82f)).epsilon(1e-4)); // cosine, not raw dot
    REQUIRE(hits.value[0].score > hits.value[1].score);

    // Replace "x" so it now points along z: the z query sees both z entries.
    REQUIRE(ops->upsert(st, "x", "now z", std::vector<float>{0.0f, 0.0f, 5.0f}, "{}").ok);
    hits = ops->query_topk(st, std::vector<float>{0.0f, 0.0f, 1.0f}, 10);
    REQUIRE(hits.ok);
    REQUIRE(hits.value.size() == 3);
    REQUIRE(hits.value[0].score == Catch::Approx(1.0f));
    REQUIRE(hits.value[1].score == Catch::Approx(1.0f));
    REQUIRE((hits.value[0].id == "x" || hits.value[0].id == "z"));

    auto erased = ops->erase(st, "x");
    REQUIRE(erased.ok);
    REQUIRE(erased.value);
    erased = ops->erase(st, "x");
    REQUIRE(erased.ok);
    REQUIRE_FALSE(erased.value);
    hits = ops->query_topk(st, std::vector<float>{0.0f, 0.0f, 1.0f}, 10);
    REQUIRE(hits.value.size() == 2);
    REQUIRE(hits.value[0].id == "z");
    REQUIRE(hits.value[0].text == "z axis");
    REQUIRE(hits.value[0].metadata_json == R"({"k":1})");

    REQUIRE(ops->clear(st).ok);
    hits = ops->query_topk(st, std::vector<float>{0.0f, 0.0f, 1.0f}, 10);
    REQUIRE(hits.ok);
    REQUIRE(hits.value.empty());
}

// ------------------------------------------------------------------ chunker

TEST_CASE("chunk_text keeps short text whole and validates options", "[rag]") {
    auto one = agent::chunk_text("  hello world \n");
    REQUIRE(one.ok);
    REQUIRE(one.value.size() == 1);
    REQUIRE(one.value[0] == "hello world");
    REQUIRE(agent::chunk_text("   \n\n ").value.empty());
    REQUIRE_FALSE(agent::chunk_text("abc", {.size = 0, .overlap = 0}).ok);
    REQUIRE_FALSE(agent::chunk_text("abc", {.size = 10, .overlap = 10}).ok);
}

TEST_CASE("chunk_text prefers paragraph breaks and respects the size bound", "[rag]") {
    const std::string text = "First paragraph is here.\n\nSecond paragraph follows it.\n\nThird one ends the text.";
    auto res = agent::chunk_text(text, {.size = 40, .overlap = 0});
    REQUIRE(res.ok);
    REQUIRE(res.value.size() == 3);
    REQUIRE(res.value[0] == "First paragraph is here.");
    REQUIRE(res.value[1] == "Second paragraph follows it.");
    REQUIRE(res.value[2] == "Third one ends the text.");
    for (auto c : res.value) REQUIRE(c.data() >= text.data()); // views into the input
}

TEST_CASE("chunk_text splits long runs at word boundaries with overlap", "[rag]") {
    std::string text;
    for (int i = 0; i < 200; ++i) text += "word" + std::to_string(i) + " ";
    auto res = agent::chunk_text(text, {.size = 100, .overlap = 30});
    REQUIRE(res.ok);
    REQUIRE(res.value.size() > 5);
    for (size_t i = 0; i < res.value.size(); ++i) {
        const auto c = res.value[i];
        REQUIRE(c.size() <= 100);
        REQUIRE(c.substr(0, 4) == "word"); // starts at a word
        const char last = c.back();
        REQUIRE(std::isdigit(static_cast<unsigned char>(last))); // ends at a word
        if (i > 0) {
            // Overlap: this chunk starts before the previous one ended.
            REQUIRE(c.data() < res.value[i - 1].data() + res.value[i - 1].size());
            REQUIRE(c.data() > res.value[i - 1].data());
        }
    }
    REQUIRE(res.value.back().ends_with("word199"));
}

TEST_CASE("chunk_text never splits a UTF-8 sequence", "[rag]") {
    std::string text;
    for (int i = 0; i < 100; ++i) text += "\xC3\xA9"; // "é" x100, no spaces
    auto res = agent::chunk_text(text, {.size = 15, .overlap = 0});
    REQUIRE(res.ok);
    size_t total = 0;
    for (auto c : res.value) {
        REQUIRE(c.size() <= 15);
        REQUIRE(c.size() % 2 == 0);
        REQUIRE((static_cast<unsigned char>(c[0]) & 0xC0) != 0x80);
        total += c.size();
    }
    REQUIRE(total == text.size());
}

// ------------------------------------------------------------------ plumbing (fake engine)

TEST_CASE("embed and rerank need an engine with the capability", "[rag]") {
    agent::Config cfg;
    cfg.workspace_dir = kWorkspace;
    auto mock = agent::init(cfg); // AiProvider::Mock
    REQUIRE(mock.ok);
    const std::string_view texts[] = {"hello"};
    auto e = agent::embed(mock.value, texts);
    REQUIRE_FALSE(e.ok);
    REQUIRE(e.error.code == agent::ErrorCode::UnsupportedProvider);

    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_MOCK_ENGINE_PATH; // TEXT_GEN only
    cfg.model = "mock.bin";
    auto text_only = agent::init(cfg);
    REQUIRE(text_only.ok);
    e = agent::embed(text_only.value, texts);
    REQUIRE_FALSE(e.ok);
    REQUIRE(e.error.code == agent::ErrorCode::UnsupportedProvider);
    auto r = agent::rerank(text_only.value, "q", texts);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.error.code == agent::ErrorCode::UnsupportedProvider);
}

TEST_CASE("embed returns one unit vector per text across batches", "[rag]") {
    agent::Session session = fake_session();
    std::vector<std::string> owned;
    for (int i = 0; i < 40; ++i) owned.push_back("text number " + std::to_string(i));
    std::vector<std::string_view> texts(owned.begin(), owned.end());

    auto res = agent::embed(session, texts);
    REQUIRE(res.ok);
    REQUIRE(res.value.size() == 40);
    for (const auto &v : res.value) {
        REQUIRE(v.size() == kFakeDims);
        REQUIRE(dot(v, v) == Catch::Approx(1.0f).epsilon(1e-5));
    }
    REQUIRE(dot(res.value[3], res.value[3]) > dot(res.value[3], res.value[4]) - 1e-6f);

    const std::string_view empty[] = {"ok", ""};
    REQUIRE_FALSE(agent::embed(session, empty).ok);
    REQUIRE(agent::embed(session, std::span<const std::string_view>{}).value.empty());
}

TEST_CASE("rerank returns one score per document in order", "[rag]") {
    agent::Session session = fake_session();
    std::vector<std::string> owned;
    for (int i = 0; i < 20; ++i) owned.push_back(i == 17 ? "the red apple" : "nothing here " + std::to_string(i));
    owned.push_back("");
    std::vector<std::string_view> docs(owned.begin(), owned.end());
    auto res = agent::rerank(session, "red apple", docs);
    REQUIRE(res.ok);
    REQUIRE(res.value.size() == docs.size());
    REQUIRE(res.value[17] == Catch::Approx(2.0f));
    REQUIRE(res.value[0] == Catch::Approx(0.0f));
    REQUIRE(res.value.back() == Catch::Approx(0.0f));
    REQUIRE_FALSE(agent::rerank(session, "", docs).ok);
}

TEST_CASE("rag_index and rag_query find the matching chunk; reindex replaces chunks", "[rag]") {
    agent::Session embedder = fake_session();
    agent::RagIndex index;
    for (const auto &d : kDocs) REQUIRE(agent::rag_index(embedder, index, d[0], d[1]).ok);
    REQUIRE(index.config.dimensions == static_cast<int>(kFakeDims));

    auto hits = agent::rag_query(embedder, index, "what does photosynthesis turn sunlight into", 2);
    REQUIRE(hits.ok);
    REQUIRE(hits.value.size() == 2);
    REQUIRE(hits.value[0].doc_id == "plants");
    REQUIRE(hits.value[0].id == "plants#0");
    REQUIRE(hits.value[0].metadata_json == R"({"doc_id":"plants","chunk":0,"offset":0})");
    REQUIRE(hits.value[0].score == hits.value[0].similarity);

    // Re-index "plants" as several small chunks, then as one: stale chunks disappear.
    const std::string long_text = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota.";
    REQUIRE(agent::rag_index(embedder, index, "plants", long_text, {.size = 20, .overlap = 0}).ok);
    hits = agent::rag_query(embedder, index, "theta iota", 10);
    REQUIRE(hits.ok);
    size_t plants_chunks = 0;
    for (const auto &h : hits.value) plants_chunks += h.doc_id == "plants";
    REQUIRE(plants_chunks == 3);
    REQUIRE(hits.value[0].id == "plants#2");
    REQUIRE(agent::rag_index(embedder, index, "plants", "Just one chunk now.").ok);
    hits = agent::rag_query(embedder, index, "theta iota", 10);
    plants_chunks = 0;
    for (const auto &h : hits.value) plants_chunks += h.doc_id == "plants";
    REQUIRE(plants_chunks == 1);

    REQUIRE(agent::rag_erase(index, "plants").ok);
    hits = agent::rag_query(embedder, index, "just one chunk now", 10);
    for (const auto &h : hits.value) REQUIRE(h.doc_id != "plants");
    REQUIRE(hits.value.size() == 3);
}

TEST_CASE("rag_query reorders candidates with a rerank session", "[rag]") {
    agent::Session embedder = fake_session();
    agent::Session reranker = fake_session();
    agent::RagIndex index;
    // The two "capital" docs share many words with the query; the reranker's word-overlap score
    // decides between them independently of the vector order.
    REQUIRE(agent::rag_index(embedder, index, "a", "capital capital capital city").ok);
    REQUIRE(agent::rag_index(embedder, index, "b", "France capital is Paris and many tourists visit every year").ok);
    REQUIRE(agent::rag_index(embedder, index, "c", "bananas are yellow").ok);

    auto plain = agent::rag_query(embedder, index, "capital of France", 1);
    REQUIRE(plain.ok);
    REQUIRE(plain.value[0].doc_id == "a"); // vector similarity alone prefers the repeated word
    auto reranked = agent::rag_query(embedder, index, "capital of France", 1, &reranker);
    REQUIRE(reranked.ok);
    REQUIRE(reranked.value.size() == 1);
    REQUIRE(reranked.value[0].doc_id == "b");
    REQUIRE(reranked.value[0].score == Catch::Approx(2.0f)); // "capital" + "france"
}

TEST_CASE("search_knowledge tool runs rag_query and returns JSON hits", "[rag]") {
    agent::Session embedder = fake_session();
    agent::RagIndex index;
    for (const auto &d : kDocs) REQUIRE(agent::rag_index(embedder, index, d[0], d[1]).ok);

    agent::Config cfg;
    cfg.workspace_dir = kWorkspace;
    auto chat = agent::init(cfg);
    REQUIRE(chat.ok);
    agent::Session &session = chat.value;
    session.config.tools.push_back(agent::make_rag_tool(session, embedder, index));
    agent::rebuild_tool_index(session);
    REQUIRE(session.tools_system_prompt.find("search_knowledge") != std::string::npos);

    auto out = agent::execute_tool(session, "search_knowledge", R"({"query":"largest coral reef","k":1})");
    REQUIRE(out.ok);
    REQUIRE(out.value.starts_with(R"([{"id":"reef#0","score":)"));
    REQUIRE(out.value.find("Great Barrier Reef") != std::string::npos);
    REQUIRE(out.value.find("rust#0") == std::string::npos);

    REQUIRE_FALSE(agent::execute_tool(session, "search_knowledge", R"({"k":1})").ok);
}

// ------------------------------------------------------------------ sqlite-vec vector store

TEST_CASE("sqlite-vec vector store round trip persists across reopen", "[rag][sqlite_vec]") {
    const std::string db_path = std::string(kWorkspace) + "/rag_store.sqlite3";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);

    const agent::VectorStoreOps *ops = agent::store::sqlite_vector_store();
    REQUIRE_FALSE(ops->open(agent::VectorStoreConfig{.path = db_path, .name = "bad name", .dimensions = 3}).ok);
    {
        auto opened = ops->open(agent::VectorStoreConfig{.path = db_path, .name = "kb", .dimensions = 3});
        REQUIRE(opened.ok);
        void *st = opened.value.get();
        REQUIRE(ops->upsert(st, "x", "x axis", std::vector<float>{1.0f, 0.0f, 0.0f}, R"({"a":1})").ok);
        REQUIRE(ops->upsert(st, "y", "y axis", std::vector<float>{0.0f, 2.0f, 0.0f}, "{}").ok);
        REQUIRE(ops->upsert(st, "z", "z axis", std::vector<float>{0.0f, 0.0f, 1.0f}, "{}").ok);
        REQUIRE(ops->upsert(st, "y", "y axis v2", std::vector<float>{0.0f, 1.0f, 0.1f}, "{}").ok); // replace
        REQUIRE_FALSE(ops->upsert(st, "w", "wrong", std::vector<float>{1.0f}, "{}").ok);

        auto hits = ops->query_topk(st, std::vector<float>{0.0f, 1.0f, 0.0f}, 2);
        REQUIRE(hits.ok);
        REQUIRE(hits.value.size() == 2);
        REQUIRE(hits.value[0].id == "y");
        REQUIRE(hits.value[0].text == "y axis v2");
        REQUIRE(hits.value[0].score > 0.99f);
        REQUIRE(hits.value[0].score > hits.value[1].score);
        REQUIRE(hits.value[1].id == "z"); // y v2 leans towards z

        auto erased = ops->erase(st, "z");
        REQUIRE(erased.ok);
        REQUIRE(erased.value);
        REQUIRE_FALSE(ops->erase(st, "z").value);
    }
    {
        auto reopened = ops->open(agent::VectorStoreConfig{.path = db_path, .name = "kb", .dimensions = 3});
        REQUIRE(reopened.ok);
        void *st = reopened.value.get();
        auto hits = ops->query_topk(st, std::vector<float>{1.0f, 0.0f, 0.0f}, 10);
        REQUIRE(hits.ok);
        REQUIRE(hits.value.size() == 2);
        REQUIRE(hits.value[0].id == "x");
        REQUIRE(hits.value[0].metadata_json == R"({"a":1})");
        REQUIRE(hits.value[0].score == Catch::Approx(1.0f).margin(1e-5));
        REQUIRE(ops->clear(st).ok);
        REQUIRE(ops->query_topk(st, std::vector<float>{1.0f, 0.0f, 0.0f}, 10).value.empty());
    }
    std::filesystem::remove(db_path, ec);
}

TEST_CASE("RagIndex over sqlite-vec works with the pipeline", "[rag][sqlite_vec]") {
    agent::Session embedder = fake_session();
    agent::RagIndex index{.store = agent::store::sqlite_vector_store(), .config = {.path = ":memory:", .name = "docs"}};
    for (const auto &d : kDocs) REQUIRE(agent::rag_index(embedder, index, d[0], d[1]).ok);
    auto hits = agent::rag_query(embedder, index, "rust memory safety garbage collector", 1);
    REQUIRE(hits.ok);
    REQUIRE(hits.value.size() == 1);
    REQUIRE(hits.value[0].doc_id == "rust");
}

// ------------------------------------------------------------------ real models

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
namespace {

agent::Config llama_task_config(const char *model, const char *mode) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = AGENT_TEST_LLAMA_ENGINE_PATH;
    cfg.model = model;
    cfg.workspace_dir = kWorkspace;
    cfg.context_window = 512;
    cfg.engine_options = std::string(R"({"mode":")") + mode + "\"}";
    return cfg;
}

const char *embed_model() { return std::getenv("AGENT_TEST_EMBED_MODEL"); }
const char *rerank_model() { return std::getenv("AGENT_TEST_RERANK_MODEL"); }

const char *kParagraphs = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by "
                          "Gustave Eiffel's company and completed in 1889 for the World's Fair.\n\n"
                          "Photosynthesis is the process plants use to convert light energy into chemical energy. Chlorophyll in the "
                          "leaves absorbs sunlight, and carbon dioxide and water are turned into glucose and oxygen.\n\n"
                          "Rust is a general-purpose programming language that emphasizes performance, type safety and concurrency. "
                          "Its borrow checker enforces memory safety without a garbage collector.\n\n"
                          "The Great Barrier Reef is the world's largest coral reef system, located in the Coral Sea off the coast of "
                          "Queensland, Australia. It can be seen from outer space.\n\n"
                          "Sourdough bread is leavened with a starter culture of wild yeast and lactic acid bacteria, which gives the "
                          "loaf its tangy flavour and chewy crumb.";

} // namespace

TEST_CASE("llama rejects an unknown mode and tasks that don't match the mode", "[rag][llama][model]") {
    if (!embed_model()) SKIP("AGENT_TEST_EMBED_MODEL not set");
    REQUIRE_FALSE(agent::init(llama_task_config(embed_model(), "bogus")).ok);

    auto s = agent::init(llama_task_config(embed_model(), "embed"));
    REQUIRE(s.ok);
    const std::string_view docs[] = {"a document"};
    auto r = agent::rerank(s.value, "query", docs);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.error.message.find("mode") != std::string::npos);
    agent::FlowMemory memory;
    REQUIRE_FALSE(agent::run_turn(s.value, memory, "hello").ok);

    // A model without a classification head can't load in rerank mode.
    REQUIRE_FALSE(agent::init(llama_task_config(embed_model(), "rerank")).ok);
}

TEST_CASE("llama embeddings: similar sentences score higher than unrelated ones", "[rag][llama][model]") {
    if (!embed_model()) SKIP("AGENT_TEST_EMBED_MODEL not set");
    auto s = agent::init(llama_task_config(embed_model(), "embed"));
    REQUIRE(s.ok);
    const std::string_view texts[] = {
        "The cat is sleeping on the sofa.",
        "A kitten naps on the couch.",
        "Quarterly tax filings are due at the end of April.",
        "The stock market fell sharply after the interest rate announcement.",
    };
    auto e = agent::embed(s.value, texts);
    REQUIRE(e.ok);
    REQUIRE(e.value.size() == 4);
    const size_t dims = e.value[0].size();
    REQUIRE(dims >= 64);
    for (const auto &v : e.value) {
        REQUIRE(v.size() == dims);
        REQUIRE(dot(v, v) == Catch::Approx(1.0f).epsilon(1e-3));
    }
    const float cat_kitten = dot(e.value[0], e.value[1]);
    const float cat_tax = dot(e.value[0], e.value[2]);
    const float cat_stocks = dot(e.value[0], e.value[3]);
    INFO("cat/kitten " << cat_kitten << " cat/tax " << cat_tax << " cat/stocks " << cat_stocks);
    REQUIRE(cat_kitten > cat_tax + 0.1f);
    REQUIRE(cat_kitten > cat_stocks + 0.1f);

    // Same text, same vector; batching (one call with many texts) doesn't change results.
    const std::string_view one[] = {texts[1]};
    auto single = agent::embed(s.value, one);
    REQUIRE(single.ok);
    REQUIRE(dot(single.value[0], e.value[1]) == Catch::Approx(1.0f).epsilon(1e-3));
}

TEST_CASE("llama rag_query returns the right chunk for a question", "[rag][llama][model]") {
    if (!embed_model()) SKIP("AGENT_TEST_EMBED_MODEL not set");
    auto s = agent::init(llama_task_config(embed_model(), "embed"));
    REQUIRE(s.ok);
    agent::RagIndex index;
    REQUIRE(agent::rag_index(s.value, index, "notes", kParagraphs, {.size = 300, .overlap = 0}).ok);

    struct Case {
        const char *question;
        const char *needle;
    } cases[] = {
        {"How do plants turn sunlight into energy?", "Photosynthesis"},
        {"Which programming language has a borrow checker?", "Rust"},
        {"When was the Eiffel Tower finished?", "Eiffel"},
        {"What makes sourdough taste sour?", "Sourdough"},
    };
    for (const auto &c : cases) {
        auto hits = agent::rag_query(s.value, index, c.question, 3);
        REQUIRE(hits.ok);
        REQUIRE(hits.value.size() == 3);
        INFO(c.question << " -> " << hits.value[0].text);
        REQUIRE(hits.value[0].text.find(c.needle) != std::string::npos);
        REQUIRE(hits.value[0].doc_id == "notes");
    }
}

TEST_CASE("llama reranker orders documents by relevance", "[rag][llama][model]") {
    if (!rerank_model()) SKIP("AGENT_TEST_RERANK_MODEL not set");
    auto s = agent::init(llama_task_config(rerank_model(), "rerank"));
    REQUIRE(s.ok);
    const std::string_view docs[] = {
        "Bananas are a good source of potassium.",
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
    };
    auto scores = agent::rerank(s.value, "What is the capital of France?", docs);
    REQUIRE(scores.ok);
    REQUIRE(scores.value.size() == 3);
    INFO("scores " << scores.value[0] << " " << scores.value[1] << " " << scores.value[2]);
    REQUIRE(scores.value[1] > scores.value[2]);
    REQUIRE(scores.value[2] > scores.value[0]);

    const std::string_view texts[] = {"x"};
    REQUIRE_FALSE(agent::embed(s.value, texts).ok); // rerank instance only reranks
}

TEST_CASE("llama rag_query with a reranker picks the best passage", "[rag][llama][model]") {
    if (!embed_model() || !rerank_model()) SKIP("AGENT_TEST_EMBED_MODEL / AGENT_TEST_RERANK_MODEL not set");
    auto e = agent::init(llama_task_config(embed_model(), "embed"));
    auto r = agent::init(llama_task_config(rerank_model(), "rerank"));
    REQUIRE(e.ok);
    REQUIRE(r.ok);
    agent::RagIndex index;
    REQUIRE(agent::rag_index(e.value, index, "notes", kParagraphs, {.size = 300, .overlap = 0}).ok);
    auto hits = agent::rag_query(e.value, index, "Where is the largest coral reef?", 2, &r.value);
    REQUIRE(hits.ok);
    REQUIRE(hits.value.size() == 2);
    INFO(hits.value[0].text);
    REQUIRE(hits.value[0].text.find("Great Barrier Reef") != std::string::npos);
    REQUIRE(hits.value[0].score > hits.value[1].score);
}

TEST_CASE("llama embeddings round trip through the sqlite-vec store", "[rag][llama][model][sqlite_vec]") {
    if (!embed_model()) SKIP("AGENT_TEST_EMBED_MODEL not set");
    const std::string db_path = std::string(kWorkspace) + "/rag_llama.sqlite3";
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    auto s = agent::init(llama_task_config(embed_model(), "embed"));
    REQUIRE(s.ok);
    {
        agent::RagIndex index{.store = agent::store::sqlite_vector_store(), .config = {.path = db_path, .name = "notes"}};
        REQUIRE(agent::rag_index(s.value, index, "notes", kParagraphs, {.size = 300, .overlap = 0}).ok);
    }
    agent::RagIndex reopened{.store = agent::store::sqlite_vector_store(), .config = {.path = db_path, .name = "notes"}};
    auto hits = agent::rag_query(s.value, reopened, "Which language avoids a garbage collector?", 1);
    REQUIRE(hits.ok);
    REQUIRE(hits.value.size() == 1);
    REQUIRE(hits.value[0].text.find("Rust") != std::string::npos);
    // The sqlite store answers with a hybrid (full-text + vector) query, so similarity is the fused rank score (about 1/60 per list),
    // not a cosine: only its being positive is meaningful here.
    REQUIRE(hits.value[0].similarity > 0.0f);
    std::filesystem::remove(db_path, ec);
}

TEST_CASE("search_knowledge tool with a real embedder", "[rag][llama][model]") {
    if (!embed_model()) SKIP("AGENT_TEST_EMBED_MODEL not set");
    auto e = agent::init(llama_task_config(embed_model(), "embed"));
    REQUIRE(e.ok);
    agent::RagIndex index;
    REQUIRE(agent::rag_index(e.value, index, "notes", kParagraphs, {.size = 300, .overlap = 0}).ok);
    agent::Config cfg;
    cfg.workspace_dir = kWorkspace;
    auto chat = agent::init(cfg);
    REQUIRE(chat.ok);
    chat.value.config.tools.push_back(agent::make_rag_tool(chat.value, e.value, index, nullptr, 1));
    agent::rebuild_tool_index(chat.value);
    auto out = agent::execute_tool(chat.value, "search_knowledge", R"({"query":"what is chlorophyll for?"})");
    REQUIRE(out.ok);
    INFO(out.value);
    REQUIRE(out.value.find("Photosynthesis") != std::string::npos);
}
#endif
