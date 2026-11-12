// RAG over a folder of documents: chunk + embed every .md / .txt file, retrieve the chunks closest to a
// question (optionally reranked) and let a chat model answer from them.
//
//   rag_docs --llama <agent-engine-llama.so> --embed <embedding.gguf> --chat <chat.gguf> --docs <folder>
//            [--rerank <reranker.gguf>] [--question TEXT] [--k 4] [--workspace DIR]
//
// Example: --embed bge-small-en-v1.5-q8_0.gguf --chat qwen2.5-0.5b-instruct-q4_k_m.gguf --docs docs/

#include <agent-cpp/agent.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string llama, embed, chat, rerank, docs;
    std::string question = "How do I run an engine out of process, and what happens when it crashes?";
    std::string workspace = ".workspace";
    int k = 4;
};

void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage: %s --llama <agent-engine-llama.so> --embed <embedding.gguf> --chat <chat.gguf> --docs <folder>\n"
                 "          [--rerank <reranker.gguf>] [--question TEXT] [--k N] [--workspace DIR]\n",
                 argv0);
}

bool parse(int argc, char **argv, Options &o) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string_view a = argv[i];
        if (a == "--k") {
            o.k = std::atoi(argv[i + 1]);
            continue;
        }
        std::string *dst = a == "--llama"       ? &o.llama
                           : a == "--embed"     ? &o.embed
                           : a == "--chat"      ? &o.chat
                           : a == "--rerank"    ? &o.rerank
                           : a == "--docs"      ? &o.docs
                           : a == "--question"  ? &o.question
                           : a == "--workspace" ? &o.workspace
                                                : nullptr;
        if (!dst) return false;
        *dst = argv[i + 1];
    }
    return argc % 2 == 1 && o.k > 0;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char **argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        usage(argv[0]);
        return 2;
    }
    std::string missing;
    const std::pair<const std::string *, const char *> required[] = {
        {&o.llama, "--llama"}, {&o.embed, "--embed"}, {&o.chat, "--chat"}, {&o.docs, "--docs"}};
    for (const auto &[value, flag] : required) {
        if (value->empty()) missing += std::string("  ") + flag + "\n";
        else if (!fs::exists(*value)) missing += std::string("  ") + flag + " " + *value + ": not found\n";
    }
    if (!missing.empty()) {
        std::fprintf(stderr, "rag_docs: not configured; missing:\n%s", missing.c_str());
        usage(argv[0]);
        return 2;
    }

    agent::AgentRuntime runtime;
    agent::Config base;
    base.provider = agent::AiProvider::Engine;
    base.engine = o.llama;
    base.workspace_dir = o.workspace;
    base.auto_default_tools = false;

    agent::Config embed_cfg = base;
    embed_cfg.model = o.embed;
    embed_cfg.context_window = 512;
    embed_cfg.engine_options = R"({"mode":"embed"})";
    auto embedder = agent::init(embed_cfg);
    if (!embedder.ok) {
        std::fprintf(stderr, "rag_docs: embedding model: %s\n", embedder.error.message.c_str());
        return 1;
    }

    agent::Result<agent::Session> reranker;
    if (!o.rerank.empty()) {
        agent::Config rerank_cfg = base;
        rerank_cfg.model = o.rerank;
        rerank_cfg.context_window = 512;
        rerank_cfg.engine_options = R"({"mode":"rerank"})";
        reranker = agent::init(rerank_cfg);
        if (!reranker.ok) {
            std::fprintf(stderr, "rag_docs: reranker: %s\n", reranker.error.message.c_str());
            return 1;
        }
    }

    // 1. Index: one document per file, doc_id = path relative to the folder. Chunks stay well under the
    //    embedding model's 512-token window.
    agent::RagIndex index; // in-memory; agent::store::sqlite_vector_store() persists it
    const auto t_index = std::chrono::steady_clock::now();
    int files = 0;
    for (const fs::directory_entry &entry : fs::recursive_directory_iterator(o.docs)) {
        const std::string ext = entry.path().extension().string();
        if (!entry.is_regular_file() || (ext != ".md" && ext != ".txt")) continue;
        std::ifstream in(entry.path(), std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        const std::string doc_id = fs::relative(entry.path(), o.docs).generic_string();
        auto r = agent::rag_index(embedder.value, index, doc_id, text.str(), {.size = 800, .overlap = 100});
        if (!r.ok) {
            std::fprintf(stderr, "rag_docs: indexing %s failed: %s\n", doc_id.c_str(), r.error.message.c_str());
            return 1;
        }
        ++files;
    }
    std::printf("indexed %d files in %.1f s\n", files, seconds_since(t_index));
    if (files == 0) {
        std::fprintf(stderr, "rag_docs: no .md or .txt files under %s\n", o.docs.c_str());
        return 1;
    }

    // 2. Retrieve (and rerank, when a reranker is given: 4*k candidates, reordered, top k kept).
    const auto t_query = std::chrono::steady_clock::now();
    auto hits = agent::rag_query(embedder.value, index, o.question, o.k, reranker.ok ? &reranker.value : nullptr);
    if (!hits.ok) {
        std::fprintf(stderr, "rag_docs: query failed: %s\n", hits.error.message.c_str());
        return 1;
    }
    std::printf("question: %s\nretrieved in %.2f s:\n", o.question.c_str(), seconds_since(t_query));
    std::string context;
    for (const agent::RagHit &h : hits.value) {
        std::printf("  %-40s score %.3f\n", h.id.c_str(), h.score);
        context += "[" + h.id + "]\n" + h.text + "\n\n";
    }

    // 3. Answer from the retrieved chunks. (An agent can also search on its own: make_rag_tool adds a
    //    "search_knowledge" tool; small models answer more reliably with the context injected.)
    agent::Config chat_cfg = base;
    chat_cfg.model = o.chat;
    chat_cfg.context_window = 4096;
    chat_cfg.max_tokens = 256;
    chat_cfg.temperature = 0.2f;
    auto chat = agent::init(chat_cfg);
    if (!chat.ok) {
        std::fprintf(stderr, "rag_docs: chat model: %s\n", chat.error.message.c_str());
        return 1;
    }
    agent::Conversation conversation;
    conversation.history.push_back({.role = "system",
                                    .content = "Answer the question using only the documentation excerpts below. Cite the "
                                               "excerpt ids in brackets. If they do not contain the answer, say so.\n\n" +
                                               context});
    const auto t_answer = std::chrono::steady_clock::now();
    auto answer = agent::run_turn(chat.value, conversation, o.question);
    if (!answer.ok) {
        std::fprintf(stderr, "rag_docs: answer failed: %s\n", answer.error.message.c_str());
        return 1;
    }
    std::printf("answer (%.1f s):\n%s\n", seconds_since(t_answer), answer.value.c_str());
    return 0;
}
