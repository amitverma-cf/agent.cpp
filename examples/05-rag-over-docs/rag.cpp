// Answer a question from a folder of documents: embed and index the files, retrieve the best chunks, let a chat model answer from them.
// Run it from the repository root (it indexes docs/). Needs models/bge-small-en-v1.5-q8_0.gguf and models/qwen2.5-0.5b-instruct-q4_k_m.gguf.

#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

int main() {
    agent::AgentRuntime runtime;

    // One session embeds, another answers. The embedding model needs the engine's "embed" mode.
    agent::Config embed_config;
    embed_config.context_window = 512;
    embed_config.engine_options = R"({"mode":"embed"})";
    auto embedder = agent::open_session(agent::engines::llama, "models/bge-small-en-v1.5-q8_0.gguf", embed_config);
    if (!embedder.ok) return std::fprintf(stderr, "embedding model: %s\n", embedder.error.message.c_str()), 1;

    agent::Config chat_config;
    chat_config.context_window = 4096;
    chat_config.temperature = 0.2f;
    auto chat = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", chat_config);
    if (!chat.ok) return std::fprintf(stderr, "chat model: %s\n", chat.error.message.c_str()), 1;

    // 1. Index every .md file under docs/ (in memory; a SQLite vector store keeps it between runs).
    agent::RagIndex index;
    int files = 0;
    for (const auto &entry : fs::recursive_directory_iterator("docs")) {
        if (!entry.is_regular_file() || entry.path().extension() != ".md") continue;
        std::ifstream in(entry.path(), std::ios::binary);
        std::stringstream text;
        text << in.rdbuf();
        auto r = agent::rag_index(embedder.value, index, entry.path().generic_string(), text.str(), {.size = 800, .overlap = 100});
        if (!r.ok) return std::fprintf(stderr, "indexing: %s\n", r.error.message.c_str()), 1;
        ++files;
    }
    std::printf("indexed %d files\n", files);

    // 2. Retrieve the four best chunks for the question.
    const std::string question = "What happens when an engine crashes?";
    auto hits = agent::rag_query(embedder.value, index, question, 4);
    if (!hits.ok) return std::fprintf(stderr, "query: %s\n", hits.error.message.c_str()), 1;
    std::string context;
    for (const agent::RagHit &hit : hits.value) {
        std::printf("  %-40s score %.3f\n", hit.id.c_str(), hit.score);
        context += "[" + hit.id + "]\n" + hit.text + "\n\n";
    }

    // 3. Answer from them.
    agent::Conversation conversation;
    conversation.history.push_back(
        {.role = "system", .content = "Answer using only these excerpts and cite them in brackets:\n\n" + context});
    auto answer = agent::run_turn(chat.value, conversation, question);
    if (!answer.ok) return std::fprintf(stderr, "answer: %s\n", answer.error.message.c_str()), 1;
    std::printf("\n%s\n", answer.value.c_str());
    return 0;
}
