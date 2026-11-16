#pragma once

// The searchable memory of a project: every note and every message of the saved chats, in one SQLite file
// (<workspace>/memory/index.sqlite3) that is searched by words (full text, BM25) and by meaning (vectors) together. With an embedding
// model installed the vectors are real; without one the index still works by words alone.

#include <agent-cpp/agent.hpp>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// Embeds texts; an empty result means "no embedding model": the index then goes by words.
using EmbedFn = std::function<std::vector<std::vector<float>>(const std::vector<std::string> &texts)>;

struct MemoryHit {
    std::string id;   // "note:<key>" or "chat:<conversation id>:<message number>"
    std::string text; // the note or the message (cut to the indexed length)
    float score = 0.0f;
};

class MemoryIndex {
  public:
    // Opens (creates) the index of a workspace. `dimensions` is the size of the embedding model's vectors, 0 when there is none.
    static Result<MemoryIndex> open(const std::string &workspace, int dimensions);

    // Adds or replaces entries. Texts longer than 1500 bytes are cut.
    Result<void> put(const std::vector<std::pair<std::string, std::string>> &id_text, const EmbedFn &embed);
    Result<void> erase(std::string_view id);
    // Best first. Entries that share no word with the query are dropped when there are no real vectors.
    std::vector<MemoryHit> search(std::string_view query, int k, const EmbedFn &embed) const;
    // True when the id is already indexed with this exact text (so nothing has to be embedded again).
    bool has(std::string_view id) const;
    int dimensions() const { return dims_; }
    size_t size() const;

  private:
    std::shared_ptr<void> state_;
    std::shared_ptr<void> seen_; // ids indexed in this run, with a hash of their text
    const VectorStoreOps *ops_ = nullptr;
    int dims_ = 0;
};

// "chat:<id>:<n>" -> the conversation id; "" for other ids.
std::string memory_hit_chat(std::string_view id);

} // namespace agent::cli
