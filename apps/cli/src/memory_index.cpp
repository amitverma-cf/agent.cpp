#include "memory_index.hpp"

#include "chat_memory.hpp"

#include <agent-cpp/store_sqlite.hpp>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace agent::cli {

namespace {

constexpr size_t kMaxText = 1500;
constexpr int kWordDims = 4; // the vector size used when there is no embedding model (every entry gets the same vector)

// What is indexed already (id and a hash of its text), kept next to the database so a new run does not embed everything again.
struct Seen {
    std::mutex mu;
    std::unordered_map<std::string, size_t> hash_by_id;
    std::string file;

    void load() {
        std::ifstream in(file);
        std::string line;
        while (std::getline(in, line)) {
            const size_t tab = line.rfind('\t');
            if (tab == std::string::npos) continue;
            const size_t h = static_cast<size_t>(std::strtoull(line.c_str() + tab + 1, nullptr, 10));
            if (h == 0) hash_by_id.erase(line.substr(0, tab));
            else hash_by_id[line.substr(0, tab)] = h;
        }
    }
    void note(const std::string &id, size_t h) { // caller holds mu
        if (h == 0) hash_by_id.erase(id);
        else hash_by_id[id] = h;
        std::ofstream out(file, std::ios::app);
        out << id << '\t' << h << '\n';
    }
};

size_t text_hash(const std::string &t) {
    const size_t h = std::hash<std::string>{}(t);
    return h == 0 ? 1 : h;
}

std::string lower_text(std::string_view s) {
    std::string r(s);
    for (char &c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

std::string cut(std::string_view text) {
    std::string t(text.substr(0, kMaxText));
    while (!t.empty() && (static_cast<unsigned char>(t.back()) & 0xC0) == 0x80) t.pop_back(); // not in the middle of a UTF-8 character
    return t;
}

} // namespace

std::string memory_hit_chat(std::string_view id) {
    if (id.substr(0, 5) != "chat:") return {};
    const size_t last = id.rfind(':');
    return last <= 5 ? std::string() : std::string(id.substr(5, last - 5));
}

Result<MemoryIndex> MemoryIndex::open(const std::string &workspace, int dimensions) {
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(workspace) / "memory";
    std::filesystem::create_directories(dir, ec);
    MemoryIndex index;
    index.ops_ = agent::store::sqlite_vector_store();
    index.dims_ = dimensions > 0 ? dimensions : 0;
    VectorStoreConfig cfg;
    cfg.path = (dir / "index.sqlite3").string();
    // One table per vector size: a different embedding model (or none) gets its own table, filled again from the saved chats.
    cfg.name = index.dims_ > 0 ? "mem_d" + std::to_string(index.dims_) : "mem_words";
    cfg.dimensions = index.dims_ > 0 ? index.dims_ : kWordDims;
    auto state = index.ops_->open(cfg);
    if (!state.ok) return fail<MemoryIndex>(state.error.code, state.error.message);
    index.state_ = std::move(state.value);
    auto seen = std::make_shared<Seen>();
    seen->file = (dir / ("index." + cfg.name + ".ids")).string();
    seen->load();
    index.seen_ = std::move(seen);
    return ok(std::move(index));
}

bool MemoryIndex::has(std::string_view id) const {
    auto &seen = *static_cast<Seen *>(seen_.get());
    std::lock_guard<std::mutex> lock(seen.mu);
    return seen.hash_by_id.count(std::string(id)) != 0;
}

size_t MemoryIndex::size() const {
    auto &seen = *static_cast<Seen *>(seen_.get());
    std::lock_guard<std::mutex> lock(seen.mu);
    return seen.hash_by_id.size();
}

Result<void> MemoryIndex::put(const std::vector<std::pair<std::string, std::string>> &id_text, const EmbedFn &embed) {
    auto &seen = *static_cast<Seen *>(seen_.get());
    std::vector<std::string> ids, texts;
    for (const auto &[id, text] : id_text) {
        if (id.empty() || text.empty()) continue;
        std::string t = cut(text);
        const size_t h = text_hash(t);
        {
            std::lock_guard<std::mutex> lock(seen.mu);
            const auto it = seen.hash_by_id.find(id);
            if (it != seen.hash_by_id.end() && it->second == h) continue; // unchanged
        }
        ids.push_back(id);
        texts.push_back(std::move(t));
    }
    if (ids.empty()) return ok();
    std::vector<std::vector<float>> vectors;
    if (dims_ > 0 && embed) vectors = embed(texts);
    const bool real = vectors.size() == texts.size();
    const std::vector<float> flat(kWordDims, 0.5f);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (dims_ > 0 && (!real || static_cast<int>(vectors[i].size()) != dims_))
            return fail(ErrorCode::DecodeFailed, "the embedding model did not return vectors for the memory index");
        auto r = ops_->upsert(state_.get(), ids[i], texts[i], dims_ > 0 ? std::span<const float>(vectors[i]) : std::span<const float>(flat),
                              "{}");
        if (!r.ok) return r;
        std::lock_guard<std::mutex> lock(seen.mu);
        seen.note(ids[i], text_hash(texts[i]));
    }
    return ok();
}

Result<void> MemoryIndex::erase(std::string_view id) {
    auto &seen = *static_cast<Seen *>(seen_.get());
    {
        std::lock_guard<std::mutex> lock(seen.mu);
        seen.note(std::string(id), 0);
    }
    auto r = ops_->erase(state_.get(), id);
    if (!r.ok) return fail(r.error.code, r.error.message);
    return ok();
}

std::vector<MemoryHit> MemoryIndex::search(std::string_view query, int k, const EmbedFn &embed) const {
    std::vector<MemoryHit> out;
    if (query.empty() || k <= 0) return out;
    std::vector<float> q;
    if (dims_ > 0 && embed) {
        auto v = embed({std::string(query)});
        if (v.size() == 1 && static_cast<int>(v[0].size()) == dims_) q = std::move(v[0]);
    }
    const bool real = !q.empty();
    if (dims_ > 0 && !real) return out; // the table has real vectors but the model is gone: the caller falls back to words
    const std::vector<float> flat(kWordDims, 0.5f);
    const std::span<const float> vec = real ? std::span<const float>(q) : std::span<const float>(flat);
    auto hits =
        ops_->query_hybrid ? ops_->query_hybrid(state_.get(), query, vec, real ? k : 4 * k) : ops_->query_topk(state_.get(), vec, k);
    if (!hits.ok) return out;
    const auto words = search_words(query);
    for (VectorHit &h : hits.value) {
        if (!real) { // no meaning to go by: keep only entries that share a word with the query
            const std::string hay = lower_text(h.text);
            bool any = false;
            for (const std::string &w : words)
                if (hay.find(w) != std::string::npos) any = true;
            if (!any) continue;
        }
        out.push_back(MemoryHit{std::move(h.id), std::move(h.text), h.score});
        if (static_cast<int>(out.size()) >= k) break;
    }
    return out;
}

} // namespace agent::cli
