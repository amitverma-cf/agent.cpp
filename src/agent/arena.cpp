#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cstring>
#include <new>

namespace agent {

namespace {
constexpr size_t kMinChunk = 16 * 1024;
}

Arena::Arena(size_t capacity) {
    if (capacity > 0) {
        std::unique_ptr<char[]> mem(new (std::nothrow) char[capacity]);
        if (mem) chunks_.push_back(Chunk{std::move(mem), capacity}); // on failure the first allocate() tries again
    }
}

Arena::~Arena() = default;
Arena::Arena(Arena &&other) noexcept
    : chunks_(std::move(other.chunks_)), offset_(other.offset_), full_(other.full_), high_water_(other.high_water_) {
    other.chunks_.clear();
    other.offset_ = other.full_ = other.high_water_ = 0;
}

Arena &Arena::operator=(Arena &&other) noexcept {
    if (this != &other) {
        chunks_ = std::move(other.chunks_);
        offset_ = other.offset_;
        full_ = other.full_;
        high_water_ = other.high_water_;
        other.chunks_.clear();
        other.offset_ = other.full_ = other.high_water_ = 0;
    }
    return *this;
}

// Bump-allocates from the current chunk; nullptr when it does not fit.
void *Arena::bump(size_t size, size_t alignment) {
    if (chunks_.empty()) return nullptr;
    char *base = chunks_.back().data.get();
    const size_t mask = alignment - 1;
    const size_t aligned_ptr = (reinterpret_cast<size_t>(base) + offset_ + mask) & ~mask;
    const size_t aligned_offset = aligned_ptr - reinterpret_cast<size_t>(base);
    if (aligned_offset + size > chunks_.back().size) return nullptr;
    offset_ = aligned_offset + size;
    return reinterpret_cast<void *>(aligned_ptr);
}

void *Arena::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    if (alignment == 0) alignment = 1;
    if (void *p = bump(size, alignment)) return p;

    // The current chunk is full: add one that fits the request and is twice the previous one, so a long
    // turn needs only a few chunks. Earlier allocations stay where they are.
    const size_t previous = chunks_.empty() ? 0 : chunks_.back().size;
    const size_t chunk_size = std::max({size + alignment, previous * 2, kMinChunk});
    std::unique_ptr<char[]> mem(new (std::nothrow) char[chunk_size]);
    if (!mem) return nullptr; // really out of memory
    full_ += offset_;
    chunks_.push_back(Chunk{std::move(mem), chunk_size});
    offset_ = 0;
    return bump(size, alignment);
}

size_t Arena::capacity() const {
    size_t total = 0;
    for (const Chunk &c : chunks_) total += c.size;
    return total;
}

size_t Arena::used() const { return full_ + offset_; }

size_t Arena::high_water() const { return std::max(high_water_, used()); }

void Arena::reset() {
    high_water_ = std::max(high_water_, used());
    if (chunks_.size() > 1) {
        // The last turn outgrew the first block: replace the chunks by one block that fits the peak (plus
        // slack), so the next turn of that size allocates nothing.
        const size_t size = high_water_ + high_water_ / 4 + kMinChunk;
        std::unique_ptr<char[]> mem(new (std::nothrow) char[size]);
        if (mem) {
            chunks_.clear();
            chunks_.push_back(Chunk{std::move(mem), size});
        } else {
            chunks_.resize(1); // out of memory: fall back to the first block
        }
    }
    offset_ = 0;
    full_ = 0;
}

std::string_view Arena::allocate_string(std::string_view src) {
    if (src.empty()) return {};
    void *mem = allocate(src.size(), 1);
    if (!mem) return {};
    std::memcpy(mem, src.data(), src.size());
    return std::string_view(static_cast<const char *>(mem), src.size());
}

} // namespace agent
