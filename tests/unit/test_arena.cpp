#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

TEST_CASE("Arena allocate returns distinct, non-overlapping regions", "[arena]") {
    agent::Arena arena(1024);

    void *a = arena.allocate(64);
    void *b = arena.allocate(64);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a != b);
    REQUIRE(static_cast<char *>(b) >= static_cast<char *>(a) + 64);
}

TEST_CASE("Arena allocate respects alignment", "[arena]") {
    agent::Arena arena(1024);

    arena.allocate(1); // misalign the offset
    void *aligned = arena.allocate(16, 16);
    REQUIRE(aligned != nullptr);
    REQUIRE(reinterpret_cast<uintptr_t>(aligned) % 16 == 0);
}

TEST_CASE("Arena grows instead of failing when the block is full", "[arena]") {
    agent::Arena arena(64);

    void *first = arena.allocate(32);
    REQUIRE(first != nullptr);
    std::memset(first, 0xAB, 32);

    void *beyond = arena.allocate(4096); // does not fit in the first block: a new chunk is added
    REQUIRE(beyond != nullptr);
    REQUIRE(arena.blocks() == 2);
    REQUIRE(arena.capacity() >= 64 + 4096);

    // Pointers handed out before the growth are still valid and untouched.
    for (int i = 0; i < 32; ++i) REQUIRE(static_cast<unsigned char *>(first)[i] == 0xAB);
}

TEST_CASE("Arena growth keeps a long run of allocations distinct and aligned", "[arena]") {
    agent::Arena arena(128);
    std::vector<std::pair<char *, size_t>> blocks;
    for (size_t i = 1; i <= 500; ++i) {
        const size_t size = 8 + (i * 37) % 700;
        auto *p = static_cast<char *>(arena.allocate(size, 16));
        REQUIRE(p != nullptr);
        REQUIRE(reinterpret_cast<uintptr_t>(p) % 16 == 0);
        std::memset(p, static_cast<int>(i & 0xff), size);
        blocks.emplace_back(p, size);
    }
    for (size_t i = 0; i < blocks.size(); ++i)
        for (size_t b = 0; b < blocks[i].second; ++b) REQUIRE(static_cast<unsigned char>(blocks[i].first[b]) == ((i + 1) & 0xff));
}

TEST_CASE("Arena allocate(0) returns nullptr; a zero-capacity Arena allocates on first use", "[arena]") {
    agent::Arena arena(1024);
    REQUIRE(arena.allocate(0) == nullptr);

    agent::Arena empty_arena(0);
    REQUIRE(empty_arena.capacity() == 0);
    REQUIRE(empty_arena.allocate(16) != nullptr);
    REQUIRE(empty_arena.capacity() > 0);
}

TEST_CASE("Arena reset reclaims the whole capacity in O(1)", "[arena]") {
    agent::Arena arena(64);

    REQUIRE(arena.allocate(60) != nullptr);
    REQUIRE(arena.used() >= 60);
    arena.reset();
    REQUIRE(arena.used() == 0);
    REQUIRE(arena.blocks() == 1);
    REQUIRE(arena.capacity() == 64); // a turn that fit in the first block never allocates
    REQUIRE(arena.allocate(60) != nullptr);
    REQUIRE(arena.capacity() == 64);
}

TEST_CASE("Arena reset after a growing turn folds everything into one block sized to the peak", "[arena]") {
    agent::Arena arena(64);
    for (int i = 0; i < 40; ++i) REQUIRE(arena.allocate(1000) != nullptr);
    REQUIRE(arena.blocks() > 1);
    const size_t peak = arena.used();
    REQUIRE(peak >= 40000);

    arena.reset();
    REQUIRE(arena.blocks() == 1);
    REQUIRE(arena.capacity() >= peak);
    REQUIRE(arena.high_water() >= peak);

    // The next turn of the same size fits without growing: no allocation in the steady state.
    const size_t capacity = arena.capacity();
    for (int i = 0; i < 40; ++i) REQUIRE(arena.allocate(1000) != nullptr);
    REQUIRE(arena.blocks() == 1);
    REQUIRE(arena.capacity() == capacity);
}

TEST_CASE("Arena::allocate_string copies bytes and returns an independent view", "[arena]") {
    agent::Arena arena(1024);

    std::string source = "hello arena";
    std::string_view view = arena.allocate_string(source);
    REQUIRE(view == "hello arena");
    REQUIRE(view.data() != source.data());

    REQUIRE(arena.allocate_string("").empty());
}

TEST_CASE("Arena::allocate_span provides a typed span backed by the arena", "[arena]") {
    agent::Arena arena(1024);

    auto span = arena.allocate_span<int>(4);
    REQUIRE(span.size() == 4);
    for (int i = 0; i < 4; ++i) span[i] = i * i;
    REQUIRE(span[2] == 4);

    REQUIRE(arena.allocate_span<int>(0).empty());
}

TEST_CASE("Arena move construction transfers ownership and empties the source", "[arena]") {
    agent::Arena arena(128);
    void *p = arena.allocate(16);
    REQUIRE(p != nullptr);

    agent::Arena moved(std::move(arena));
    // The moved-from arena is empty (zero capacity) and still usable: it grows on the next allocation.
    REQUIRE(arena.capacity() == 0);
    REQUIRE(arena.used() == 0);
    REQUIRE(arena.allocate(1) != nullptr);
    // The moved-to arena keeps working.
    REQUIRE(moved.allocate(16) != nullptr);
}
