#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <cstdint>

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

TEST_CASE("Arena allocate returns nullptr on overflow", "[arena]") {
    agent::Arena arena(64);

    void *ok = arena.allocate(32);
    REQUIRE(ok != nullptr);

    void *overflow = arena.allocate(64); // only ~32 bytes left
    REQUIRE(overflow == nullptr);
}

TEST_CASE("Arena allocate(0) and a zero-capacity Arena both return nullptr", "[arena]") {
    agent::Arena arena(1024);
    REQUIRE(arena.allocate(0) == nullptr);

    agent::Arena empty_arena(0);
    REQUIRE(empty_arena.allocate(16) == nullptr);
}

TEST_CASE("Arena reset reclaims the whole capacity in O(1)", "[arena]") {
    agent::Arena arena(64);

    REQUIRE(arena.allocate(60) != nullptr);
    REQUIRE(arena.allocate(60) == nullptr); // out of space

    arena.reset();
    REQUIRE(arena.allocate(60) != nullptr); // space reclaimed
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
    // The moved-from arena has zero capacity; further allocation must fail, not crash.
    REQUIRE(arena.allocate(1) == nullptr);
    // The moved-to arena keeps working.
    REQUIRE(moved.allocate(16) != nullptr);
}
