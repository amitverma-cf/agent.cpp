// The C API (agent_c.h), used the way a C program would: opaque handles, plain strings, caller buffers.

#include <agent-cpp/agent_c.h>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct Collected {
    std::string text;
};

void collect(const char *text, size_t length, void *user) { static_cast<Collected *>(user)->text.append(text, length); }

std::string workspace(const char *name) {
    const auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(dir);
    return dir.string();
}

} // namespace

TEST_CASE("c api: open, chat turns with streaming, close", "[c_api]") {
    char err[256] = {};
    const std::string ws = workspace("agent_test_c_api_chat");
    CHECK(std::strlen(agent_c_version()) > 0);

    agent_c_session *session = agent_c_open(AGENT_TEST_MOCK_ENGINE_PATH, "mock.bin", nullptr, ws.c_str(), 0, 0, err, sizeof(err));
    INFO(err);
    REQUIRE(session != nullptr);
    agent_c_conversation *chat = agent_c_conversation_new("c-chat");
    REQUIRE(chat != nullptr);

    Collected streamed;
    char reply[256] = {};
    size_t needed = 0;
    REQUIRE(agent_c_run_turn(session, chat, "hello from C", collect, &streamed, reply, sizeof(reply), &needed, err, sizeof(err)) ==
            AGENT_C_OK);
    CHECK(std::string(reply) == "echo: hello from C");
    CHECK(streamed.text == "echo: hello from C");
    CHECK(needed == std::strlen(reply) + 1);

    // A second turn on the same conversation; no streaming callback.
    REQUIRE(agent_c_run_turn(session, chat, "again", nullptr, nullptr, reply, sizeof(reply), nullptr, err, sizeof(err)) == AGENT_C_OK);
    CHECK(std::string(reply) == "echo: again");

    // A reply buffer that is too small: the answer is cut, and the size needed is reported.
    char tiny[4] = {};
    CHECK(agent_c_run_turn(session, chat, "a long enough message", nullptr, nullptr, tiny, sizeof(tiny), &needed, err, sizeof(err)) ==
          AGENT_C_BUFFER_TOO_SMALL);
    CHECK(std::string(tiny) == "ech");
    CHECK(needed > sizeof(tiny));

    agent_c_cancel(session); // harmless when nothing runs
    agent_c_conversation_free(chat);
    agent_c_close(session);
    agent_c_close(nullptr); // closing nothing is allowed
}

TEST_CASE("c api: errors come back as a status and a message", "[c_api]") {
    char err[256] = {};
    CHECK(agent_c_open(nullptr, "m", nullptr, "/tmp/x", 0, 0, err, sizeof(err)) == nullptr);
    CHECK(std::string(err).find("engine") != std::string::npos);
    CHECK(agent_c_open("no-such-engine-anywhere", "m", nullptr, workspace("agent_test_c_api_err").c_str(), 0, 0, err, sizeof(err)) ==
          nullptr);
    CHECK(std::strlen(err) > 0);

    char reply[32] = {};
    CHECK(agent_c_run_turn(nullptr, nullptr, "x", nullptr, nullptr, reply, sizeof(reply), nullptr, err, sizeof(err)) == AGENT_C_ERROR);
    CHECK(agent_c_embed(nullptr, "x", nullptr, 0, nullptr, err, sizeof(err)) == AGENT_C_ERROR);
}

TEST_CASE("c api: embeddings and one-shot tasks through the mock engine", "[c_api]") {
    char err[256] = {};
    const std::string ws = workspace("agent_test_c_api_tasks");
    agent_c_session *session = agent_c_open(AGENT_TEST_MOCK_ALL_ENGINE_PATH, "mock.bin", nullptr, ws.c_str(), 0, 0, err, sizeof(err));
    INFO(err);
    REQUIRE(session != nullptr);

    std::vector<float> v(16, 0.0f);
    size_t dims = 0;
    REQUIRE(agent_c_embed(session, "hello", v.data(), v.size(), &dims, err, sizeof(err)) == AGENT_C_OK);
    CHECK(dims > 0);
    float norm = 0;
    for (size_t i = 0; i < dims; ++i) norm += v[i] * v[i];
    CHECK(norm > 0.98f);
    CHECK(norm < 1.02f);
    float one = 0;
    CHECK(agent_c_embed(session, "hello", &one, 1, &dims, err, sizeof(err)) == AGENT_C_BUFFER_TOO_SMALL);
    CHECK(dims > 1);

    // Speech recognition on a file: the mock engine answers "mock transcript" for any audio.
    const std::string wav = (std::filesystem::path(ws) / "in.wav").string();
    std::FILE *f = std::fopen(wav.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fputs("RIFF....WAVEfmt ", f);
    std::fclose(f);
    char text[128] = {}, file[256] = {};
    REQUIRE(agent_c_run_task(session, AGENT_CAP_ASR, "", wav.c_str(), text, sizeof(text), file, sizeof(file), err, sizeof(err)) ==
            AGENT_C_OK);
    CHECK(std::string(text) == "mock transcript");

    // Text to speech: a WAV file comes back.
    REQUIRE(agent_c_run_task(session, AGENT_CAP_TTS, "say this", nullptr, text, sizeof(text), file, sizeof(file), err, sizeof(err)) ==
            AGENT_C_OK);
    CHECK(std::strlen(file) > 0);
    CHECK(std::filesystem::exists(file));

    CHECK(agent_c_run_task(session, 0, "x", nullptr, text, sizeof(text), file, sizeof(file), err, sizeof(err)) == AGENT_C_ERROR);
    agent_c_close(session);
}
