// Helpers shared by the speech engine plugin tests (whisper, parakeet, qwen3tts).
#pragma once

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <catch_amalgamated.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace speech_test {

inline agent_str c_str(std::string_view s) { return agent_str{s.data(), s.size()}; }
inline std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

inline const char *env(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline const agent_engine_v1 *load(const char *path, std::string_view name) {
    auto loaded = agent::load_engine_plugin(path);
    REQUIRE(loaded.ok);
    REQUIRE(loaded.value == name);
    return agent::find_engine(std::string(name));
}

inline int32_t create(const agent_engine_v1 *e, const std::string &cfg, void **inst, std::string *err_out = nullptr) {
    char err[512] = {0};
    const int32_t rc = e->create(c_str(cfg), inst, err, sizeof(err));
    if (err_out) *err_out = err;
    return rc;
}

struct Captured {
    size_t audio_bytes = 0;
    size_t audio_chunks = 0;
    std::string text;
    uint64_t last_step = 0, total = 0;
    volatile int32_t cancel = 0;
    bool cancel_on_first_chunk = false;
    bool cancel_on_progress = false;
};

inline void on_chunk(void *ud, uint32_t kind, const void *data, size_t size) {
    auto *c = static_cast<Captured *>(ud);
    if (kind == AGENT_CHUNK_AUDIO) {
        c->audio_bytes += size;
        ++c->audio_chunks;
    } else if (kind == AGENT_CHUNK_TEXT) {
        c->text.append(static_cast<const char *>(data), size);
    }
    if (c->cancel_on_first_chunk) c->cancel = 1;
}

inline void on_progress(void *ud, uint64_t step, uint64_t total) {
    auto *c = static_cast<Captured *>(ud);
    c->last_step = step;
    c->total = total;
    if (c->cancel_on_progress) c->cancel = 1;
}

// One user message with optional text, audio FILE and F32 audio TENSOR parts.
struct OneMessage {
    agent_part parts[3]{};
    int64_t shape[1]{};
    agent_message msg{};
    agent_request req{.struct_size = sizeof(agent_request)};

    OneMessage(uint32_t task, std::string_view text, std::string_view audio_path, Captured *cap, const char *out_dir = "/tmp",
               const std::vector<float> *tensor = nullptr) {
        size_t n = 0;
        if (!text.empty()) {
            parts[n].kind = AGENT_PART_TEXT;
            parts[n++].text = c_str(text);
        }
        if (!audio_path.empty()) {
            parts[n].kind = AGENT_PART_FILE;
            parts[n].path = c_str(audio_path);
            parts[n++].mime = c_str("audio/wav");
        }
        if (tensor) {
            shape[0] = static_cast<int64_t>(tensor->size());
            parts[n].kind = AGENT_PART_TENSOR;
            parts[n].name = c_str("audio");
            parts[n].dtype = AGENT_DTYPE_F32;
            parts[n].shape = shape;
            parts[n].ndim = 1;
            parts[n].data = tensor->data();
            parts[n++].size = tensor->size() * sizeof(float);
        }
        msg.role = c_str("user");
        msg.parts = parts;
        msg.n_parts = n;
        req.task = task;
        req.messages = &msg;
        req.n_messages = 1;
        req.output_dir = c_str(out_dir);
        if (cap) {
            req.on_chunk = on_chunk;
            req.on_progress = on_progress;
            req.user_data = cap;
            req.cancel = &cap->cancel;
        }
    }
};

inline std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// 16-bit PCM mono WAV with a 44-byte header (as the tests' fixtures and the TTS engines write it).
struct Pcm {
    std::vector<float> samples;
    uint32_t rate = 0;
};

inline Pcm read_pcm16_wav(const std::string &path) {
    const std::vector<uint8_t> bytes = read_file(path);
    REQUIRE(bytes.size() > 44);
    REQUIRE(std::memcmp(bytes.data(), "RIFF", 4) == 0);
    Pcm out;
    std::memcpy(&out.rate, bytes.data() + 24, 4);
    uint16_t bits = 0, channels = 0;
    std::memcpy(&channels, bytes.data() + 22, 2);
    std::memcpy(&bits, bytes.data() + 34, 2);
    REQUIRE(bits == 16);
    REQUIRE(channels == 1);
    // Find the data chunk (fixtures may carry LIST chunks before it).
    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        uint32_t size = 0;
        std::memcpy(&size, bytes.data() + pos + 4, 4);
        if (std::memcmp(bytes.data() + pos, "data", 4) == 0) {
            const size_t n = std::min<size_t>(size, bytes.size() - pos - 8) / 2;
            out.samples.resize(n);
            for (size_t i = 0; i < n; ++i) {
                int16_t s = 0;
                std::memcpy(&s, bytes.data() + pos + 8 + 2 * i, 2);
                out.samples[i] = static_cast<float>(s) / 32768.0f;
            }
            return out;
        }
        pos += 8 + size + (size & 1);
    }
    FAIL("no data chunk in " << path);
    return out;
}

} // namespace speech_test
