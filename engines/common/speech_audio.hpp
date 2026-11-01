// Header-only audio helpers shared by the speech engine plugins (whisper, parakeet, qwen3tts).
//
// Plugins are self-contained shared libraries that cannot link libagent, so this lives here as
// inline functions compiled into each plugin. No exceptions are thrown on purpose: failures are
// reported through return values. (std::vector can still throw std::bad_alloc on exhaustion, like
// everywhere else in agent.cpp.)
//
//   - parse_wav / read_wav: RIFF/WAVE, PCM 8/16/24/32-bit or IEEE float 32-bit, any channel count
//     (down-mixed to mono), any sample rate.
//   - write_wav: 16-bit PCM mono.
//   - resample: linear interpolation, good enough for speech going into 16/24 kHz models.
//   - load_audio_part: the audio part of an agent_message (FILE / BYTES WAV, or an F32 TENSOR of
//     mono samples at a caller-given rate).
#pragma once

#include <agent-cpp/engine_abi.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <vector>

namespace agent_speech {

inline std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }
inline agent_str to_c(const std::string &s) { return agent_str{s.data(), s.size()}; }

inline bool file_exists(const std::string &path) {
    struct stat st{};
    return !path.empty() && ::stat(path.c_str(), &st) == 0;
}

struct Audio {
    std::vector<float> samples; // mono, [-1, 1]
    int32_t rate = 0;
};

inline uint32_t rd_u32(const uint8_t *p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
inline uint16_t rd_u16(const uint8_t *p) { return static_cast<uint16_t>(p[0] | p[1] << 8); }

// Returns an empty string on success, else what is wrong with the data.
inline std::string parse_wav(const uint8_t *data, size_t size, Audio &out) {
    if (size < 12 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) return "not a RIFF/WAVE file";
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    bool have_fmt = false;
    size_t pos = 12;
    while (pos + 8 <= size) {
        const uint8_t *chunk = data + pos;
        const size_t chunk_size = rd_u32(chunk + 4);
        const size_t avail = size - pos - 8;
        const uint8_t *body = chunk + 8;
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (chunk_size < 16 || avail < 16) return "WAV fmt chunk is truncated";
            format = rd_u16(body);
            channels = rd_u16(body + 2);
            rate = rd_u32(body + 4);
            bits = rd_u16(body + 14);
            if (format == 0xFFFE && chunk_size >= 26 && avail >= 26) format = rd_u16(body + 24); // WAVE_FORMAT_EXTENSIBLE
            have_fmt = true;
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            if (!have_fmt) return "WAV data chunk before fmt chunk";
            const bool pcm = format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
            const bool flt = format == 3 && bits == 32;
            if (!pcm && !flt) return "unsupported WAV sample format (PCM 8/16/24/32-bit or float 32-bit)";
            if (channels == 0 || rate == 0) return "WAV has no channels or no sample rate";
            const size_t bytes = chunk_size < avail ? chunk_size : avail; // tolerate a truncated last chunk
            const size_t frame = static_cast<size_t>(bits / 8) * channels;
            const size_t n = bytes / frame;
            out.samples.resize(n);
            out.rate = static_cast<int32_t>(rate);
            for (size_t i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (size_t c = 0; c < channels; ++c) {
                    const uint8_t *s = body + i * frame + c * (bits / 8);
                    float v = 0.0f;
                    if (flt) {
                        uint32_t u = rd_u32(s);
                        std::memcpy(&v, &u, sizeof(v));
                    } else if (bits == 8) {
                        v = (static_cast<float>(s[0]) - 128.0f) / 128.0f;
                    } else if (bits == 16) {
                        v = static_cast<float>(static_cast<int16_t>(rd_u16(s))) / 32768.0f;
                    } else if (bits == 24) {
                        const int32_t x = static_cast<int32_t>(uint32_t(s[0]) << 8 | uint32_t(s[1]) << 16 | uint32_t(s[2]) << 24) >> 8;
                        v = static_cast<float>(x) / 8388608.0f;
                    } else {
                        v = static_cast<float>(static_cast<int32_t>(rd_u32(s))) / 2147483648.0f;
                    }
                    sum += v;
                }
                out.samples[i] = sum / static_cast<float>(channels);
            }
            return {};
        }
        pos += 8 + chunk_size + (chunk_size & 1);
    }
    return "WAV has no data chunk";
}

inline std::string read_wav(const std::string &path, Audio &out) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::vector<uint8_t> bytes;
    uint8_t buf[65536];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
    std::fclose(f);
    std::string err = parse_wav(bytes.data(), bytes.size(), out);
    return err.empty() ? err : path + ": " + err;
}

// 16-bit PCM mono; samples are clamped to [-1, 1]. Returns false when the file cannot be written.
inline bool write_wav(const std::string &path, const float *samples, size_t n, int32_t rate) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t data_bytes = static_cast<uint32_t>(n * 2);
    uint8_t h[44];
    auto put32 = [&](size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) h[at + static_cast<size_t>(i)] = static_cast<uint8_t>(v >> (8 * i));
    };
    auto put16 = [&](size_t at, uint16_t v) {
        h[at] = static_cast<uint8_t>(v);
        h[at + 1] = static_cast<uint8_t>(v >> 8);
    };
    std::memcpy(h, "RIFF", 4);
    put32(4, 36 + data_bytes);
    std::memcpy(h + 8, "WAVEfmt ", 8);
    put32(16, 16);
    put16(20, 1);
    put16(22, 1);
    put32(24, static_cast<uint32_t>(rate));
    put32(28, static_cast<uint32_t>(rate) * 2);
    put16(32, 2);
    put16(34, 16);
    std::memcpy(h + 36, "data", 4);
    put32(40, data_bytes);
    bool ok = std::fwrite(h, 1, sizeof(h), f) == sizeof(h);
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; ++i) {
        float v = samples[i] < -1.0f ? -1.0f : (samples[i] > 1.0f ? 1.0f : samples[i]);
        pcm[i] = static_cast<int16_t>(v * 32767.0f);
    }
    ok = ok && (n == 0 || std::fwrite(pcm.data(), sizeof(int16_t), n, f) == n);
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

inline void resample(Audio &a, int32_t to_rate) {
    if (a.rate == to_rate || a.rate <= 0 || to_rate <= 0 || a.samples.empty()) {
        if (a.samples.empty()) a.rate = to_rate;
        return;
    }
    const double ratio = static_cast<double>(a.rate) / to_rate;
    const size_t n_out = static_cast<size_t>(static_cast<double>(a.samples.size()) / ratio);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double src = static_cast<double>(i) * ratio;
        const size_t i0 = static_cast<size_t>(src);
        const size_t i1 = i0 + 1 < a.samples.size() ? i0 + 1 : i0;
        const float t = static_cast<float>(src - static_cast<double>(i0));
        out[i] = a.samples[i0] * (1.0f - t) + a.samples[i1] * t;
    }
    a.samples.swap(out);
    a.rate = to_rate;
}

// True for parts that carry audio: an F32 tensor, or FILE / BYTES with an audio/* mime (a FILE
// without mime counts when its path ends in .wav).
inline bool is_audio_part(const agent_part &p) {
    if (p.kind == AGENT_PART_TENSOR) return p.dtype == AGENT_DTYPE_F32;
    if (p.kind != AGENT_PART_FILE && p.kind != AGENT_PART_BYTES) return false;
    const std::string_view mime = sv(p.mime);
    if (mime.starts_with("audio/")) return true;
    if (p.kind == AGENT_PART_FILE && mime.empty()) return sv(p.path).ends_with(".wav");
    return false;
}

struct UserInput {
    std::string_view text;            // last TEXT part of the last user message
    const agent_part *audio = nullptr; // last audio part of the last user message
};

inline UserInput last_user_input(const agent_request *req) {
    UserInput in;
    for (size_t i = req->n_messages; i-- > 0;) {
        const agent_message &m = req->messages[i];
        if (sv(m.role) != "user") continue;
        for (size_t p = 0; p < m.n_parts; ++p) {
            const agent_part &part = m.parts[p];
            if (part.kind == AGENT_PART_TEXT && part.text.size > 0) in.text = sv(part.text);
            else if (is_audio_part(part)) in.audio = &part;
        }
        break;
    }
    return in;
}

// Decodes an audio part to mono samples resampled to target_rate. A TENSOR is taken as mono F32 at
// tensor_rate. Returns an empty string on success.
inline std::string load_audio_part(const agent_part &part, int32_t tensor_rate, int32_t target_rate, Audio &out) {
    out = Audio{};
    if (part.kind == AGENT_PART_TENSOR) {
        if (part.dtype != AGENT_DTYPE_F32 || (part.size % sizeof(float)) != 0 || (part.size && !part.data))
            return "an audio tensor must be F32 mono samples";
        out.samples.resize(part.size / sizeof(float));
        if (part.size) std::memcpy(out.samples.data(), part.data, part.size);
        out.rate = tensor_rate;
    } else if (part.kind == AGENT_PART_FILE) {
        std::string err = read_wav(std::string(sv(part.path)), out);
        if (!err.empty()) return err;
    } else if (part.kind == AGENT_PART_BYTES) {
        if (!part.data) return "audio BYTES part has no data";
        std::string err = parse_wav(static_cast<const uint8_t *>(part.data), part.size, out);
        if (!err.empty()) return "<bytes>: " + err;
    } else {
        return "unsupported audio part";
    }
    if (out.samples.empty() || out.rate <= 0) return "audio input is empty";
    resample(out, target_rate);
    if (out.samples.size() > 0x7fffffffu) return "audio input too long";
    return {};
}

// <output_dir>/<prefix>-<unix ms>-<n>.wav, skipping names that already exist.
inline std::string unique_wav_path(std::string_view dir, const char *prefix) {
    static std::atomic<uint64_t> counter{0};
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string base(dir);
    if (!base.empty() && base.back() != '/' && base.back() != '\\') base += '/';
    for (;;) {
        std::string path = base + prefix + "-" + std::to_string(ms) + "-" + std::to_string(counter.fetch_add(1)) + ".wav";
        if (!file_exists(path)) return path;
    }
}

inline std::string_view trim(std::string_view t) {
    while (!t.empty() && (t.front() == ' ' || t.front() == '\n' || t.front() == '\t')) t.remove_prefix(1);
    while (!t.empty() && (t.back() == ' ' || t.back() == '\n' || t.back() == '\t')) t.remove_suffix(1);
    return t;
}

} // namespace agent_speech
