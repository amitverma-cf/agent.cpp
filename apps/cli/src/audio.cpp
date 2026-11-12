#include "audio.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>

#if !defined(AGENT_CLI_NO_AUDIO_DEVICE)
#include <miniaudio.h> // declarations only; the implementation is compiled in miniaudio_impl.c
#endif

namespace agent::cli::audio {

namespace fs = std::filesystem;

// -------------------------- WAV --------------------------

namespace {

uint32_t le32(const uint8_t *p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint16_t le16(const uint8_t *p) { return static_cast<uint16_t>(p[0] | p[1] << 8); }

void put32(std::vector<uint8_t> &out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put16(std::vector<uint8_t> &out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
}

// One sample of the given format at p, as [-1, 1].
float decode_sample(const uint8_t *p, uint16_t format, uint16_t bits) {
    if (format == 3 && bits == 32) {
        float f;
        std::memcpy(&f, p, 4);
        return f;
    }
    switch (bits) {
    case 8: return (static_cast<float>(p[0]) - 128.0f) / 128.0f;
    case 16: return static_cast<float>(static_cast<int16_t>(le16(p))) / 32768.0f;
    case 24: {
        int32_t v = static_cast<int32_t>(uint32_t(p[0]) << 8 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 24) >> 8;
        return static_cast<float>(v) / 8388608.0f;
    }
    case 32: return static_cast<float>(static_cast<double>(static_cast<int32_t>(le32(p))) / 2147483648.0);
    default: return 0.0f;
    }
}

} // namespace

Result<Wav> parse_wav(std::span<const uint8_t> bytes) {
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0)
        return fail<Wav>(ErrorCode::ParseError, "not a RIFF/WAVE file");
    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    bool have_fmt = false;
    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const uint8_t *chunk = bytes.data() + pos;
        const uint32_t size = le32(chunk + 4);
        const size_t body = pos + 8;
        const size_t avail = bytes.size() - body;
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (size < 16 || avail < 16) return fail<Wav>(ErrorCode::ParseError, "WAV fmt chunk is truncated");
            format = le16(chunk + 8);
            channels = le16(chunk + 10);
            rate = le32(chunk + 12);
            bits = le16(chunk + 22);
            if (format == 0xFFFE && size >= 26 && avail >= 26) format = le16(chunk + 32); // WAVE_FORMAT_EXTENSIBLE sub-format
            have_fmt = true;
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            if (!have_fmt) return fail<Wav>(ErrorCode::ParseError, "WAV data chunk before fmt chunk");
            const bool pcm = format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
            const bool flt = format == 3 && bits == 32;
            if (!pcm && !flt) return fail<Wav>(ErrorCode::ParseError, "unsupported WAV sample format (PCM 8/16/24/32-bit or float 32-bit)");
            if (channels == 0 || rate == 0) return fail<Wav>(ErrorCode::ParseError, "WAV has no channels or no sample rate");
            const size_t frame = size_t(bits / 8) * channels;
            const size_t n = std::min<size_t>(size, avail) / frame; // tolerate a data size past EOF (streams)
            Wav wav;
            wav.rate = static_cast<int>(rate);
            wav.samples.resize(n);
            const uint8_t *p = chunk + 8;
            for (size_t i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < channels; ++c) sum += decode_sample(p + i * frame + size_t(c) * (bits / 8), format, bits);
                wav.samples[i] = sum / static_cast<float>(channels);
            }
            return ok(std::move(wav));
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    return fail<Wav>(ErrorCode::ParseError, "WAV has no data chunk");
}

Result<Wav> read_wav(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail<Wav>(ErrorCode::FilesystemError, "cannot open " + path);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto wav = parse_wav(bytes);
    if (!wav.ok) return fail<Wav>(wav.error.code, path + ": " + wav.error.message);
    return wav;
}

std::vector<uint8_t> encode_wav(std::span<const float> samples, int rate) {
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 2);
    std::vector<uint8_t> out;
    out.reserve(44 + data_bytes);
    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    put32(out, 36 + data_bytes);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    put32(out, 16);
    put16(out, 1); // PCM
    put16(out, 1); // mono
    put32(out, static_cast<uint32_t>(rate));
    put32(out, static_cast<uint32_t>(rate) * 2);
    put16(out, 2);
    put16(out, 16);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    put32(out, data_bytes);
    for (float s : samples) {
        const float c = std::clamp(s, -1.0f, 1.0f);
        put16(out, static_cast<uint16_t>(static_cast<int16_t>(std::lrint(c * 32767.0f))));
    }
    return out;
}

Result<void> write_wav(const std::string &path, std::span<const float> samples, int rate) {
    const std::vector<uint8_t> bytes = encode_wav(samples, rate);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return fail(ErrorCode::FilesystemError, "cannot write " + path);
    f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) return fail(ErrorCode::FilesystemError, "cannot write " + path);
    return ok();
}

std::vector<float> resample(std::span<const float> samples, int from_rate, int to_rate) {
    if (from_rate == to_rate || samples.empty() || from_rate <= 0 || to_rate <= 0) return {samples.begin(), samples.end()};
    const double step = static_cast<double>(from_rate) / to_rate;
    const size_t n = static_cast<size_t>(static_cast<double>(samples.size()) / step);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * step;
        const size_t j = static_cast<size_t>(x);
        const float frac = static_cast<float>(x - static_cast<double>(j));
        const float a = samples[std::min(j, samples.size() - 1)];
        const float b = samples[std::min(j + 1, samples.size() - 1)];
        out[i] = a + (b - a) * frac;
    }
    return out;
}

// -------------------------- dispatch --------------------------

size_t read(AudioInput &in, float *out, size_t max, int timeout_ms, bool &eof) {
    eof = false;
    return in.ops ? in.ops->read(in.state.get(), out, max, timeout_ms, eof) : 0;
}
void discard(AudioInput &in) {
    if (in.ops && in.ops->discard) in.ops->discard(in.state.get());
}
Result<void> play(AudioOutput &out, std::span<const float> samples, int rate) {
    if (!out.ops) return fail(ErrorCode::InvalidConfig, "no audio output");
    return out.ops->play(out.state.get(), samples, rate);
}
void stop(AudioOutput &out) {
    if (out.ops) out.ops->stop(out.state.get());
}
bool playing(AudioOutput &out) { return out.ops && out.ops->playing(out.state.get()); }
void drain(AudioOutput &out) {
    if (out.ops) out.ops->drain(out.state.get());
}
void tick(AudioOutput &out, size_t samples_16k) {
    if (out.ops && out.ops->tick) out.ops->tick(out.state.get(), samples_16k);
}

// -------------------------- file backend --------------------------

namespace {

struct FileInput {
    std::vector<float> samples;
    size_t pos = 0;
};

size_t file_read(void *state, float *out, size_t max, int, bool &eof) {
    auto *s = static_cast<FileInput *>(state);
    const size_t n = std::min(max, s->samples.size() - s->pos);
    if (n) std::memcpy(out, s->samples.data() + s->pos, n * sizeof(float));
    s->pos += n;
    eof = s->pos >= s->samples.size();
    return n;
}

void file_discard(void *) {}

constexpr InputOps kFileInputOps = {file_read, file_discard};

struct FileOutput {
    std::string dir;
    std::string prefix;
    bool realtime = false;
    FileOutputInfo info;
    // The reply "playing" in input time (realtime mode only).
    std::vector<float> pending;
    int pending_rate = 0;
    double played_seconds = 0.0;
};

Result<void> write_next(FileOutput &s, std::span<const float> samples, int rate) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s-%03zu.wav", s.prefix.c_str(), s.info.written.size() + 1);
    const std::string path = (fs::path(s.dir) / name).string();
    if (auto r = write_wav(path, samples, rate); !r.ok) return r;
    s.info.written.push_back(path);
    return ok();
}

void finish_pending(FileOutput &s, bool interrupted) {
    if (s.pending_rate == 0) return;
    size_t n = s.pending.size();
    if (interrupted) {
        n = std::min(n, static_cast<size_t>(s.played_seconds * s.pending_rate));
        ++s.info.interrupted;
    }
    write_next(s, std::span<const float>(s.pending.data(), n), s.pending_rate);
    s.pending.clear();
    s.pending_rate = 0;
    s.played_seconds = 0.0;
}

Result<void> file_play(void *state, std::span<const float> samples, int rate) {
    auto *s = static_cast<FileOutput *>(state);
    if (!s->realtime) return write_next(*s, samples, rate);
    finish_pending(*s, false);
    s->pending.assign(samples.begin(), samples.end());
    s->pending_rate = rate;
    s->played_seconds = 0.0;
    return ok();
}

void file_stop(void *state) { finish_pending(*static_cast<FileOutput *>(state), true); }
bool file_playing(void *state) { return static_cast<FileOutput *>(state)->pending_rate != 0; }
void file_drain(void *state) { finish_pending(*static_cast<FileOutput *>(state), false); }

void file_tick(void *state, size_t samples_16k) {
    auto *s = static_cast<FileOutput *>(state);
    if (s->pending_rate == 0) return;
    s->played_seconds += static_cast<double>(samples_16k) / kCaptureRate;
    if (s->played_seconds * s->pending_rate >= static_cast<double>(s->pending.size())) finish_pending(*s, false);
}

constexpr OutputOps kFileOutputOps = {file_play, file_stop, file_playing, file_drain, file_tick};

} // namespace

Result<AudioInput> open_file_input(const std::string &wav_path) {
    auto wav = read_wav(wav_path);
    if (!wav.ok) return fail<AudioInput>(wav.error.code, wav.error.message);
    auto state = std::make_shared<FileInput>();
    state->samples = resample(wav.value.samples, wav.value.rate, kCaptureRate);
    AudioInput in;
    in.ops = &kFileInputOps;
    in.state = std::move(state);
    in.name = "file:" + wav_path;
    return ok(std::move(in));
}

Result<AudioOutput> open_file_output(const std::string &dir, std::string_view prefix, bool realtime_in_input_time) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return fail<AudioOutput>(ErrorCode::FilesystemError, "cannot create " + dir + ": " + ec.message());
    auto state = std::make_shared<FileOutput>();
    state->dir = dir;
    state->prefix = std::string(prefix);
    state->realtime = realtime_in_input_time;
    AudioOutput out;
    out.ops = &kFileOutputOps;
    out.state = std::move(state);
    out.name = "file:" + dir;
    return ok(std::move(out));
}

const FileOutputInfo *file_output_info(const AudioOutput &out) {
    if (out.ops != &kFileOutputOps) return nullptr;
    return &static_cast<const FileOutput *>(out.state.get())->info;
}

// -------------------------- device backend (miniaudio) --------------------------

#if defined(AGENT_CLI_NO_AUDIO_DEVICE)

bool device_backend_available() { return false; }
Result<AudioInput> open_device_input() {
    return fail<AudioInput>(ErrorCode::UnsupportedProvider,
                            "this agent was built without sound device support (AGENT_CLI_AUDIO_DEVICE=OFF)");
}
Result<AudioOutput> open_device_output() {
    return fail<AudioOutput>(ErrorCode::UnsupportedProvider,
                             "this agent was built without sound device support (AGENT_CLI_AUDIO_DEVICE=OFF)");
}

#else

namespace {

std::string ma_error(const char *what, ma_result r) { return std::string(what) + ": " + ma_result_description(r); }

// Microphone: the device thread appends to a bounded buffer, read() takes from its front.
struct DeviceInput {
    ma_device device{};
    bool started = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<float> buffer;
    size_t max_samples = static_cast<size_t>(kCaptureRate) * 60; // keep at most a minute unread

    ~DeviceInput() {
        if (started) ma_device_uninit(&device);
    }
};

void capture_callback(ma_device *device, void *, const void *input, ma_uint32 frames) {
    auto *s = static_cast<DeviceInput *>(device->pUserData);
    if (!input) return;
    const auto *in = static_cast<const float *>(input);
    {
        std::lock_guard<std::mutex> lk(s->mu);
        s->buffer.insert(s->buffer.end(), in, in + frames);
        if (s->buffer.size() > s->max_samples)
            s->buffer.erase(s->buffer.begin(), s->buffer.begin() + static_cast<std::ptrdiff_t>(s->buffer.size() - s->max_samples));
    }
    s->cv.notify_one();
}

size_t device_read(void *state, float *out, size_t max, int timeout_ms, bool &eof) {
    auto *s = static_cast<DeviceInput *>(state);
    eof = false;
    std::unique_lock<std::mutex> lk(s->mu);
    s->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !s->buffer.empty(); });
    const size_t n = std::min(max, s->buffer.size());
    if (n) {
        std::memcpy(out, s->buffer.data(), n * sizeof(float));
        s->buffer.erase(s->buffer.begin(), s->buffer.begin() + static_cast<std::ptrdiff_t>(n));
    }
    return n;
}

void device_discard(void *state) {
    auto *s = static_cast<DeviceInput *>(state);
    std::lock_guard<std::mutex> lk(s->mu);
    s->buffer.clear();
}

constexpr InputOps kDeviceInputOps = {device_read, device_discard};

// Speakers: opened on the first play() at that reply's rate; later replies at another rate are resampled.
struct DeviceOutput {
    ma_device device{};
    bool started = false;
    int rate = 0;
    std::mutex mu;
    std::vector<float> queue;
    size_t pos = 0;

    ~DeviceOutput() {
        if (started) ma_device_uninit(&device);
    }
};

void playback_callback(ma_device *device, void *output, const void *, ma_uint32 frames) {
    auto *s = static_cast<DeviceOutput *>(device->pUserData);
    auto *out = static_cast<float *>(output);
    std::lock_guard<std::mutex> lk(s->mu);
    const size_t n = std::min<size_t>(frames, s->queue.size() - s->pos);
    if (n) std::memcpy(out, s->queue.data() + s->pos, n * sizeof(float));
    std::fill(out + n, out + frames, 0.0f);
    s->pos += n;
    if (s->pos >= s->queue.size()) {
        s->queue.clear();
        s->pos = 0;
    }
}

Result<void> device_play(void *state, std::span<const float> samples, int rate) {
    auto *s = static_cast<DeviceOutput *>(state);
    if (!s->started) {
        ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
        cfg.playback.format = ma_format_f32;
        cfg.playback.channels = 1;
        cfg.sampleRate = static_cast<ma_uint32>(rate);
        cfg.dataCallback = playback_callback;
        cfg.pUserData = s;
        if (ma_result r = ma_device_init(nullptr, &cfg, &s->device); r != MA_SUCCESS)
            return fail(ErrorCode::ProviderInitFailed, ma_error("cannot open the speaker", r));
        s->started = true;
        s->rate = rate;
        if (ma_result r = ma_device_start(&s->device); r != MA_SUCCESS)
            return fail(ErrorCode::ProviderInitFailed, ma_error("cannot start the speaker", r));
    }
    std::vector<float> converted;
    if (rate != s->rate) converted = resample(samples, rate, s->rate);
    const std::span<const float> src = rate != s->rate ? std::span<const float>(converted) : samples;
    std::lock_guard<std::mutex> lk(s->mu);
    s->queue.insert(s->queue.end(), src.begin(), src.end());
    return ok();
}

void device_stop(void *state) {
    auto *s = static_cast<DeviceOutput *>(state);
    std::lock_guard<std::mutex> lk(s->mu);
    s->queue.clear();
    s->pos = 0;
}

bool device_playing(void *state) {
    auto *s = static_cast<DeviceOutput *>(state);
    std::lock_guard<std::mutex> lk(s->mu);
    return s->pos < s->queue.size();
}

void device_drain(void *state) {
    while (device_playing(state)) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

constexpr OutputOps kDeviceOutputOps = {device_play, device_stop, device_playing, device_drain, nullptr};

} // namespace

bool device_backend_available() { return true; }

Result<AudioInput> open_device_input() {
    auto state = std::make_shared<DeviceInput>();
    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate = kCaptureRate; // miniaudio converts from the device's native rate
    cfg.dataCallback = capture_callback;
    cfg.pUserData = state.get();
    if (ma_result r = ma_device_init(nullptr, &cfg, &state->device); r != MA_SUCCESS)
        return fail<AudioInput>(ErrorCode::ProviderInitFailed, ma_error("cannot open the microphone", r));
    state->started = true;
    if (state->device.pContext && state->device.pContext->backend == ma_backend_null)
        return fail<AudioInput>(ErrorCode::ProviderInitFailed, "no microphone (miniaudio found only its null backend)");
    if (ma_result r = ma_device_start(&state->device); r != MA_SUCCESS)
        return fail<AudioInput>(ErrorCode::ProviderInitFailed, ma_error("cannot start the microphone", r));
    AudioInput in;
    in.ops = &kDeviceInputOps;
    in.name = std::string("device:") + state->device.capture.name;
    in.state = std::move(state);
    return ok(std::move(in));
}

Result<AudioOutput> open_device_output() {
    // The device itself opens on the first play() (its rate follows the TTS model). Check up front
    // that a playback device exists so a box without sound fails before models are loaded.
    ma_context context;
    if (ma_result r = ma_context_init(nullptr, 0, nullptr, &context); r != MA_SUCCESS)
        return fail<AudioOutput>(ErrorCode::ProviderInitFailed, ma_error("no audio backend", r));
    ma_device_info *playback = nullptr;
    ma_uint32 n_playback = 0;
    const ma_result r = ma_context_get_devices(&context, &playback, &n_playback, nullptr, nullptr);
    const bool null_backend = context.backend == ma_backend_null;
    ma_context_uninit(&context);
    if (r != MA_SUCCESS || n_playback == 0 || null_backend)
        return fail<AudioOutput>(ErrorCode::ProviderInitFailed, "no speaker (playback device) found");
    AudioOutput out;
    out.ops = &kDeviceOutputOps;
    out.state = std::make_shared<DeviceOutput>();
    out.name = "device:default";
    return ok(std::move(out));
}

#endif

} // namespace agent::cli::audio
