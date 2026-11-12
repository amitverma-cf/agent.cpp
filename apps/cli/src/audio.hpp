#pragma once

// Audio I/O for the `agent` CLI (agent_cli_core; libagent never links it). Capture is always mono
// f32 at 16 kHz; playback takes mono f32 at any rate. Two backends sit behind the same small op
// tables (no vtables):
//   - device: miniaudio (vendor/miniaudio), the default microphone and speakers;
//   - file:   microphone input read from a WAV file, speaker output written to WAV files. Used by
//             tests, by `agent voice --input-wav/--output-dir`, and when no sound device exists.

#include <agent-cpp/agent.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli::audio {

inline constexpr int kCaptureRate = 16000;

// -------------------------- WAV files --------------------------

struct Wav {
    std::vector<float> samples; // mono, [-1, 1]
    int rate = 0;
};

// RIFF/WAVE with PCM 8/16/24/32-bit or IEEE float 32-bit samples; multi-channel input is
// down-mixed to mono. Unknown chunks are skipped.
Result<Wav> parse_wav(std::span<const uint8_t> bytes);
Result<Wav> read_wav(const std::string &path);
// 16-bit PCM mono (samples are clamped to [-1, 1]).
std::vector<uint8_t> encode_wav(std::span<const float> samples, int rate);
Result<void> write_wav(const std::string &path, std::span<const float> samples, int rate);

// Linear-interpolation resampler; good enough for speech going into 16 kHz models.
std::vector<float> resample(std::span<const float> samples, int from_rate, int to_rate);

// -------------------------- Backends --------------------------

struct InputOps {
    // Copies up to max captured samples (16 kHz mono) into out and returns how many. Waits up to
    // timeout_ms for audio on a device. Sets eof when a file input has nothing left.
    size_t (*read)(void *state, float *out, size_t max, int timeout_ms, bool &eof);
    // Drops audio captured so far but not read yet (half-duplex: what the mic heard while the
    // agent was talking). No-op for files.
    void (*discard)(void *state);
};

struct AudioInput {
    const InputOps *ops = nullptr;
    std::shared_ptr<void> state;
    std::string name; // "file:<path>" or "device:<name>"
};

struct OutputOps {
    // Queues samples (mono, any rate) for playback; returns at once on a device.
    Result<void> (*play)(void *state, std::span<const float> samples, int rate);
    // Barge-in: stops playback and drops whatever is still queued.
    void (*stop)(void *state);
    bool (*playing)(void *state);
    // Blocks until everything queued has been played.
    void (*drain)(void *state);
    // samples_16k of input time went by (the file backend plays in input time for barge-in).
    void (*tick)(void *state, size_t samples_16k);
};

struct AudioOutput {
    const OutputOps *ops = nullptr;
    std::shared_ptr<void> state;
    std::string name;
};

size_t read(AudioInput &in, float *out, size_t max, int timeout_ms, bool &eof);
void discard(AudioInput &in);
Result<void> play(AudioOutput &out, std::span<const float> samples, int rate);
void stop(AudioOutput &out);
bool playing(AudioOutput &out);
void drain(AudioOutput &out);
void tick(AudioOutput &out, size_t samples_16k);

// Reads the whole WAV up front (resampled to 16 kHz) and hands it out in order.
Result<AudioInput> open_file_input(const std::string &wav_path);

struct FileOutputInfo {
    std::vector<std::string> written; // paths of the WAVs written so far, in order
    size_t interrupted = 0;           // replies cut short by stop()
};

// Each play() becomes <dir>/<prefix>-NNN.wav. With realtime_in_input_time, a reply "plays" for its
// duration measured by tick() (input time), so stop() during that window truncates the file the
// way barge-in cuts a real speaker; otherwise playback finishes immediately.
Result<AudioOutput> open_file_output(const std::string &dir, std::string_view prefix, bool realtime_in_input_time);
// The bookkeeping of a file output (nullptr for other backends).
const FileOutputInfo *file_output_info(const AudioOutput &out);

// miniaudio devices. Fail with ErrorCode::UnsupportedProvider when the binary was built without them
// (AGENT_CLI_AUDIO_DEVICE=OFF) and with a description when no device can be opened.
bool device_backend_available();
Result<AudioInput> open_device_input();
Result<AudioOutput> open_device_output();

} // namespace agent::cli::audio
