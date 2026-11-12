#pragma once

// `agent voice`: the testable part of the voice loop (agent_cli_core). Speech detection (an energy
// gate, or the audiocpp VAD engine behind the same op table), utterance segmentation, the
// capture -> detect -> ASR -> LLM -> TTS -> playback loop with barge-in, flag handling and the
// audiocpp engine options built from GGUF model files. The engines themselves are wired in voice.cpp.

#include "audio.hpp"
#include "cli_core.hpp"

#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// -------------------------- speech detection --------------------------

// Sample positions at 16 kHz, counted from the detector's last reset. end is exclusive.
struct SpeechSegment {
    int64_t start = 0;
    int64_t end = 0;
};

// Frame-RMS gate used when no VAD model is given. A frame is speech when its RMS is above
// max(threshold, noise_ratio * noise floor); speech starts after min_speech_ms of speech frames
// and ends after min_silence_ms of quiet ones.
struct EnergyGateOptions {
    float threshold = 0.01f; // absolute RMS floor (about -40 dBFS)
    float noise_ratio = 3.0f;
    int frame_ms = 30;
    int min_speech_ms = 120;
    int min_silence_ms = 600;
    int max_speech_ms = 20000; // longer speech is split
};

struct EnergyGate {
    EnergyGateOptions options;
    // state
    std::vector<float> partial; // samples of the frame being filled
    int64_t pos = 0;            // samples consumed in whole frames
    float noise_floor = -1.0f;  // < 0 until the first frame
    bool speaking = false;
    int64_t speech_start = 0;
    int64_t last_voiced_end = 0;
    int voiced_run = 0;
    int64_t voiced_run_start = 0;
    int quiet_run = 0;
};

void energy_gate_reset(EnergyGate &gate);
// Feeds samples; appends segments that finished. Returns whether speech is ongoing.
bool energy_gate_feed(EnergyGate &gate, std::span<const float> samples, std::vector<SpeechSegment> &finished);
// End of input: closes an open segment.
void energy_gate_flush(EnergyGate &gate, std::vector<SpeechSegment> &finished);
// Offline convenience: every segment in samples.
std::vector<SpeechSegment> detect_speech_energy(std::span<const float> samples, const EnergyGateOptions &options = {});

struct SpeechDetectorOps {
    // Returns whether speech is ongoing after this chunk; finished segments are appended.
    Result<bool> (*feed)(void *state, std::span<const float> samples, std::vector<SpeechSegment> &finished);
    Result<void> (*flush)(void *state, std::vector<SpeechSegment> &finished);
    void (*reset)(void *state);
};

struct SpeechDetector {
    const SpeechDetectorOps *ops = nullptr;
    void *state = nullptr;
    std::string name; // "energy" or "silero"
};

SpeechDetector energy_detector(EnergyGate &gate);

// The audiocpp engine's VAD output: [{"start":s,"end":s[,"open":true]}, ...] in seconds.
struct VadJsonSegment {
    double start = 0.0;
    double end = 0.0;
    bool open = false;
};
Result<std::vector<VadJsonSegment>> parse_vad_segments(std::string_view json);

// -------------------------- the loop --------------------------

enum class VoiceEvent {
    Listening, // ready for the next utterance
    Heard,     // text = transcript
    Skipped,   // utterance too short, or the transcript has no words
    Reply,     // text = reply text (after the LLM turn)
    Spoken,    // text = what the output did with the reply (e.g. the WAV path)
    BargeIn,   // the user spoke over the reply; playback stopped
    Error,     // text = message; the loop goes on
};

struct VoiceHooks {
    // 16 kHz mono utterance -> transcript.
    Result<std::string> (*transcribe)(std::span<const float> utterance, void *user_data) = nullptr;
    // transcript -> reply text (one LLM turn, tools included).
    Result<std::string> (*respond)(std::string_view transcript, void *user_data) = nullptr;
    // reply text -> speech.
    Result<audio::Wav> (*speak)(std::string_view reply, void *user_data) = nullptr;
    void (*on_event)(VoiceEvent event, std::string_view text, void *user_data) = nullptr;
    void *user_data = nullptr;
};

struct VoiceLoopOptions {
    bool barge_in = false; // keep listening while the reply plays; speech stops playback
    int max_turns = 0;     // 0 = until Ctrl-C / end of the input file
    int pre_roll_ms = 200; // audio kept before and after a detected segment
    int post_roll_ms = 150;
    int min_utterance_ms = 250;
    int barge_in_ms = 200;       // speech needed during playback before it stops
    size_t chunk_samples = 1600; // 100 ms reads
    const std::atomic<bool> *stop = nullptr;
};

struct VoiceLoopStats {
    int turns = 0;
    int skipped = 0;
    int barge_ins = 0;
    int errors = 0;
};

// Runs until stop, max_turns, or the end of a file input (then waits for playback to finish).
// Hook failures are reported through on_event(Error) and the loop continues, except Cancelled,
// which ends it. Without barge-in the loop is half-duplex: audio captured while a reply is being
// produced or played is dropped, so the agent does not hear itself.
Result<VoiceLoopStats> run_voice_loop(audio::AudioInput &input, audio::AudioOutput &output, SpeechDetector &detector,
                                      const VoiceHooks &hooks, const VoiceLoopOptions &options);

// false for transcripts with no words, such as "", "[BLANK_AUDIO]" or "(music)".
bool is_meaningful_transcript(std::string_view transcript);
// Reply text made fit for TTS: tool-call / think markup, code blocks and markdown removed.
std::string speakable_text(std::string_view reply);

// -------------------------- settings --------------------------

struct VoiceSettings {
    std::string model;     // LLM: catalogue name or path (empty = config default)
    std::string engine;    // LLM engine flag
    std::string asr_model; // audio.cpp ASR GGUF, or a JSON "asr" object
    std::string tts_model; // audio.cpp TTS GGUF (e.g. pocket_tts), or a JSON "tts" object
    std::string vad_model; // audio.cpp VAD GGUF (firered_vad), or a JSON "vad" object; empty = energy gate
    std::string input_wav; // file backend input
    std::string output_dir;
    std::string workspace;
    std::string engine_options;
    std::string resume;
    int context_window = 0; // 0 = config
    int max_tokens = 0;
    int max_turns = 0;
    std::string voice; // TTS voice id (a preset name, e.g. alba); empty = the model default
    bool barge_in = false;
    bool auto_approve = false;
    bool verbose = false;
    bool inprocess = false;
};

Result<VoiceSettings> voice_settings_from(const CommandLine &cl);

// audiocpp engine_options for each component, from a GGUF model file (or passed through when the argument is already a
// JSON object). The VAD is streaming: the engine keeps the audio since the last reset (see docs/api/engines.md).
Result<std::string> audiocpp_asr_options(std::string_view file_or_json);
Result<std::string> audiocpp_tts_options(std::string_view file_or_json, std::string_view voice = {});
Result<std::string> audiocpp_vad_options(std::string_view file_or_json);

} // namespace agent::cli
