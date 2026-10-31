# Guide: a local voice assistant

This guide sets up `agent voice`: you speak, a local model answers out loud. Everything runs on your
machine: Silero VAD finds where you speak, Whisper (or Parakeet) writes it down, a llama.cpp chat model
answers (with the same tools and permission prompts as `agent chat`), and Kokoro reads the answer out.

```
microphone ──16 kHz──> VAD (sherpa) ──utterance──> ASR (sherpa) ──text──> LLM turn (llama) ──text──> TTS (sherpa) ──> speaker
     ^                                                                                                                   │
     └──────────────────── half-duplex: the mic is ignored while the reply plays (or --barge-in) ────────────────────────┘
```

## 1. Build with the speech engine

The `sherpa` engine wraps the prebuilt sherpa-onnx C API. Download the shared release for your platform from
<https://github.com/k2-fsa/sherpa-onnx/releases> (tested: `sherpa-onnx-v2.0.1-linux-x64-shared.tar.bz2`) and
point CMake at it:

```sh
cmake -B build -G Ninja -DAGENT_ENGINE_SHERPA=ON -DSHERPA_ONNX_ROOT=/path/to/sherpa-onnx-v2.0.1-linux-x64-shared
cmake --build build
./build/agent setup --yes          # registers the llama and sherpa plugins it finds
```

Microphone and speaker support (miniaudio) is built into the CLI by default. On Linux it uses ALSA,
PulseAudio or JACK, whichever is installed; nothing extra is needed at build time.

## 2. Download the models

All of these have licences without usage restrictions.

```sh
mkdir -p ~/models && cd ~/models
base=https://github.com/k2-fsa/sherpa-onnx/releases/download
# speech to text: Whisper tiny.en (MIT), ~100 MB
curl -LO $base/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2 && tar xjf sherpa-onnx-whisper-tiny.en.tar.bz2
# text to speech: Kokoro (Apache-2.0), ~330 MB, plus the English lexicon
curl -LO $base/tts-models/kokoro-en-v0_19.tar.bz2 && tar xjf kokoro-en-v0_19.tar.bz2
curl -L -o kokoro-en-v0_19/lexicon-en-us.txt $base/tts-models/lexicon-en-us.txt   # if the archive has none
# voice activity detection: Silero VAD (MIT), 2 MB
curl -LO $base/asr-models/silero_vad.onnx
# the chat model
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --name qwen
```

For better recognition use a larger Whisper (`base.en`, `small.en`) or NVIDIA Parakeet TDT
(`sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`, CC-BY-4.0); `--asr-model` detects the family from the files.

## 3. Talk

```sh
agent voice --model qwen --asr-model ~/models/sherpa-onnx-whisper-tiny.en \
            --tts-model ~/models/kokoro-en-v0_19 --vad-model ~/models/silero_vad.onnx
```

```
[listening]
you: What is 2 + 2?
agent: 2 + 2 = 4
[spoken] 2.00 s of speech at 24000 Hz
[listening]
```

Ctrl-C quits (and cancels a model call that is running). The conversation is saved; `agent voice --resume
voice-20261031-161204` or `agent chat --resume voice-20261031-161204` continues it.

Tips:

- **Use the VAD.** Without `--vad-model` an energy gate decides what is speech. It works in a quiet room;
  fans, music or keyboard noise make it cut badly.
- **Headphones** let you use `--barge-in`: the assistant keeps listening while it talks and stops when you
  interrupt. With loudspeakers the microphone would hear the reply and interrupt it (there is no echo
  cancellation), so barge-in is off by default and the loop is half-duplex.
- **Short answers.** The voice system prompt asks for one to three plain sentences. Markdown, code blocks and
  tool-call markup are removed before speaking, and numbers and `+ - = % / x` are spelled out (Kokoro's
  lexicon has no digits; they would be skipped silently).
- **Other voices:** `--speaker-id N` picks a Kokoro voice (kokoro-en-v0_19 has 11).
- Engine logs go to `<workspace>/logs/engine.log`; `--verbose` prints them.

## 4. Without a sound card

`--input-wav` reads the "microphone" from a WAV file and `--output-dir` writes every reply as
`reply-NNN.wav`. That is how the end-to-end test runs, and it is handy for scripted checks:

```sh
agent voice --model qwen --asr-model ~/models/sherpa-onnx-whisper-tiny.en --tts-model ~/models/kokoro-en-v0_19 \
            --vad-model ~/models/silero_vad.onnx --input-wav question.wav --output-dir replies --max-turns 1
```

With `--barge-in`, the file backend plays each reply in input time, so speech in the file that overlaps a
reply cuts that reply's WAV short, as a real interruption would.

## 5. From C++

The loop is built from public pieces you can use directly:

- one `Session` per component (`Config::engine = "sherpa"`, `engine_options` = `{"asr": {...}}`,
  `{"tts": {...}}` or `{"vad": {..., "streaming": true}}`);
- audio goes in as a `TensorView` (F32, 16 kHz mono) on the user message of `infer()`; ASR returns the
  transcript as the message text, VAD returns `[{"start":..,"end":..}]` JSON (see the sherpa section of
  [api/engines.md](../api/engines.md#the-sherpa-engine));
- TTS returns a WAV file in `message.media`;
- the LLM turn is an ordinary `run_turn` on a llama session.

Reset `session.arena` before each direct `infer()` call; responses live in the arena.

A graph file (`agent run`) cannot express the loop yet: a graph runs on one engine session, and the CLI
does not register ASR / TTS function nodes.

## Status

The file backend, the loop logic and the speech engines are covered by tests (including an end-to-end
run: Kokoro asks "What is two plus two?", Whisper tiny.en hears "What is 2 + 2?", Qwen2.5-0.5B answers
"2 + 2 = 4", Kokoro speaks it). The microphone / speaker path has not been tested on real hardware yet.
