# Guide: a local voice assistant

This guide sets up `agent voice`: you speak, a local model answers out loud. Everything runs on your
machine: a VAD model finds where you speak, an ASR model writes it down, a llama.cpp chat model
answers (with the same tools and permission prompts as `agent chat`), and a TTS model reads the answer out.
The speech models all run in the audio.cpp engine.

```
microphone ──16 kHz──> VAD (audiocpp) ──utterance──> ASR (audiocpp) ──text──> LLM turn (llama) ──text──> TTS (audiocpp) ──> speaker
     ^                                                                                                                   │
     └──────────────────── half-duplex: the mic is ignored while the reply plays (or --barge-in) ────────────────────────┘
```

## 1. Build with the speech engine

The `audiocpp` engine wraps [audio.cpp](https://github.com/0xShug0/audio.cpp) (Apache-2.0), which is a git
submodule in `vendor/audio.cpp`:

```sh
git submodule update --init vendor/audio.cpp
cmake -B build -G Ninja -DAGENT_ENGINE_AUDIOCPP=ON
cmake --build build
./build/agent setup --yes          # registers the llama and audiocpp plugins it finds
```

Microphone and speaker support (miniaudio) is built into the CLI by default. On Linux it uses ALSA,
PulseAudio or JACK, whichever is installed; nothing extra is needed at build time.

## 2. Download the models

The speech models are single GGUF files from the Hugging Face repository `audio-cpp/audio.cpp-gguf`. The hub
aliases are `parakeet-tdt` (speech to text), `pocket-tts` (text to speech, English) and `firered-vad` (voice activity
detection). Parakeet TDT is CC-BY-4.0 (attribution required). For text to speech in 30+ languages, use `supertonic-3`.

```sh
agent model pull parakeet-tdt        # speech to text
agent model pull pocket-tts          # text to speech
agent model pull firered-vad         # voice activity detection
agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --name qwen   # the chat model
```

Pass the GGUF file of each speech model to `--asr-model`, `--tts-model` and `--vad-model`. See
[api/engines.md](../api/engines.md#the-audiocpp-engine) for the model options.

Other speech-to-text aliases are `moonshine-tiny` and `crisperwhisper-large`; `--asr-model` takes any of their GGUF files.

## 3. Talk

```sh
agent voice --model qwen --asr-model ~/models/parakeet-tdt.gguf \
            --tts-model ~/models/pocket-tts.gguf --vad-model ~/models/firered-vad.gguf
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
  tool-call markup are removed before speaking, and numbers and `+ - = % / x` are spelled out.
- **Other voices:** `--voice ID` picks a voice preset, if the TTS model has one.
- Engine logs go to `<workspace>/logs/engine.log`; `--verbose` prints them.

## 4. Without a sound card

`--input-wav` reads the "microphone" from a WAV file and `--output-dir` writes every reply as
`reply-NNN.wav`. That is how the end-to-end test runs, and it is handy for scripted checks:

```sh
agent voice --model qwen --asr-model ~/models/parakeet-tdt.gguf --tts-model ~/models/pocket-tts.gguf \
            --vad-model ~/models/firered-vad.gguf --input-wav question.wav --output-dir replies --max-turns 1
```

With `--barge-in`, the file backend plays each reply in input time, so speech in the file that overlaps a
reply cuts that reply's WAV short, as a real interruption would.

## 5. From C++

The loop is built from public pieces you can use directly:

- one `Session` per component (`Config::engine = "audiocpp"`, `engine_options` = `{"asr": {...}}`,
  `{"tts": {...}}` or `{"vad": {..., "streaming": true}}`);
- audio goes in as a `TensorView` (F32, 16 kHz mono) on the user message of `infer()`; ASR returns the
  transcript as the message text, VAD returns `[{"start":..,"end":..}]` JSON (see the audiocpp section of
  [api/engines.md](../api/engines.md#the-audiocpp-engine));
- TTS returns a WAV file in `message.media`;
- the LLM turn is an ordinary `run_turn` on a llama session.

Reset `session.arena` before each direct `infer()` call; responses live in the arena.

A graph file (`agent run`) cannot express the loop yet: a graph runs on one engine session, and the CLI
does not register ASR / TTS function nodes.

## Status

The file backend and the loop logic are covered by tests. The microphone / speaker path has not been tested on
real hardware yet.
