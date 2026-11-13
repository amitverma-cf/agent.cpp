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
submodule in `vendor/audio.cpp`. It is built by default once that submodule is checked out, so a clone with
`--recurse-submodules` already has it. To fetch it into an existing clone and force the engine on:

```sh
git submodule update --init vendor/audio.cpp
cmake -B build -G Ninja -DAGENT_ENGINE_AUDIOCPP=ON
cmake --build build
./build/agent setup --yes          # records the llama and audiocpp plugins it finds
```

Microphone and speaker support (miniaudio) is built into the CLI by default (`AGENT_CLI_AUDIO_DEVICE`). On Linux it uses
ALSA, PulseAudio or JACK, whichever is installed; nothing extra is needed at build time.

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

`agent model pull` stores each file in the models folder of the CLI's home (`<home>/models/`, see
[workspace.md](../workspace.md)). The commands below write that location as `<home>/models/...`; use the path the
download reports, or look for the `.gguf` file under that folder. Pass the GGUF file of each speech model to
`--asr-model`, `--tts-model` and `--vad-model`. See [api/engines.md](../api/engines.md#the-audiocpp-engine) for the
model options.

Other speech-to-text aliases are `moonshine-tiny` and `crisperwhisper-large`; `--asr-model` takes any of their GGUF files.

## 3. Talk

```sh
agent voice --model qwen --asr-model <home>/models/parakeet-tdt-0.6b-v3-q8_0.gguf \
            --tts-model <home>/models/pocket-tts-english-q8_0.gguf \
            --vad-model <home>/models/firered-vad-f32.gguf
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

`--input-wav` reads the "microphone" from a WAV file, and `--output-dir` receives the WAV file of each reply. That is
how the end-to-end test runs, and it is handy for scripted checks:

```sh
agent voice --model qwen --asr-model <home>/models/parakeet-tdt-0.6b-v3-q8_0.gguf --tts-model <home>/models/pocket-tts-english-q8_0.gguf \
            --vad-model <home>/models/firered-vad-f32.gguf --input-wav question.wav --output-dir replies --max-turns 1
```

With `--barge-in`, the file backend plays each reply in input time, so speech in the file that overlaps a
reply cuts that reply's WAV short, as a real interruption would.

## 5. From C++

The speech parts are ordinary sessions on the audiocpp engine. Give each model file by its role with `open_session`
(see [getting-started-library.md](getting-started-library.md)); the dotted roles are the audiocpp ones:

```cpp
#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    auto speech = agent::open_session(agent::engines::audiocpp, {
        {"asr.model", "models/parakeet-tdt-0.6b-v3-q8_0.gguf"},
        {"tts.model", "models/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf"},
        {"tts.voice", "alba"}, // a voice preset that ships with Pocket TTS
    });
    if (!speech.ok) return std::fprintf(stderr, "open: %s\n", speech.error.message.c_str()), 1;

    auto spoken = agent::speak(speech.value, "The quick brown fox jumps over the lazy dog.");  // files[0] is a WAV
    if (!spoken.ok || spoken.value.files.empty()) return std::fprintf(stderr, "speak: %s\n", spoken.error.message.c_str()), 1;

    auto heard = agent::transcribe(speech.value, spoken.value.files[0]);
    if (!heard.ok) return std::fprintf(stderr, "transcribe: %s\n", heard.error.message.c_str()), 1;
    std::printf("transcribed: %s\n", heard.value.text.c_str());
    return 0;
}
```

This is the same program as [`examples/10-speech`](../../examples/10-speech/speech.cpp), which takes no arguments and runs
when the models are in `models/` (see [examples/README.md](../../examples/README.md)).

The voice loop adds a voice activity model (`vad.model`) and the llama chat session:

- the VAD model returns the start and end of each utterance as JSON (see the audiocpp section of
  [api/engines.md](../api/engines.md#the-audiocpp-engine) for the streaming option);
- ASR returns the transcript as the message text;
- TTS returns a WAV file in `files[0]`;
- the LLM turn is an ordinary `run_turn` on a llama session.

Audio goes into a direct `infer()` call as a `TensorView` (F32, 16 kHz, mono) on the user message. The responses of an `infer()`
point into the session's arena, so reset `session.arena` before each direct `infer()` call.

A graph file (`agent run`) cannot express the loop yet: a graph runs on one engine session, and the CLI does not register
ASR or TTS function nodes.

## Status

The file backend and the loop logic are covered by tests. The microphone and speaker path has not been tested on real
hardware yet.
