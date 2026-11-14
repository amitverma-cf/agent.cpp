# 10 speech

One audio.cpp session with `asr.model`, `tts.model` and the `tts.voice` preset (`alba`): `agent::speak` writes a WAV, `agent::transcribe` reads it back. Needs
`-DAGENT_ENGINE_AUDIOCPP=ON` and models such as `agent model pull audio-cpp/audio.cpp-gguf:Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf --name parakeet-tdt` and the Pocket TTS files (see [the voice guide](../../docs/guides/voice-assistant.md)). The smoke test runs the same
code on the mock engine.
