# 10 speech

One audio.cpp session with `asr.model`, `tts.model` and the `tts.voice` preset (`alba`): `agent::speak` writes a WAV, `agent::transcribe` reads it back. Needs
`-DAGENT_ENGINE_AUDIOCPP=ON` and models such as `agent model pull parakeet-tdt` and `agent model pull pocket-tts`. The smoke test runs the same
code on the mock engine.
