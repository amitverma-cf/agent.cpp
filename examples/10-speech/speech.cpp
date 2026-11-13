// Speak a sentence, then transcribe the audio back: text to speech and speech recognition in one audio.cpp session.
// Needs models/parakeet-tdt-0.6b-v3-q8_0.gguf and models/PocketTTS-GGUF/english/ (examples/download-models.sh) and the audiocpp engine.

#include <agent-cpp/agent.hpp>
#include <cstdio>

int main() {
    agent::AgentRuntime runtime;

    auto session =
        agent::open_session(agent::engines::audiocpp, {{"asr.model", "models/parakeet-tdt-0.6b-v3-q8_0.gguf"},
                                                       {"tts.model", "models/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf"},
                                                       {"tts.voice", "alba"}}); // a voice preset that ships with Pocket TTS
    if (!session.ok) return std::fprintf(stderr, "open: %s\n", session.error.message.c_str()), 1;

    auto spoken = agent::speak(session.value, "The quick brown fox jumps over the lazy dog.");
    if (!spoken.ok || spoken.value.files.empty()) return std::fprintf(stderr, "speak: %s\n", spoken.error.message.c_str()), 1;
    std::printf("spoken:      %s\n", spoken.value.files[0].c_str());

    auto heard = agent::transcribe(session.value, spoken.value.files[0]);
    if (!heard.ok) return std::fprintf(stderr, "transcribe: %s\n", heard.error.message.c_str()), 1;
    std::printf("transcribed: %s\n", heard.value.text.c_str());
    return 0;
}
