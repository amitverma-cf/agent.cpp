// The curated model registry: short names for models that work with the engines in this repo.
// `tested` means the entry was pulled with `agent model pull` and run end to end with agent.cpp
// (see local test notes); the others exist on Hugging Face at the given file and size, and use the same
// engine as a tested sibling.

#include <agent-cpp/hub.hpp>
#include <algorithm>

namespace agent::hub {

namespace {

constexpr RegistryPart kQwen05[] = {{"Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kQwen35_4[] = {{"unsloth/Qwen3.5-4B-MTP-GGUF:Q5_K_M", ""}};
constexpr RegistryPart kQwen3_17[] = {{"Qwen/Qwen3-1.7B-GGUF:Q8_0", ""}};
constexpr RegistryPart kQwen3_4[] = {{"Qwen/Qwen3-4B-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kLlama1[] = {{"bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kLlama3[] = {{"bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kGemma1[] = {{"ggml-org/gemma-3-1b-it-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kPhi4[] = {{"unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kSmolVlm[] = {{"ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0", ""}};
constexpr RegistryPart kGemma4Vl[] = {{"ggml-org/gemma-3-4b-it-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kQwenVl3[] = {{"ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kGlmOcr[] = {{"ggml-org/GLM-OCR-GGUF:Q8_0", ""}};
constexpr RegistryPart kVoxtral[] = {{"ggml-org/Voxtral-Mini-3B-2507-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kBge[] = {{"ggml-org/bge-small-en-v1.5-Q8_0-GGUF", ""}};
constexpr RegistryPart kNomic[] = {{"nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0", ""}};
constexpr RegistryPart kEmbGemma[] = {{"ggml-org/embeddinggemma-300M-qat-q4_0-GGUF:q4_0", ""}};
constexpr RegistryPart kEmbGemma2[] = {{"ggml-org/embeddinggemma-2-GGUF:Q8_0", ""}};
constexpr RegistryPart kJina[] = {{"gpustack/jina-reranker-v1-tiny-en-GGUF:Q8_0", ""}};
constexpr RegistryPart kBgeRerank[] = {{"gpustack/bge-reranker-v2-m3-GGUF:Q4_K_M", ""}};
constexpr RegistryPart kSdxs[] = {{"concedo/sdxs-512-tinySDdistilled-GGUF:Q8_0", ""}};
constexpr RegistryPart kSdTurbo[] = {{"Green-Sky/SD-Turbo-GGUF:sd_turbo-f16-q8_0.gguf", ""}};
constexpr RegistryPart kPocketTts[] = {{"audio-cpp/audio.cpp-gguf:PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf", ""},
                                       {"audio-cpp/audio.cpp-gguf:PocketTTS-GGUF/english/embeddings/alba.safetensors", ""}};
constexpr RegistryPart kSupertonic[] = {{"audio-cpp/audio.cpp-gguf:Supertonic-3-GGUF/supertonic-3-q8_0.gguf", ""}};
constexpr RegistryPart kParakeet[] = {{"audio-cpp/audio.cpp-gguf:Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf", ""}};
constexpr RegistryPart kMoonshine[] = {{"audio-cpp/audio.cpp-gguf:Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf", ""}};
constexpr RegistryPart kCrisperWhisper[] = {{"audio-cpp/audio.cpp-gguf:CrisperWhisper2.0-GGUF/crisperwhisper-2-large-q4_k.gguf", ""}};
constexpr RegistryPart kFireRedVad[] = {{"audio-cpp/audio.cpp-gguf:FireRedVAD-GGUF/firered-vad-f32.gguf", ""}};
constexpr RegistryPart kNemotronDiar[] = {{"audio-cpp/Nemotron-3-Diarization-GGUF:nemotron-3-diarization-q8_0.gguf", ""}};
constexpr RegistryPart kMiniLm[] = {{"Xenova/all-MiniLM-L6-v2:onnx/model_quantized.onnx", ""}};
constexpr RegistryPart kTrellis[] = {{"LocalAI-io/TRELLIS.2-4B-GGUF:ss_flow_f16.gguf", ""},
                                     {"LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF:dino_f16.gguf", ""},
                                     {"LocalAI-io/TRELLIS-image-large-GGUF:ss_dec_f16.gguf", ""}};
constexpr RegistryPart kJulia[] = {{"ggml-org/Julia-1-GGUF:Q8_0", ""}};

constexpr RegistryEntry kRegistry[] = {
    // alias, task, engine, parts, size MiB, engine options, notes, tested
    {"qwen3.5-4b", "text_gen", "llama", kQwen35_4, 3704, "",
     "recommended all-round model for 8 GB: graph suite 7/7, function calling 88% on BFCL, image input (projector pulled automatically), "
     "prompt-lookup speculation 1.9x; has an MTP head",
     true},
    {"qwen2.5-0.5b", "text_gen", "llama", kQwen05, 469, "", "chat, tool calls, decide; the default test model", true},
    {"qwen3-1.7b", "text_gen", "llama", kQwen3_17, 1749, "", "thinking model; Q8_0", true},
    {"qwen3-4b", "text_gen", "llama", kQwen3_4, 2382, "", "best small tool-calling model that fits 8 GB", true},
    {"llama3.2-1b", "text_gen", "llama", kLlama1, 770, "", "", true},
    {"llama3.2-3b", "text_gen", "llama", kLlama3, 1926, "", "", true},
    {"gemma3-1b", "text_gen", "llama", kGemma1, 769, "", "", true},
    {"phi4-mini", "text_gen", "llama", kPhi4, 2376, "", "3.8B", true},
    {"smolvlm-256m", "vision", "llama", kSmolVlm, 267, "", "image input; the projector is pulled and used automatically", true},
    {"gemma3-4b-vl", "vision", "llama", kGemma4Vl, 2374, "", "image input + projector", true},
    {"qwen2.5-vl-3b", "vision", "llama", kQwenVl3, 1840, "", "image input + projector", true},
    {"glm-ocr", "ocr", "llama", kGlmOcr, 1368, "", "OCR; licence unknown on the hub, pass --allow-restricted", true},
    {"voxtral-mini-3b", "audio_in", "llama", kVoxtral, 2358, "",
     "audio input + projector; the audio is encoded, but its transcription was not convincing in testing", false},
    {"bge-small", "embed", "llama", kBge, 35, "{\"mode\":\"embed\"}", "384 dims", true},
    {"nomic-embed", "embed", "llama", kNomic, 139, "{\"mode\":\"embed\"}", "768 dims", true},
    {"embeddinggemma", "embed", "llama", kEmbGemma, 265, "{\"mode\":\"embed\"}", "", true},
    {"embeddinggemma-2", "embed", "llama", kEmbGemma2, 825, "{\"mode\":\"embed\"}",
     "text, images and audio in one 768-dimensional space (Matryoshka 512/256/128); the projector is pulled with it. Text wants the prefix "
     "\"task: search result | query: \" (or \"title: none | text: \" for documents), images and audio take none",
     true},
    {"jina-reranker-tiny", "rerank", "llama", kJina, 35, "{\"mode\":\"rerank\"}", "ranked the right passage first in testing", true},
    {"bge-reranker-v2-m3", "rerank", "llama", kBgeRerank, 418, "{\"mode\":\"rerank\"}", "multilingual", true},
    {"sdxs-512", "image_gen", "sdcpp", kSdxs, 0, "", "one-step 512px; use {\"steps\":1,\"cfg_scale\":1}", true},
    {"sd-turbo", "image_gen", "sdcpp", kSdTurbo, 1930, "", "", true},
    {"parakeet-tdt", "asr", "audiocpp", kParakeet, 874, "", "25 European languages, word timestamps; ~18x realtime on 4 CPU cores", true},
    {"moonshine-tiny", "asr", "audiocpp", kMoonshine, 58, "", "English, ~100x realtime on 4 CPU cores", true},
    {"crisperwhisper-large", "asr", "audiocpp", kCrisperWhisper, 937, "",
     "Whisper large based, 99 languages, verbatim; slower than realtime on 4 CPU cores", true},
    {"pocket-tts", "tts", "audiocpp", kPocketTts, 120, "",
     "small English voice model; speaks with the alba voice unless tts.voice names another, more voices on its model card", true},
    {"supertonic-3", "tts", "audiocpp", kSupertonic, 434, "", "30+ languages, 44.1 kHz; the default speech model, no extra dependencies",
     true},
    {"firered-vad", "vad", "audiocpp", kFireRedVad, 3, "", "voice activity detection", true},
    {"nemotron-diar", "diarize", "audiocpp", kNemotronDiar, 102, "",
     "speaker diarization; separated two different voices (speaker 0, 1, 0) with the right turn times in testing", true},
    {"minilm-onnx", "generic", "onnx", kMiniLm, 22, "", "feed input_ids / attention_mask / token_type_ids tensors", true},
    {"trellis-coarse", "mesh_gen", "trellis", kTrellis, 3300, "", "image to 3D, coarse path; ~6 min and 3.5 GB RAM on 4 CPU cores", true},
    {"julia-1", "decide", "llama", kJulia, 160, "", "native System One decision model; runs in process through llama.cpp's own handler",
     false},
};

} // namespace

std::span<const RegistryEntry> model_registry() { return kRegistry; }

const RegistryEntry *find_registry_entry(std::string_view alias) {
    const auto it = std::find_if(std::begin(kRegistry), std::end(kRegistry), [&](const RegistryEntry &e) { return e.alias == alias; });
    return it == std::end(kRegistry) ? nullptr : &*it;
}

} // namespace agent::hub
