#!/bin/sh
# Downloads the models the examples use into ./models (about 2.7 GB). Run it from the folder you run the examples in.
# Any one example needs only the files named at the top of its source.
set -e
mkdir -p models/PocketTTS-GGUF/english/embeddings
get() { [ -s "$2" ] || curl -fL --progress-bar -o "$2" "$1"; }

HF=https://huggingface.co
get $HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf models/qwen2.5-0.5b-instruct-q4_k_m.gguf
get $HF/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf models/bge-small-en-v1.5-q8_0.gguf
get $HF/concedo/sdxs-512-tinySDdistilled-GGUF/resolve/main/sdxs-512-tinySDdistilled_Q8_0.gguf models/sdxs-512-tinySDdistilled_Q8_0.gguf
get $HF/audio-cpp/audio.cpp-gguf/resolve/main/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf models/parakeet-tdt-0.6b-v3-q8_0.gguf
get $HF/audio-cpp/audio.cpp-gguf/resolve/main/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf models/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf
get $HF/audio-cpp/audio.cpp-gguf/resolve/main/PocketTTS-GGUF/english/embeddings/alba.safetensors models/PocketTTS-GGUF/english/embeddings/alba.safetensors
echo "models are in ./models"
