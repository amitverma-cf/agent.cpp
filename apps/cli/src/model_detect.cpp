#include "model_detect.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>

namespace agent::cli {

namespace {

std::string lower_copy(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool has_any(const std::string &text, std::initializer_list<std::string_view> words) {
    return std::any_of(words.begin(), words.end(), [&](std::string_view w) { return text.find(w) != std::string::npos; });
}

bool read_u32(std::ifstream &in, uint32_t &v) { return static_cast<bool>(in.read(reinterpret_cast<char *>(&v), 4)); }
bool read_u64(std::ifstream &in, uint64_t &v) { return static_cast<bool>(in.read(reinterpret_cast<char *>(&v), 8)); }

bool read_string(std::ifstream &in, std::string &out, size_t max_keep) {
    uint64_t len = 0;
    if (!read_u64(in, len) || len > (1ull << 32)) return false;
    if (len > max_keep) {
        in.seekg(static_cast<std::streamoff>(len), std::ios::cur);
        out.clear();
        return static_cast<bool>(in);
    }
    out.resize(static_cast<size_t>(len));
    return len == 0 || static_cast<bool>(in.read(out.data(), static_cast<std::streamsize>(len)));
}

size_t fixed_size(uint32_t type) {
    switch (type) {
    case 0:
    case 1:
    case 7: return 1;
    case 2:
    case 3: return 2;
    case 4:
    case 5:
    case 6: return 4;
    case 10:
    case 11:
    case 12: return 8;
    default: return 0;
    }
}

// Skips one value of a GGUF metadata type (strings and arrays of strings are walked, the numeric kinds are skipped in one seek).
bool skip_value(std::ifstream &in, uint32_t type) {
    if (const size_t n = fixed_size(type)) {
        in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
        return static_cast<bool>(in);
    }
    if (type == 8) {
        std::string ignored;
        return read_string(in, ignored, 0);
    }
    if (type == 9) {
        uint32_t item = 0;
        uint64_t count = 0;
        if (!read_u32(in, item) || !read_u64(in, count)) return false;
        if (const size_t n = fixed_size(item)) {
            in.seekg(static_cast<std::streamoff>(n * count), std::ios::cur);
            return static_cast<bool>(in);
        }
        for (uint64_t i = 0; i < count; ++i)
            if (!skip_value(in, item)) return false;
        return true;
    }
    return false;
}

bool ends_with(const std::string &s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

std::string read_gguf_architecture(const std::string &path, bool *pooling) {
    if (pooling) *pooling = false;
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    if (!in || !in.read(magic, 4) || std::string_view(magic, 4) != "GGUF") return {};
    uint32_t version = 0;
    uint64_t tensors = 0, kvs = 0;
    if (!read_u32(in, version) || !read_u64(in, tensors) || !read_u64(in, kvs) || kvs > (1u << 20)) return {};
    std::string architecture;
    for (uint64_t i = 0; i < kvs; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_string(in, key, 256) || !read_u32(in, type)) break;
        if (key == "general.architecture" && type == 8) {
            if (!read_string(in, architecture, 256)) break;
            continue;
        }
        if (ends_with(key, ".pooling_type") && pooling) *pooling = true;
        if (!skip_value(in, type)) break;
        // The architecture comes first in files written by llama.cpp; the pooling type is one of the first few keys after it.
        if (!architecture.empty() && i > 64) break;
    }
    return architecture;
}

ModelKind detect_model(const std::string &path, std::string_view repo, std::string_view file, bool has_projector) {
    ModelKind kind;
    const std::string name = lower_copy(std::string(repo) + " " + std::string(file));

    // Not GGUF: the extension says it.
    if (ends_with(lower_copy(path), ".onnx")) {
        kind = {"onnx", "generic", "", "", "an ONNX file"};
        return kind;
    }
    if (ends_with(lower_copy(path), ".safetensors") || ends_with(lower_copy(path), ".ckpt")) {
        kind = {"sdcpp", "image_gen", "", "", "a checkpoint file (stable-diffusion.cpp reads these)"};
        return kind;
    }

    bool pooling = false;
    kind.architecture = read_gguf_architecture(path, &pooling);
    const std::string arch = lower_copy(kind.architecture);

    if (has_any(name, {"trellis"})) {
        kind.engine = "trellis";
        kind.task = "mesh_gen";
        kind.why = "TRELLIS.2 model (image to 3D)";
        return kind;
    }
    if (name.find("audio.cpp") != std::string::npos || name.find("audio-cpp") != std::string::npos ||
        has_any(name, {"parakeet", "moonshine", "whisper", "firered", "nemotron", "supertonic", "pocket-tts", "pocket_tts", "kokoro",
                       "diarization", "sortformer", "-vad", "_vad", "/vad"})) {
        kind.engine = "audiocpp";
        kind.task = has_any(name, {"diariz", "sortformer"})                    ? "diarize"
                    : has_any(name, {"vad"})                                   ? "vad"
                    : has_any(name, {"tts", "supertonic", "kokoro", "pocket"}) ? "tts"
                                                                               : "asr";
        kind.why = "a speech model (audio.cpp)";
        return kind;
    }
    if (has_any(arch, {"sd1", "sd2", "sdxl", "sd3", "flux", "wan", "qwen_image", "z_image", "chroma"}) ||
        has_any(name, {"stable-diffusion", "stable_diffusion", "sdxl", "sd-turbo", "sdxs", "sd_turbo", "flux", "-sd3", "sd1.5", "wan2"})) {
        kind.engine = "sdcpp";
        kind.task = has_any(name, {"wan"}) ? "video_gen" : "image_gen";
        kind.why = "an image generation model (stable-diffusion.cpp)";
        return kind;
    }
    if (!kind.architecture.empty() || ends_with(lower_copy(path), ".gguf")) {
        kind.engine = "llama";
        if (has_any(name, {"rerank"})) {
            kind.task = "rerank";
            kind.engine_options = "{\"mode\":\"rerank\"}";
            kind.why = "a reranker (llama.cpp)";
        } else if (pooling || has_any(arch, {"bert", "nomic-bert"}) || has_any(name, {"embed"})) {
            kind.task = "embed";
            kind.engine_options = "{\"mode\":\"embed\"}";
            kind.why = "an embedding model (llama.cpp)";
        } else if (has_any(name, {"ocr"})) {
            kind.task = "ocr";
            kind.why = "an OCR model (llama.cpp)";
        } else {
            kind.task = "text_gen";
            kind.why = has_projector ? "a language model with image or audio input (llama.cpp)" : "a language model (llama.cpp)";
        }
        return kind;
    }
    kind.why = "could not tell which engine runs this file; pass --engine NAME";
    return kind;
}

} // namespace agent::cli
