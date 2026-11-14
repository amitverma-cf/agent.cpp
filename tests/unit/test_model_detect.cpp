#include "model_detect.hpp"

#include <catch_amalgamated.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>

using namespace agent::cli;
namespace fs = std::filesystem;

namespace {

void put_u32(std::ofstream &o, uint32_t v) { o.write(reinterpret_cast<const char *>(&v), 4); }
void put_u64(std::ofstream &o, uint64_t v) { o.write(reinterpret_cast<const char *>(&v), 8); }
void put_str(std::ofstream &o, const std::string &s) {
    put_u64(o, s.size());
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// A GGUF header with general.architecture, and optionally "<arch>.pooling_type" (an embedding model) and a string array in between.
std::string write_gguf(const char *name, const std::string &arch, bool pooling) {
    const fs::path p = fs::temp_directory_path() / name;
    std::ofstream o(p, std::ios::binary);
    o.write("GGUF", 4);
    put_u32(o, 3);
    put_u64(o, 0);               // tensors
    put_u64(o, pooling ? 3 : 2); // metadata entries
    put_str(o, "general.architecture");
    put_u32(o, 8);
    put_str(o, arch);
    put_str(o, "tokenizer.ggml.tokens"); // an array of strings is skipped
    put_u32(o, 9);
    put_u32(o, 8);
    put_u64(o, 2);
    put_str(o, "a");
    put_str(o, "bb");
    if (pooling) {
        put_str(o, arch + ".pooling_type");
        put_u32(o, 4);
        put_u32(o, 1);
    }
    return p.string();
}

} // namespace

TEST_CASE("detect_model: the file and its name say which engine runs it", "[cli][detect]") {
    SECTION("a language model") {
        const auto k = detect_model(write_gguf("detect_llm.gguf", "qwen3", false), "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", false);
        CHECK(k.engine == "llama");
        CHECK(k.task == "text_gen");
        CHECK(k.architecture == "qwen3");
    }
    SECTION("an embedding model is told by its pooling type") {
        const auto k = detect_model(write_gguf("detect_emb.gguf", "bert", true), "x/y-GGUF", "y.gguf", false);
        CHECK(k.engine == "llama");
        CHECK(k.task == "embed");
        CHECK(k.engine_options == "{\"mode\":\"embed\"}");
    }
    SECTION("a reranker") {
        const auto k = detect_model(write_gguf("detect_rr.gguf", "bert", false), "gpustack/bge-reranker-v2-m3-GGUF", "r.gguf", false);
        CHECK(k.task == "rerank");
    }
    SECTION("speech, image and 3D models are told by their names") {
        CHECK(
            detect_model(write_gguf("detect_a.gguf", "x", false), "audio-cpp/audio.cpp-gguf", "Parakeet-TDT/parakeet.gguf", false).engine ==
            "audiocpp");
        CHECK(detect_model(write_gguf("detect_b.gguf", "x", false), "Green-Sky/SD-Turbo-GGUF", "sd_turbo-f16-q8_0.gguf", false).engine ==
              "sdcpp");
        CHECK(detect_model(write_gguf("detect_c.gguf", "x", false), "LocalAI-io/TRELLIS.2-4B-GGUF", "ss_flow_f16.gguf", false).engine ==
              "trellis");
    }
    SECTION("ONNX and checkpoint files are told by their extension") {
        CHECK(detect_model("model.onnx", "Xenova/all-MiniLM-L6-v2", "onnx/model.onnx", false).engine == "onnx");
        CHECK(detect_model("sd15.safetensors", "o/r", "sd15.safetensors", false).engine == "sdcpp");
    }
    SECTION("a file that is none of these") { CHECK(detect_model("notes.txt", "o/r", "notes.txt", false).engine.empty()); }
}
