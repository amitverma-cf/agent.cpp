#pragma once

// Which engine a downloaded model needs. `agent model pull` takes any Hugging Face link (a repository or one file), looks at the file
// that came down and puts it in the folder of the engine that can run it (models/llama, models/sdcpp, models/audiocpp, models/onnx,
// models/trellis). There is no list of suggested models: the file itself, and its name, decide.

#include <string>
#include <string_view>

namespace agent::cli {

struct ModelKind {
    std::string engine;         // llama, sdcpp, audiocpp, onnx, trellis; empty when it cannot be told
    std::string task;           // text_gen, embed, rerank, image_gen, asr, tts, vad, diarize, mesh_gen, generic; empty when unknown
    std::string engine_options; // JSON the engine needs to serve this model ("{\"mode\":\"embed\"}"), or empty
    std::string architecture;   // general.architecture of a GGUF file
    std::string why;            // one line for the person: what the decision was based on
};

// Reads the GGUF header (architecture and pooling type) and looks at the file extension, `file` and `repo` names.
// `has_projector`: the pull found a multimodal projector next to the model (a vision or audio input model).
ModelKind detect_model(const std::string &path, std::string_view repo, std::string_view file, bool has_projector);

// "general.architecture" of a GGUF file; empty when the file is not GGUF or has none. `pooling` is set when the model has a pooling type
// (an embedding model).
std::string read_gguf_architecture(const std::string &path, bool *pooling = nullptr);

} // namespace agent::cli
