#pragma once

// llama.cpp's own engine, in process. The scheduler behind llama-server and llama-cli (slots, prompt cache, checkpoints,
// speculative decoding including multi-token prediction, tool-call grammars, image and audio input) is a library inside
// llama.cpp (tools/server). The plugin calls its request handlers directly with in-memory requests: no socket is opened and
// nothing listens.
//
// The plugin uses it for models whose GGUF has an MTP head. The file says so: upstream looks for the tensor
// blk.<last>.nextn.eh_proj.weight, so no model name is involved.

#include <agent-cpp/engine_abi.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace upstream {

struct Settings {
    std::string model;
    std::string mmproj;
    int context = 4096; // tokens per sequence
    int parallel = 1;   // sequences served at once
    int threads = 0;    // 0 = llama.cpp's choice
    int gpu_layers = -1;
    std::vector<std::pair<std::string, float>> loras;
    std::string kv_type;  // "", "f16", "q8_0" or "q4_0"
    bool mtp = true;      // use the MTP head when the file has one
    int draft_max = 3;    // tokens drafted per step
};

struct ToolCall {
    std::string id, name, arguments;
};

struct Result {
    std::string content, reasoning;
    std::vector<ToolCall> calls;
    int prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
    double prompt_ms = 0, predicted_ms = 0;
    int drafted = 0, accepted = 0; // speculative decoding, when the answer reports it
    std::string finish_reason;
};

// True when the GGUF has a multi-token-prediction head (llama.cpp's own check).
bool gguf_has_mtp(const std::string &path);

class Engine {
  public:
    // Loads the model and starts the scheduler's loop. Returns null with `error` set on failure.
    static std::unique_ptr<Engine> create(const Settings &settings, std::string &error);
    ~Engine();

    // body is an OpenAI chat completion request. on_text gets each piece of the reply as it arrives (and turns streaming on).
    bool chat(const std::string &body, const volatile int32_t *cancel, const std::function<void(const std::string &)> &on_text, Result &out,
              std::string &error);

    bool mtp_active() const;
    int count_tokens(const std::string &text) const; // with the model's own tokenizer

  private:
    Engine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// An OpenAI chat completion body from an ABI request: messages with their text, tool calls and tool results, images and audio as
// data URLs, and the tools in `tools_json`. max_tokens <= 0 and temperature < 0 leave the server's defaults.
std::string chat_body_from_request(const agent_request *req, int max_tokens, float temperature, bool stream, std::string &error);

} // namespace upstream
