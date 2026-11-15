#pragma once

// What models are installed and what each can do, for the chat's Settings, its system prompt and the agent_run tool.

#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

struct InstalledModel {
    std::string name;       // catalogue name, exactly as typed after --model
    std::string capability; // chat, vision, ocr, asr, tts, vad, diarize, image, video, mesh, embed, rerank or other
    std::string engine;     // llama, audiocpp, sdcpp, onnx, trellis, api
    std::string task;       // the `agent task` capability (text_gen, tts, ...) or empty
    bool remote = false;    // answers through a remote service (the api engine)
};

// Every catalogue entry with its capability, in catalogue order.
std::vector<InstalledModel> installed_models(const std::string &catalog_path);

// Models that can hold a conversation (chat, vision, remote): the choices of Settings > Model.
bool can_chat(const InstalledModel &m);

// The capability word a task needs: "image_gen" -> "image", "text_gen" -> "chat"; other tasks keep their name.
std::string capability_for_task(std::string_view task);

// The paragraph of the system prompt that lists the installed models by capability, with exact names, and says how to get others.
std::string installed_models_prompt(const std::vector<InstalledModel> &models);

// `--model NAME` of an `agent task TASK` call: NAME itself when it is installed or a file, otherwise the installed model for that task.
// `found` is false when NAME is not installed and no installed model fits the task. `requested` empty: returns "" (found = true).
std::string model_for_task(const std::vector<InstalledModel> &installed, std::string_view task, std::string_view requested, bool &found);

// Error text for a person who types plain English: command hints ("run `agent setup`", "agent model pull ...") are dropped.
std::string beginner_error(std::string text);

} // namespace agent::cli
