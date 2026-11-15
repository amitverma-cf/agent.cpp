#include "chat_models.hpp"

#include "model_detect.hpp"

#include <agent-cpp/hub.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace agent::cli {

namespace fs = std::filesystem;

std::string capability_for_task(std::string_view task) {
    if (task == "text_gen") return "chat";
    if (task == "image_gen" || task == "image_edit") return "image";
    if (task == "video_gen") return "video";
    if (task == "mesh_gen") return "mesh";
    return std::string(task);
}

bool can_chat(const InstalledModel &m) { return m.capability == "chat" || m.capability == "vision" || (m.remote && m.task != "embed"); }

std::vector<InstalledModel> installed_models(const std::string &catalog_path) {
    std::vector<InstalledModel> out;
    auto catalog = agent::hub::list_models(catalog_path);
    if (!catalog.ok) return out;
    for (const auto &e : catalog.value) {
        InstalledModel m;
        m.name = e.name;
        if (e.repo == "api") {
            m.remote = true;
            m.engine = "api";
            m.task = "text_gen";
            m.capability = "chat";
            out.push_back(std::move(m));
            continue;
        }
        const ModelKind kind = detect_model(e.path, e.repo, e.file, !e.mmproj.empty());
        m.engine = kind.engine;
        m.task = kind.task;
        m.capability = capability_for_task(kind.task);
        if (m.capability == "chat" && !e.mmproj.empty()) m.capability = "vision";
        if (m.capability.empty()) m.capability = "other";
        out.push_back(std::move(m));
    }
    return out;
}

std::string installed_models_prompt(const std::vector<InstalledModel> &models) {
    std::string text = " Installed models (use these exact names; they are the only models on this computer):";
    static constexpr const char *kOrder[] = {"chat", "vision", "ocr",    "asr", "tts",     "image", "video",
                                             "mesh", "embed",  "rerank", "vad", "diarize", "other"};
    bool any = false;
    for (const char *cap : kOrder) {
        std::string names;
        for (const InstalledModel &m : models)
            if (m.capability == cap) names += (names.empty() ? "" : ", ") + m.name + (m.remote ? " (remote)" : "");
        if (names.empty()) continue;
        any = true;
        const std::string c = cap;
        const std::string label = c == "vision"  ? "vision (reads images)"
                                  : c == "ocr"   ? "ocr (text in images)"
                                  : c == "asr"   ? "asr (speech to text)"
                                  : c == "tts"   ? "tts (text to speech)"
                                  : c == "image" ? "image (draw or edit pictures)"
                                                 : c;
        text += " " + label + ": " + names + ";";
    }
    if (!any) text += " none yet;";
    text +=
        " Another model can be downloaded with agent_run [\"model\",\"pull\",\"SPEC\"] (a Hugging Face link, owner/repo or "
        "owner/repo:file); agent_run [\"engine\",\"ls\"] lists the engines. Never invent a model name, a URL or a tool name: a name that "
        "is not in this list or in a tool result does not exist. Reading text in an image needs the ocr or vision model (agent_run task "
        "ocr); never state what an image says without a tool result that shows it.";
    return text;
}

std::string model_for_task(const std::vector<InstalledModel> &installed, std::string_view task, std::string_view requested, bool &found) {
    found = true;
    if (requested.empty()) return {};
    for (const InstalledModel &m : installed)
        if (m.name == requested) return std::string(requested);
    std::error_code ec;
    if (fs::exists(std::string(requested), ec)) return std::string(requested); // a path to a model file
    const std::string wanted = capability_for_task(task);
    for (const InstalledModel &m : installed)
        if (m.capability == wanted || (wanted == "chat" && m.capability == "vision")) return m.name;
    found = false;
    return {};
}

std::string beginner_error(std::string text) {
    // Drop `agent ...` code spans and the lines that only tell the reader to type a command.
    size_t at = 0;
    while ((at = text.find('`', at)) != std::string::npos) {
        const size_t end = text.find('`', at + 1);
        if (end == std::string::npos) break;
        if (text.compare(at + 1, 6, "agent ") == 0) {
            size_t from = at;
            while (from > 0 && text[from - 1] == ' ') --from;
            text.erase(from, end + 1 - from);
            at = from;
        } else {
            at = end + 1;
        }
    }
    std::string out;
    size_t from = 0;
    while (from <= text.size()) {
        size_t nl = text.find('\n', from);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(from, nl - from);
        from = nl + 1;
        std::string low = line;
        for (char &c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const bool hint = low.find("agent model pull") != std::string::npos || low.find("agent setup") != std::string::npos ||
                          low.find("agent engine ") != std::string::npos || low.find("agent runtime ") != std::string::npos ||
                          low.find("agent secret ") != std::string::npos || low.rfind("try:", 0) == 0;
        if (hint) {
            const size_t paren = line.rfind(" (");
            if (paren != std::string::npos && !line.empty() && line.back() == ')') line.erase(paren);
            else line.clear();
        }
        if (!line.empty()) out += line + "\n";
        if (nl == text.size()) break;
    }
    return out.empty() ? std::string("the command failed") : out;
}

} // namespace agent::cli
