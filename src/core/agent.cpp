#include <agent-cpp/agent.h>
#include <filesystem>

namespace agent {

#ifdef AGENT_HAS_LLAMACPP
void init_llama_cpp(Session &session);
std::string generate_text_llama_cpp(Session &session, std::string_view prompt);
void stream_text_llama_cpp(Session &session, std::string_view prompt,
                           const std::function<void(std::string_view)> &on_token);
#endif

#ifdef AGENT_HAS_OPENAICOMPATIBLE
std::string generate_text_openai_compatible(Session &session, std::string_view prompt);
void stream_text_openai_compatible(Session &session, std::string_view prompt,
                                   const std::function<void(std::string_view)> &on_token);
#endif

Session init(const Config &config) {
    if (!config.workspace_dir.empty()) {
        std::filesystem::create_directories(std::filesystem::path(config.workspace_dir) / "");
    }
    Session session{.config = config, .state = nullptr};
    if (config.provider == Provider::LlamaCpp) {
#ifdef AGENT_HAS_LLAMACPP
        init_llama_cpp(session);
#else
        std::cerr << "Error: LlamaCpp provider requested but not compiled." << "\n";
#endif
    }
    return session;
}

std::string generate_text(Session &session, std::string_view prompt) {
    switch (session.config.provider) {
    case Provider::Mock:
        return "Mock Response";

    case Provider::LlamaCpp:
#ifdef AGENT_HAS_LLAMACPP
        return generate_text_llama_cpp(session, prompt);
#else
        return "Error: Rebuild with AGENT_LLAMA_DIR pointing to binaries.";
#endif

    case Provider::OpenAICompatible:
#ifdef AGENT_HAS_OPENAICOMPATIBLE
        return generate_text_openai_compatible(session, prompt);
#else
        return "Error: Rebuild with AGENT_HAS_OPENAICOMPATIBLE pointing to binaries.";
#endif

    default:
        return std::string(prompt);
    }
}

void stream_text(Session &session, std::string_view prompt, const std::function<void(std::string_view)> &on_token) {
    switch (session.config.provider) {
    case Provider::Mock:
        on_token("Mock Response Streamed");
        break;

    case Provider::LlamaCpp:
#ifdef AGENT_HAS_LLAMACPP
        stream_text_llama_cpp(session, prompt, on_token);
#else
        on_token("Error: Rebuild with AGENT_LLAMA_DIR pointing to binaries.");
#endif
        break;

    case Provider::OpenAICompatible:
#ifdef AGENT_HAS_OPENAICOMPATIBLE
        stream_text_openai_compatible(session, prompt, on_token);
#else
        on_token("Error: Rebuild with AGENT_HAS_OPENAICOMPATIBLE pointing to binaries.");
#endif
        break;

    default:
        on_token(std::string(prompt));
        break;
    }
}

} // namespace agent