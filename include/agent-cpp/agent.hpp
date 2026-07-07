#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace agent {

enum class LogLevel { Debug, Info, Warning, Error };

enum class AiProvider { Mock, LlamaCpp, OpenAICompatible };

enum class MemoryProvider { InMemory, RocksDB };

struct MemoryConfig {
    MemoryProvider provider = MemoryProvider::InMemory;
    std::string db_path = ".workspace/agent_db";
};

enum class ErrorCode {
    Ok,
    InvalidConfig,
    UnsupportedProvider,
    FilesystemError,
    ProviderInitFailed,
    ModelLoadFailed,
    DecodeFailed,
    NetworkError,
    HttpError,
    ParseError,
    Cancelled,
    KeyNotFound
};

struct Error {
    ErrorCode code = ErrorCode::Ok;
    std::string message;
};

template <typename T> struct Result {
    bool ok = false;
    T value{};
    Error error{};
};

template <> struct Result<void> {
    bool ok = true;
    Error error{};
};

inline Error make_error(ErrorCode code, std::string message) {
    return Error{.code = code, .message = std::move(message)};
}

template <typename T> inline Result<T> ok(T value) {
    return Result<T>{.ok = true, .value = std::move(value), .error = {}};
}

inline Result<void> ok() {
    return Result<void>{.ok = true, .error = {}};
}

template <typename T> inline Result<T> fail(ErrorCode code, std::string message) {
    return Result<T>{.ok = false, .value = {}, .error = make_error(code, std::move(message))};
}

inline Result<void> fail(ErrorCode code, std::string message) {
    return Result<void>{.ok = false, .error = make_error(code, std::move(message))};
}

using LogCallback = void (*)(LogLevel level, std::string_view message, void *user_data);

using TokenCallback = void (*)(std::string_view token, void *user_data);

struct Usage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

struct GenerationResult {
    std::string text;
    Usage usage;
};

struct Config {
    AiProvider provider = AiProvider::Mock;
    std::string base_url;
    std::string api_key;
    std::string model;
    std::string workspace_dir;
    MemoryConfig memory;

    // Hyperparameters
    int context_window = 2048;
    int max_tokens = 512;
    float temperature = 0.7f;

    LogCallback logger = nullptr;
    void *logger_user_data = nullptr;
};

/**
 * NOTE: Session is NOT thread-safe. One Session instance must not be accessed from multiple threads concurrently.
 */
struct Session {
    Config config;
    std::shared_ptr<void> provider_state;
    std::shared_ptr<void> memory_state;

    Session() = default;
    ~Session() = default;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = default;
    Session& operator=(Session&&) = default;
};

struct Message {
    std::string role;
    std::string content;
};

struct Conversation {
    Session &session;
    std::vector<Message> history;

    Conversation(Session &sess);
};

void init_backend();
void free_backend();

struct AgentRuntime {
    AgentRuntime() { init_backend(); }
    ~AgentRuntime() { free_backend(); }
    AgentRuntime(const AgentRuntime&) = delete;
    AgentRuntime& operator=(const AgentRuntime&) = delete;
    AgentRuntime(AgentRuntime&&) = delete;
    AgentRuntime& operator=(AgentRuntime&&) = delete;
};

Result<Session> init(const Config &config);
Result<GenerationResult> generate_text(Session &session, std::string_view prompt);
Result<void> stream_text(Session &session, std::string_view prompt, TokenCallback on_token, void *user_data = nullptr);

Result<void> init_memory(Session &session);
Result<void> store_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_memory(Session &session, std::string_view key);
Result<void> clear_memory(Session &session);

Result<void> add_message(Conversation &conv, std::string role, std::string content);
Result<std::string> complete_conversation(Conversation &conv);
Result<std::vector<Message>> get_conversation_history(Conversation &conv);
Result<void> sync_conversation(Conversation &conv);

} // namespace agent
