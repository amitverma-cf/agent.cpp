#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace agent {

enum class LogLevel { Debug, Info, Warning, Error };

enum class Provider { Mock, LlamaCpp, OpenAICompatible };

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
    Cancelled
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

struct Config {
    Provider provider = Provider::Mock;
    std::string base_url;
    std::string api_key;
    std::string model;
    std::string workspace_dir;
    LogCallback logger = nullptr;
    void *logger_user_data = nullptr;
};

/**
 * NOTE: Session is NOT thread-safe. One Session instance must not be accessed from multiple threads concurrently.
 */
struct Session {
    Config config;
    std::shared_ptr<void> state;
};

void init_backend();
void free_backend();

Result<Session> init(const Config &config);
Result<std::string> generate_text(Session &session, std::string_view prompt);
Result<void> stream_text(Session &session, std::string_view prompt, TokenCallback on_token, void *user_data = nullptr);

} // namespace agent
