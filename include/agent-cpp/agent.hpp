#pragma once

#include <filesystem>
#include <memory>
#include <simdjson.h>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace agent {

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

class Arena {
  public:
    explicit Arena(size_t capacity = 1024 * 1024);
    ~Arena();
    Arena(const Arena &) = delete;
    Arena &operator=(const Arena &) = delete;
    Arena(Arena &&other) noexcept;
    Arena &operator=(Arena &&other) noexcept;
    void *allocate(size_t size, size_t alignment = alignof(std::max_align_t));
    void reset();
    std::string_view allocate_string(std::string_view src);
    template <typename T> std::span<T> allocate_span(size_t count) {
        if (count == 0)
            return {};
        void *mem = allocate(sizeof(T) * count, alignof(T));
        if (!mem)
            return {};
        return std::span<T>(static_cast<T *>(mem), count);
    }

  private:
    size_t capacity_;
    char *ptr_;
    size_t offset_;
};

using TokenCallback = void (*)(std::string_view token, void *user_data);

struct Usage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

enum class DType { Float32, Float16, Int8, Int32, Int64, UInt8 };

struct TensorView {
    std::string_view name;
    DType dtype = DType::Float32;
    std::span<const int64_t> shape;
    std::span<const uint8_t> data;
};

struct TensorData {
    std::string name;
    DType dtype = DType::Float32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

enum class LogLevel { Debug, Info, Warning, Error };
using LogCallback = void (*)(LogLevel level, std::string_view message, void *user_data);

struct ToolCallView {
    std::string_view id;
    std::string_view name;
    std::string_view arguments;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

struct Tool {
    std::string_view name;
    std::string_view description;
    std::string_view parameter_schema;
    using CallbackFn = Result<std::string> (*)(std::string_view arguments, void *user_data);
    CallbackFn callback = nullptr;
    void *user_data = nullptr;
};

namespace tools {
Tool get_calculator_tool();
std::vector<Tool> get_filesystem_tools(const std::string *workspace_dir);
Tool get_terminal_tool(const std::string *workspace_dir);
} // namespace tools

struct MessageView {
    std::string_view role;
    std::string_view content;
    std::span<const ToolCallView> tool_calls;
    std::string_view tool_call_id;
    std::span<const TensorView> tensors;
};

struct ChatRequest {
    std::span<const MessageView> messages;
    int max_tokens = 0;
    float temperature = -1.0f;
    bool stream = false;
    TokenCallback on_token = nullptr;
    void *token_user_data = nullptr;
};

struct ChatResponse {
    MessageView message;
    Usage usage;
    std::span<const TensorView> output_tensors;
};

enum class MemoryProvider { InMemory, RocksDB };

struct MemoryConfig {
    MemoryProvider provider = MemoryProvider::InMemory;
    std::string db_path = "memory/db";
};

enum class AiProvider { Mock, LlamaCpp, OpenAICompatible, OnnxRuntime };

enum class EventType { OnTurnStart, OnInferenceSubmit, OnInferenceComplete, OnToolCall, OnStateTransition };

struct Event {
    EventType type;
    std::span<const uint8_t> payload;
};

using EventHookFn = void (*)(const Event &event, void *user_data);

struct EventHook {
    EventType type;
    EventHookFn callback;
    void *user_data;
};

struct Config {
    AiProvider provider = AiProvider::Mock;
    std::string base_url;
    std::string api_key;
    std::string model;
    std::string workspace_dir;
    MemoryConfig memory;
    std::vector<Tool> tools;

    int context_window = 2048;
    int max_tokens = 512;
    float temperature = 0.7f;
    size_t arena_capacity = 0;

    bool auto_default_tools = true;

    LogCallback logger = nullptr;
    void *logger_user_data = nullptr;
};

struct Session {
    Config config;
    Arena arena;
    simdjson::ondemand::parser json_parser;
    std::vector<EventHook> event_hooks;
    std::shared_ptr<void> provider_state;
    std::shared_ptr<void> memory_state;
    std::shared_ptr<std::string> workspace_dir_ref;

    struct ConversationState *active_conversation = nullptr;

    int total_prompt_tokens = 0;
    int total_completion_tokens = 0;
    int total_tokens = 0;

    // Stable heap indirection to `this`, since Session is move-only and gets relocated after
    // tool registration (e.g. by init()'s return-by-value). Tool callbacks that need Session&
    // receive a pointer to this cell as user_data; the dispatcher refreshes it before every call.
    std::shared_ptr<Session *> self_ref;

    Session() = default;
    ~Session() = default;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = default;
    Session &operator=(Session &&) = default;
};

struct Message {
    std::string role;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    std::vector<TensorData> tensors;
    mutable int token_count = -1;
};

struct ConversationState {
    std::string id;
    std::vector<Message> history;
};

struct AgentRuntime {
    AgentRuntime();
    ~AgentRuntime();
    AgentRuntime(const AgentRuntime &) = delete;
    AgentRuntime &operator=(const AgentRuntime &) = delete;
    AgentRuntime(AgentRuntime &&) = delete;
    AgentRuntime &operator=(AgentRuntime &&) = delete;
};

struct FSMState {
    std::string_view state_id;
    using TransitionFn = Result<std::string_view> (*)(Session &session, void *context);
    TransitionFn on_transition = nullptr;
};

struct FSMExecutor {
    std::span<const FSMState> registered_states;
};

void init_backend();
void free_backend();
Result<Session> init(const Config &config);

Result<ChatResponse> execute_turn(Session &session, const ChatRequest &request);
Result<void> init_memory(Session &session);
Result<void> store_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_memory(Session &session, std::string_view key);
Result<void> clear_memory(Session &session);
void register_event_hook(Session &session, EventType type, EventHookFn callback, void *user_data = nullptr);
void trigger_event(Session &session, EventType type, std::span<const uint8_t> payload);

Result<std::string_view> run_conversation_turn(Session &session, ConversationState &state, std::string_view user_prompt,
                                               bool stream = false, TokenCallback on_token = nullptr,
                                               void *token_user_data = nullptr);

Result<void> run_fsm(Session &session, void *context, FSMExecutor &executor);

Result<void> compress_context(Session &session, ConversationState &state, size_t keep_recent = 6);

Result<void> init_tools(Session &session);
Result<std::string> execute_tool(Session &session, std::string_view name, std::string_view arguments);

} // namespace agent
