#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <simdjson.h>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agent {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

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
    KeyNotFound,
    ToolCallLimitExceeded,
    SandboxViolation,
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

inline Result<void> ok() { return Result<void>{.ok = true, .error = {}}; }

template <typename T> inline Result<T> fail(ErrorCode code, std::string message) {
    return Result<T>{.ok = false, .value = {}, .error = make_error(code, std::move(message))};
}

inline Result<void> fail(ErrorCode code, std::string message) {
    return Result<void>{.ok = false, .error = make_error(code, std::move(message))};
}

// ---------------------------------------------------------------------------
// Per-turn bump allocator
// ---------------------------------------------------------------------------

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
    size_t capacity_ = 0;
    char *ptr_ = nullptr;
    size_t offset_ = 0;
};

// ---------------------------------------------------------------------------
// Model I/O views (arena-backed; live only for the current infer() call)
// ---------------------------------------------------------------------------

using TokenStreamFn = void (*)(std::string_view token, void *user_data);

struct TokenUsage {
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
using LogFn = void (*)(LogLevel level, std::string_view message, void *user_data);

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

struct WorkspaceContext {
    const std::string *workspace_dir = nullptr;
    bool sandbox = true;
};

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
Tool get_terminal_tool(const std::string *workspace_dir);

}

struct MessageView {
    std::string_view role;
    std::string_view content;
    std::span<const ToolCallView> tool_calls;
    std::string_view tool_call_id;
    std::span<const TensorView> tensors;
};

struct InferRequest {
    std::span<const MessageView> messages;
    int max_tokens = 0;
    float temperature = -1.0f;
    bool stream = false;
    TokenStreamFn on_token = nullptr;
    void *token_user_data = nullptr;
};

struct InferResponse {
    MessageView message;
    TokenUsage usage;
    std::span<const TensorView> output_tensors;
};

enum class MemoryProvider { Sqlite };

struct MemoryConfig {
    MemoryProvider provider = MemoryProvider::Sqlite;
    std::string db_path = "memory/db.sqlite3";

    size_t max_cache_bytes = 16 * 1024 * 1024;
    size_t flush_dirty_threshold_bytes = 4 * 1024 * 1024;
    int flush_interval_ms = 500;
};

enum class AiProvider { Mock, LlamaCpp, OpenAICompatible, OnnxRuntime };

enum class EventType {
    OnTurnStart,
    OnInferenceSubmit,
    OnInferenceComplete,
    OnToolCall,
    OnToolResult,
    OnStateTransition,
    OnCompression,
    OnPrune,
    OnCronFire,
    OnSubAgentComplete,
};

struct Event {
    EventType type;
    std::span<const uint8_t> payload;
};

using EventHookFn = void (*)(const Event &event, void *user_data);

struct EventHook {
    EventType type;
    EventHookFn callback = nullptr;
    void *user_data = nullptr;
};

struct TurnStats {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int tool_calls = 0;
    int compressions = 0;
    int prune_cycles = 0;
    int turns = 0;
};

struct SessionStats {
    int total_prompt_tokens = 0;
    int total_completion_tokens = 0;
    int total_tool_calls = 0;
    int total_turns = 0;
    int total_compressions = 0;
    int total_prune_cycles = 0;
};

struct Message {
    std::string role;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    std::vector<TensorData> tensors;
    mutable int token_count = -1;
};

struct FlowMemory {
    std::string id;
    std::vector<Message> history;
    TurnStats stats;
};

struct Session;

struct FlowHook {
    enum class When { Pre, Post };
    When when = When::Pre;
    std::string_view tool_name;

    using HookFn = void (*)(Session &session, FlowMemory &memory, std::string_view tool_name,
                            std::string_view args, std::string_view result, void *user_data);
    HookFn callback = nullptr;
    void *user_data = nullptr;
};

struct FlowState {
    std::string_view state_id;
    std::string_view system_prompt;
    std::span<const std::string_view> allowed_tools;
    std::span<const FlowHook> hooks;

    bool isolated_memory = false;

    using ContextProviderFn = std::string_view (*)(Session &session, FlowMemory &memory,
                                                   void *context);
    ContextProviderFn context_provider = nullptr;

    using TransitionFn = Result<std::string_view> (*)(Session &session, FlowMemory &memory,
                                                      void *context);
    TransitionFn on_transition = nullptr;
};

struct Flow {
    std::string id;
    std::span<const FlowState> states;

    FlowMemory memory;
    std::unordered_map<std::string, FlowMemory> isolated_memories;

    size_t current_state_index = 0;
    bool has_started = false;
    bool has_finished = false;
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

    int max_tool_call_rounds = 10;
    bool sandbox_filesystem = true;
    bool auto_default_tools = true;
    bool enable_file_logging = false;

    int retry_max_attempts = 3;
    int retry_base_delay_ms = 500;

    int max_parallel_subagents = 4;

    LogFn logger = nullptr;
    void *logger_user_data = nullptr;
};

struct DataSource {
    std::string name;
    std::shared_ptr<void> state;
    using QueryFn = Result<std::string> (*)(void *state, std::string_view query, void *user_data);
    QueryFn query = nullptr;
    void *user_data = nullptr;
};

struct Session {
    Config config;

    Arena arena;
    std::vector<Arena> worker_arenas;

    std::vector<EventHook> event_hooks;

    std::shared_ptr<void> provider_state;
    std::shared_ptr<void> memory_state;

    std::shared_ptr<std::string> workspace_path;

    std::shared_ptr<void> tool_context;

    std::vector<std::shared_ptr<DataSource>> retained_data_sources;

    SessionStats stats;
    std::shared_ptr<std::mutex> stats_mutex = std::make_shared<std::mutex>();
    std::shared_ptr<std::mutex> log_mutex = std::make_shared<std::mutex>();
    std::shared_ptr<std::mutex> event_mutex = std::make_shared<std::mutex>();

    std::unordered_map<std::string, size_t> tool_by_name;
    std::string tools_system_prompt;

    std::unordered_map<std::string, DataSource> data_sources;

    std::shared_ptr<FILE> log_file;

    std::shared_ptr<Session *> session_ptr_cell;

    Session() = default;
    ~Session() = default;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    Session(Session &&) = default;
    Session &operator=(Session &&) = default;
};

struct AgentRuntime {
    AgentRuntime();
    ~AgentRuntime();
    AgentRuntime(const AgentRuntime &) = delete;
    AgentRuntime &operator=(const AgentRuntime &) = delete;
    AgentRuntime(AgentRuntime &&) = delete;
    AgentRuntime &operator=(AgentRuntime &&) = delete;
};

struct SubAgentResult {
    std::string id;
    bool ok = false;
    Error error;
    TurnStats stats;
};

struct CronTask {
    std::string id;
    using TaskFn = Result<void> (*)(Session &session, void *user_data);
    TaskFn callback = nullptr;
    void *user_data = nullptr;
    std::chrono::steady_clock::time_point next_fire;
    std::chrono::milliseconds interval{0};
};

class AgentScheduler {
  public:
    Result<void> spawn(std::string id, Flow flow, void *context = nullptr);

    Result<void> add_cron(std::string id, CronTask::TaskFn fn, void *user_data,
                          std::chrono::milliseconds delay,
                          std::chrono::milliseconds interval = std::chrono::milliseconds{0});
    Result<void> cancel_cron(std::string_view id);

    Result<void> pump(Session &session);
    Result<void> run_until_done(Session &session);

    bool all_done() const;
    std::vector<SubAgentResult> results() const;

  private:
    struct SubAgent {
        std::string id;
        Flow flow;
        void *context = nullptr;
        bool done = false;
        SubAgentResult result;
    };
    struct CronJob {
        CronTask task;
        bool done = false;
    };

    std::vector<SubAgent> subagents_;
    std::vector<CronJob> cron_jobs_;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void init_backend();
void free_backend();
Result<Session> init(const Config &config);

Result<InferResponse> infer(Session &session, const InferRequest &request);

Result<void> init_memory(Session &session);
Result<void> store_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_memory(Session &session, std::string_view key);
Result<void> clear_memory(Session &session);

Result<void> register_data_source(Session &session, DataSource ds);
Result<std::string> query_data_source(Session &session, std::string_view name,
                                      std::string_view query);

Tool bind_data_source_tool(Session &session, std::shared_ptr<DataSource> source,
                           std::string_view description, std::string_view parameter_schema);

Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(std::string name,
                                                                std::string db_path,
                                                                int dimensions);

Result<void> sqlite_vec_insert(DataSource &source, std::string_view text,
                               std::span<const float> embedding);

void register_event_hook(Session &session, EventType type, EventHookFn callback,
                         void *user_data = nullptr);
void trigger_event(Session &session, EventType type, std::span<const uint8_t> payload);

Result<std::string> run_turn(Session &session, FlowMemory &memory, std::string_view user_prompt,
                             bool stream = false, TokenStreamFn on_token = nullptr,
                             void *token_user_data = nullptr);

Result<void> run_flow(Session &session, void *context, Flow &flow);
Result<bool> step_flow(Session &session, void *context, Flow &flow);

Result<void> compress_context(Session &session, FlowMemory &memory, size_t keep_recent = 6);

Result<void> save_flow_memory(Session &session, const FlowMemory &memory);
Result<void> load_flow_memory(Session &session, FlowMemory &memory);

Result<void> init_tools(Session &session);
Result<std::string> execute_tool(Session &session, std::string_view name,
                                 std::string_view arguments);

void rebuild_tool_index(Session &session);

SessionStats get_stats(const Session &session);
void reset_stats(Session &session);

Result<void> clear_provider_kv_cache(Session &session);

} // namespace agent
