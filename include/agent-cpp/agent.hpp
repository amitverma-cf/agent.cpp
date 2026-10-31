#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct agent_engine_v1; // <agent-cpp/engine_abi.h>

namespace agent {

// -------------------------- Errors --------------------------
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
    PermissionDenied,
    // The out-of-process engine host died (crash, abort, exit, killed by the watchdog) or could not be
    // restarted. See Config::engine_isolation.
    EngineCrashed,
    // A ModelRegistry could not fit a model in its RAM/VRAM budget (every other loaded model is in use
    // or pinned). See acquire_model.
    OutOfMemory,
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

inline Error make_error(ErrorCode code, std::string message) { return Error{.code = code, .message = std::move(message)}; }

template <typename T> inline Result<T> ok(T value) { return Result<T>{.ok = true, .value = std::move(value), .error = {}}; }

inline Result<void> ok() { return Result<void>{.ok = true, .error = {}}; }

template <typename T> inline Result<T> fail(ErrorCode code, std::string message) {
    return Result<T>{.ok = false, .value = {}, .error = make_error(code, std::move(message))};
}

inline Result<void> fail(ErrorCode code, std::string message) {
    return Result<void>{.ok = false, .error = make_error(code, std::move(message))};
}

// -------------------------- Per-turn bump allocator --------------------------
class Arena {
  public:
    explicit Arena(size_t capacity = 0);
    ~Arena();
    Arena(const Arena &) = delete;
    Arena &operator=(const Arena &) = delete;
    Arena(Arena &&other) noexcept;
    Arena &operator=(Arena &&other) noexcept;

    void *allocate(size_t size, size_t alignment = alignof(std::max_align_t));
    void reset();
    std::string_view allocate_string(std::string_view src);

    template <typename T> std::span<T> allocate_span(size_t count) {
        if (count == 0) return {};
        void *mem = allocate(sizeof(T) * count, alignof(T));
        if (!mem) return {};
        return std::span<T>(static_cast<T *>(mem), count);
    }

  private:
    size_t capacity_ = 0;
    char *ptr_ = nullptr;
    size_t offset_ = 0;
};

// -------------------------- Model I/O views (arena-backed; live only for the current infer() call) --------------------------
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

// Media referenced by file path: images, audio, video, meshes. Inputs can live anywhere; files that
// engines generate are written to <workspace>/artifacts/. mime e.g. "image/png", "audio/wav".
struct MediaView {
    std::string_view path;
    std::string_view mime;
};

struct Media {
    std::string path;
    std::string mime;
};

// Progress of a long-running engine job (image/video/3D generation). total may be 0 if unknown.
using ProgressFn = void (*)(uint64_t step, uint64_t total, void *user_data);

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
    // Always invoke via agent::execute_tool, never call this function pointer directly. Some
    // built-in tools (e.g. use_skill, compress_context) close over a Session* cell that
    // execute_tool refreshes right before dispatch; calling callback() directly can run against
    // a stale or dangling Session pointer if the Session has since moved.
    CallbackFn callback = nullptr;
    void *user_data = nullptr;
};

// Per-call tool dispatch policy, checked by execute_tool before a Tool::callback runs.
enum class PermissionDecision { Allow, Ask, Deny };

// Decide whether a tool call should proceed. Called with the raw JSON arguments string.
using PermissionCheckFn = PermissionDecision (*)(std::string_view tool_name, std::string_view arguments, void *user_data);
// Resolves an Ask decision; returns true to approve the call.
using PermissionPromptFn = bool (*)(std::string_view tool_name, std::string_view arguments, void *user_data);

// OS-level sandbox for a child process (engine host, run_command, MCP server). See
// docs/security-sandbox.md. Linux: Landlock + seccomp + rlimits + a new session; macOS: a
// sandbox_init() profile (untested); Windows: a Job Object (untested; no filesystem/network rules).
// Missing kernel features degrade to a logged warning, never to a failed spawn.
enum class SandboxNetwork { Allow, Deny };

struct SandboxPolicy {
    bool enabled = false;
    // false: leave the filesystem alone (network, rlimits and process-group parts still apply).
    bool filesystem = true;
    // Extra paths, on top of what each use adds (engine: plugin, model, workspace ro, artifacts rw;
    // run_command: workspace rw; MCP: the server's program dir ro, working_directory rw).
    std::vector<std::string> read_only_paths;
    std::vector<std::string> read_write_paths;
    // Read-only system dirs (/usr, /lib*, selected /etc files, /proc, /sys) and the basic device
    // nodes plus GPUs read-write (/dev/null, /dev/urandom, /dev/dri, /dev/nvidia*, ...).
    bool system_paths = true;
    bool allow_tmp = true; // $TMPDIR (or /tmp) and /var/tmp read-write
    SandboxNetwork network = SandboxNetwork::Allow;
    uint64_t memory_limit_bytes = 0; // RLIMIT_AS / Job Object process memory; 0 = none
    // RLIMIT_NPROC (Linux: counts every process and thread of the user, not just this tree; ignored
    // for root) / Job Object active process limit. 0 = none.
    int max_processes = 0;
};

namespace tools {

struct WorkspaceContext {
    const std::string *workspace_dir = nullptr;
    bool sandbox = true;
    // Applied to run_command when it is built from this context (get_terminal_tool(ctx)).
    SandboxPolicy command_sandbox;
};

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
// No OS sandbox (backward compatible); prefer the WorkspaceContext overload.
Tool get_terminal_tool(const std::string *workspace_dir);
// run_command under ctx->command_sandbox (workspace read-write when ctx->sandbox is true, the
// whole filesystem otherwise). ctx must outlive the Tool. Used by the default tool set.
Tool get_terminal_tool(std::shared_ptr<WorkspaceContext> ctx);

// Opt-in example policy: Ask before run_command/delete_path/move_path, Allow everything else.
// Not wired in automatically -- set Config::permission_check to it explicitly if wanted.
PermissionDecision default_permission_policy(std::string_view tool_name, std::string_view arguments, void *user_data);

} // namespace tools

struct MessageView {
    std::string_view role;
    std::string_view content;
    std::span<const ToolCallView> tool_calls;
    std::string_view tool_call_id;
    std::span<const TensorView> tensors;
    std::span<const MediaView> media;
};

struct InferRequest {
    std::span<const MessageView> messages;
    int max_tokens = 0;
    float temperature = -1.0f;
    bool stream = false;
    TokenStreamFn on_token = nullptr;
    void *token_user_data = nullptr;
    ProgressFn on_progress = nullptr;
    void *progress_user_data = nullptr;
    const volatile int32_t *cancel = nullptr; // engines stop with ErrorCode::Cancelled when *cancel != 0
};

struct InferResponse {
    MessageView message;
    TokenUsage usage;
    std::span<const TensorView> output_tensors;
};

struct MemoryConfig;

// Pluggable key-value store behind store_memory/retrieve_memory/clear_memory and conversation
// persistence. open() runs once from init(); the state it returns is passed to every other call.
struct StoreOps {
    using OpenFn = Result<std::shared_ptr<void>> (*)(const MemoryConfig &config, std::string_view workspace_dir);
    using PutFn = Result<void> (*)(void *state, std::string_view key, std::string_view value);
    using GetFn = Result<std::string> (*)(void *state, std::string_view key);
    using ClearFn = Result<void> (*)(void *state);

    OpenFn open = nullptr;
    PutFn put = nullptr;
    GetFn get = nullptr;
    ClearFn clear = nullptr;
};

struct MemoryConfig {
    // nullptr = built-in in-memory store (nothing persists past the Session). The optional
    // agent-store-sqlite library provides agent::store::sqlite_store().
    const StoreOps *store = nullptr;
    // Backend-specific location. Empty = backend default (SQLite: <workspace>/memory/db.sqlite3).
    std::string db_path;

    // Write-back cache tuning, used by caching stores (SQLite).
    size_t max_cache_bytes = 16 * 1024 * 1024;
    size_t flush_dirty_threshold_bytes = 4 * 1024 * 1024;
    int flush_interval_ms = 500;
};

// Engine = an engine plugin (see <agent-cpp/engine_abi.h>), selected by Config::engine.
enum class AiProvider { Mock, Engine };

// Where an engine plugin runs (AiProvider::Engine). InProcess: dlopen'ed into this process (fastest;
// a crash in the engine takes the process down). Process: in an agent-engine-host child process
// speaking the engine ABI over pipes; a crash fails the in-flight call with ErrorCode::EngineCrashed
// and the next call restarts the host. Default: the engines.json "isolation" of the engine when it
// was loaded with load_engine_manifest, else InProcess.
enum class EngineIsolation { Default, InProcess, Process };

enum class EventType {
    OnTurnStart,
    OnInferenceSubmit,
    OnInferenceComplete,
    OnToolCall,
    OnToolResult,
    OnStateTransition, // a graph node starts; payload = node name
    OnCompression,
    OnPrune,
    OnSubAgentComplete, // a Parallel node's branch finished (fired after the join); payload = the branch's entry node name
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
    std::vector<Media> media;
    mutable int token_count = -1;
};

struct FlowMemory {
    std::string id;
    std::vector<Message> history;
    TurnStats stats;
};

struct Session;

// Callback fired around each tool call made inside run_turn (and around tool calls approved on a
// graph resume). Attach them to a graph Agent node via GraphNode::hooks; they are active only while
// that node runs. tool_name empty = every tool, otherwise only that tool (the view must outlive the
// node). memory is the conversation that issued the call; result is empty for Pre.
struct ToolHook {
    enum class When { Pre, Post };
    When when = When::Pre;
    std::string_view tool_name;

    using HookFn = void (*)(Session &session, FlowMemory &memory, std::string_view tool_name, std::string_view args,
                            std::string_view result, void *user_data);
    HookFn callback = nullptr;
    void *user_data = nullptr;
};

struct Config {
    AiProvider provider = AiProvider::Mock;
    // AiProvider::Engine only: a registered engine name, or a path to an engine shared library.
    std::string engine;
    // AiProvider::Engine only: JSON object forwarded to the engine as "options" in its create() config.
    std::string engine_options;

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

    // Concurrent engines only: worker arenas (one per thread) for the branches of a graph Parallel
    // node; the llama engine also sizes its continuous-batching sequence slots from it.
    int max_parallel_subagents = 4;

    LogFn logger = nullptr;
    void *logger_user_data = nullptr;

    // Progress reports from long-running engine jobs during run_turn.
    ProgressFn on_progress = nullptr;
    void *progress_user_data = nullptr;

    // Directory of skill subdirectories (each containing a SKILL.md). Empty = feature inert.
    std::string skills_dir;

    // Per-call tool permission gate. nullptr = Allow everything (default, backward compatible).
    PermissionCheckFn permission_check = nullptr;
    // Resolves an Ask decision from permission_check. Only consulted when permission_check
    // returns PermissionDecision::Ask; if nullptr, Ask is treated as Deny.
    PermissionPromptFn permission_prompt = nullptr;
    void *permission_user_data = nullptr;

    // Engine isolation (AiProvider::Engine). See docs/api/engines.md, "Isolation".
    EngineIsolation engine_isolation = EngineIsolation::Default;
    // Process isolation: the agent-engine-host executable. Empty = $AGENT_ENGINE_HOST, else next to
    // the current executable, else <exe dir>/../bin, else PATH.
    std::string engine_host_path;
    // Process isolation: per-request watchdog (run / count_tokens / reset_cache, not create). When a
    // request takes longer the host is killed and the call fails with ErrorCode::EngineCrashed. 0 = none.
    int engine_timeout_ms = 0;
    // Process isolation: after request_cancel(), how long the engine gets to stop before the host is killed.
    int engine_cancel_grace_ms = 2000;
    // Process isolation: automatic host restarts allowed per rolling minute. Past it, calls fail with
    // ErrorCode::EngineCrashed until the window frees up.
    int engine_max_restarts = 3;

    // OS sandbox for run_command (default tool set). The workspace is read-write and the rest of
    // the filesystem is off-limits while sandbox_filesystem is true; the network is allowed by
    // default (set network = SandboxNetwork::Deny to block it). See docs/security-sandbox.md.
    SandboxPolicy tool_sandbox{.enabled = true};
    // OS sandbox for the agent-engine-host (process isolation only): plugin dir, model path and
    // engine_options paths read-only, workspace read-only, workspace/artifacts read-write, no
    // network, no /tmp. memory_limit_bytes becomes the host's RLIMIT_AS.
    SandboxPolicy engine_sandbox{.enabled = true, .allow_tmp = false, .network = SandboxNetwork::Deny};
};

struct DataSource {
    std::string name;
    std::shared_ptr<void> state;
    using QueryFn = Result<std::string> (*)(void *state, std::string_view query, void *user_data);
    QueryFn query = nullptr;
    void *user_data = nullptr;
};

// A SKILL.md-based capability package (https://agentskills.io). Only name/description are kept
// in context by default; the full body is loaded on demand via read_skill_body/get_skill_loader_tool.
struct Skill {
    std::string name;
    std::string description;
    std::string path; // directory containing this skill's SKILL.md
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
    std::string tools_system_prompt;  // tool list + usage instructions, followed by skills_system_prompt
    std::string skills_system_prompt; // skill names + descriptions only

    std::unordered_map<std::string, DataSource> data_sources;
    std::unordered_map<std::string, Skill> skills;

    std::shared_ptr<FILE> log_file;

    std::shared_ptr<Session *> session_ptr_cell;

    // Set by request_cancel(); read by engines mid-job. Shared so it survives Session moves.
    std::shared_ptr<int32_t> cancel_flag = std::make_shared<int32_t>(0);

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

// -------------------------- Lifecycle --------------------------

void init_backend();
void free_backend();
Result<Session> init(const Config &config);

Result<InferResponse> infer(Session &session, const InferRequest &request);

Result<void> init_memory(Session &session);
Result<void> store_memory(Session &session, std::string_view key, std::string_view value);
Result<std::string> retrieve_memory(Session &session, std::string_view key);
Result<void> clear_memory(Session &session);

Result<void> register_data_source(Session &session, DataSource ds);
Result<std::string> query_data_source(Session &session, std::string_view name, std::string_view query);

Tool bind_data_source_tool(Session &session, std::shared_ptr<DataSource> source, std::string_view description,
                           std::string_view parameter_schema);

// Scans skills_dir for one subdirectory per skill (each containing a SKILL.md), populates
// session.skills, and folds each skill's name+description into the tools system prompt.
Result<void> load_skills_dir(Session &session, std::string_view skills_dir);
// Full SKILL.md content for a loaded skill, read on demand (not kept in context otherwise).
Result<std::string> read_skill_body(Session &session, std::string_view name);

// Engine plugins. Engines are process-wide and stay loaded until exit.
// register_engine adds an engine linked into the host; load_engine_plugin dlopens a shared library
// exporting agent_engine_get_v1 and returns the engine's name.
Result<void> register_engine(const agent_engine_v1 *engine);
Result<std::string> load_engine_plugin(std::string_view path);
const agent_engine_v1 *find_engine(std::string_view name);
// Loads every plugin listed in an engines.json manifest: {"engines": [{"path": "agent-engine-llama.so"}, ...]}.
// Relative paths resolve against the manifest's directory. Returns the loaded engine names; stops at
// the first plugin that fails to load. An entry may set "isolation": "process" | "inprocess" (the
// EngineIsolation::Default for sessions using it) and "name"; "process" entries are only recorded,
// not loaded into this process.
Result<std::vector<std::string>> load_engine_manifest(std::string_view manifest_path);

void register_event_hook(Session &session, EventType type, EventHookFn callback, void *user_data = nullptr);
void trigger_event(Session &session, EventType type, std::span<const uint8_t> payload);

Result<std::string> run_turn(Session &session, FlowMemory &memory, std::string_view user_prompt, bool stream = false,
                             TokenStreamFn on_token = nullptr, void *token_user_data = nullptr);

// Like run_turn, but the user message also carries media (images, audio, ...) for engines that accept them.
// (A separate name, not an overload: a Media array would otherwise silently convert to run_turn's `bool stream`.)
Result<std::string> run_turn_with_media(Session &session, FlowMemory &memory, std::string_view user_prompt, std::span<const Media> media,
                                        bool stream = false, TokenStreamFn on_token = nullptr, void *token_user_data = nullptr);

// Cancels in-flight engine work and makes every later inference on this Session fail with
// ErrorCode::Cancelled until clear_cancel() is called. Safe to call from any thread.
void request_cancel(Session &session);
void clear_cancel(Session &session);

Result<void> compress_context(Session &session, FlowMemory &memory, size_t keep_recent = 6);

Result<void> save_flow_memory(Session &session, const FlowMemory &memory);
Result<void> load_flow_memory(Session &session, FlowMemory &memory);

Result<void> init_tools(Session &session);
Result<std::string> execute_tool(Session &session, std::string_view name, std::string_view arguments);

void rebuild_tool_index(Session &session);

SessionStats get_stats(const Session &session);
void reset_stats(Session &session);

Result<void> clear_provider_kv_cache(Session &session);

} // namespace agent

// ============================================================================================
// -------------------------- Graph runtime (Phase 3, v1) --------------------------
// LangGraph / Google ADK style workflows: named nodes (Agent, Function, Tool) joined by static or
// conditional edges, ADK workflow builders (make_sequential / make_parallel / make_loop), scoped
// key/value state, a per-run event log, checkpoints keyed by thread_id, and interrupts for human
// approval. See docs/api/graph.md. Graphs are built once at setup (owning strings are fine there);
// the per-turn inference path inside Agent nodes is the ordinary arena-backed run_turn.
// ============================================================================================

#include <map>

namespace agent {

// Router return value (and edge target) meaning "this graph is done". Inside a graph composed by
// make_sequential / make_loop / make_parallel, END finishes the current step, not the whole run.
inline constexpr std::string_view kGraphEnd = "__end__";

// Scoped key/value state shared by every node of a run.
//   "temp:<k>"            discarded when run_graph / resume_graph returns; never checkpointed
//   "user:<k>", "app:<k>" persisted through the Session's StoreOps; visible to every thread
//   anything else         per-thread state (GraphRun::thread_id), carried in checkpoints
// Write through set_state() so the runtime logs, persists and conflict-checks the write.
struct GraphState {
    std::map<std::string, std::string, std::less<>> values;
    // Agent node conversations, keyed by GraphNode::history_key (default: the node name).
    std::map<std::string, FlowMemory, std::less<>> histories;

    // Keys written via set_state() that the runtime has not processed yet.
    std::vector<std::string> written;
    // Set by request_interrupt().
    bool interrupt_requested = false;
    std::string interrupt_reason;
};

void set_state(GraphState &state, std::string_view key, std::string value);
// The value of key, or fallback when unset. The view stays valid until key is written again.
std::string_view get_state(const GraphState &state, std::string_view key, std::string_view fallback = {});
bool has_state(const GraphState &state, std::string_view key);
// Called from a Function node: pause the run after this node returns. On resume_graph the same node
// runs again with the human's reply in "temp:human_input" (check it to avoid interrupting forever).
void request_interrupt(GraphState &state, std::string_view reason);

using GraphFunctionFn = Result<void> (*)(Session &session, GraphState &state, void *user_data);
// Returns the next node name, or kGraphEnd. The view must outlive the call (string literal, node
// name, or a value in state).
using GraphRouterFn = Result<std::string_view> (*)(Session &session, const GraphState &state, void *user_data);
// Builds a Tool node's JSON arguments from state.
using GraphToolArgsFn = Result<std::string> (*)(Session &session, const GraphState &state, void *user_data);
// make_loop termination test, checked after every iteration.
using GraphUntilFn = Result<bool> (*)(Session &session, const GraphState &state, void *user_data);
// Agent node: extra system-prompt text computed per run of the node (e.g. retrieved context). An
// empty result adds nothing; an error fails the node.
using GraphContextFn = Result<std::string> (*)(Session &session, const GraphState &state, void *user_data);

enum class GraphNodeKind {
    Agent,    // run_turn on the node's FlowMemory (see history_key) with system_prompt + allowed_tools
    Function, // GraphFunctionFn
    Tool,     // execute_tool(tool_name, args)
    Parallel, // fan out to branches, join, then follow this node's own edge (make_parallel)
    Loop,     // loop controller (make_loop)
};

struct GraphNode {
    std::string name;
    GraphNodeKind kind = GraphNodeKind::Function;

    // Agent: the user message is state[input_key] when set, otherwise prompt. The final reply is
    // written to state[output_key] when output_key is non-empty. Tool: input_key may hold the
    // JSON arguments; the tool result goes to output_key.
    std::string system_prompt;
    std::vector<std::string> allowed_tools; // empty = every registered tool
    std::string prompt;
    std::string input_key;
    std::string output_key;
    // Agent: pause the run before executing model-requested tool calls (with a permission_check
    // configured, only before calls it rates Ask). The calls are exposed in
    // GraphRun::pending_tool_calls; resume_graph approves or denies them.
    bool interrupt_before_tools = false;
    // Agent: key into GraphState::histories. Empty = the node name (a private conversation). Agent
    // nodes with the same history_key continue one shared conversation.
    std::string history_key;
    // Agent: called before each run of the node; its text is prepended to system_prompt for that
    // run only (nothing is written to the history). Receives user_data.
    GraphContextFn context_provider = nullptr;
    // Agent: Pre/Post tool hooks active while this node runs. Inside a Parallel node's branches they
    // may be called from worker threads.
    std::vector<ToolHook> hooks;
    // Agent: run the turn on this model of GraphRun::models (acquire_model) instead of the run's
    // Session. Empty = the run's Session. The run fails when it is set and GraphRun::models is null.
    std::string model;

    GraphFunctionFn function = nullptr; // Function

    std::string tool_name;               // Tool
    GraphToolArgsFn tool_args = nullptr; // Tool: overrides input_key; neither = "{}"

    std::vector<std::string> branches; // Parallel: entry node of each branch

    std::string loop_body;        // Loop: entry node of the body
    int max_iterations = 1;       // Loop
    GraphUntilFn until = nullptr; // Loop: optional early exit

    void *user_data = nullptr; // passed to function / tool_args / until / context_provider

    // Where the run goes when this node's transition resolves to kGraphEnd. Empty = the graph ends.
    // Set by the builders to chain composed steps; a branch of a Parallel node ends when it reaches
    // the Parallel node itself.
    std::string exit_to;
};

// Static edge (router == nullptr): from -> to. Conditional edge: router picks the target. A node
// has at most one outgoing edge; a node without one transitions to kGraphEnd.
struct GraphEdge {
    std::string from;
    std::string to;
    GraphRouterFn router = nullptr;
    void *user_data = nullptr;
};

struct Graph {
    std::string name;
    std::string entry;
    std::vector<GraphNode> nodes;
    std::vector<GraphEdge> edges;
    int max_steps = 1000; // guard against routing cycles, per run_graph / resume_graph call
};

// ADK-style builders. Node names must stay unique across the composed graph.
Graph make_graph(GraphNode node);
// Runs steps in order: each step's end continues at the next step's entry.
Graph make_sequential(std::string name, std::vector<Graph> steps);
// Fans out to every branch (concurrently when the provider supports parallel inference), joins,
// then continues. Branches must write distinct state keys; a key written by two branches fails the run.
Graph make_parallel(std::string name, std::vector<Graph> branches);
// Runs body until until() returns true (checked after each iteration) or max_iterations is reached.
Graph make_loop(std::string name, Graph body, int max_iterations, GraphUntilFn until = nullptr, void *user_data = nullptr);

// Exit: checkpoint only when run_graph / resume_graph returns. Async: after every node, written by a
// background thread (bounded queue) that is flushed before the call returns; a crash can lose the
// checkpoints still queued, so a resume may re-run up to that many completed nodes. Sync: after
// every node, before the next one starts.
enum class GraphDurability { Exit, Async, Sync };

enum class GraphRunStatus { Pending, Running, Completed, Interrupted, Failed };

struct ModelRegistry; // model registry section below

enum class GraphEventType { NodeStart, NodeEnd, StateWrite, ToolCall, ToolResult, Interrupt, Resume, Checkpoint, NodeFailed };

struct GraphEvent {
    GraphEventType type = GraphEventType::NodeStart;
    std::string node;
    std::string key;          // StateWrite: state key. ToolCall/ToolResult: tool name. Checkpoint: store key.
    int64_t timestamp_ms = 0; // wall clock, ms since the Unix epoch
    std::string payload;      // short (truncated) value, arguments, result, reason or error
};

// One stored checkpoint of a thread, as listed by CheckpointerOps::list / list_graph_checkpoints.
struct GraphCheckpointInfo {
    uint64_t step = 0;
    int64_t timestamp_ms = 0; // when it was written (ms since the Unix epoch)
    std::string metadata;     // JSON object written by the runtime: {"graph", "step", "next", "status", ...}
};

struct CheckpointerConfig {
    // Backend-specific location. In-memory: unused. SQLite: file path, or empty / ":memory:".
    std::string path;
};

// Pluggable graph checkpointer, same pattern as StoreOps / VectorStoreOps. open() returns the state
// passed to every other call. Implementations must be safe to call from several threads on one
// state (the Async writer calls put from a background thread). Steps of one thread only grow;
// putting an existing (thread_id, step) replaces it.
struct CheckpointerOps {
    using OpenFn = Result<std::shared_ptr<void>> (*)(const CheckpointerConfig &config);
    using PutFn = Result<void> (*)(void *state, std::string_view thread_id, uint64_t step, std::string_view blob,
                                   std::string_view metadata);
    // The checkpoint with the highest step. ErrorCode::KeyNotFound when the thread has none.
    using GetLatestFn = Result<std::string> (*)(void *state, std::string_view thread_id);
    // ErrorCode::KeyNotFound when that step was not stored.
    using GetFn = Result<std::string> (*)(void *state, std::string_view thread_id, uint64_t step);
    // Every stored checkpoint of the thread, oldest step first (empty when there are none).
    using ListFn = Result<std::vector<GraphCheckpointInfo>> (*)(void *state, std::string_view thread_id);
    using DeleteThreadFn = Result<void> (*)(void *state, std::string_view thread_id);

    OpenFn open = nullptr;
    PutFn put = nullptr;
    GetLatestFn get_latest = nullptr;
    GetFn get = nullptr;
    ListFn list = nullptr;
    DeleteThreadFn delete_thread = nullptr;
};

// Checkpoints kept in process memory (nothing persists past the Checkpointer). The optional
// agent-store-sqlite library provides agent::store::sqlite_checkpointer().
const CheckpointerOps *in_memory_checkpointer();

// An opened checkpointer. Copies share the same state and Async writer. ops == nullptr (the
// default) stores checkpoints through the Session's StoreOps under checkpoint:<thread_id>:<step>,
// as before CheckpointerOps existed.
struct Checkpointer {
    const CheckpointerOps *ops = nullptr;
    std::shared_ptr<void> state;  // from ops->open
    std::shared_ptr<void> writer; // runtime-owned Async background writer (one thread per checkpointer)
};

// Opens ops with config. async_queue_capacity bounds the Async writer's queue: when that many
// checkpoints are waiting, the run blocks until the writer catches up.
Result<Checkpointer> open_checkpointer(const CheckpointerOps *ops, const CheckpointerConfig &config = {}, size_t async_queue_capacity = 64);

struct GraphRun {
    // Checkpoints and per-thread state are keyed by this. Empty = "default".
    std::string thread_id;
    GraphDurability durability = GraphDurability::Async;
    // Where checkpoints go. Default: the Session's StoreOps. Set it on the GraphRun passed to
    // resume_graph too.
    Checkpointer checkpointer;

    // Seed values before run_graph (prefer set_state); the final state afterwards.
    GraphState state;

    GraphRunStatus status = GraphRunStatus::Pending;
    std::string next_node; // node that runs next; the interrupted / failed node when paused
    uint64_t step = 0;     // nodes completed in this thread (checkpoint:<thread_id>:<step>)
    Error error;           // set when status == Failed

    std::string interrupt_node;
    std::string interrupt_reason;
    std::vector<ToolCall> pending_tool_calls; // interrupt_before_tools: the calls awaiting approval

    std::vector<GraphEvent> events; // appended to by every run_graph / resume_graph call

    // Resolves GraphNode::model. Not owned; must outlive the run_graph / resume_graph call.
    ModelRegistry *models = nullptr;
};

// Runs graph from graph.entry. Per-thread state and Agent histories from this thread's latest
// checkpoint (if any) are loaded first, then user:/app: keys from the store, then run.state's seed.
// Returns ok when the run completed or was interrupted (check run.status); a node failure returns
// the error with run.status == Failed, after checkpointing so resume_graph can retry the node.
Result<void> run_graph(Session &session, const Graph &graph, GraphRun &run);
// Reloads checkpoint:<thread_id>:latest and continues from its next node. human_input (if any) is
// written to "temp:human_input". Pending tool calls are approved when human_input is "approve",
// "approved", "yes", "y", "allow" or "ok" (case-insensitive) and denied otherwise.
Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run,
                          std::string_view human_input = {});
// Same, with an explicit decision for pending tool calls (Allow approves; Deny and Ask deny).
Result<void> resume_graph(Session &session, const Graph &graph, std::string_view thread_id, GraphRun &run, PermissionDecision decision,
                          std::string_view human_input = {});

// Stored checkpoints of a thread, oldest first.
Result<std::vector<GraphCheckpointInfo>> list_graph_checkpoints(Session &session, std::string_view thread_id,
                                                                const Checkpointer &checkpointer = {});
// Deletes a thread's checkpoints (and its persisted events); a later run_graph starts it fresh.
Result<void> delete_graph_thread(Session &session, std::string_view thread_id, const Checkpointer &checkpointer = {});

} // namespace agent

// ============================================================================================
// -------------------------- RAG: embeddings, rerank, vector stores --------------------------
// Non-chat engine tasks (AGENT_CAP_EMBED / AGENT_CAP_RERANK), a pluggable VectorStoreOps table with an
// in-memory brute-force default, a character/paragraph chunker, and the index -> query -> rerank
// pipeline plus a "search_knowledge" tool. Embedding, reranking and chat normally use different
// models, so each runs in its own Session (e.g. the llama engine with engine_options
// {"mode":"embed"} / {"mode":"rerank"}). See docs/api/rag.md.
// ============================================================================================

namespace agent {

// One pooled, L2-normalized embedding per text, in order. session must be an AiProvider::Engine
// session whose engine performs AGENT_CAP_EMBED. Texts go to the engine in batches. One session must
// not run embed / rerank / run_turn concurrently from several threads.
Result<std::vector<std::vector<float>>> embed(Session &session, std::span<const std::string_view> texts);
// One relevance score per document, in document order (higher = more relevant; the scale is the
// model's, e.g. raw reranker logits). session's engine must perform AGENT_CAP_RERANK.
Result<std::vector<float>> rerank(Session &session, std::string_view query, std::span<const std::string_view> docs);

struct VectorStoreConfig {
    // Backend-specific location. In-memory: unused. sqlite-vec: file path, or empty / ":memory:".
    std::string path;
    // Collection (table) name: [A-Za-z_][A-Za-z0-9_]{0,63}.
    std::string name = "rag";
    // Embedding dimensionality. 0 in RagIndex::config = taken from the first embedding.
    int dimensions = 0;
};

struct VectorHit {
    std::string id;
    std::string text;
    std::string metadata_json;
    float score = 0.0f; // cosine similarity, 1 = same direction
};

// Pluggable vector store, same pattern as StoreOps. open() returns the state passed to every other
// call. Implementations must be safe to call from several threads on one state.
struct VectorStoreOps {
    using OpenFn = Result<std::shared_ptr<void>> (*)(const VectorStoreConfig &config);
    // Inserts or replaces the entry with this id.
    using UpsertFn = Result<void> (*)(void *state, std::string_view id, std::string_view text, std::span<const float> embedding,
                                      std::string_view metadata_json);
    // Up to k entries, most similar first.
    using QueryTopkFn = Result<std::vector<VectorHit>> (*)(void *state, std::span<const float> embedding, int k);
    // true when an entry was removed, false when id was not present.
    using EraseFn = Result<bool> (*)(void *state, std::string_view id);
    using ClearFn = Result<void> (*)(void *state);

    OpenFn open = nullptr;
    UpsertFn upsert = nullptr;
    QueryTopkFn query_topk = nullptr;
    EraseFn erase = nullptr;
    ClearFn clear = nullptr;
};

// Brute-force cosine search over vectors kept in process memory (nothing persists). The optional
// agent-store-sqlite library provides agent::store::sqlite_vector_store() (sqlite-vec).
const VectorStoreOps *in_memory_vector_store();

struct ChunkOptions {
    size_t size = 1000;   // maximum bytes per chunk
    size_t overlap = 150; // bytes repeated from the end of the previous chunk; must be < size
};

// Splits text into chunks of at most options.size bytes, breaking preferably at paragraph breaks
// (blank lines), then at line / sentence ends, then at whitespace, never inside a UTF-8 sequence.
// Consecutive chunks overlap by up to options.overlap bytes, starting at a word boundary. The views
// point into text; surrounding whitespace is trimmed and empty chunks are dropped.
Result<std::vector<std::string_view>> chunk_text(std::string_view text, const ChunkOptions &options = {});

// A vector store holding document chunks under ids "<doc_id>#<n>", with metadata
// {"doc_id": ..., "chunk": n, "offset": byte offset in the document}.
struct RagIndex {
    const VectorStoreOps *store = nullptr; // nullptr = in_memory_vector_store()
    VectorStoreConfig config;
    // Opened by the first rag_index / rag_query (dimensions from that embedding when
    // config.dimensions is 0) or explicitly by open_rag_index. Not thread-safe until opened.
    std::shared_ptr<void> state;
};

// Opens index.store with index.config (dimensions overrides config.dimensions when > 0).
Result<void> open_rag_index(RagIndex &index, int dimensions = 0);

struct RagHit {
    std::string id; // "<doc_id>#<n>"
    std::string doc_id;
    std::string text;
    std::string metadata_json;
    float similarity = 0.0f; // cosine similarity from the vector store
    float score = 0.0f;      // ranking score: the rerank score when reranked, otherwise similarity
};

// Chunks text, embeds the chunks with embed_session and upserts them as "<doc_id>#<n>". Re-indexing
// a doc_id replaces its chunks (stale higher-numbered chunks are erased).
Result<void> rag_index(Session &embed_session, RagIndex &index, std::string_view doc_id, std::string_view text,
                       const ChunkOptions &options = {});
// Removes every chunk of doc_id.
Result<void> rag_erase(RagIndex &index, std::string_view doc_id);
// Embeds query and returns the k most similar chunks. With rerank_session, max(4*k, 20) candidates
// are fetched, reordered by the reranker, and the top k returned.
Result<std::vector<RagHit>> rag_query(Session &embed_session, RagIndex &index, std::string_view query, int k,
                                      Session *rerank_session = nullptr);

// A "search_knowledge" tool ({"query": string, "k": integer (optional)}) that runs rag_query and
// returns [{"id", "score", "text"}, ...] as JSON. The tool's state is retained by chat_session;
// embed_session, index and rerank_session are referenced and must outlive it (and not be moved).
// Calls through one tool are serialized. Add it with chat_session.config.tools.push_back(...) and
// rebuild_tool_index(chat_session).
Tool make_rag_tool(Session &chat_session, Session &embed_session, RagIndex &index, Session *rerank_session = nullptr, int k = 4);

} // namespace agent

// ============================================================================================
// -------------------------- MCP stdio client (Phase 3) --------------------------
// Model Context Protocol servers run as child processes speaking newline-delimited JSON-RPC 2.0 on
// stdin/stdout. Each remote tool becomes an ordinary Tool named "<server>.<tool>" whose callback
// forwards to tools/call, so allowed_tools, the permission gate and graph Tool nodes apply as for
// any other tool. The Session owns the server processes; they stop on disconnect or when the
// Session is destroyed. See docs/api/mcp.md.
// ============================================================================================

namespace agent {

struct McpServerConfig {
    std::string name;    // tool prefix; letters, digits, '_' and '-' only; unique per Session
    std::string command; // program to start (looked up on PATH; not run through a shell)
    std::vector<std::string> args;
    std::map<std::string, std::string> env; // added to / overriding the inherited environment
    int timeout_ms = 30000;                 // per request (initialize, each tools/list page, tools/call)
    std::string working_directory;          // empty = inherit the host's
    // Opt-in OS sandbox (off by default: servers such as filesystem or npx-launched ones need broad
    // access). When enabled, the program's directory is added read-only and working_directory
    // read-write; list everything else the server needs. See docs/security-sandbox.md.
    SandboxPolicy sandbox;
};

// Starts the server, runs the initialize handshake and tools/list (following pagination), and
// registers every remote tool on the Session (then rebuild_tool_index). Returns the registered tools.
Result<std::vector<Tool>> connect_mcp_server(Session &session, const McpServerConfig &config);
// Unregisters the server's tools and shuts the process down (close stdin, then terminate).
Result<void> disconnect_mcp_server(Session &session, std::string_view name);
void disconnect_mcp_servers(Session &session);
// Names of the connected servers, in connection order.
std::vector<std::string> list_mcp_servers(const Session &session);

} // namespace agent

// ============================================================================================
// -------------------------- agent.json graph loader (Phase 3) --------------------------
// Declarative graphs for CLI users: nodes, edges (static or routed on a state value), workflow
// nodes (sequential / parallel / loop), MCP servers to connect, and engine/model hints. Function
// nodes reference C++ functions registered by name. See docs/api/agent-json.md.
// ============================================================================================

namespace agent {

// Process-wide registry used by "kind": "function" nodes ("function": "<name>"). Registering an
// existing name replaces it. user_data must outlive every graph loaded with it.
void register_graph_function(std::string_view name, GraphFunctionFn fn, void *user_data = nullptr);
void unregister_graph_function(std::string_view name);

struct LoadedGraph {
    Graph graph;
    std::string description;
    // Hints for the host (the CLI picks the engine/model from them); the loader does not act on them.
    std::string engine;
    std::string model;
    std::vector<McpServerConfig> mcp_servers; // as declared
    // Non-fatal findings: unknown fields, unreachable nodes, cycles without a conditional edge, ...
    std::vector<std::string> warnings;
    // Owns the route tables, loop conditions and static tool arguments that graph's user_data
    // pointers refer to. Keep the LoadedGraph alive while graph is in use (moving it is fine).
    std::shared_ptr<void> storage;
};

// Parses and validates a graph without a Session: no MCP servers are started and tool names are
// not checked. path_or_json is JSON text when it starts with '{' or '[' (after whitespace), else a file path.
Result<LoadedGraph> parse_graph_json(std::string_view path_or_json);
// parse_graph_json, then connects the declared mcp_servers (a server already connected under the
// same name is reused) and checks that every Tool node names a registered tool. On failure, servers
// connected by this call are disconnected again.
Result<LoadedGraph> load_graph_json(Session &session, std::string_view path_or_json);

} // namespace agent

// ============================================================================================
// -------------------------- Model registry (Phase 2) --------------------------
// Named model templates loaded lazily into Sessions under a RAM/VRAM budget. acquire_model returns
// a lease on the model's Session, loading it with agent::init (and evicting least-recently-used idle,
// unpinned models first) when needed. A leased or pinned model is never evicted. Thread-safe. Layered
// on top of Session creation: a loaded model is an ordinary Session. See docs/api/models.md.
// ============================================================================================

namespace agent {

struct ModelSpec;

// Called once after a model's Session is created and before the first lease is handed out (e.g. to
// register event hooks, add tools or share a store). An error unloads the model and fails the acquire.
using ModelLoadFn = Result<void> (*)(Session &session, const ModelSpec &spec, void *user_data);

struct ModelSpec {
    std::string name; // unique per registry
    // Template passed to agent::init: provider, engine, model path, engine_options, engine_isolation,
    // context_window, max_tokens, workspace_dir, tools, permission gate, store, ...
    Config config;
    // Memory the loaded model needs. 0 = estimate_model_memory() from the file size(s) on disk.
    uint64_t estimated_bytes = 0;
    // Never evicted once loaded (unload_model still unloads it).
    bool pinned = false;
    ModelLoadFn on_load = nullptr;
    void *on_load_user_data = nullptr;
};

struct ModelBudget {
    uint64_t ram_bytes = 0;  // 0 = unlimited
    uint64_t vram_bytes = 0; // 0 = unlimited. Best effort: only models with engine option gpu_layers > 0 count
    // Load a model even when evicting every idle model does not make it fit (the acquire then
    // succeeds over budget instead of failing with ErrorCode::OutOfMemory).
    bool allow_overcommit = false;
    // How long acquire_model waits for leases to be released when the model does not fit because the
    // other models are in use. 0 = fail at once.
    int wait_ms = 0;
};

struct ModelMemoryEstimate {
    uint64_t file_bytes = 0; // model file(s) on disk: model path (a directory is summed), split-GGUF
                             // shards, and files named by string values in engine_options (mmproj, ...)
    uint64_t ram_bytes = 0;  // counted against ModelBudget::ram_bytes
    uint64_t vram_bytes = 0; // counted against ModelBudget::vram_bytes (engine option gpu_layers > 0)
};

// The memory a spec will need once loaded: estimated_bytes when set, otherwise file size x a per-engine
// factor (+ a KV-cache estimate for llama). Fails with FilesystemError when the model path is missing.
// The factors are documented in docs/api/models.md.
Result<ModelMemoryEstimate> estimate_model_memory(const ModelSpec &spec);

// Resident set size of this process in bytes (/proc/self/statm, GetProcessMemoryInfo, task_info);
// 0 when unavailable. Models running under EngineIsolation::Process live in agent-engine-host
// children and do not show up here.
uint64_t process_rss_bytes();

// A handle: copies share one registry. The registry's state lives until the last handle and the last
// lease are gone; models still loaded then are unloaded.
struct ModelRegistry {
    explicit ModelRegistry(ModelBudget budget = {});
    std::shared_ptr<void> state;
};

// A model in use. The Session stays loaded (never evicted) while any copy of the lease exists;
// destroying the last copy or release_model() returns it to the registry. One Session serves every
// lease on the same model: concurrent run_turn calls on it need turn_mutex (or a graph Parallel node,
// which uses the Session's worker arenas).
struct ModelLease {
    Session *session = nullptr;
    std::string name;
    std::shared_ptr<std::mutex> turn_mutex; // shared by every lease on this loaded model
    std::shared_ptr<void> hold;             // internal: the lease count

    Session &operator*() const { return *session; }
    Session *operator->() const { return session; }
    explicit operator bool() const { return session != nullptr; }
};

struct LoadedModelInfo {
    std::string name;
    uint64_t ram_bytes = 0;  // estimate counted against the RAM budget
    uint64_t vram_bytes = 0; // estimate counted against the VRAM budget
    bool pinned = false;
    int leases = 0;           // outstanding leases (in use when > 0)
    int64_t loaded_at_ms = 0; // wall clock, ms since the Unix epoch
    int64_t last_used_ms = 0; // last acquire or release
    int64_t load_time_ms = 0; // how long agent::init took
};

// Adds a model. Re-registering a name replaces the spec (an idle loaded model is unloaded first; a
// leased one fails with InvalidConfig). Fails with InvalidConfig for an empty name.
Result<void> register_model(ModelRegistry &registry, ModelSpec spec);
bool has_model(const ModelRegistry &registry, std::string_view name);
// Lease on the named model, loading it first if needed. KeyNotFound for an unregistered name;
// OutOfMemory when it cannot fit (see ModelBudget); the agent::init error when loading fails.
Result<ModelLease> acquire_model(ModelRegistry &registry, std::string_view name);
// Returns the lease early (the same as destroying it). Safe to call twice.
void release_model(ModelLease &lease);
// Unloads a loaded model now. Not loaded = ok. InvalidConfig when it is leased.
Result<void> unload_model(ModelRegistry &registry, std::string_view name);
// Loaded (and loading) models, most recently used first.
std::vector<LoadedModelInfo> loaded_models(const ModelRegistry &registry);
// Replaces the budget and evicts idle, unpinned models until the loaded ones fit it.
void set_budget(ModelRegistry &registry, const ModelBudget &budget);
ModelBudget get_budget(const ModelRegistry &registry);

} // namespace agent
