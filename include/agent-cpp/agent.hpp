#pragma once

#include <cstdint>
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

namespace tools {

struct WorkspaceContext {
    const std::string *workspace_dir = nullptr;
    bool sandbox = true;
};

std::vector<Tool> get_filesystem_tools(std::shared_ptr<WorkspaceContext> ctx);
Tool get_terminal_tool(const std::string *workspace_dir);

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

enum class EventType {
    OnTurnStart,
    OnInferenceSubmit,
    OnInferenceComplete,
    OnToolCall,
    OnToolResult,
    OnStateTransition,
    OnCompression,
    OnPrune,
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
    std::vector<Media> media;
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

    using HookFn = void (*)(Session &session, FlowMemory &memory, std::string_view tool_name, std::string_view args,
                            std::string_view result, void *user_data);
    HookFn callback = nullptr;
    void *user_data = nullptr;
};

struct FlowState {
    std::string_view state_id;
    std::string_view system_prompt;
    std::span<const std::string_view> allowed_tools;
    std::span<const FlowHook> hooks;

    bool isolated_memory = false;

    using ContextProviderFn = std::string_view (*)(Session &session, FlowMemory &memory, void *context);
    ContextProviderFn context_provider = nullptr;

    using TransitionFn = Result<std::string_view> (*)(Session &session, FlowMemory &memory, void *context);
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

struct SubAgentResult {
    std::string id;
    bool ok = false;
    Error error;
    TurnStats stats;
};

class AgentScheduler {
  public:
    Result<void> spawn(std::string id, Flow flow, void *context = nullptr);

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
    std::vector<SubAgent> subagents_;
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
// the first plugin that fails to load.
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

Result<void> run_flow(Session &session, void *context, Flow &flow);
Result<bool> step_flow(Session &session, void *context, Flow &flow);

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
    // Conversation history of each Agent node, keyed by node name.
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

enum class GraphNodeKind {
    Agent,    // run_turn on the node's own FlowMemory with system_prompt + allowed_tools
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

    GraphFunctionFn function = nullptr; // Function

    std::string tool_name;               // Tool
    GraphToolArgsFn tool_args = nullptr; // Tool: overrides input_key; neither = "{}"

    std::vector<std::string> branches; // Parallel: entry node of each branch

    std::string loop_body;        // Loop: entry node of the body
    int max_iterations = 1;       // Loop
    GraphUntilFn until = nullptr; // Loop: optional early exit

    void *user_data = nullptr; // passed to function / tool_args / until

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

// Exit: checkpoint only when run_graph / resume_graph returns. Async: after every node (v1 writes
// synchronously, identical to Sync; reserved for a background writer). Sync: after every node,
// before the next one starts.
enum class GraphDurability { Exit, Async, Sync };

enum class GraphRunStatus { Pending, Running, Completed, Interrupted, Failed };

enum class GraphEventType { NodeStart, NodeEnd, StateWrite, ToolCall, ToolResult, Interrupt, Resume, Checkpoint, NodeFailed };

struct GraphEvent {
    GraphEventType type = GraphEventType::NodeStart;
    std::string node;
    std::string key;          // StateWrite: state key. ToolCall/ToolResult: tool name. Checkpoint: store key.
    int64_t timestamp_ms = 0; // wall clock, ms since the Unix epoch
    std::string payload;      // short (truncated) value, arguments, result, reason or error
};

struct GraphRun {
    // Checkpoints and per-thread state are keyed by this. Empty = "default".
    std::string thread_id;
    GraphDurability durability = GraphDurability::Async;

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

} // namespace agent
