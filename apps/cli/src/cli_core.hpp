#pragma once

// Internal library of the `agent` CLI (target agent_cli_core): the pure, testable logic --
// argument parsing, config.json / engines.json round trips, model/engine resolution,
// slash-command parsing -- plus a few thin OS helpers (cli_platform.cpp). Not a public API.

#include <agent-cpp/agent.hpp>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// -------------------------- Argument parsing --------------------------

struct Flag {
    std::string name;  // long name without dashes ("model")
    std::string value; // empty for boolean flags
};

struct CommandLine {
    std::string command;                  // "setup", "engine", "model", "chat", "run", "help", "version"
    std::string subcommand;               // "ls" / "add" / "rm" / "pull" for engine and model
    std::vector<std::string> positionals; // after the subcommand
    std::vector<Flag> flags;
    bool help = false; // -h / --help anywhere

    bool has(std::string_view flag) const;
    // Value of a flag, or fallback when it was not given.
    std::string_view get(std::string_view flag, std::string_view fallback = {}) const;
};

// argv without the program name. Validates the command, its subcommand and its flags.
// Global flags (any command): --help/-h, --config-dir DIR.
Result<CommandLine> parse_command_line(std::span<const std::string_view> args);

std::string usage_text();

// The one-line summary printed after every chat turn from the last model call: prompt speed, generation
// speed and KV cache size, e.g. "prompt 412 tok (380 cached) 548 tok/s | gen 63 tok 91 tok/s | KV 475/2048 q4_0 3.1 MiB".
// Empty when the engine reported none of it (non-LLM engines). Speeds use the engine's own timings.
std::string format_turn_footer(const TokenUsage &u);

// -------------------------- JSON helpers --------------------------

std::string json_escape(std::string_view s);

// -------------------------- Hardware --------------------------

struct HardwareInfo {
    int cpu_cores = 0;
    bool nvidia_smi = false;            // nvidia-smi found on PATH
    bool nvidia_device = false;         // /dev/nvidia0 exists
    std::vector<std::string> dri_nodes; // /dev/dri/renderD* (Vulkan-capable GPUs usually show up here)
};

// "cuda" when an NVIDIA GPU is visible, "vulkan" when a DRI render node is, "cpu" otherwise.
std::string suggest_backend(const HardwareInfo &hw);

// -------------------------- config.json --------------------------

// ROADMAP §6 #9 starter pack, minus trellis (no engine yet).
std::span<const std::string_view> starter_pack();

struct CliConfig {
    int version = 1;
    std::string workspace;     // empty = default_workspace_dir()
    std::string models_dir;    // empty = agent::hub::default_models_dir()
    std::string default_model; // catalogue name or path; empty = first chat model in models.json
    std::vector<std::string> capabilities;
    std::string backend = "cpu"; // cpu | cuda | vulkan (recorded; prebuilt downloads do not exist yet)
    HardwareInfo hardware;
    int context_window = 4096;
    int max_tokens = 512;
    float temperature = 0.7f;
    // RAM budget of the model registry behind `chat` /model switches and multi-model `run` graphs.
    // 0 = unlimited. Exceeding it evicts the least recently used idle model.
    int64_t memory_budget_mb = 0;
    // JavaScript tooling for MCP servers and skills, chosen in `agent setup`: "bun" (bun x / bun install), "node" (a private
    // Node.js with pnpm; npm is never used) or "nub" (the nub toolkit, a pnpm-compatible CLI over Node.js).
    std::string js_runtime = "bun";
};

std::string config_to_json(const CliConfig &config);
Result<CliConfig> config_from_json(std::string_view json);
// Missing file -> default CliConfig. Saves atomically (".tmp" + rename), creating the directory.
Result<CliConfig> load_config(const std::string &path);
Result<void> save_config(const std::string &path, const CliConfig &config);

// -------------------------- engines.json --------------------------

// Same file agent::load_engine_manifest reads ({"engines": [{"path": ...}]}); the CLI adds
// name / version / caps / backend for display and lookup.
struct EngineEntry {
    std::string name;
    std::string path;
    std::string version;
    std::vector<std::string> caps;
    std::string backend = "cpu";
    std::string isolation; // "process" | "inprocess" | empty (= process in the CLI)
};

std::string engines_to_json(std::span<const EngineEntry> engines);
Result<std::vector<EngineEntry>> engines_from_json(std::string_view json);
Result<std::vector<EngineEntry>> load_engines(const std::string &path);
Result<void> save_engines(const std::string &path, std::span<const EngineEntry> engines);
// Replaces the entry with the same name, or appends.
void upsert_engine(std::vector<EngineEntry> &engines, EngineEntry entry);
bool remove_engine(std::vector<EngineEntry> &engines, std::string_view name);
const EngineEntry *find_engine_entry(std::span<const EngineEntry> engines, std::string_view name);
// The CLI runs engines out of process unless --inprocess is given or engines.json marks the engine
// "isolation": "inprocess".
EngineIsolation cli_engine_isolation(bool inprocess_flag, std::string_view entry_isolation);
// The engine sandbox opens the GPU device nodes for any backend but "cpu" (cuda, vulkan, rocm, ...).
bool cli_engine_gpu(std::string_view backend);
// --no-sandbox (a global flag): turns the sandboxes of this run off. The default is on; where the OS cannot provide one, it is
// off anyway and a warning says so.
void set_no_sandbox(bool value);
void apply_sandbox_choice(Config &cfg);

// Names of the agent_capability bits set in caps ("text_gen", "tool_call", ...).
std::vector<std::string> caps_to_names(uint64_t caps);
// "agent-engine-llama.so" / ".dll" / ".dylib" for the current platform.
std::string plugin_file_name(std::string_view engine_name);
// Comma/space separated list -> names; empty entries dropped, duplicates removed.
std::vector<std::string> split_list(std::string_view list);

// -------------------------- Model / engine resolution --------------------------

struct ModelRef {
    std::string name;
    std::string path;
};

using ExistsFn = bool (*)(std::string_view path);

// spec is an existing file path, or a catalogue name (models.json). Empty spec: config default,
// else the first catalogue entry an engine can run as chat (.gguf).
Result<ModelRef> resolve_model(std::string_view spec, std::string_view default_model, std::span<const ModelRef> catalog, ExistsFn exists);

// Engine that runs a model file, by extension: .gguf -> llama, .onnx -> onnx,
// .safetensors/.ckpt/.sft -> sdcpp. Empty when unknown.
std::string_view engine_for_model(std::string_view model_path);

struct EngineRef {
    std::string name; // empty when engine_flag is a path (known only after loading)
    std::string path; // empty when the engine is not in engines.json (caller may search for it)
};

// engine_flag: existing plugin path, or an engine name; empty = chosen from the model's file type.
Result<EngineRef> resolve_engine(std::string_view engine_flag, std::string_view model_path, std::span<const EngineEntry> engines,
                                 ExistsFn exists);

std::string mime_for_path(std::string_view path);

// -------------------------- Chat --------------------------

enum class SlashKind { None, Exit, Clear, Model, Image, Open, Stats, Help, Cli, Unknown };

// SlashKind::Cli: a slash command that runs an ordinary `agent` subcommand and shows its output: /doctor, /pull SPEC, /fit SPEC,
// /recommend [TEXT], /search [TEXT], /setup, /wake, /schedule ... (arg = the name, rest = the arguments as typed).
struct SlashCommand {
    SlashKind kind = SlashKind::None;
    std::string arg;  // first argument (path / name / index); quotes removed
    std::string rest; // remaining text (the /image prompt)
    std::string text; // SlashKind::None: the message to send ("//x" -> "/x")
};

SlashCommand parse_slash_command(std::string_view line);
// The words after `agent` that run a SlashKind::Cli command ({"model", "pull", SPEC}, {"doctor"}, ...); empty for any other command.
std::vector<std::string> slash_cli_argv(const SlashCommand &cmd);

// Hides tool-call markup (<tool_call>...</tool_call>, the Hermes / Qwen format) from streamed tokens;
// the chat shows each call as a "[tool] ..." line instead. Tags split across tokens are handled.
struct StreamFilter {
    std::string pending;
    bool inside = false;
};
// Text of token that should be printed now.
std::string filter_stream(StreamFilter &filter, std::string_view token);
// End of the turn: whatever was held back (an unfinished tag prefix outside a call).
std::string flush_stream(StreamFilter &filter);

// Conversation index kept in the store under "cli:conversations": one id per line, newest last.
std::string add_conversation_id(std::string_view index, std::string_view id);
std::vector<std::string> conversation_ids(std::string_view index);

// "[#########-----------]  45.0%  221.1/491.4 MiB"
std::string format_progress(uint64_t done, uint64_t total, int width = 30);

// Truncates to max_bytes (on a UTF-8 boundary) and replaces newlines with spaces, appending "..."
std::string one_line(std::string_view text, size_t max_bytes);

// -------------------------- Platform (cli_platform.cpp) --------------------------

// $AGENT_CONFIG_DIR, else %APPDATA%\agent (Windows), else $XDG_CONFIG_HOME/agent, else ~/.config/agent.
std::string default_config_dir();
// %LOCALAPPDATA%\agent\workspace (Windows), else $XDG_DATA_HOME/agent/workspace, else ~/.local/share/agent/workspace.
std::string default_workspace_dir();
std::string executable_dir();
// Where `setup` and the engine fallback look for plugins: next to the binary, <exe>/engines and
// its subdirectories (build tree), <exe>/../lib/agent/engines (install prefix), then extra.
std::vector<std::string> engine_search_dirs(std::string_view extra_dir = {});
// First <dir>/agent-engine-<name><suffix> that exists; empty when none.
std::string find_plugin(std::string_view engine_name, std::span<const std::string> dirs);

HardwareInfo detect_hardware();
// Installed RAM in bytes; 0 when it cannot be read.
uint64_t total_ram_bytes();

bool file_exists(std::string_view path);
bool path_exists(std::string_view path); // file or folder
bool stdin_is_tty();
// Prints prompt on stderr and reads one line from the terminal without echoing it (passwords, keys). False when
// stdin is not a terminal or reading failed.
bool read_hidden_line(const char *prompt, std::string &out);
bool stream_is_tty(std::FILE *stream);
// Turns on ANSI escape processing for a console stream (Windows 10+ needs this; elsewhere a no-op
// that returns true). False when the stream is not a console or the console cannot do it.
bool enable_ansi(std::FILE *stream);
// Columns of the terminal behind stream; 0 when unknown.
int terminal_width(std::FILE *stream);

enum class ReadStatus { Line, Eof, Interrupted };
// Reads one line (without the trailing newline / CR). Interrupted = a signal arrived (Ctrl-C).
ReadStatus read_line(std::FILE *in, std::string &line);

// Points fd 2 at path (append) so engine libraries that log straight to stderr (llama.cpp, ...)
// do not flood the terminal. Returns a stream that still writes to the original stderr, or nullptr
// on failure (stderr is then unchanged).
std::FILE *redirect_stderr(const std::string &path);
// Undoes redirect_stderr and closes original.
void restore_stderr(std::FILE *original);

// Hands the file to the OS default application (xdg-open / open / ShellExecute); waits only for the
// opener itself, not for the application it launches.
Result<void> open_with_os(const std::string &path);

} // namespace agent::cli
