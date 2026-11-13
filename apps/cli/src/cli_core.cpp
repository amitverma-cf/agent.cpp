#include "cli_core.hpp"

#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <simdjson.h>

namespace agent::cli {

namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

std::string lower(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool contains(std::span<const std::string_view> list, std::string_view v) { return std::find(list.begin(), list.end(), v) != list.end(); }

// -------------------------- Command table --------------------------

struct SubSpec {
    std::string_view name;
    int min_positionals;
    int max_positionals;
};

struct CommandSpec {
    std::string_view name;
    std::span<const SubSpec> subs; // empty = no subcommand
    int min_positionals;           // used when subs is empty
    int max_positionals;
    std::span<const std::string_view> value_flags;
    std::span<const std::string_view> bool_flags;
};

constexpr std::string_view kGlobalValue[] = {"config-dir", "home"};
constexpr std::string_view kGlobalBool[] = {"help", "no-sandbox"};

constexpr std::string_view kSetupValue[] = {"caps", "workspace", "models-dir", "engines-dir", "backend", "model", "js-runtime"};
constexpr std::string_view kSetupBool[] = {"yes", "no-models", "download-engines", "install-runtimes"};

constexpr SubSpec kEngineSubs[] = {{"ls", 0, 0}, {"add", 1, 1}, {"rm", 1, 1}, {"pull", 1, 1}};
constexpr std::string_view kEngineValue[] = {"tag", "repo"};

constexpr SubSpec kSecretSubs[] = {{"set", 1, 1}, {"ls", 0, 0}, {"rm", 1, 1}, {"bind", 2, 2}, {"unbind", 2, 2}};
constexpr SubSpec kMcpSubs[] = {{"ls", 0, 0}, {"search", 1, 1}, {"add", 1, 1}, {"rm", 1, 1}, {"serve", 0, 0}};
constexpr std::string_view kMcpValue[] = {"workspace"};
constexpr std::string_view kMcpBool[] = {"project", "yes", "read-only"};
constexpr SubSpec kSkillSubs[] = {{"ls", 0, 0}, {"search", 1, 1}, {"add", 1, 1}, {"rm", 1, 1}};
constexpr std::string_view kSkillBool[] = {"project", "yes"};
constexpr SubSpec kSandboxSubs[] = {{"ls", 0, 0}, {"grant", 1, 1}, {"revoke", 1, 1}};
constexpr std::string_view kSandboxValue[] = {"for"};
constexpr std::string_view kSandboxBool[] = {"write", "project"};
constexpr std::string_view kWakeValue[] = {"workspace"};
constexpr std::string_view kWakeBool[] = {"dry-run", "yes"};
constexpr SubSpec kScheduleSubs[] = {{"ls", 0, 0}, {"add", 1, 64}, {"rm", 1, 1}};
constexpr std::string_view kScheduleValue[] = {"every", "daily", "cron"};
constexpr std::string_view kScheduleBool[] = {"yes"};
constexpr SubSpec kRuntimeSubs[] = {{"ls", 0, 0}, {"install", 1, 1}, {"rm", 1, 1}, {"path", 1, 1}, {"use", 1, 1}};
constexpr std::string_view kRuntimeValue[] = {"version"};
constexpr std::string_view kCleanBool[] = {"downloads", "cache", "logs", "yes"};
constexpr std::string_view kUpdateValue[] = {"repo", "tag"};
constexpr std::string_view kUpdateBool[] = {"check", "yes", "allow-unsigned"};
constexpr std::string_view kSecretValue[] = {"identity"};
constexpr std::string_view kSecretBool[] = {"stdin"};

constexpr SubSpec kModelSubs[] = {{"ls", 0, 0}, {"search", 0, 1}, {"info", 1, 1}, {"pull", 1, 1}, {"rm", 1, 1}};
constexpr std::string_view kModelValue[] = {"name", "quant", "mmproj", "include", "task", "engine"};
constexpr std::string_view kModelBool[] = {"allow-restricted", "keep-file", "no-mmproj"};

constexpr std::string_view kChatValue[] = {"model",      "engine",         "resume", "workspace", "ctx",
                                           "max-tokens", "engine-options", "prompt", "grammar"};
constexpr std::string_view kChatBool[] = {"yes", "list", "verbose", "inprocess", "plain", "jsonl", "json"};

constexpr std::string_view kRunValue[] = {"model", "engine", "workspace", "thread", "set", "resume", "input", "engine-options"};
constexpr std::string_view kRunBool[] = {"yes", "inprocess", "json", "swap-models"};

constexpr std::string_view kVoiceValue[] = {"model",     "engine", "asr-model",  "tts-model",      "vad-model", "input-wav", "output-dir",
                                            "workspace", "ctx",    "max-tokens", "engine-options", "resume",    "max-turns", "voice"};
constexpr std::string_view kVoiceBool[] = {"yes", "verbose", "inprocess", "barge-in"};

constexpr std::string_view kTaskValue[] = {"model",      "engine", "engine-options", "text", "file", "params", "workspace",
                                           "max-tokens", "doc",    "tensor",         "tools"};
constexpr std::string_view kTaskBool[] = {"inprocess", "all-values"};

constexpr CommandSpec kCommands[] = {
    {"setup", {}, 0, 0, kSetupValue, kSetupBool},
    {"init", {}, 0, 1, {}, {}},
    {"secret", kSecretSubs, 0, 0, kSecretValue, kSecretBool},
    {"doctor", {}, 0, 0, {}, {}},
    {"mcp", kMcpSubs, 0, 0, kMcpValue, kMcpBool},
    {"skill", kSkillSubs, 0, 0, {}, kSkillBool},
    {"sandbox", kSandboxSubs, 0, 0, kSandboxValue, kSandboxBool},
    {"schedule", kScheduleSubs, 0, 0, kScheduleValue, kScheduleBool},
    {"wake", {}, 0, 0, kWakeValue, kWakeBool},
    {"runtime", kRuntimeSubs, 0, 0, kRuntimeValue, {}},
    {"clean", {}, 0, 0, {}, kCleanBool},
    {"engine", kEngineSubs, 0, 0, kEngineValue, {}},
    {"model", kModelSubs, 0, 0, kModelValue, kModelBool},
    {"chat", {}, 0, 0, kChatValue, kChatBool},
    {"run", {}, 1, 1, kRunValue, kRunBool},
    {"voice", {}, 0, 0, kVoiceValue, kVoiceBool},
    {"task", {}, 1, 1, kTaskValue, kTaskBool},
    {"help", {}, 0, 1, {}, {}},
    {"version", {}, 0, 0, {}, {}},
    {"completion", {}, 1, 1, {}, {}},
    {"update", {}, 0, 0, kUpdateValue, kUpdateBool},
};

const CommandSpec *find_command(std::string_view name) {
    for (const auto &c : kCommands)
        if (c.name == name) return &c;
    return nullptr;
}

std::string_view expand_short(std::string_view s) {
    if (s == "-h") return "help";
    if (s == "-y") return "yes";
    if (s == "-m") return "model";
    if (s == "-e") return "engine";
    if (s == "-p") return "prompt";
    return {};
}

} // namespace

bool CommandLine::has(std::string_view flag) const {
    return std::any_of(flags.begin(), flags.end(), [&](const Flag &f) { return f.name == flag; });
}

std::string_view CommandLine::get(std::string_view flag, std::string_view fallback) const {
    for (auto it = flags.rbegin(); it != flags.rend(); ++it)
        if (it->name == flag) return it->value;
    return fallback;
}

Result<CommandLine> parse_command_line(std::span<const std::string_view> args) {
    CommandLine cl;
    const CommandSpec *spec = nullptr;
    std::vector<std::string> positionals;
    bool only_positionals = false;

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (!only_positionals && a == "--") {
            only_positionals = true;
            continue;
        }
        const bool is_long = !only_positionals && a.size() > 2 && a.substr(0, 2) == "--";
        const std::string_view short_name = only_positionals ? std::string_view() : expand_short(a);
        if (!is_long && short_name.empty() && !only_positionals && a.size() > 1 && a.front() == '-')
            return fail<CommandLine>(ErrorCode::InvalidConfig, "unknown option '" + std::string(a) + "'");
        if (!is_long && short_name.empty()) {
            if (!spec) {
                spec = find_command(a);
                if (!spec) return fail<CommandLine>(ErrorCode::InvalidConfig, "unknown command '" + std::string(a) + "'");
                cl.command = std::string(a);
            } else {
                positionals.emplace_back(a);
            }
            continue;
        }

        std::string_view name = is_long ? a.substr(2) : short_name;
        std::string value;
        bool has_inline_value = false;
        if (const size_t eq = name.find('='); is_long && eq != std::string_view::npos) {
            value = std::string(name.substr(eq + 1));
            name = name.substr(0, eq);
            has_inline_value = true;
        }
        const bool takes_value = contains(kGlobalValue, name) || (spec && contains(spec->value_flags, name));
        const bool is_bool = contains(kGlobalBool, name) || (spec && contains(spec->bool_flags, name));
        if (!takes_value && !is_bool) {
            std::string msg = "unknown option '--" + std::string(name) + "'";
            if (spec) msg += " for '" + cl.command + "'";
            return fail<CommandLine>(ErrorCode::InvalidConfig, msg);
        }
        if (takes_value && !has_inline_value) {
            if (i + 1 >= args.size())
                return fail<CommandLine>(ErrorCode::InvalidConfig, "option '--" + std::string(name) + "' needs a value");
            value = std::string(args[++i]);
        }
        if (is_bool && has_inline_value)
            return fail<CommandLine>(ErrorCode::InvalidConfig, "option '--" + std::string(name) + "' takes no value");
        if (name == "help") cl.help = true;
        else cl.flags.push_back(Flag{std::string(name), std::move(value)});
    }

    if (!spec) {
        if (cl.help) cl.command = "help";
        return ok(std::move(cl));
    }
    if (cl.help) return ok(std::move(cl)); // "agent chat --help": no positional validation

    int min_pos = spec->min_positionals, max_pos = spec->max_positionals;
    size_t first = 0;
    if (!spec->subs.empty()) {
        if (positionals.empty()) return fail<CommandLine>(ErrorCode::InvalidConfig, "'" + cl.command + "' needs a subcommand");
        const SubSpec *sub = nullptr;
        for (const auto &s : spec->subs)
            if (s.name == positionals[0]) sub = &s;
        if (!sub) return fail<CommandLine>(ErrorCode::InvalidConfig, "unknown subcommand '" + cl.command + " " + positionals[0] + "'");
        cl.subcommand = positionals[0];
        min_pos = sub->min_positionals;
        max_pos = sub->max_positionals;
        first = 1;
    }
    const int n = static_cast<int>(positionals.size() - first);
    std::string what = cl.command + (cl.subcommand.empty() ? "" : " " + cl.subcommand);
    if (n < min_pos) return fail<CommandLine>(ErrorCode::InvalidConfig, "'" + what + "' is missing an argument");
    if (n > max_pos) return fail<CommandLine>(ErrorCode::InvalidConfig, "too many arguments for '" + what + "'");
    cl.positionals.assign(positionals.begin() + static_cast<std::ptrdiff_t>(first), positionals.end());
    return ok(std::move(cl));
}

std::string format_turn_footer(const TokenUsage &u) {
    if (u.prefill_ms <= 0.0f && u.decode_ms <= 0.0f && u.kv_max_tokens <= 0) return {};
    const char *kv_name[] = {"", "f32", "f16", "q8_0", "q4_0"};
    char buf[200];
    std::string out;
    const int fresh = std::max(0, u.prompt_tokens - u.cached_tokens);
    int n = std::snprintf(buf, sizeof(buf), "prompt %d tok", u.prompt_tokens);
    out.append(buf, static_cast<size_t>(n));
    if (u.cached_tokens > 0) {
        n = std::snprintf(buf, sizeof(buf), " (%d cached)", u.cached_tokens);
        out.append(buf, static_cast<size_t>(n));
    }
    if (u.prefill_ms > 0.0f && fresh > 0) {
        n = std::snprintf(buf, sizeof(buf), " %.0f tok/s", static_cast<double>(fresh) / (static_cast<double>(u.prefill_ms) / 1000.0));
        out.append(buf, static_cast<size_t>(n));
    }
    n = std::snprintf(buf, sizeof(buf), " | gen %d tok", u.completion_tokens);
    out.append(buf, static_cast<size_t>(n));
    if (u.decode_ms > 0.0f && u.completion_tokens > 1) {
        n = std::snprintf(buf, sizeof(buf), " %.0f tok/s",
                          static_cast<double>(u.completion_tokens - 1) / (static_cast<double>(u.decode_ms) / 1000.0));
        out.append(buf, static_cast<size_t>(n));
    }
    if (u.kv_max_tokens > 0) {
        n = std::snprintf(buf, sizeof(buf), " | KV %d/%d", u.kv_tokens, u.kv_max_tokens);
        out.append(buf, static_cast<size_t>(n));
        if (u.kv_type > 0 && u.kv_type < 5) (out += ' ') += kv_name[u.kv_type];
        if (u.kv_bytes > 0) {
            n = std::snprintf(buf, sizeof(buf), " %.1f MiB", static_cast<double>(u.kv_bytes) / (1024.0 * 1024.0));
            out.append(buf, static_cast<size_t>(n));
        }
    }
    return out;
}

std::string usage_all() {
    return "usage: agent <command> [options]   (agent alone, in a terminal, opens the chat screen)\n"
           "\n"
           "commands:\n"
           "  init [DIR]                 create .agents/ (skills, MCP config, private memory) in a project folder\n"
           "  doctor                     what is installed, how much disk each area of the home uses, what is safe to clean\n"
           "  clean --downloads|--cache|--logs [--yes]   remove partial downloads, rebuildable caches, or old logs (never models or "
           "memory)\n"
           "  secret set NAME            store a key in the OS credential store (typed without echo, or --stdin); never shown again\n"
           "  secret bind NAME CONSUMER  allow mcp:NAME / tool:NAME / group:NAME to use it (see `secret ls`); unbind, rm, ls too\n"
           "  setup                      choose the global home, detect hardware, find engines, pick and smoke-test models\n"
           "        [--home DIR] [--model NAME|owner/repo[:quant]]... [--no-models]\n"
           "        [--yes] [--caps llama,audiocpp,sdcpp,onnx] [--workspace DIR] [--models-dir DIR]\n"
           "        [--engines-dir DIR] [--backend cpu|cuda|vulkan] [--js-runtime bun|node|nub]\n"
           "        [--download-engines] [--install-runtimes]\n"
           "  runtime ls|use|install|path|rm   the private JavaScript and Python runtimes (bun, node, pnpm, nub, uv) kept in the home\n"
           "        runtime use bun|node|nub   choose the JavaScript tooling; runtime install NAME|js   download one\n"
           "  skill search|add|ls|rm     skills from skills.sh (add owner/repo/skill [--yes] [--project]); a skill is text the model "
           "follows\n"
           "  mcp search|add|ls|rm       MCP servers from the official registry (add NAME [--yes] [--project]); they run in the sandbox\n"
           "  wake [--dry-run]           resume the graph runs paused at a wait node whose time has come (start it with agent schedule)\n"
           "  schedule ls|add|rm         run an agent command on a schedule with cron / Task Scheduler: add NAME (--every 15m|--daily "
           "09:00|--cron EXPR) -- run graph.json --yes\n"
           "  sandbox ls|grant|revoke    folders outside the workspace that tools may use: grant PATH [--write] [--for "
           "run_command|files|mcp:NAME]\n"
           "  engine ls                  list engines in engines.json\n"
           "  engine add <plugin>        validate an engine plugin and record it\n"
           "  engine pull <name>         download a prebuilt engine from the GitHub release (--tag, --repo)\n"
           "  engine rm <name>           forget an engine\n"
           "  model pull <spec>          download from Hugging Face (owner/repo[:file-or-QUANT])\n"
           "        owner/repo:*  pulls every file of a folder model (TRELLIS); [--include STR[,STR]] narrows it\n"
           "        [--name NAME] [--quant Q4_K_M] [--mmproj auto|none|QUANT|FILE] [--no-mmproj] [--allow-restricted]\n"
           "        (vision / audio / OCR repos: the projector is fetched too, recorded, and used automatically)\n"
           "  model search [TEXT] [--task T]  curated registry: short names for tested models (pull by name)\n"
           "  model info <repo>          list a Hugging Face repo: quants with sizes, projectors, licence\n"
           "        (pick a file, then: model pull owner/repo:QUANT)\n"
           "  model ls                   list downloaded models (models.json)\n"
           "  model rm <name> [--keep-file]\n"
           "  chat                       interactive chat with streaming and tools (full-screen in a terminal)\n"
           "        [-m|--model NAME|PATH] [-e|--engine NAME|PATH] [--resume ID] [--list] [--plain]\n"
           "        [--workspace DIR] [--ctx N] [--max-tokens N] [--engine-options JSON] [-y|--yes] [--verbose]\n"
           "        [--inprocess]\n"
           "        -p|--prompt TEXT    answer once and exit: the answer goes to stdout, tools and notes to stderr (- reads the prompt\n"
           "                            from stdin); tool calls are denied unless -y\n"
           "        --jsonl             editor integration: JSON lines in and out on stdio (see docs/cli.md)\n"
           "        --json | --grammar FILE   constrain the answer: valid JSON only, or a GBNF grammar (llama engine)\n"
           "  run <graph.json>           run a declarative graph\n"
           "        [-m|--model NAME|PATH] [-e|--engine NAME|PATH] [--workspace DIR] [--thread ID] [--set k=v]\n"
           "        [--resume ID] [--input TEXT] [--engine-options JSON] [-y|--yes] [--inprocess] [--json]\n"
           "  voice                      talk to the agent: mic -> VAD -> ASR -> LLM -> TTS -> speaker\n"
           "        --asr-model FILE --tts-model FILE [--vad-model FILE] [-m|--model NAME|PATH]\n"
           "        [--input-wav FILE] [--output-dir DIR] [--barge-in] [--max-turns N] [--voice ID]\n"
           "        [--resume ID] [-e|--engine NAME|PATH] [--workspace DIR] [--ctx N] [--max-tokens N]\n"
           "        [--engine-options JSON] [-y|--yes] [--verbose] [--inprocess]\n"
           "  task <capability>          run one engine capability once and print every output part\n"
           "        (text_gen|ocr|asr|tts|vad|diarize|image_gen|image_edit|video_gen|mesh_gen|embed|rerank|decide|classify|generic)\n"
           "        [-m|--model NAME|PATH] -e|--engine NAME|PATH [--text TEXT] [--file PATH] [--params JSON]\n"
           "        [--engine-options JSON] [--workspace DIR] [--max-tokens N] [--inprocess]\n"
           "        rerank: --text QUERY --doc DOCUMENT [--doc DOCUMENT ...]\n"
           "        generic / onnx: --tensor name:dtype:d1,d2=v1,v2,... (repeatable; dtype f32|i64|i32|u8)\n"
           "  version                    print the version\n"
           "\n"
           "global options: -h|--help, --no-sandbox (run engines, tools and commands without the OS sandbox), --home DIR (the global "
           "workspace: default $AGENT_HOME or the folder chosen at setup),\n"
           "                 --config-dir DIR (old layout: default $AGENT_CONFIG_DIR or the platform config dir)\n"
           "engines run in a separate agent-engine-host process (a crash fails the turn, not the CLI);\n"
           "--inprocess, or \"isolation\": \"inprocess\" in engines.json, loads them into the CLI instead\n"
           "chat commands: /exit /clear /model [NAME] /image <path> [prompt] /open [n] /stats /help\n";
}

namespace {
struct CommandBlurb {
    std::string_view name, summary;
};
constexpr CommandBlurb kOverview[] = {
    {"chat", "talk to a model in a full-screen chat with tools (also: just run `agent`)"},
    {"setup", "first-time setup: home folder, hardware, engines, a first model"},
    {"model", "download, list, search and remove models"},
    {"task", "run one engine capability once: asr, tts, image_gen, embed, ..."},
    {"run", "run a declarative agent graph"},
    {"voice", "talk to the agent with a microphone and speakers"},
    {"engine", "list, add, download and remove engine plugins"},
    {"skill", "search and install skills"},
    {"mcp", "search and install MCP servers"},
    {"sandbox", "folders that tools may use outside the workspace"},
    {"secret", "API keys in the OS credential store"},
    {"runtime", "the private bun / node / uv runtimes"},
    {"init", "create a .agents/ folder in a project"},
    {"doctor", "what is installed, disk use, what is safe to clean"},
    {"clean", "remove partial downloads, caches or old logs"},
    {"wake", "resume graph runs that are waiting for a time (see schedule)"},
    {"schedule", "run an agent command on a schedule (cron / Task Scheduler)"},
    {"completion", "print a completion script: bash, zsh, fish or powershell"},
    {"update", "check for a newer release and install it (agent update --check only looks)"},
    {"version", "print the version"},
};
} // namespace

std::string usage_text(std::string_view command) {
    if (command.empty()) {
        std::string out = "usage: agent <command> [options]     (agent alone, in a terminal, opens the chat)\n\n";
        for (const auto &b : kOverview) {
            out += "  ";
            out += b.name;
            out.append(b.name.size() < 10 ? 10 - b.name.size() : 1, ' ');
            out += b.summary;
            out += '\n';
        }
        out += "\nagent <command> --help shows the options of one command.\n"
               "global options: -h|--help, --no-sandbox (run without the OS sandbox), --home DIR (default $AGENT_HOME or the folder "
               "chosen at setup)\n";
        return out;
    }
    // The detailed lines of one command: its first line starts with two spaces and the name, the lines under it are indented deeper.
    const std::string all = usage_all();
    std::string out, current;
    size_t pos = all.find("commands:\n");
    pos = pos == std::string::npos ? 0 : pos + 10;
    while (pos < all.size()) {
        size_t end = all.find('\n', pos);
        if (end == std::string::npos) end = all.size();
        const std::string_view line(all.data() + pos, end - pos);
        if (line.empty()) break;
        if (line.size() > 2 && line[0] == ' ' && line[1] == ' ' && line[2] != ' ') {
            const size_t e = line.find_first_of(" \t", 2);
            current = std::string(line.substr(2, e == std::string_view::npos ? std::string_view::npos : e - 2));
        }
        if (current == command) {
            out.append(line);
            out += '\n';
        }
        pos = end + 1;
    }
    if (out.empty()) return usage_text();
    return "usage: agent " + std::string(command) + " ...\n\n" + out + "\nglobal options: -h|--help, --no-sandbox, --home DIR\n";
}

std::string closest_command(std::string_view typed) {
    auto distance = [](std::string_view a, std::string_view b) {
        std::vector<size_t> row(b.size() + 1);
        for (size_t j = 0; j <= b.size(); ++j) row[j] = j;
        for (size_t i = 1; i <= a.size(); ++i) {
            size_t diagonal = row[0];
            row[0] = i;
            for (size_t j = 1; j <= b.size(); ++j) {
                const size_t above = row[j];
                row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] == b[j - 1] ? 0 : 1)});
                diagonal = above;
            }
        }
        return row[b.size()];
    };
    std::string best;
    size_t best_distance = 3; // at most two edits away
    for (const auto &b : kOverview) {
        const size_t d = distance(typed, b.name);
        if (d < best_distance) {
            best_distance = d;
            best = std::string(b.name);
        }
    }
    return best;
}

// -------------------------- JSON helpers --------------------------

std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

namespace {

std::string json_str(std::string_view s) { return "\"" + json_escape(s) + "\""; }

std::string string_array(std::span<const std::string> items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? ", " : "") + json_str(items[i]);
    return out + "]";
}

std::string get_str(simdjson::dom::element obj, std::string_view key, std::string fallback = {}) {
    std::string_view v;
    if (obj[key].get_string().get(v)) return fallback;
    return std::string(v);
}

int64_t get_int(simdjson::dom::element obj, std::string_view key, int64_t fallback) {
    int64_t v = 0;
    if (obj[key].get_int64().get(v)) return fallback;
    return v;
}

double get_num(simdjson::dom::element obj, std::string_view key, double fallback) {
    double v = 0;
    if (obj[key].get_double().get(v)) {
        int64_t i = 0;
        if (obj[key].get_int64().get(i)) return fallback;
        return static_cast<double>(i);
    }
    return v;
}

bool get_bool(simdjson::dom::element obj, std::string_view key, bool fallback) {
    bool v = false;
    if (obj[key].get_bool().get(v)) return fallback;
    return v;
}

std::vector<std::string> get_str_array(simdjson::dom::element obj, std::string_view key) {
    std::vector<std::string> out;
    simdjson::dom::array arr;
    if (obj[key].get_array().get(arr)) return out;
    for (simdjson::dom::element e : arr) {
        std::string_view v;
        if (!e.get_string().get(v)) out.emplace_back(v);
    }
    return out;
}

Result<std::string> read_text_file(const std::string &path, bool &exists) {
    std::error_code ec;
    exists = std::filesystem::exists(path, ec);
    if (!exists) return ok(std::string());
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail<std::string>(ErrorCode::FilesystemError, "cannot read " + path);
    return ok(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
}

Result<void> write_text_file_atomic(const std::string &path, std::string_view text) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return fail(ErrorCode::FilesystemError, "cannot write " + tmp);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out.flush()) return fail(ErrorCode::FilesystemError, "failed writing " + tmp);
    }
    std::filesystem::rename(tmp, p, ec);
    if (ec) return fail(ErrorCode::FilesystemError, "cannot replace " + path + ": " + ec.message());
    return ok();
}

} // namespace

// -------------------------- Hardware --------------------------

std::string suggest_backend(const HardwareInfo &hw) {
    if (hw.nvidia_smi || hw.nvidia_device) return "cuda";
    if (!hw.dri_nodes.empty()) return "vulkan";
    return "cpu";
}

// -------------------------- config.json --------------------------

std::span<const std::string_view> starter_pack() {
    static constexpr std::string_view kPack[] = {"llama", "audiocpp", "sdcpp", "onnx"};
    return kPack;
}

std::string config_to_json(const CliConfig &c) {
    char temp[32];
    std::snprintf(temp, sizeof(temp), "%g", static_cast<double>(c.temperature));
    std::string out = "{\n";
    out += "  \"version\": " + std::to_string(c.version) + ",\n";
    out += "  \"workspace\": " + json_str(c.workspace) + ",\n";
    out += "  \"models_dir\": " + json_str(c.models_dir) + ",\n";
    out += "  \"default_model\": " + json_str(c.default_model) + ",\n";
    out += "  \"capabilities\": " + string_array(c.capabilities) + ",\n";
    out += "  \"backend\": " + json_str(c.backend) + ",\n";
    out += "  \"hardware\": {\"cpu_cores\": " + std::to_string(c.hardware.cpu_cores) +
           ", \"nvidia_smi\": " + (c.hardware.nvidia_smi ? "true" : "false") +
           ", \"nvidia_device\": " + (c.hardware.nvidia_device ? "true" : "false") +
           ", \"dri_nodes\": " + string_array(c.hardware.dri_nodes) + "},\n";
    out += "  \"context_window\": " + std::to_string(c.context_window) + ",\n";
    out += "  \"max_tokens\": " + std::to_string(c.max_tokens) + ",\n";
    out += "  \"temperature\": " + std::string(temp) + ",\n";
    out += "  \"memory_budget_mb\": " + std::to_string(c.memory_budget_mb) + ",\n";
    out += "  \"js_runtime\": \"" + c.js_runtime + "\",\n";
    out += "  \"search_url\": " + json_str(c.search_url) + "\n";
    out += "}\n";
    return out;
}

Result<CliConfig> config_from_json(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc) || !doc.is_object())
        return fail<CliConfig>(ErrorCode::ParseError, "config.json is not a JSON object");
    CliConfig c;
    c.version = static_cast<int>(get_int(doc, "version", 1));
    c.workspace = get_str(doc, "workspace");
    c.models_dir = get_str(doc, "models_dir");
    c.default_model = get_str(doc, "default_model");
    c.capabilities = get_str_array(doc, "capabilities");
    c.backend = get_str(doc, "backend", "cpu");
    simdjson::dom::element hw;
    if (!doc["hardware"].get(hw) && hw.is_object()) {
        c.hardware.cpu_cores = static_cast<int>(get_int(hw, "cpu_cores", 0));
        c.hardware.nvidia_smi = get_bool(hw, "nvidia_smi", false);
        c.hardware.nvidia_device = get_bool(hw, "nvidia_device", false);
        c.hardware.dri_nodes = get_str_array(hw, "dri_nodes");
    }
    c.context_window = static_cast<int>(get_int(doc, "context_window", c.context_window));
    c.max_tokens = static_cast<int>(get_int(doc, "max_tokens", c.max_tokens));
    c.temperature = static_cast<float>(get_num(doc, "temperature", c.temperature));
    c.memory_budget_mb = get_int(doc, "memory_budget_mb", 0);
    if (c.memory_budget_mb < 0) return fail<CliConfig>(ErrorCode::InvalidConfig, "config.json: memory_budget_mb must be >= 0");
    c.search_url = std::string(get_str(doc, "search_url"));
    c.js_runtime = std::string(get_str(doc, "js_runtime", "bun"));
    if (c.js_runtime != "bun" && c.js_runtime != "node" && c.js_runtime != "nub")
        return fail<CliConfig>(ErrorCode::InvalidConfig, "config.json: js_runtime must be \"bun\", \"node\" or \"nub\"");
    if (c.context_window <= 0 || c.max_tokens <= 0)
        return fail<CliConfig>(ErrorCode::InvalidConfig, "config.json: context_window and max_tokens must be positive");
    return ok(std::move(c));
}

Result<CliConfig> load_config(const std::string &path) {
    bool exists = false;
    auto text = read_text_file(path, exists);
    if (!text.ok) return fail<CliConfig>(text.error.code, text.error.message);
    if (!exists || trim(text.value).empty()) return ok(CliConfig{});
    auto c = config_from_json(text.value);
    if (!c.ok) c.error.message = path + ": " + c.error.message;
    return c;
}

Result<void> save_config(const std::string &path, const CliConfig &config) { return write_text_file_atomic(path, config_to_json(config)); }

// -------------------------- engines.json --------------------------

std::string engines_to_json(std::span<const EngineEntry> engines) {
    std::string out = "{\n  \"engines\": [";
    for (size_t i = 0; i < engines.size(); ++i) {
        const auto &e = engines[i];
        out += i ? ",\n    " : "\n    ";
        out += "{\"name\": " + json_str(e.name) + ", \"path\": " + json_str(e.path) + ", \"version\": " + json_str(e.version) +
               ", \"backend\": " + json_str(e.backend) + ", \"caps\": " + string_array(e.caps);
        if (!e.isolation.empty()) out += ", \"isolation\": " + json_str(e.isolation);
        out += "}";
    }
    out += engines.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

Result<std::vector<EngineEntry>> engines_from_json(std::string_view json) {
    using R = std::vector<EngineEntry>;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc)) return fail<R>(ErrorCode::ParseError, "engines.json is not valid JSON");
    simdjson::dom::array arr;
    if (doc["engines"].get_array().get(arr)) return fail<R>(ErrorCode::ParseError, "engines.json has no \"engines\" array");
    R out;
    for (simdjson::dom::element e : arr) {
        if (!e.is_object()) return fail<R>(ErrorCode::ParseError, "engines.json entry is not an object");
        EngineEntry entry;
        entry.path = get_str(e, "path");
        if (entry.path.empty()) return fail<R>(ErrorCode::ParseError, "engines.json entry without a \"path\"");
        entry.name = get_str(e, "name");
        if (entry.name.empty()) {
            // Hand-written manifest: derive the name from agent-engine-<name>.<ext>.
            std::string stem = std::filesystem::path(entry.path).stem().string();
            constexpr std::string_view prefix = "agent-engine-";
            entry.name = stem.rfind(prefix, 0) == 0 ? stem.substr(prefix.size()) : stem;
        }
        entry.version = get_str(e, "version");
        entry.backend = get_str(e, "backend", "cpu");
        entry.caps = get_str_array(e, "caps");
        entry.isolation = get_str(e, "isolation");
        if (!entry.isolation.empty() && entry.isolation != "process" && entry.isolation != "inprocess")
            return fail<R>(ErrorCode::ParseError, "engines.json: \"isolation\" must be \"process\" or \"inprocess\"");
        out.push_back(std::move(entry));
    }
    return ok(std::move(out));
}

Result<std::vector<EngineEntry>> load_engines(const std::string &path) {
    using R = std::vector<EngineEntry>;
    bool exists = false;
    auto text = read_text_file(path, exists);
    if (!text.ok) return fail<R>(text.error.code, text.error.message);
    if (!exists || trim(text.value).empty()) return ok(R{});
    auto r = engines_from_json(text.value);
    if (!r.ok) r.error.message = path + ": " + r.error.message;
    return r;
}

Result<void> save_engines(const std::string &path, std::span<const EngineEntry> engines) {
    return write_text_file_atomic(path, engines_to_json(engines));
}

void upsert_engine(std::vector<EngineEntry> &engines, EngineEntry entry) {
    for (auto &e : engines)
        if (e.name == entry.name) {
            e = std::move(entry);
            return;
        }
    engines.push_back(std::move(entry));
}

bool remove_engine(std::vector<EngineEntry> &engines, std::string_view name) {
    const auto before = engines.size();
    std::erase_if(engines, [&](const EngineEntry &e) { return e.name == name; });
    return engines.size() != before;
}

EngineIsolation cli_engine_isolation(bool inprocess_flag, std::string_view entry_isolation) {
    return inprocess_flag || entry_isolation == "inprocess" ? EngineIsolation::InProcess : EngineIsolation::Process;
}

bool cli_engine_gpu(std::string_view backend) { return !backend.empty() && backend != "cpu"; }

namespace {
bool g_no_sandbox = false;
}

namespace {
// All the words that can follow `agent <command>`: subcommands and --flags.
std::string command_words(const CommandSpec &c) {
    std::string words;
    for (const SubSpec &sub : c.subs) (words += sub.name) += ' ';
    const auto add = [&](std::span<const std::string_view> flags) {
        for (const auto f : flags) (words += "--") += f, words += ' ';
    };
    add(c.value_flags);
    add(c.bool_flags);
    words += "--help";
    return words;
}
} // namespace

std::string with_role_model(std::string_view engine_options, std::string_view role, std::string_view model_path) {
    const std::string quoted_role = "\"" + std::string(role) + "\"";
    if (model_path.empty() || engine_options.find(quoted_role) != std::string_view::npos) return std::string(engine_options);
    const std::string entry = quoted_role + ":{\"model\":\"" + json_escape(model_path) + "\"}";
    const size_t open = engine_options.find('{');
    if (open == std::string_view::npos) return "{" + entry + "}";
    const size_t close = engine_options.rfind('}');
    const bool empty_object =
        engine_options.substr(open + 1, close == std::string_view::npos ? 0 : close - open - 1).find_first_not_of(" \t\r\n") ==
        std::string_view::npos;
    return std::string(engine_options.substr(0, open + 1)) + entry + (empty_object ? "" : ",") +
           std::string(engine_options.substr(open + 1));
}

std::string with_option(std::string_view engine_options, std::string_view key, std::string_view value_json) {
    const std::string quoted = "\"" + std::string(key) + "\"";
    if (engine_options.find(quoted) != std::string_view::npos) return std::string(engine_options);
    const std::string entry = quoted + ":" + std::string(value_json);
    const size_t open = engine_options.find('{');
    if (open == std::string_view::npos) return "{" + entry + "}";
    const size_t close = engine_options.rfind('}');
    const bool empty_object =
        engine_options.substr(open + 1, close == std::string_view::npos ? 0 : close - open - 1).find_first_not_of(" \t\r\n") ==
        std::string_view::npos;
    return std::string(engine_options.substr(0, open + 1)) + entry + (empty_object ? "" : ",") +
           std::string(engine_options.substr(open + 1));
}

std::string model_engine_folder(std::string_view spec, std::string_view explicit_engine, std::string_view registry_engine) {
    const auto valid = [](std::string_view name) { return !name.empty() && name.find_first_of("/\\.") == std::string_view::npos; };
    if (valid(explicit_engine)) return std::string(explicit_engine);
    if (valid(registry_engine)) return std::string(registry_engine);
    std::string lower(spec);
    for (char &c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const size_t colon = lower.rfind(':');
    const std::string suffix = colon == std::string::npos ? std::string() : lower.substr(colon + 1);
    const auto ends = [&](std::string_view e) {
        return suffix.size() >= e.size() && suffix.compare(suffix.size() - e.size(), e.size(), e) == 0;
    };
    // The audio.cpp repository holds GGUF files too, but they are not llama.cpp's.
    if (lower.find("audio.cpp") != std::string::npos || lower.find("audio-cpp") != std::string::npos) return "audiocpp";
    if (ends(".onnx")) return "onnx";
    if (ends(".safetensors") || ends(".ckpt")) return "sdcpp";
    if (ends(".gguf")) return "llama";
    // A bare quant name (Q4_K_M, IQ3_XS, BF16, F16) or a GGUF repository.
    const bool quant = !suffix.empty() && suffix.find('.') == std::string::npos && suffix.find('/') == std::string::npos;
    if (quant || lower.find("gguf") != std::string::npos) return "llama";
    return {};
}

std::string completion_script(std::string_view shell) {
    std::string names;
    for (const CommandSpec &c : kCommands) (names += c.name) += ' ';
    const std::string global = "--no-sandbox --home --config-dir --help";
    if (shell == "bash" || shell == "zsh") {
        std::string s;
        if (shell == "zsh") s += "autoload -U +X bashcompinit && bashcompinit\n";
        s += "_agent() {\n  local cur=\"${COMP_WORDS[COMP_CWORD]}\" cmd=\"${COMP_WORDS[1]}\"\n  COMPREPLY=()\n";
        s += "  if [ \"$COMP_CWORD\" -eq 1 ]; then COMPREPLY=($(compgen -W \"" + names + global + "\" -- \"$cur\")); return; fi\n";
        s += "  case \"$cmd\" in\n";
        for (const CommandSpec &c : kCommands)
            s += "    " + std::string(c.name) + ") COMPREPLY=($(compgen -W \"" + command_words(c) + " " + global + "\" -- \"$cur\")) ;;\n";
        s += "  esac\n}\ncomplete -o default -F _agent agent\n";
        return s;
    }
    if (shell == "fish") {
        std::string s = "complete -c agent -f\n";
        for (const auto &blurb : kOverview)
            s += "complete -c agent -n '__fish_use_subcommand' -a '" + std::string(blurb.name) + "' -d '" + std::string(blurb.summary) +
                 "'\n";
        for (const CommandSpec &c : kCommands) {
            for (const SubSpec &sub : c.subs)
                s += "complete -c agent -n '__fish_seen_subcommand_from " + std::string(c.name) + "' -a '" + std::string(sub.name) + "'\n";
            const auto add = [&](std::span<const std::string_view> flags, const char *arg) {
                for (const auto f : flags)
                    s += "complete -c agent -n '__fish_seen_subcommand_from " + std::string(c.name) + "' -l " + std::string(f) + arg + "\n";
            };
            add(c.value_flags, " -r");
            add(c.bool_flags, "");
        }
        return s;
    }
    if (shell == "powershell") {
        std::string s = "Register-ArgumentCompleter -Native -CommandName agent -ScriptBlock {\n  param($wordToComplete, $commandAst, "
                        "$cursorPosition)\n";
        s += "  $words = $commandAst.CommandElements | ForEach-Object { $_.ToString() }\n";
        s += "  $table = @{\n";
        for (const CommandSpec &c : kCommands) s += "    '" + std::string(c.name) + "' = '" + command_words(c) + " " + global + "'\n";
        s += "  }\n  if ($words.Count -le 1 -or ($words.Count -eq 2 -and $wordToComplete)) {\n    '" + names + global +
             "'.Split(' ') | Where-Object { $_ -like \"$wordToComplete*\" }\n  } elseif ($table.ContainsKey($words[1])) {\n";
        s += "    $table[$words[1]].Split(' ') | Where-Object { $_ -like \"$wordToComplete*\" }\n  }\n}\n";
        return s;
    }
    return {};
}

void set_no_sandbox(bool value) { g_no_sandbox = value; }

std::string sandbox_status_line() {
    if (g_no_sandbox) return "off (--no-sandbox): engines, tools and commands run with your own permissions";
    const SandboxStatus status = sandbox_status();
    if (status.supported) return "on (" + status.detail + ")";
    return "not supported on this system (" + status.detail + "): running unsandboxed";
}

std::string sandbox_notice() {
    if (g_no_sandbox) return "Sandbox off (--no-sandbox): tools, commands and engines run with your own permissions.";
    const SandboxStatus status = sandbox_status();
    if (status.supported) return {};
    return "Sandbox not supported on this system (" + status.detail + "): running unsandboxed.";
}

void apply_sandbox_choice(Config &cfg) {
    if (!g_no_sandbox) return;
    // The user chose to run without a sandbox: engines, tools and commands all run with the user's own permissions.
    cfg.engine_sandbox.enabled = false;
    cfg.tool_sandbox.enabled = false;
}

const EngineEntry *find_engine_entry(std::span<const EngineEntry> engines, std::string_view name) {
    for (const auto &e : engines)
        if (e.name == name) return &e;
    return nullptr;
}

std::vector<std::string> caps_to_names(uint64_t caps) {
    static constexpr struct {
        uint64_t bit;
        std::string_view name;
    } kNames[] = {
        {AGENT_CAP_TEXT_GEN, "text_gen"},
        {AGENT_CAP_TOOL_CALL, "tool_call"},
        {AGENT_CAP_VISION_IN, "vision_in"},
        {AGENT_CAP_AUDIO_IN, "audio_in"},
        {AGENT_CAP_VIDEO_IN, "video_in"},
        {AGENT_CAP_OCR, "ocr"},
        {AGENT_CAP_ASR, "asr"},
        {AGENT_CAP_TTS, "tts"},
        {AGENT_CAP_VAD, "vad"},
        {AGENT_CAP_DIARIZE, "diarize"},
        {AGENT_CAP_IMAGE_GEN, "image_gen"},
        {AGENT_CAP_IMAGE_EDIT, "image_edit"},
        {AGENT_CAP_VIDEO_GEN, "video_gen"},
        {AGENT_CAP_MESH_GEN, "mesh_gen"},
        {AGENT_CAP_EMBED, "embed"},
        {AGENT_CAP_RERANK, "rerank"},
        {AGENT_CAP_DECIDE, "decide"},
        {AGENT_CAP_CLASSIFY, "classify"},
        {AGENT_CAP_GENERIC, "generic"},
    };
    std::vector<std::string> out;
    for (const auto &n : kNames)
        if (caps & n.bit) out.emplace_back(n.name);
    return out;
}

std::string plugin_file_name(std::string_view engine_name) {
#if defined(_WIN32)
    constexpr std::string_view suffix = ".dll";
#elif defined(__APPLE__)
    constexpr std::string_view suffix = ".dylib";
#else
    constexpr std::string_view suffix = ".so";
#endif
    return "agent-engine-" + std::string(engine_name) + std::string(suffix);
}

std::vector<std::string> split_list(std::string_view list) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= list.size()) {
        size_t j = list.find_first_of(", \t", i);
        if (j == std::string_view::npos) j = list.size();
        std::string item = lower(trim(list.substr(i, j - i)));
        if (!item.empty() && std::find(out.begin(), out.end(), item) == out.end()) out.push_back(std::move(item));
        i = j + 1;
    }
    return out;
}

// -------------------------- Model / engine resolution --------------------------

std::string_view engine_for_model(std::string_view model_path) {
    const std::string ext = lower(std::filesystem::path(model_path).extension().string());
    if (ext == ".gguf") return "llama";
    if (ext == ".onnx") return "onnx";
    if (ext == ".safetensors" || ext == ".ckpt" || ext == ".sft") return "sdcpp";
    return {};
}

Result<ModelRef> resolve_model(std::string_view spec, std::string_view default_model, std::span<const ModelRef> catalog, ExistsFn exists) {
    std::string_view want = trim(spec);
    if (want.empty()) want = trim(default_model);
    if (want.empty()) {
        for (const auto &m : catalog)
            if (engine_for_model(m.path) == "llama") return ok(m);
        return fail<ModelRef>(ErrorCode::InvalidConfig, "no model yet. Run `agent setup` (it suggests models that fit this machine and "
                                                        "downloads one), or `agent model pull <name>`, or pass --model PATH");
    }
    for (const auto &m : catalog)
        if (m.name == want) {
            if (!exists(m.path))
                return fail<ModelRef>(ErrorCode::FilesystemError,
                                      "model '" + m.name + "' is in models.json but its file is missing: " + m.path);
            return ok(m);
        }
    if (exists(want)) {
        std::string name = std::filesystem::path(want).stem().string();
        return ok(ModelRef{std::move(name), std::string(want)});
    }
    std::string msg = "no model file or catalogue entry named '" + std::string(want) + "'";
    if (want.find('/') != std::string_view::npos && want.find(':') != std::string_view::npos)
        msg += "; download it first with `agent model pull " + std::string(want) + "`";
    else msg += "; see `agent model ls`";
    return fail<ModelRef>(ErrorCode::KeyNotFound, msg);
}

Result<EngineRef> resolve_engine(std::string_view engine_flag, std::string_view model_path, std::span<const EngineEntry> engines,
                                 ExistsFn exists) {
    engine_flag = trim(engine_flag);
    if (!engine_flag.empty()) {
        if (const EngineEntry *e = find_engine_entry(engines, engine_flag)) return ok(EngineRef{e->name, e->path});
        if (exists(engine_flag)) return ok(EngineRef{"", std::string(engine_flag)});
        if (engine_flag.find('/') != std::string_view::npos || engine_flag.find('\\') != std::string_view::npos)
            return fail<EngineRef>(ErrorCode::FilesystemError, "engine plugin not found: " + std::string(engine_flag));
        return ok(EngineRef{std::string(engine_flag), ""});
    }
    const std::string_view name = engine_for_model(model_path);
    if (name.empty())
        return fail<EngineRef>(ErrorCode::UnsupportedProvider,
                               "cannot tell which engine runs '" + std::string(model_path) + "' from its file type; pass --engine NAME");
    if (const EngineEntry *e = find_engine_entry(engines, name)) return ok(EngineRef{e->name, e->path});
    return ok(EngineRef{std::string(name), ""});
}

std::string mime_for_path(std::string_view path) {
    const std::string ext = lower(std::filesystem::path(path).extension().string());
    static constexpr struct {
        std::string_view ext, mime;
    } kMimes[] = {
        {".png", "image/png"},   {".jpg", "image/jpeg"},  {".jpeg", "image/jpeg"},       {".gif", "image/gif"},   {".bmp", "image/bmp"},
        {".webp", "image/webp"}, {".wav", "audio/wav"},   {".mp3", "audio/mpeg"},        {".flac", "audio/flac"}, {".ogg", "audio/ogg"},
        {".mp4", "video/mp4"},   {".webm", "video/webm"}, {".glb", "model/gltf-binary"},
    };
    for (const auto &m : kMimes)
        if (m.ext == ext) return std::string(m.mime);
    return "application/octet-stream";
}

// -------------------------- Chat --------------------------

SlashCommand parse_slash_command(std::string_view line) {
    SlashCommand cmd;
    std::string_view t = trim(line);
    if (t.size() >= 2 && t[0] == '/' && t[1] == '/') {
        cmd.text = std::string(t.substr(1));
        return cmd;
    }
    if (t.empty() || t.front() != '/') {
        cmd.text = std::string(t);
        return cmd;
    }
    t.remove_prefix(1);
    const size_t sp = t.find_first_of(" \t");
    const std::string name = lower(t.substr(0, sp));
    std::string_view args = sp == std::string_view::npos ? std::string_view() : trim(t.substr(sp));

    const std::string_view whole_args = args;

    // First argument, optionally "quoted" (paths with spaces).
    if (!args.empty() && (args.front() == '"' || args.front() == '\'')) {
        const char q = args.front();
        const size_t end = args.find(q, 1);
        if (end == std::string_view::npos) {
            cmd.arg = std::string(args.substr(1));
            args = {};
        } else {
            cmd.arg = std::string(args.substr(1, end - 1));
            args = trim(args.substr(end + 1));
        }
    } else if (!args.empty()) {
        const size_t end = args.find_first_of(" \t");
        cmd.arg = std::string(args.substr(0, end));
        args = end == std::string_view::npos ? std::string_view() : trim(args.substr(end));
    }
    cmd.rest = std::string(args);

    // Commands that run an `agent` subcommand: everything after the name is passed on as it was typed.
    for (std::string_view cli_name : {"doctor", "setup"})
        if (name == cli_name) {
            cmd.kind = SlashKind::Cli;
            cmd.arg = name;
            cmd.rest = std::string(whole_args);
            return cmd;
        }

    if (name == "exit" || name == "quit" || name == "q") cmd.kind = SlashKind::Exit;
    else if (name == "clear") cmd.kind = SlashKind::Clear;
    else if (name == "model") cmd.kind = SlashKind::Model;
    else if (name == "image") cmd.kind = SlashKind::Image;
    else if (name == "open") cmd.kind = SlashKind::Open;
    else if (name == "stats") cmd.kind = SlashKind::Stats;
    else if (name == "diff") cmd.kind = SlashKind::Diff;
    else if (name == "undo") cmd.kind = SlashKind::Undo;
    else if (name == "compact") cmd.kind = SlashKind::Compact;
    else if (name == "mode") cmd.kind = SlashKind::Mode;
    else if (name == "thinking") cmd.kind = SlashKind::Thinking;
    else if (name == "permissions") cmd.kind = SlashKind::Permissions;
    else if (name == "rename") cmd.kind = SlashKind::Rename;
    else if (name == "delete") cmd.kind = SlashKind::Delete;
    else if (name == "help" || name == "?") cmd.kind = SlashKind::Help;
    else {
        cmd.kind = SlashKind::Unknown;
        cmd.arg = name;
    }
    return cmd;
}

std::vector<std::string> slash_cli_argv(const SlashCommand &cmd) {
    if (cmd.kind != SlashKind::Cli) return {};
    // The arguments as words; "double" and 'single' quotes keep spaces together.
    std::vector<std::string> words;
    std::string word;
    char quote = 0;
    bool open = false;
    for (char c : cmd.rest) {
        if (quote) {
            if (c == quote) quote = 0;
            else word += c;
        } else if (c == '"' || c == '\'') {
            quote = c;
            open = true;
        } else if (c == ' ' || c == '\t') {
            if (open || !word.empty()) words.push_back(std::move(word));
            word.clear();
            open = false;
        } else {
            word += c;
        }
    }
    if (open || !word.empty()) words.push_back(std::move(word));

    std::vector<std::string> argv;
    const std::string &name = cmd.arg;
    if (name == "setup") argv = {"setup"}; // alone, /setup asks its questions in the chat (chat.cpp); with flags it is passed on as typed
    else argv = {name};
    argv.insert(argv.end(), words.begin(), words.end());
    return argv;
}

namespace {

// Length of the longest suffix of text that is a proper prefix of tag.
size_t partial_tag_suffix(std::string_view text, std::string_view tag) {
    for (size_t n = std::min(text.size(), tag.size() - 1); n > 0; --n)
        if (text.substr(text.size() - n) == tag.substr(0, n)) return n;
    return 0;
}

} // namespace

std::string filter_stream(StreamFilter &f, std::string_view token) {
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    f.pending += token;
    std::string out;
    for (;;) {
        if (!f.inside) {
            const size_t pos = f.pending.find(kOpen);
            if (pos != std::string::npos) {
                out += f.pending.substr(0, pos);
                f.pending.erase(0, pos + kOpen.size());
                f.inside = true;
                continue;
            }
            const size_t keep = partial_tag_suffix(f.pending, kOpen);
            out += f.pending.substr(0, f.pending.size() - keep);
            f.pending.erase(0, f.pending.size() - keep);
            return out;
        }
        const size_t pos = f.pending.find(kClose);
        if (pos != std::string::npos) {
            f.pending.erase(0, pos + kClose.size());
            if (!f.pending.empty() && f.pending.front() == '\n') f.pending.erase(0, 1);
            f.inside = false;
            continue;
        }
        const size_t keep = partial_tag_suffix(f.pending, kClose);
        f.pending.erase(0, f.pending.size() - keep);
        return out;
    }
}

std::string flush_stream(StreamFilter &f) {
    std::string out = f.inside ? std::string() : std::move(f.pending);
    f.pending.clear();
    f.inside = false;
    return out;
}

std::string add_conversation_id(std::string_view index, std::string_view id) {
    std::vector<std::string> ids = conversation_ids(index);
    std::erase(ids, std::string(id));
    ids.emplace_back(id);
    std::string out;
    for (const auto &i : ids) out += i + "\n";
    return out;
}

std::vector<std::string> conversation_ids(std::string_view index) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < index.size()) {
        size_t j = index.find('\n', i);
        if (j == std::string_view::npos) j = index.size();
        std::string_view id = trim(index.substr(i, j - i));
        if (!id.empty()) out.emplace_back(id);
        i = j + 1;
    }
    return out;
}

std::string format_progress(uint64_t done, uint64_t total, int width) {
    constexpr double kMiB = 1024.0 * 1024.0;
    char buf[96];
    if (total == 0) {
        std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(done) / kMiB);
        return buf;
    }
    if (done > total) done = total;
    const double frac = static_cast<double>(done) / static_cast<double>(total);
    const int filled = static_cast<int>(frac * width);
    std::string bar = "[" + std::string(static_cast<size_t>(filled), '#') + std::string(static_cast<size_t>(width - filled), '-') + "]";
    std::snprintf(buf, sizeof(buf), " %5.1f%%  %.1f/%.1f MiB", frac * 100.0, static_cast<double>(done) / kMiB,
                  static_cast<double>(total) / kMiB);
    return bar + buf;
}

std::string one_line(std::string_view text, size_t max_bytes) {
    std::string out;
    out.reserve(std::min(text.size(), max_bytes) + 3);
    for (char c : text) {
        if (out.size() >= max_bytes) {
            // Do not cut a UTF-8 sequence in half.
            while (!out.empty() && (static_cast<unsigned char>(out.back()) & 0xC0) == 0x80) out.pop_back();
            if (!out.empty() && (static_cast<unsigned char>(out.back()) & 0x80)) out.pop_back();
            return out + "...";
        }
        out += (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
    }
    return out;
}

std::string load_project_rules(const std::string &workspace, const std::string &home_root, size_t max_bytes) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, fs::path>> files;
    std::error_code ec;
    const fs::path ws(workspace);
    files.emplace_back("AGENTS.md", ws / "AGENTS.md");
    files.emplace_back(".agents/AGENTS.md", ws / ".agents" / "AGENTS.md");
    std::vector<fs::path> rules;
    for (const auto &entry : fs::directory_iterator(ws / ".agents" / "rules", ec))
        if (entry.is_regular_file(ec) && entry.path().extension() == ".md") rules.push_back(entry.path());
    std::sort(rules.begin(), rules.end());
    for (const fs::path &r : rules) files.emplace_back(".agents/rules/" + r.filename().string(), r);
    if (!home_root.empty()) files.emplace_back("your AGENTS.md", fs::path(home_root) / "AGENTS.md");

    std::string out;
    for (const auto &[label, path] : files) {
        if (!fs::is_regular_file(path, ec)) continue;
        std::ifstream in(path, std::ios::binary);
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        while (!body.empty() && (body.back() == '\n' || body.back() == '\r' || body.back() == ' ')) body.pop_back();
        if (body.empty()) continue;
        const std::string head = "\n\n## Rules from " + label + "\n";
        if (out.size() + head.size() >= max_bytes) break;
        const size_t room = max_bytes - out.size() - head.size();
        out += head;
        if (body.size() > room) {
            out += body.substr(0, room) + "\n[cut: the rules are longer than " + std::to_string(max_bytes) + " bytes]";
            break;
        }
        out += body;
    }
    if (out.empty()) return {};
    return "\n\nFollow these rules written by the user for this project." + out;
}

} // namespace agent::cli
