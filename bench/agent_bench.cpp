// agent_bench: throughput and latency of agent.cpp pipelines (AGENT_BUILD_BENCH).
//
//   agent_bench [--llama <agent-engine-llama.so> --model <chat.gguf>] [--mock <agent-engine-mock.so>]
//               [--sherpa <agent-engine-sherpa.so> --tts-model <kokoro dir>] [--host <agent-engine-host>]
//               [--reps 3] [--gen-tokens 128] [--prompt-words 200] [--branches 4] [--threads 0]
//               [--turns 2000] [--json results.json] [--md results.md]
//
// Sections (each runs when its inputs are given):
//   llm        prompt tok/s, generation tok/s, time to first token, peak RSS; in process vs process isolation
//   parallel   one Agent branch vs N parallel branches on one llama instance (continuous batching)
//   overhead   per-turn cost of the core on the mock engine, in process vs through agent-engine-host
//   tts        time to first audio chunk (engine ABI, streaming) and whole-utterance time through libagent
// Prints markdown tables to stdout; --json / --md also write them to files.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct Options {
    std::string llama, model, mock, sherpa, tts_model, host, json_out, md_out;
    std::string workspace = (fs::temp_directory_path() / "agent-bench").string();
    int reps = 3, gen_tokens = 128, prompt_words = 200, branches = 4, threads = 0, turns = 2000;
};

// ---- memory ------------------------------------------------------------------------------------

#if defined(__linux__)
uint64_t status_kib(const std::string &path, const char *field) {
    std::ifstream in(path);
    std::string line;
    const size_t n = std::strlen(field);
    while (std::getline(in, line))
        if (line.compare(0, n, field) == 0) return std::strtoull(line.c_str() + n + 1, nullptr, 10);
    return 0;
}
#endif

// Resets this process's peak RSS (VmHWM) so the next reading covers only what follows.
void reset_peak_rss() {
#if defined(__linux__)
    std::ofstream("/proc/self/clear_refs") << "5";
#endif
}

double peak_rss_mib() {
#if defined(__linux__)
    return static_cast<double>(status_kib("/proc/self/status", "VmHWM:")) / 1024.0;
#else
    return static_cast<double>(agent::process_rss_bytes()) / (1024.0 * 1024.0);
#endif
}

// Peak RSS of this process's live agent-engine-host children (largest), 0 when none is found.
double host_peak_rss_mib() {
    double best = 0;
#if defined(__linux__)
    std::error_code ec;
    for (const auto &task : fs::directory_iterator("/proc/self/task", ec)) {
        std::ifstream children(task.path() / "children");
        for (long pid; children >> pid;) {
            const std::string dir = "/proc/" + std::to_string(pid);
            std::ifstream comm(dir + "/comm");
            std::string name;
            std::getline(comm, name);
            if (name.rfind("agent-engine-ho", 0) != 0) continue;
            best = std::max(best, static_cast<double>(status_kib(dir + "/status", "VmHWM:")) / 1024.0);
        }
    }
#endif
    return best;
}

// ---- results -----------------------------------------------------------------------------------

struct Table {
    std::string title, note;
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
};

std::string fmt(double v, int decimals = 1) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

std::string markdown(const std::vector<Table> &tables) {
    std::ostringstream md;
    for (const Table &t : tables) {
        md << "### " << t.title << "\n\n";
        if (!t.note.empty()) md << t.note << "\n\n";
        md << "|";
        for (const auto &c : t.columns) md << " " << c << " |";
        md << "\n|";
        for (size_t i = 0; i < t.columns.size(); ++i) md << "---|";
        md << "\n";
        for (const auto &r : t.rows) {
            md << "|";
            for (const auto &cell : r) md << " " << cell << " |";
            md << "\n";
        }
        md << "\n";
    }
    return md.str();
}

std::string json_escape(std::string_view s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) {
            out += ' ';
            continue;
        }
        out += c;
    }
    return out;
}

std::string json(const std::vector<Table> &tables) {
    std::ostringstream js;
    js << "{\"tables\": [";
    for (size_t t = 0; t < tables.size(); ++t) {
        js << (t ? ", " : "") << "{\"title\": \"" << json_escape(tables[t].title) << "\", \"rows\": [";
        for (size_t r = 0; r < tables[t].rows.size(); ++r) {
            js << (r ? ", " : "") << "{";
            for (size_t c = 0; c < tables[t].columns.size(); ++c)
                js << (c ? ", " : "") << "\"" << json_escape(tables[t].columns[c]) << "\": \"" << json_escape(tables[t].rows[r][c]) << "\"";
            js << "}";
        }
        js << "]}";
    }
    js << "]}\n";
    return js.str();
}

// ---- sessions ----------------------------------------------------------------------------------

agent::Config llama_config(const Options &o, agent::EngineIsolation isolation) {
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = o.llama;
    cfg.model = o.model;
    cfg.workspace_dir = o.workspace;
    cfg.auto_default_tools = false;
    cfg.context_window = 4096;
    cfg.max_tokens = o.gen_tokens;
    cfg.temperature = 0.0f;
    cfg.engine_isolation = isolation;
    cfg.engine_host_path = o.host;
    if (o.threads > 0) cfg.engine_options = "{\"threads\":" + std::to_string(o.threads) + "}";
    return cfg;
}

const char *isolation_name(agent::EngineIsolation i) { return i == agent::EngineIsolation::Process ? "process" : "in-process"; }

std::string long_prompt(int words) {
    static const char *kText[] = {"The", "history", "of", "lighthouses", "spans", "centuries", "of", "maritime", "trade", "and",
                                  "engineering", "progress", "along", "dangerous", "coasts", "where", "ships", "needed", "light", "."};
    std::string p = "Read these notes, then write a long, detailed story about a lighthouse keeper.\nNotes: ";
    for (int i = 0; i < words; ++i) {
        p += kText[i % 20];
        p += ' ';
    }
    return p;
}

struct Stream {
    Clock::time_point start;
    double first_ms = -1;
};

void on_token(std::string_view, void *user_data) {
    auto *s = static_cast<Stream *>(user_data);
    if (s->first_ms < 0) s->first_ms = ms_since(s->start);
}

// ---- llm ---------------------------------------------------------------------------------------

void bench_llm(const Options &o, std::vector<Table> &tables) {
    Table t{.title = "LLM: prefill, generation, time to first token",
            .note = "`infer()` with streaming, temperature 0, KV cache cleared before every repetition; medians of " +
                    std::to_string(o.reps) + " runs. Prompt tok/s = prompt tokens / TTFT.",
            .columns = {"isolation", "prompt tok", "prompt tok/s", "gen tok", "gen tok/s", "TTFT ms", "total ms", "load ms",
                        "peak RSS MiB (this process / host)"}};
    const std::string prompt = long_prompt(o.prompt_words);
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        reset_peak_rss();
        const auto t_load = Clock::now();
        auto sess = agent::init(llama_config(o, isolation));
        const double load_ms = ms_since(t_load);
        if (!sess.ok) {
            std::fprintf(stderr, "llm (%s): %s\n", isolation_name(isolation), sess.error.message.c_str());
            continue;
        }
        std::vector<double> ttft, total, pps, gps;
        int prompt_tok = 0, gen_tok = 0;
        for (int r = 0; r < o.reps + 1; ++r) { // the first run warms up and is dropped
            agent::clear_provider_kv_cache(sess.value);
            agent::MessageView msg{.role = "user", .content = prompt};
            Stream s{.start = Clock::now()};
            auto res = agent::infer(sess.value, {.messages = {&msg, 1}, .max_tokens = o.gen_tokens, .temperature = 0.0f, .stream = true,
                                                 .on_token = on_token, .token_user_data = &s});
            const double ms = ms_since(s.start);
            if (!res.ok) {
                std::fprintf(stderr, "llm (%s): %s\n", isolation_name(isolation), res.error.message.c_str());
                break;
            }
            if (r == 0) continue;
            prompt_tok = res.value.usage.prompt_tokens;
            gen_tok = res.value.usage.completion_tokens;
            ttft.push_back(s.first_ms);
            total.push_back(ms);
            pps.push_back(prompt_tok / (s.first_ms / 1000.0));
            gps.push_back((gen_tok - 1) / ((ms - s.first_ms) / 1000.0));
        }
        if (ttft.empty()) continue;
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        const double host_rss = host_peak_rss_mib();
        t.rows.push_back({isolation_name(isolation), std::to_string(prompt_tok), fmt(median(pps)), std::to_string(gen_tok),
                          fmt(median(gps)), fmt(median(ttft)), fmt(median(total), 0), fmt(load_ms, 0),
                          fmt(peak_rss_mib(), 0) + " / " + (host_rss > 0 ? fmt(host_rss, 0) : std::string("-"))});
    }
    tables.push_back(std::move(t));
}

// ---- parallel branches -------------------------------------------------------------------------

void bench_parallel(const Options &o, std::vector<Table> &tables) {
    Table t{.title = "Graph: one branch vs parallel branches",
            .note = "A `make_parallel` node with N Agent branches on one llama instance (`max_parallel_subagents` = " +
                    std::to_string(std::max(4, o.branches)) + "), " + std::to_string(o.gen_tokens) +
                    " tokens per branch, temperature 0. Aggregate tok/s = all generated tokens / wall time.",
            .columns = {"isolation", "branches", "gen tok (all)", "wall s", "aggregate tok/s", "per-branch tok/s"}};
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        agent::Config cfg = llama_config(o, isolation);
        cfg.max_parallel_subagents = std::max(4, o.branches);
        auto sess = agent::init(cfg);
        if (!sess.ok) {
            std::fprintf(stderr, "parallel (%s): %s\n", isolation_name(isolation), sess.error.message.c_str());
            continue;
        }
        for (int n : {1, o.branches}) {
            std::vector<agent::Graph> branches;
            for (int b = 0; b < n; ++b) {
                const std::string name = "b" + std::to_string(b);
                branches.push_back(agent::make_graph(agent::GraphNode{
                    .name = name,
                    .kind = agent::GraphNodeKind::Agent,
                    .prompt = "Write a long story about lighthouse number " + std::to_string(b + 1) + ".",
                    .output_key = name}));
            }
            const agent::Graph g = agent::make_parallel("fan", std::move(branches));
            agent::GraphRun run;
            run.thread_id = std::string("bench-") + isolation_name(isolation) + "-" + std::to_string(n);
            run.durability = agent::GraphDurability::Exit;
            agent::delete_graph_thread(sess.value, run.thread_id);
            agent::clear_provider_kv_cache(sess.value);
            const auto t0 = Clock::now();
            auto res = agent::run_graph(sess.value, g, run);
            const double s = ms_since(t0) / 1000.0;
            if (!res.ok) {
                std::fprintf(stderr, "parallel (%s, %d): %s\n", isolation_name(isolation), n, res.error.message.c_str());
                continue;
            }
            int tokens = 0;
            for (const auto &[key, mem] : run.state.histories) tokens += mem.stats.completion_tokens;
            t.rows.push_back({isolation_name(isolation), std::to_string(n), std::to_string(tokens), fmt(s, 2), fmt(tokens / s),
                              fmt(tokens / s / n)});
        }
    }
    tables.push_back(std::move(t));
}

// ---- core overhead -----------------------------------------------------------------------------

void bench_overhead(const Options &o, std::vector<Table> &tables) {
    Table t{.title = "Core overhead per turn (mock engine)",
            .note = "`run_turn` on the mock engine plugin (replies instantly), " + std::to_string(o.turns) +
                    " turns after 50 warm-up turns, conversation cleared every turn: the cost of the core and, for process "
                    "isolation, the pipe round trip to agent-engine-host.",
            .columns = {"isolation", "turns", "mean µs", "p50 µs", "p99 µs"}};
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        agent::Config cfg;
        cfg.provider = agent::AiProvider::Engine;
        cfg.engine = o.mock;
        cfg.workspace_dir = o.workspace;
        cfg.auto_default_tools = false;
        cfg.engine_isolation = isolation;
        cfg.engine_host_path = o.host;
        auto sess = agent::init(cfg);
        if (!sess.ok) {
            std::fprintf(stderr, "overhead (%s): %s\n", isolation_name(isolation), sess.error.message.c_str());
            continue;
        }
        std::vector<double> us;
        for (int i = 0; i < o.turns + 50; ++i) {
            agent::FlowMemory memory;
            const auto t0 = Clock::now();
            auto r = agent::run_turn(sess.value, memory, "hello");
            const double v = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            if (!r.ok) {
                std::fprintf(stderr, "overhead: %s\n", r.error.message.c_str());
                break;
            }
            if (i >= 50) us.push_back(v);
        }
        if (us.empty()) continue;
        double sum = 0;
        for (double v : us) sum += v;
        std::sort(us.begin(), us.end());
        t.rows.push_back({isolation_name(isolation), std::to_string(us.size()), fmt(sum / static_cast<double>(us.size())),
                          fmt(us[us.size() / 2]), fmt(us[us.size() * 99 / 100])});
    }
    tables.push_back(std::move(t));
}

// ---- tts ---------------------------------------------------------------------------------------

std::string kokoro_options(const std::string &dir) {
    auto in = [&](const char *f) { return (fs::path(dir) / f).string(); };
    std::string lexicon = fs::exists(in("lexicon-us-en.txt")) ? in("lexicon-us-en.txt") : in("lexicon-en-us.txt");
    return std::string("{\"threads\":2,\"tts\":{\"kokoro\":{\"model\":\"") + in("model.onnx") + "\",\"voices\":\"" + in("voices.bin") +
           "\",\"tokens\":\"" + in("tokens.txt") + "\",\"lexicon\":\"" + lexicon + "\",\"data_dir\":\"" + in("espeak-ng-data") +
           "\",\"lang\":\"en-us\"},\"speaker_id\":0,\"speed\":1.0}}";
}

constexpr const char *kTtsText = "Local inference keeps your data on your own machine. This sentence is here so that the speech "
                                 "engine has a second sentence to stream after the first one.";

struct AudioStream {
    Clock::time_point start;
    double first_ms = -1;
    size_t bytes = 0;
};

void on_audio(void *user_data, uint32_t kind, const void *, size_t size) {
    auto *a = static_cast<AudioStream *>(user_data);
    if (kind != AGENT_CHUNK_AUDIO) return;
    if (a->first_ms < 0) a->first_ms = ms_since(a->start);
    a->bytes += size;
}

int sample_rate_of(std::string_view text) { // "Generated 2.00 s of speech (24000 Hz)."
    const size_t at = text.find('(');
    return at == std::string_view::npos ? 0 : std::atoi(std::string(text.substr(at + 1)).c_str());
}

void bench_tts(const Options &o, std::vector<Table> &tables) {
    Table t{.title = "TTS: time to first audio",
            .note = "Kokoro (sherpa engine), two sentences. `engine ABI` calls the plugin's `run()` directly to see its streamed "
                    "`AGENT_CHUNK_AUDIO` chunks (libagent does not forward audio chunks to callers yet); the libagent rows time "
                    "`run_task(AGENT_CAP_TTS)` until the WAV is written. Medians of " +
                    std::to_string(o.reps) + " runs after one warm-up.",
            .columns = {"path", "first audio ms", "total ms", "audio s", "real-time factor", "peak RSS MiB"}};
    const std::string options = kokoro_options(o.tts_model);
    auto name = agent::load_engine_plugin(o.sherpa);
    if (!name.ok) {
        std::fprintf(stderr, "tts: %s\n", name.error.message.c_str());
        return;
    }
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    };

    // 1. Engine ABI, in process: streaming chunks.
    {
        reset_peak_rss();
        const agent_engine_v1 *engine = agent::find_engine(name.value);
        const std::string config = "{\"model\":\"\",\"options\":" + options + "}";
        void *inst = nullptr;
        char err[512] = {0};
        if (engine->create(agent_str{config.data(), config.size()}, &inst, err, sizeof(err)) != AGENT_OK) {
            std::fprintf(stderr, "tts: create failed: %s\n", err);
            return;
        }
        const std::string out_dir = (fs::path(o.workspace) / "artifacts").string();
        fs::create_directories(out_dir);
        std::vector<double> first, total;
        double audio_s = 0;
        for (int r = 0; r < o.reps + 1; ++r) {
            agent_part part{};
            part.kind = AGENT_PART_TEXT;
            part.text = agent_str{kTtsText, std::strlen(kTtsText)};
            agent_message msg{};
            msg.role = agent_str{"user", 4};
            msg.parts = &part;
            msg.n_parts = 1;
            AudioStream a{.start = Clock::now()};
            agent_request req{};
            req.struct_size = sizeof(agent_request);
            req.task = AGENT_CAP_TTS;
            req.messages = &msg;
            req.n_messages = 1;
            req.output_dir = agent_str{out_dir.data(), out_dir.size()};
            req.on_chunk = on_audio;
            req.user_data = &a;
            agent_response resp{};
            resp.struct_size = sizeof(agent_response);
            const int32_t rc = engine->run(inst, &req, &resp);
            const double ms = ms_since(a.start);
            if (rc != AGENT_OK) {
                std::fprintf(stderr, "tts: run failed: %s\n", engine->last_error(inst));
                break;
            }
            int rate = 0;
            for (size_t p = 0; p < resp.message.n_parts; ++p)
                if (resp.message.parts[p].kind == AGENT_PART_TEXT)
                    rate = sample_rate_of(std::string_view(resp.message.parts[p].text.data, resp.message.parts[p].text.size));
            engine->free_response(inst, &resp);
            if (r == 0) continue;
            first.push_back(a.first_ms);
            total.push_back(ms);
            if (rate > 0) audio_s = static_cast<double>(a.bytes / sizeof(float)) / rate;
        }
        engine->destroy(inst);
        if (!total.empty())
            t.rows.push_back({"engine ABI (in process)", fmt(median(first), 0), fmt(median(total), 0), fmt(audio_s, 2),
                              audio_s > 0 ? fmt(median(total) / 1000.0 / audio_s, 2) : "-", fmt(peak_rss_mib(), 0)});
    }

    // 2. libagent run_task, in process and through agent-engine-host.
    for (auto isolation : {agent::EngineIsolation::InProcess, agent::EngineIsolation::Process}) {
        reset_peak_rss();
        agent::Config cfg;
        cfg.provider = agent::AiProvider::Engine;
        cfg.engine = o.sherpa;
        cfg.engine_options = options;
        cfg.workspace_dir = o.workspace;
        cfg.auto_default_tools = false;
        cfg.engine_isolation = isolation;
        cfg.engine_host_path = o.host;
        auto sess = agent::init(cfg);
        if (!sess.ok) {
            std::fprintf(stderr, "tts (%s): %s\n", isolation_name(isolation), sess.error.message.c_str());
            continue;
        }
        std::vector<double> total;
        for (int r = 0; r < o.reps + 1; ++r) {
            agent::MessageView msg{.role = "user", .content = kTtsText};
            const auto t0 = Clock::now();
            auto res = agent::run_task(sess.value, AGENT_CAP_TTS, {.messages = {&msg, 1}});
            const double ms = ms_since(t0);
            if (!res.ok) {
                std::fprintf(stderr, "tts (%s): %s\n", isolation_name(isolation), res.error.message.c_str());
                break;
            }
            if (r > 0) total.push_back(ms);
        }
        const double host = host_peak_rss_mib();
        if (!total.empty())
            t.rows.push_back({std::string("libagent run_task (") + isolation_name(isolation) + ")", "-", fmt(median(total), 0), "-",
                              "-", fmt(peak_rss_mib(), 0) + (host > 0 ? " (host " + fmt(host, 0) + ")" : std::string())});
    }
    tables.push_back(std::move(t));
}

bool parse(int argc, char **argv, Options &o) {
    std::map<std::string, std::string *> strings = {{"--llama", &o.llama},   {"--model", &o.model},         {"--mock", &o.mock},
                                                    {"--sherpa", &o.sherpa}, {"--tts-model", &o.tts_model}, {"--host", &o.host},
                                                    {"--json", &o.json_out}, {"--md", &o.md_out},           {"--workspace", &o.workspace}};
    std::map<std::string, int *> ints = {{"--reps", &o.reps},         {"--gen-tokens", &o.gen_tokens}, {"--prompt-words", &o.prompt_words},
                                         {"--branches", &o.branches}, {"--threads", &o.threads},       {"--turns", &o.turns}};
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) return false;
        if (auto s = strings.find(argv[i]); s != strings.end()) *s->second = argv[i + 1];
        else if (auto n = ints.find(argv[i]); n != ints.end()) *n->second = std::atoi(argv[i + 1]);
        else return false;
    }
    return o.reps > 0 && o.gen_tokens > 1 && o.branches > 0 && o.turns > 0;
}

} // namespace

int main(int argc, char **argv) {
    Options o;
    if (!parse(argc, argv, o) || (o.llama.empty() && o.mock.empty() && o.sherpa.empty())) {
        std::fprintf(stderr,
                     "usage: %s [--llama <plugin> --model <gguf>] [--mock <plugin>] [--sherpa <plugin> --tts-model <kokoro dir>]\n"
                     "          [--host <agent-engine-host>] [--reps N] [--gen-tokens N] [--prompt-words N] [--branches N]\n"
                     "          [--threads N] [--turns N] [--workspace DIR] [--json FILE] [--md FILE]\n",
                     argv[0]);
        return 2;
    }
    agent::AgentRuntime runtime;
    std::vector<Table> tables;
    if (!o.mock.empty()) bench_overhead(o, tables);
    if (!o.llama.empty() && !o.model.empty()) {
        bench_llm(o, tables);
        bench_parallel(o, tables);
    }
    if (!o.sherpa.empty() && !o.tts_model.empty()) bench_tts(o, tables);

    const std::string md = markdown(tables);
    std::fputs(md.c_str(), stdout);
    if (!o.md_out.empty()) std::ofstream(o.md_out) << md;
    if (!o.json_out.empty()) std::ofstream(o.json_out) << json(tables);
    return tables.empty() ? 1 : 0;
}
