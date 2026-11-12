// agent voice: a spoken conversation. Microphone (or a WAV file) -> speech detection (audio.cpp VAD, else an energy gate)
// -> ASR (audio.cpp) -> one LLM turn with tools (the conversation is stored
// like `agent chat`) -> TTS (audio.cpp) -> speaker (or WAV files). The loop itself is in
// voice_core.cpp; this file loads the engines and wires them in as hooks.

#include "voice.hpp"

#include "commands.hpp"

#include <agent-cpp/store_sqlite.hpp>
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>

#if !defined(_WIN32)
#include <signal.h>
#endif

namespace agent::cli {

namespace {

namespace fs = std::filesystem;

constexpr std::string_view kIndexKey = "cli:conversations"; // shared with agent chat (resumable there too)
constexpr std::string_view kVoicePrompt =
    "You are agent, a helpful voice assistant running locally on the user's computer. The user speaks to you and hears your "
    "replies through text-to-speech, so answer in one to three short, plain spoken sentences: no lists, no markdown, no code "
    "unless asked. Use the available tools when they help; file paths are relative to the workspace.";

std::atomic<bool> g_stop{false}; // Ctrl-C: also cancels the running engine call
constexpr int kMaxFlags = 4;
std::atomic<int32_t *> g_cancel_flags[kMaxFlags] = {};
std::FILE *g_err = stderr;
bool g_at_line_start = true;

void on_sigint(int) {
    g_stop.store(true);
    for (auto &f : g_cancel_flags)
        if (int32_t *flag = f.load()) std::atomic_ref<int32_t>(*flag).store(1, std::memory_order_relaxed);
}

void install_sigint() {
#if defined(_WIN32)
    std::signal(SIGINT, on_sigint);
#else
    struct sigaction sa {};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // no SA_RESTART: a blocked permission prompt returns
    sigaction(SIGINT, &sa, nullptr);
#endif
}

void out(std::string_view s) {
    if (s.empty()) return;
    std::fwrite(s.data(), 1, s.size(), stdout);
    g_at_line_start = s.back() == '\n';
    std::fflush(stdout);
}

void ensure_newline() {
    if (!g_at_line_start) out("\n");
}

struct VoiceState {
    Context *ctx = nullptr;
    VoiceSettings settings;
    std::string workspace;
    bool interactive = false;

    ModelRef model;
    LoadedEngine llm_engine;
    LoadedEngine speech_engine;
    Session llm, asr, tts, vad;
    bool has_vad = false;
    Conversation memory;
    std::vector<std::string> always_allowed;
    StreamFilter filter;
    size_t streamed = 0;
    bool vad_reset_pending = true;
    EnergyGate gate;
};

void log_fn(LogLevel level, std::string_view message, void *user_data) {
    const auto *st = static_cast<const VoiceState *>(user_data);
    if (!st->settings.verbose && (level == LogLevel::Debug || level == LogLevel::Info)) return;
    const char *prefix = level == LogLevel::Error ? "error" : level == LogLevel::Warning ? "warning" : "log";
    ensure_newline();
    std::fprintf(g_err, "[%s] %.*s\n", prefix, static_cast<int>(message.size()), message.data());
}

// -------------------------- tools (same policy and prompts as agent chat) --------------------------

PermissionDecision check_tool(std::string_view name, std::string_view arguments, void *user_data) {
    auto *st = static_cast<VoiceState *>(user_data);
    ensure_newline();
    out("[tool] " + std::string(name) + " " + one_line(arguments, 300) + "\n");
    PermissionDecision d = tools::default_permission_policy(name, arguments, nullptr);
    if (d == PermissionDecision::Ask && std::find(st->always_allowed.begin(), st->always_allowed.end(), name) != st->always_allowed.end())
        return PermissionDecision::Allow;
    return d;
}

bool prompt_tool(std::string_view name, std::string_view, void *user_data) {
    auto *st = static_cast<VoiceState *>(user_data);
    const std::string n(name);
    if (st->settings.auto_approve) {
        out("[permission] " + n + ": approved (--yes)\n");
        return true;
    }
    if (!st->interactive) {
        out("[permission] " + n + ": denied (stdin is not a terminal; pass --yes to approve such calls)\n");
        return false;
    }
    out("[permission] allow " + n + "? [y]es / [N]o / [a]lways this session: ");
    std::string line;
    const ReadStatus rs = read_line(stdin, line);
    g_at_line_start = true;
    if (rs != ReadStatus::Line) {
        out("\n[permission] denied\n");
        return false;
    }
    const size_t i = line.find_first_not_of(" \t");
    const char c = i == std::string::npos ? 'n' : line[i];
    if (c == 'a' || c == 'A') {
        st->always_allowed.push_back(n);
        return true;
    }
    return c == 'y' || c == 'Y';
}

void on_tool_result(const Event &event, void *) {
    const std::string_view result(reinterpret_cast<const char *>(event.payload.data()), event.payload.size());
    ensure_newline();
    out("[result] " + one_line(result, 300) + "\n");
}

void on_token(std::string_view token, void *user_data) {
    auto *st = static_cast<VoiceState *>(user_data);
    st->streamed += token.size();
    out(filter_stream(st->filter, token));
}

// -------------------------- sessions --------------------------

Config base_config(VoiceState &st, const LoadedEngine &engine) {
    Config cfg;
    cfg.provider = AiProvider::Engine;
    cfg.engine = engine.name;
    cfg.engine_isolation = cli_engine_isolation(st.settings.inprocess, engine.isolation);
    cfg.engine_sandbox.gpu = cli_engine_gpu(engine.backend);
    apply_sandbox_choice(cfg);
    cfg.workspace_dir = st.workspace;
    cfg.logger = log_fn;
    cfg.logger_user_data = &st;
    return cfg;
}

Result<Session> open_speech_session(VoiceState &st, const std::string &options) {
    Config cfg = base_config(st, st.speech_engine);
    cfg.engine_options = options;
    cfg.auto_default_tools = false;
    return init(cfg);
}

Result<void> open_llm_session(VoiceState &st) {
    Config cfg = base_config(st, st.llm_engine);
    cfg.engine_options = st.settings.engine_options;
    cfg.model = st.model.path;
    cfg.memory = {.store = store::sqlite_store()};
    cfg.context_window = st.settings.context_window > 0 ? st.settings.context_window : st.ctx->config.context_window;
    cfg.max_tokens = st.settings.max_tokens > 0 ? st.settings.max_tokens : st.ctx->config.max_tokens;
    cfg.temperature = st.ctx->config.temperature;
    cfg.enable_file_logging = true;
    cfg.permission_check = check_tool;
    cfg.permission_prompt = prompt_tool;
    cfg.permission_user_data = &st;
    auto s = init(cfg);
    if (!s.ok) return fail(s.error.code, s.error.message);
    st.llm = std::move(s.value);
    register_event_hook(st.llm, EventType::OnToolResult, on_tool_result, &st);
    return ok();
}

// One user message holding 16 kHz samples as an F32 tensor (the audiocpp engine's in-memory audio input).
Result<InferResponse> infer_audio(Session &session, std::span<const float> samples, std::string_view text) {
    session.arena.reset();
    const int64_t shape[] = {static_cast<int64_t>(samples.size())};
    const TensorView tensor{.name = "audio",
                            .dtype = DType::Float32,
                            .shape = shape,
                            .data = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(samples.data()), samples.size_bytes())};
    const MessageView msg{.role = "user", .content = text, .tensors = std::span<const TensorView>(&tensor, 1)};
    InferRequest req;
    req.messages = std::span<const MessageView>(&msg, 1);
    req.cancel = session.cancel_flag.get();
    return infer(session, req);
}

// -------------------------- hooks --------------------------

Result<std::string> transcribe(std::span<const float> utterance, void *user_data) {
    auto *st = static_cast<VoiceState *>(user_data);
    auto r = infer_audio(st->asr, utterance, {});
    if (!r.ok) return fail<std::string>(r.error.code, "ASR: " + r.error.message);
    return ok(std::string(r.value.message.content));
}

void remember_conversation(VoiceState &st) {
    auto index = retrieve_memory(st.llm, kIndexKey);
    const std::string current = index.ok ? index.value : std::string();
    const std::string updated = add_conversation_id(current, st.memory.id);
    if (updated != current) store_memory(st.llm, kIndexKey, updated);
}

Result<std::string> respond(std::string_view transcript, void *user_data) {
    auto *st = static_cast<VoiceState *>(user_data);
    ensure_newline();
    out("agent: ");
    st->streamed = 0;
    clear_cancel(st->llm);
    auto r = run_turn(st->llm, st->memory, transcript, true, on_token, st);
    out(flush_stream(st->filter));
    if (r.ok && st->streamed == 0) out(r.value);
    ensure_newline();
    remember_conversation(*st);
    if (!r.ok) return fail<std::string>(r.error.code, "LLM: " + r.error.message);
    return r;
}

Result<audio::Wav> speak(std::string_view reply, void *user_data) {
    auto *st = static_cast<VoiceState *>(user_data);
    st->tts.arena.reset();
    const MessageView msg{.role = "user", .content = reply};
    InferRequest req;
    req.messages = std::span<const MessageView>(&msg, 1);
    req.cancel = st->tts.cancel_flag.get();
    auto r = infer(st->tts, req);
    if (!r.ok) return fail<audio::Wav>(r.error.code, "TTS: " + r.error.message);
    if (r.value.message.media.empty()) return fail<audio::Wav>(ErrorCode::DecodeFailed, "TTS: the engine returned no audio");
    const std::string path(r.value.message.media[0].path);
    auto wav = audio::read_wav(path);
    std::error_code ec;
    fs::remove(path, ec); // the output backend keeps (or plays) its own copy
    if (!wav.ok) return fail<audio::Wav>(wav.error.code, "TTS: " + wav.error.message);
    return wav;
}

void on_event(VoiceEvent event, std::string_view text, void *user_data) {
    const auto *st = static_cast<const VoiceState *>(user_data);
    switch (event) {
    case VoiceEvent::Listening:
        ensure_newline();
        out("[listening]\n");
        break;
    case VoiceEvent::Heard:
        ensure_newline();
        out("you: " + std::string(text) + "\n");
        break;
    case VoiceEvent::Skipped:
        if (st->settings.verbose) out("[skipped] " + (text.empty() ? std::string("(no words)") : std::string(text)) + "\n");
        break;
    case VoiceEvent::Reply: break; // already streamed by respond()
    case VoiceEvent::Spoken: out("[spoken] " + std::string(text) + "\n"); break;
    case VoiceEvent::BargeIn:
        ensure_newline();
        out("[barge-in] playback stopped\n");
        break;
    case VoiceEvent::Error:
        ensure_newline();
        out("[error] " + std::string(text) + "\n");
        break;
    }
}

// -------------------------- the audiocpp VAD engine as a SpeechDetector --------------------------

Result<bool> vad_feed(void *state, std::span<const float> samples, std::vector<SpeechSegment> &finished) {
    auto *st = static_cast<VoiceState *>(state);
    const bool reset = st->vad_reset_pending;
    auto r = infer_audio(st->vad, samples, reset ? "reset" : "");
    if (!r.ok) return fail<bool>(r.error.code, "VAD: " + r.error.message);
    st->vad_reset_pending = false;
    auto segs = parse_vad_segments(r.value.message.content);
    if (!segs.ok) return fail<bool>(segs.error.code, segs.error.message);
    bool speaking = false;
    for (const VadJsonSegment &s : segs.value) {
        if (s.open) speaking = true;
        else finished.push_back({static_cast<int64_t>(s.start * audio::kCaptureRate), static_cast<int64_t>(s.end * audio::kCaptureRate)});
    }
    return ok(speaking);
}

Result<void> vad_flush(void *state, std::vector<SpeechSegment> &finished) {
    auto *st = static_cast<VoiceState *>(state);
    const std::vector<float> tail(audio::kCaptureRate / 100, 0.0f); // the request needs some audio: 10 ms of silence
    auto r = infer_audio(st->vad, tail, "flush");
    if (!r.ok) return fail(r.error.code, "VAD: " + r.error.message);
    auto segs = parse_vad_segments(r.value.message.content);
    if (!segs.ok) return fail(segs.error.code, segs.error.message);
    for (const VadJsonSegment &s : segs.value)
        finished.push_back({static_cast<int64_t>(s.start * audio::kCaptureRate), static_cast<int64_t>(s.end * audio::kCaptureRate)});
    return ok();
}

void vad_reset(void *state) { static_cast<VoiceState *>(state)->vad_reset_pending = true; }

constexpr SpeechDetectorOps kVadOps = {vad_feed, vad_flush, vad_reset};

std::string new_conversation_id() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "voice-%Y%m%d-%H%M%S", &tm);
    return buf;
}

int voice_fail(const Error &e) {
    ensure_newline();
    std::fprintf(g_err, "error: %s\n", e.message.c_str());
    return 1;
}

int run_voice(VoiceState &st, const std::string &engine_log);

} // namespace

int cmd_voice(const CommandLine &cl, Context &ctx) {
    auto settings = voice_settings_from(cl);
    if (!settings.ok) return report_error(settings.error);
    VoiceState st;
    st.ctx = &ctx;
    st.settings = std::move(settings.value);
    st.workspace = workspace_dir(ctx, st.settings.workspace);
    st.interactive = stdin_is_tty();

    auto model = resolve_model(st.settings.model, ctx.config.default_model, catalog_refs(ctx), file_exists);
    if (!model.ok) return report_error(model.error);
    st.model = model.value;
    auto llm = load_engine_for(ctx, st.settings.engine, st.model.path);
    if (!llm.ok) return report_error(llm.error);
    st.llm_engine = llm.value;
    auto speech = load_engine_for(ctx, "audiocpp", {});
    if (!speech.ok) return report_error(speech.error);
    st.speech_engine = speech.value;
    if (!st.settings.input_wav.empty() && !file_exists(st.settings.input_wav))
        return report_error(make_error(ErrorCode::FilesystemError, "--input-wav: no such file: " + st.settings.input_wav));

    std::string engine_log;
    if (!st.settings.verbose) {
        std::error_code ec;
        fs::create_directories(fs::path(st.workspace) / "logs", ec);
        engine_log = (fs::path(st.workspace) / "logs" / "engine.log").string();
        if (std::FILE *original = redirect_stderr(engine_log)) g_err = original;
        else engine_log.clear();
    }
    const int rc = run_voice(st, engine_log);
    for (auto &f : g_cancel_flags) f.store(nullptr);
    st.llm = Session{};
    st.asr = Session{};
    st.tts = Session{};
    st.vad = Session{};
    if (g_err != stderr) restore_stderr(g_err);
    g_err = stderr;
    return rc;
}

namespace {

int run_voice(VoiceState &st, const std::string &engine_log) {
    // Audio first: a missing device should fail before seconds of model loading.
    const bool file_input = !st.settings.input_wav.empty();
    auto input = file_input ? audio::open_file_input(st.settings.input_wav) : audio::open_device_input();
    if (!input.ok) {
        if (!file_input)
            return voice_fail(make_error(input.error.code, input.error.message + "; use --input-wav FILE to read speech from a WAV file"));
        return voice_fail(input.error);
    }
    std::string output_dir = st.settings.output_dir;
    Result<audio::AudioOutput> output = fail<audio::AudioOutput>(ErrorCode::InvalidConfig, "");
    if (output_dir.empty() && !file_input) {
        output = audio::open_device_output();
        if (!output.ok) {
            output_dir = (fs::path(st.workspace) / "voice").string();
            std::fprintf(g_err, "warning: %s; replies go to %s instead\n", output.error.message.c_str(), output_dir.c_str());
        }
    }
    if (!output.ok) {
        if (output_dir.empty()) output_dir = (fs::path(st.workspace) / "voice").string();
        output = audio::open_file_output(output_dir, "reply", st.settings.barge_in);
        if (!output.ok) return voice_fail(output.error);
    }

    auto asr_opts = audiocpp_asr_options(st.settings.asr_model);
    if (!asr_opts.ok) return voice_fail(asr_opts.error);
    auto tts_opts = audiocpp_tts_options(st.settings.tts_model, st.settings.voice);
    if (!tts_opts.ok) return voice_fail(tts_opts.error);
    Result<std::string> vad_opts = ok(std::string());
    if (!st.settings.vad_model.empty()) {
        vad_opts = audiocpp_vad_options(st.settings.vad_model);
        if (!vad_opts.ok) return voice_fail(vad_opts.error);
    }

    std::fprintf(g_err, "loading speech models and %s ...\n", st.model.path.c_str());
    auto asr = open_speech_session(st, asr_opts.value);
    if (!asr.ok) return voice_fail(make_error(asr.error.code, "ASR: " + asr.error.message));
    st.asr = std::move(asr.value);
    auto tts = open_speech_session(st, tts_opts.value);
    if (!tts.ok) return voice_fail(make_error(tts.error.code, "TTS: " + tts.error.message));
    st.tts = std::move(tts.value);
    if (!vad_opts.value.empty()) {
        auto vad = open_speech_session(st, vad_opts.value);
        if (!vad.ok) return voice_fail(make_error(vad.error.code, "VAD: " + vad.error.message));
        st.vad = std::move(vad.value);
        st.has_vad = true;
    }
    if (auto r = open_llm_session(st); !r.ok) return voice_fail(r.error);

    if (!st.settings.resume.empty()) {
        st.memory.id = st.settings.resume;
        if (auto r = load_conversation(st.llm, st.memory); !r.ok) {
            if (r.error.code == ErrorCode::KeyNotFound)
                return voice_fail(make_error(ErrorCode::KeyNotFound, "no saved conversation '" + st.memory.id + "' in " + st.workspace));
            return voice_fail(r.error);
        }
    } else {
        st.memory.id = new_conversation_id();
        st.memory.history.push_back(Message{.role = "system", .content = std::string(kVoicePrompt)});
    }

    SpeechDetector detector = st.has_vad ? SpeechDetector{&kVadOps, &st, "vad"} : energy_detector(st.gate);
    const bool process = cli_engine_isolation(st.settings.inprocess, st.llm_engine.isolation) == EngineIsolation::Process;
    std::printf("agent voice  (%s)\n", st.memory.id.c_str());
    std::printf("  model:     %s  %s  (%s %s)\n", st.model.name.c_str(), st.model.path.c_str(), st.llm_engine.name.c_str(),
                process ? "agent-engine-host" : "in-process");
    std::printf("  speech:    %s %s  asr %s, tts %s\n", st.speech_engine.name.c_str(), st.speech_engine.version.c_str(),
                st.settings.asr_model.c_str(), st.settings.tts_model.c_str());
    std::printf("  detector:  %s\n",
                st.has_vad ? ("audio.cpp VAD " + st.settings.vad_model).c_str() : "energy gate (pass --vad-model for a VAD model)");
    std::printf("  input:     %s\n  output:    %s%s\n", input.value.name.c_str(), output.value.name.c_str(),
                st.settings.barge_in ? "  (barge-in on)" : "");
    std::printf("  workspace: %s\n", st.workspace.c_str());
    if (!engine_log.empty()) std::printf("  engine log: %s (--verbose prints it here)\n", engine_log.c_str());
    std::printf("Speak. Ctrl-C quits.\n");
    std::fflush(stdout);
    g_at_line_start = true;

    g_stop.store(false);
    g_cancel_flags[0].store(st.llm.cancel_flag.get());
    g_cancel_flags[1].store(st.asr.cancel_flag.get());
    g_cancel_flags[2].store(st.tts.cancel_flag.get());
    if (st.has_vad) g_cancel_flags[3].store(st.vad.cancel_flag.get());
    install_sigint();

    VoiceHooks hooks;
    hooks.transcribe = transcribe;
    hooks.respond = respond;
    hooks.speak = speak;
    hooks.on_event = on_event;
    hooks.user_data = &st;
    VoiceLoopOptions options;
    options.barge_in = st.settings.barge_in;
    options.max_turns = st.settings.max_turns;
    options.stop = &g_stop;
    auto stats = run_voice_loop(input.value, output.value, detector, hooks, options);
    std::signal(SIGINT, SIG_DFL);

    ensure_newline();
    if (const audio::FileOutputInfo *info = audio::file_output_info(output.value))
        for (const std::string &path : info->written) out("[reply] " + path + "\n");
    if (!stats.ok) return voice_fail(stats.error);
    out(std::to_string(stats.value.turns) + " turn(s)" +
        (stats.value.barge_ins ? ", " + std::to_string(stats.value.barge_ins) + " barge-in(s)" : "") + "; saved as " + st.memory.id +
        "; resume with: agent voice --resume " + st.memory.id + " (or agent chat --resume)\n");
    return 0;
}

} // namespace

} // namespace agent::cli
