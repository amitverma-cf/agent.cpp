// agent-engine-trellis: TRELLIS.2 image -> 3D (GLB) as an agent.cpp engine plugin, over trellis.cpp (github.com/pwilkin/trellis.cpp).
//
// trellis.cpp publishes its CPU, Vulkan, CUDA and ROCm builds as programs (trellis-cli, trellis-server) and no library, so this plugin
// runs the release's `trellis-cli` for each request: the installer puts the program and its ggml libraries next to the plugin, the
// plugin builds the command line, waits for it and returns the GLB. Nothing listens on a socket (trellis-server is not used).
//   - AGENT_CAP_MESH_GEN: the last image FILE (or BYTES) part of the last user message is the input; the result is
//     <output_dir>/trellis-<unix ms>-<n>.glb as a FILE part (model/gltf-binary) plus a short text part.
//   - config "model" is the folder with the GGUF weights (what `--models` takes); options and params_json set the generation keys.
// Cancellation ends the child process.

#include <agent-cpp/engine_abi.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <csignal>
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }
agent_str to_c(const std::string &s) { return agent_str{s.data(), s.size()}; }

// ------------------------------------------------------------------ settings

// Defaults come from Config::engine_options at create(); agent_request::params_json overrides them per call.
struct GenSettings {
    int64_t seed = -1;           // -1 = random
    int resolution = 1024;       // 512 (light), 1024 (cascade, default), 1536
    int steps = 0;               // flow sampler steps; 0 = the model's default
    float guidance = -1.0f;      // sparse-structure guidance; < 0 = default (7.5)
    float guidance_shape = -1.0f; // shape guidance; < 0 = default (7.5)
    std::string background = "auto"; // "auto", "birefnet" or "threshold"
    bool texture = true;         // false: geometry only
    bool xatlas = true;          // false: box UV projection
    int faces = 0;               // face budget of the decimated mesh; 0 = default
    int atlas = 0;               // UV atlas size in px; 0 = default
    int texture_resolution = 0;  // PBR texture resolution; 0 = automatic
    bool normal_map = false;
    int gpu = 0;                 // GPU index; < 0 runs on the CPU
    std::string backend;         // force a ggml backend by name (Vulkan, CUDA, ...); empty = automatic
    int threads = 0;             // CPU threads; 0 = all
};

template <typename T> T json_number(simdjson::ondemand::object &obj, const char *key, T fallback) {
    double d = 0;
    if (obj[key].get_double().get(d)) return fallback;
    return static_cast<T>(d);
}

bool json_string(simdjson::ondemand::object &obj, const char *key, std::string &out) {
    std::string_view s;
    if (obj[key].get_string().get(s)) return false;
    out = std::string(s);
    return true;
}

bool json_bool(simdjson::ondemand::object &obj, const char *key, bool fallback) {
    bool b = false;
    if (obj[key].get_bool().get(b)) return fallback;
    return b;
}

// Applies the keys present in obj on top of g. Returns false (with error) on bad values.
bool apply_gen_settings(simdjson::ondemand::object &obj, GenSettings &g, std::string &error) {
    int64_t seed = 0;
    if (!obj["seed"].get_int64().get(seed)) g.seed = seed;
    g.resolution = json_number<int>(obj, "resolution", g.resolution);
    g.steps = json_number<int>(obj, "steps", g.steps);
    g.guidance = json_number<float>(obj, "guidance", g.guidance);
    g.guidance_shape = json_number<float>(obj, "guidance_shape", g.guidance_shape);
    json_string(obj, "background", g.background);
    g.texture = json_bool(obj, "texture", g.texture);
    g.xatlas = json_bool(obj, "xatlas", g.xatlas);
    g.faces = json_number<int>(obj, "faces", g.faces);
    g.atlas = json_number<int>(obj, "texture_size", g.atlas);
    g.texture_resolution = json_number<int>(obj, "texture_resolution", g.texture_resolution);
    g.normal_map = json_bool(obj, "normal_map", g.normal_map);
    g.gpu = json_number<int>(obj, "gpu", g.gpu);
    json_string(obj, "backend", g.backend);
    g.threads = json_number<int>(obj, "threads", g.threads);

    if (g.resolution != 512 && g.resolution != 1024 && g.resolution != 1536) {
        error = "trellis: resolution must be 512, 1024 or 1536";
        return false;
    }
    if (g.background != "auto" && g.background != "birefnet" && g.background != "threshold") {
        error = "trellis: background must be auto, birefnet or threshold (got '" + g.background + "')";
        return false;
    }
    if (g.steps < 0 || g.steps > 1000) {
        error = "trellis: steps must be in 0..1000 (0 = model default)";
        return false;
    }
    if (g.faces < 0 || g.atlas < 0 || g.atlas > 8192 || g.texture_resolution < 0 || g.threads < 0) {
        error = "trellis: faces, texture_size, texture_resolution and threads must not be negative (texture_size at most 8192)";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ the trellis-cli program

#if defined(_WIN32)
constexpr const char *kCliName = "trellis-cli.exe";
#else
constexpr const char *kCliName = "trellis-cli";
#endif

// The folder this plugin was loaded from: the installer puts trellis-cli and its libraries beside it.
fs::path plugin_folder() {
#if defined(_WIN32)
    HMODULE self = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&plugin_folder), &self)) {
        wchar_t buf[MAX_PATH * 4];
        const DWORD n = GetModuleFileNameW(self, buf, static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])));
        if (n > 0 && n < sizeof(buf) / sizeof(buf[0])) return fs::path(std::wstring(buf, n)).parent_path();
    }
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&plugin_folder), &info) && info.dli_fname) return fs::path(info.dli_fname).parent_path();
#endif
    return {};
}

bool is_file(const fs::path &p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

std::string find_cli(const std::string &configured) {
    if (!configured.empty()) return is_file(configured) ? configured : std::string();
    if (const char *env = std::getenv("AGENT_TRELLIS_CLI"); env && *env) return is_file(env) ? std::string(env) : std::string();
    const fs::path beside = plugin_folder() / kCliName;
    return is_file(beside) ? beside.string() : std::string();
}

struct RunResult {
    int exit_code = -1;
    bool cancelled = false;
    std::string tail; // the end of what the program printed, for an error message
};

void keep_tail(std::string &tail, const char *data, size_t n) {
    tail.append(data, n);
    if (tail.size() > 4096) tail.erase(0, tail.size() - 4096);
}

bool is_cancelled(const agent_request *req) { return req->cancel && *req->cancel; }

#if defined(_WIN32)

std::wstring widen(const std::string &s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// One argument as the Windows command line wants it (CommandLineToArgvW rules).
void append_quoted(std::wstring &line, const std::wstring &arg) {
    if (!line.empty()) line += L' ';
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        line += arg;
        return;
    }
    line += L'"';
    size_t backslashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }
        if (c == L'"') line.append(backslashes * 2 + 1, L'\\');
        else line.append(backslashes, L'\\');
        backslashes = 0;
        line += c;
    }
    line.append(backslashes * 2, L'\\');
    line += L'"';
}

RunResult run_program(const std::string &exe, const std::vector<std::string> &args, const agent_request *req) {
    RunResult r;
    std::wstring line;
    append_quoted(line, widen(exe));
    for (const std::string &a : args) append_quoted(line, widen(a));

    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE read_end = nullptr, write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &sa, 0)) {
        r.tail = "could not create a pipe";
        return r;
    }
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_end;
    si.hStdError = write_end;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    const std::wstring dir = widen(fs::path(exe).parent_path().string());
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, dir.empty() ? nullptr : dir.c_str(), &si, &pi)) {
        r.tail = "could not start " + exe + " (error " + std::to_string(GetLastError()) + ")";
        CloseHandle(read_end);
        CloseHandle(write_end);
        return r;
    }
    CloseHandle(write_end);
    char buf[4096];
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(read_end, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
            DWORD got = 0;
            if (ReadFile(read_end, buf, sizeof(buf), &got, nullptr) && got > 0) keep_tail(r.tail, buf, got);
            continue;
        }
        if (is_cancelled(req)) {
            TerminateProcess(pi.hProcess, 1);
            r.cancelled = true;
            break;
        }
        if (WaitForSingleObject(pi.hProcess, 100) == WAIT_OBJECT_0) {
            while (PeekNamedPipe(read_end, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
                DWORD got = 0;
                if (!ReadFile(read_end, buf, sizeof(buf), &got, nullptr) || got == 0) break;
                keep_tail(r.tail, buf, got);
            }
            break;
        }
    }
    DWORD code = 1;
    WaitForSingleObject(pi.hProcess, 5000);
    GetExitCodeProcess(pi.hProcess, &code);
    r.exit_code = static_cast<int>(code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(read_end);
    return r;
}

#else

RunResult run_program(const std::string &exe, const std::vector<std::string> &args, const agent_request *req) {
    RunResult r;
    int fds[2];
    if (pipe(fds) != 0) {
        r.tail = "could not create a pipe";
        return r;
    }
    std::vector<std::string> argv_store;
    argv_store.push_back(exe);
    argv_store.insert(argv_store.end(), args.begin(), args.end());
    std::vector<char *> argv;
    for (std::string &a : argv_store) argv.push_back(a.data());
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        r.tail = "could not start " + exe;
        close(fds[0]);
        close(fds[1]);
        return r;
    }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        const std::string dir = fs::path(exe).parent_path().string();
        if (!dir.empty() && chdir(dir.c_str()) != 0) _exit(126);
        execv(exe.c_str(), argv.data());
        _exit(127);
    }
    close(fds[1]);
    char buf[4096];
    for (;;) {
        if (is_cancelled(req)) {
            kill(pid, SIGTERM);
            r.cancelled = true;
            break;
        }
        pollfd pfd{fds[0], POLLIN, 0};
        const int rc = poll(&pfd, 1, 100);
        if (rc > 0) {
            const ssize_t n = read(fds[0], buf, sizeof(buf));
            if (n > 0) keep_tail(r.tail, buf, static_cast<size_t>(n));
            else if (n == 0) break; // the program closed its output: it is ending
        }
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    r.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return r;
}

#endif

// ------------------------------------------------------------------ instance

struct Instance {
    std::string cli;
    std::string models;
    GenSettings defaults;
    uint64_t counter = 0;

    // Response storage, valid until free_response() / the next run().
    std::string path;
    std::string text;
    agent_part parts[2]{};
    std::string error;
};

int32_t t2_engine_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(sv(config_json));
    simdjson::ondemand::document doc;
    simdjson::ondemand::object cfg;
    if (parser.iterate(json).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "trellis: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    std::string model, cli;
    json_string(cfg, "model", model);
    GenSettings defaults;
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        json_string(options, "cli", cli);
        std::string error;
        if (!apply_gen_settings(options, defaults, error)) {
            std::snprintf(err, err_cap, "%s", error.c_str());
            return AGENT_ERR_INVALID_CONFIG;
        }
    }
    if (model.empty()) {
        std::snprintf(err, err_cap, "trellis: config.model (the folder with the TRELLIS.2 GGUF files) is required");
        return AGENT_ERR_INVALID_CONFIG;
    }
    std::error_code ec;
    if (!fs::is_directory(model, ec)) {
        std::snprintf(err, err_cap, "trellis: model directory not found: %s", model.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    const std::string program = find_cli(cli);
    if (program.empty()) {
        std::snprintf(err, err_cap, "trellis: %s not found next to the plugin (add the engine with `agent engine pull trellis`; options.cli or AGENT_TRELLIS_CLI name another)", kCliName);
        return AGENT_ERR_INIT_FAILED;
    }
    auto *inst = new Instance();
    inst->cli = program;
    inst->models = model;
    inst->defaults = defaults;
    *out_instance = inst;
    return AGENT_OK;
}

void t2_engine_destroy(void *instance) { delete static_cast<Instance *>(instance); }

int32_t fail(Instance *inst, int32_t code, std::string msg) {
    inst->error = std::move(msg);
    return code;
}

const agent_message *last_user_message(const agent_request *req) {
    for (size_t i = req->n_messages; i-- > 0;)
        if (sv(req->messages[i].role) == "user") return &req->messages[i];
    return nullptr;
}

const agent_part *last_image_part(const agent_message *m) {
    if (!m) return nullptr;
    const agent_part *src = nullptr;
    for (size_t p = 0; p < m->n_parts; ++p) {
        const agent_part &part = m->parts[p];
        const std::string_view mime = sv(part.mime);
        if ((part.kind == AGENT_PART_FILE || part.kind == AGENT_PART_BYTES) && (mime.empty() || mime.starts_with("image/"))) src = &part;
    }
    return src;
}

// <output_dir>/trellis-<unix ms>-<n><ext>, skipping names that already exist.
std::string unique_path(Instance *inst, const std::string &dir, const char *stem, const char *ext) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    fs::path base(dir);
    for (;;) {
        fs::path p = base / (std::string(stem) + std::to_string(ms) + "-" + std::to_string(inst->counter++) + ext);
        std::error_code ec;
        if (!fs::exists(p, ec)) return p.string();
    }
}

std::string number_text(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(v));
    return buf;
}

int32_t t2_engine_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    inst->path.clear();
    inst->text.clear();
    if (req->task != AGENT_CAP_MESH_GEN) return fail(inst, AGENT_ERR_UNSUPPORTED, "trellis: only MESH_GEN is supported");

    const agent_part *image = last_image_part(last_user_message(req));
    if (!image) return fail(inst, AGENT_ERR_INVALID_CONFIG, "trellis: MESH_GEN needs an image FILE or BYTES part in the last user message");
    const std::string output_dir(sv(req->output_dir));
    if (output_dir.empty()) return fail(inst, AGENT_ERR_INVALID_CONFIG, "trellis: request output_dir is required");

    GenSettings g = inst->defaults;
    if (req->params_json.size) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(sv(req->params_json));
        simdjson::ondemand::document doc;
        simdjson::ondemand::object params;
        if (parser.iterate(json).get(doc) || doc.get_object().get(params))
            return fail(inst, AGENT_ERR_PARSE, "trellis: params_json is not a JSON object");
        std::string error;
        if (!apply_gen_settings(params, g, error)) return fail(inst, AGENT_ERR_INVALID_CONFIG, error);
    }
    if (is_cancelled(req)) return fail(inst, AGENT_ERR_CANCELLED, "trellis: cancelled");

    std::error_code ec;
    fs::create_directories(output_dir, ec);

    // The input: a FILE part is used where it is; BYTES are written next to the result for the length of the run.
    std::string input;
    bool temp_input = false;
    if (image->kind == AGENT_PART_FILE) {
        input = std::string(sv(image->path));
        if (!is_file(input)) return fail(inst, AGENT_ERR_FILESYSTEM, "trellis: could not read image " + input);
    } else {
        if (!image->data || image->size == 0) return fail(inst, AGENT_ERR_INVALID_CONFIG, "trellis: the input image is empty");
        input = unique_path(inst, output_dir, "trellis-input-", ".img");
        std::FILE *f = std::fopen(input.c_str(), "wb");
        const bool ok = f && std::fwrite(image->data, 1, image->size, f) == image->size;
        if (f) std::fclose(f);
        if (!ok) return fail(inst, AGENT_ERR_FILESYSTEM, "trellis: could not write " + input);
        temp_input = true;
    }

    const std::string output = unique_path(inst, output_dir, "trellis-", ".glb");
    std::vector<std::string> args = {"--image", input, "--output", output, "--models", inst->models, "--res", std::to_string(g.resolution)};
    auto add = [&](const char *flag, const std::string &value) {
        args.push_back(flag);
        args.push_back(value);
    };
    if (g.seed >= 0) add("--seed", std::to_string(g.seed));
    if (g.steps > 0) add("--steps", std::to_string(g.steps));
    if (g.guidance >= 0) add("--gss", number_text(g.guidance));
    if (g.guidance_shape >= 0) add("--gsh", number_text(g.guidance_shape));
    if (g.background != "auto") add("--bg-removal", g.background);
    if (!g.texture) args.push_back("--no-texture");
    if (!g.xatlas) args.push_back("--box-uv");
    if (g.faces > 0) add("--faces", std::to_string(g.faces));
    if (g.atlas > 0) add("--atlas", std::to_string(g.atlas));
    if (g.texture_resolution > 0) add("--tex-res", std::to_string(g.texture_resolution));
    if (g.normal_map) add("--normal-map", "on");
    add("--gpu", std::to_string(g.gpu));
    if (!g.backend.empty()) add("--backend", g.backend);
    if (g.threads > 0) add("--threads", std::to_string(g.threads));

    const RunResult run = run_program(inst->cli, args, req);
    if (temp_input) fs::remove(input, ec);
    if (run.cancelled || is_cancelled(req)) {
        fs::remove(output, ec);
        return fail(inst, AGENT_ERR_CANCELLED, "trellis: cancelled");
    }
    if (run.exit_code != 0 || !is_file(output)) {
        fs::remove(output, ec);
        std::string tail = run.tail;
        while (!tail.empty() && (tail.back() == '\n' || tail.back() == '\r' || tail.back() == ' ')) tail.pop_back();
        return fail(inst, AGENT_ERR_DECODE_FAILED,
                    "trellis: generation failed (trellis-cli exit " + std::to_string(run.exit_code) + ")" + (tail.empty() ? "" : ": " + tail));
    }
    if (req->on_progress) req->on_progress(req->user_data, 1, 1);

    const uintmax_t bytes = fs::file_size(output, ec);
    inst->path = output;
    inst->text = "generated 3D model: " + inst->path + " (" + std::to_string(ec ? 0 : static_cast<unsigned long long>(bytes / 1024)) + " KiB)";
    inst->parts[0] = agent_part{};
    inst->parts[0].kind = AGENT_PART_FILE;
    inst->parts[0].path = to_c(inst->path);
    inst->parts[0].mime = agent_str{"model/gltf-binary", 17};
    inst->parts[1] = agent_part{};
    inst->parts[1].kind = AGENT_PART_TEXT;
    inst->parts[1].text = to_c(inst->text);

    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = inst->parts;
    resp->message.n_parts = 2;
    return AGENT_OK;
}

void t2_engine_free_response(void *instance, agent_response *resp) {
    auto *inst = static_cast<Instance *>(instance);
    inst->path.clear();
    inst->text.clear();
    inst->parts[0] = agent_part{};
    inst->parts[1] = agent_part{};
    agent_response_reset(resp);
}

const char *t2_engine_last_error(void *instance) { return static_cast<Instance *>(instance)->error.c_str(); }

const agent_engine_v1 kTrellisEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "trellis",
    "0.2",
    AGENT_CAP_MESH_GEN,
    0, // one program runs at a time on the card; the host serializes run()
    t2_engine_create,
    t2_engine_destroy,
    t2_engine_run,
    t2_engine_free_response,
    t2_engine_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kTrellisEngine; }
