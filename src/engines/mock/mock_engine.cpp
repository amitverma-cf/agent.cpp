// Reference engine plugin used by the test suite. Shows the minimum an engine implements.
//   - echoes the last user text back as "echo: <text>"
//   - a user message starting with "call:<tool>" produces a tool call to <tool> with {};
//     "call:<tool> <json>" passes <json> as the arguments
//   - after tool results it replies "tool done", plus "tool media: <mime> <path>" for every FILE part
//     on the trailing tool messages
//   - "show:system" replies "<n> system: <text>|<text>" with the system messages it received
//   - declares VISION_IN / AUDIO_IN so tool media reaches it (it only lists the files)
//   - "reject-media" in engine_options' JSON: any request with a FILE part fails (like a llama model
//     without a projector)
//   - streams the reply as one text chunk when on_chunk is set
//   - a model path of "fail" makes create() fail
//   - media parts on the last user message are acknowledged as "media: <mime> <path>" lines
//   - "draw" writes a small file to output_dir and returns it as a FILE part
//   - "slow" reports 100 progress steps and stops early when *cancel is set
//   - debug triggers for the engine host tests (they take the whole process down, so only send
//     them to an out-of-process engine): "crash:segv" raises SIGSEGV, "crash:abort" calls abort(),
//     "crash:exit" calls exit(3) (like sherpa-onnx on a corrupt model), "hang" never returns
//     and ignores *cancel
//   - TENSOR parts on the last user message come back unchanged as output TENSOR parts
//   - "stream:N" streams N text chunks of one character each
//   - sandbox probes (tests/unit/test_sandbox.cpp): "read:<path>" replies "read ok: <n> bytes" or
//     "read denied: <strerror>"; "write:<path>" replies "write ok" or "write denied: <strerror>";
//     "net" tries an AF_INET socket and replies "net ok" or "net denied: <strerror>"
//   - "sleep:N" takes N ms (stops early with AGENT_ERR_CANCELLED when *cancel is set)
//   - "barrier:<tag>:N" returns only once N runs with the same tag are inside run() at the same time
//     (fails after 5 s): proof that requests overlap
//
// For non-chat tasks the texts "mock-error", "mock-empty" and "mock-slow" simulate a failing, an empty and a slow cancellable call.
//
// A third build, agent-engine-mock-all (name "mock_all", AGENT_MOCK_ALL), declares every capability and
// answers each non-chat task with deterministic stand-in output (tiny WAV / PNG / GLB files, f32 embedding,
// score and probs tensors, ASR/OCR text, VAD and diarization segments), so graphs, the CLI and the engine
// host can be tested without model files.
//
// Built twice: agent-engine-mock (flags 0, run() serialized by the host) and, with
// AGENT_MOCK_CONCURRENT, agent-engine-mock-concurrent (name "mock_concurrent",
// AGENT_ENGINE_FLAG_CONCURRENT). run() keeps its output per thread, so both are thread-safe.

#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

// The reply to a sandbox probe trigger, or empty when last_user is not one.
std::string sandbox_probe(std::string_view last_user) {
    if (last_user.starts_with("read:")) {
        const std::string path(last_user.substr(5));
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) return std::string("read denied: ") + std::strerror(errno);
        size_t n = 0;
        char buf[4096];
        for (size_t got; (got = std::fread(buf, 1, sizeof(buf), f)) > 0;) n += got;
        std::fclose(f);
        return "read ok: " + std::to_string(n) + " bytes";
    }
    if (last_user.starts_with("write:")) {
        const std::string path(last_user.substr(6));
        FILE *f = std::fopen(path.c_str(), "wb");
        if (!f) return std::string("write denied: ") + std::strerror(errno);
        std::fputs("x", f);
        std::fclose(f);
        return "write ok";
    }
    if (last_user == "net") {
#if defined(_WIN32)
        return "net unsupported";
#else
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return std::string("net denied: ") + std::strerror(errno);
        ::close(fd);
        return "net ok";
#endif
    }
    return {};
}

struct MockInstance {
    std::string model;
};

// Everything a run() returns, owned by the calling thread until its next run() / free_response().
struct MockOutput {
    std::string reply;
    std::string tool_name;
    std::string tool_args;
    std::string out_path;
    agent_part part{};
    agent_part parts[2]{};
    agent_tool_call call{};
    std::string error;
    // Echoed tensors: copies of the inputs, owned until the next run().
    std::vector<std::string> tensor_names;
    std::vector<std::vector<int64_t>> tensor_shapes;
    std::vector<std::vector<uint8_t>> tensor_data;
    std::vector<uint32_t> tensor_dtypes;
    std::vector<agent_part> out_parts;
    // Multi-task (AGENT_MOCK_ALL) outputs.
    std::vector<agent_part> multi;
    std::vector<std::string> strings;
    std::vector<std::vector<float>> fbufs;
    std::vector<std::vector<int64_t>> shapes;
};

MockOutput &thread_output() {
    thread_local MockOutput out;
    return out;
}

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

// "barrier:<tag>:N": waits until N callers with tag are inside at once.
bool barrier(std::string_view spec) {
    static std::mutex mu;
    static std::condition_variable cv;
    static std::map<std::string, int> arrived;
    const size_t colon = spec.rfind(':');
    if (colon == std::string_view::npos) return false;
    const std::string tag(spec.substr(0, colon));
    const int n = std::atoi(std::string(spec.substr(colon + 1)).c_str());
    std::unique_lock<std::mutex> lk(mu);
    const int me = ++arrived[tag];
    cv.notify_all();
    return me >= n || cv.wait_for(lk, std::chrono::seconds(5), [&] { return arrived[tag] >= n; });
}

int32_t mock_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    std::string_view cfg = sv(config_json);
    if (cfg.find("\"model\":\"fail\"") != std::string_view::npos) {
        std::snprintf(err, err_cap, "mock: refusing model 'fail'");
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    auto *inst = new MockInstance();
    inst->model = std::string(cfg);
    *out_instance = inst;
    return AGENT_OK;
}

void mock_destroy(void *instance) { delete static_cast<MockInstance *>(instance); }

#if defined(AGENT_MOCK_ALL)
// Deterministic stand-ins for every non-chat task so graphs, the CLI and the host can be exercised
// without model files. The output shapes mirror what the real engines return (llama, sdcpp, whisper,
// parakeet), recorded by running them on real models; see docs/api/engines.md.
void put_le(std::string &b, uint32_t v, int n) {
    for (int i = 0; i < n; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

bool write_file(const std::string &path, const std::string &bytes) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return true;
}

std::string tiny_wav() { // 100 ms, 24 kHz (the Kokoro rate), mono, 16-bit, 440 Hz at low volume
    const uint32_t rate = 24000, n = 2400;
    std::string d;
    for (uint32_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / rate;
        const auto v = static_cast<int16_t>(2000.0 * std::sin(6.283185307 * 440.0 * t));
        put_le(d, static_cast<uint16_t>(v), 2);
    }
    std::string w = "RIFF";
    put_le(w, 36 + n * 2, 4);
    w += "WAVEfmt ";
    put_le(w, 16, 4);
    put_le(w, 1, 2);
    put_le(w, 1, 2);
    put_le(w, rate, 4);
    put_le(w, rate * 2, 4);
    put_le(w, 2, 2);
    put_le(w, 16, 2);
    w += "data";
    put_le(w, n * 2, 4);
    return w + d;
}

std::string tiny_png() { // a valid 1x1 PNG
    static const unsigned char kPng[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
                                         0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
                                         0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x08,
                                         0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xdd, 0x8d,
                                         0xb0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
    return std::string(reinterpret_cast<const char *>(kPng), sizeof(kPng));
}

std::string tiny_glb() { // header + an empty-scene JSON chunk
    std::string json = R"({"asset":{"version":"2.0"},"scenes":[{}],"scene":0})";
    while (json.size() % 4) json.push_back(' ');
    std::string g = "glTF";
    put_le(g, 2, 4);
    put_le(g, static_cast<uint32_t>(12 + 8 + json.size()), 4);
    put_le(g, static_cast<uint32_t>(json.size()), 4);
    g += "JSON";
    return g + json;
}

// Every string and buffer a response part points at lives here until the thread's next run().
agent_part &new_part(MockOutput *o) {
    o->multi.emplace_back();
    return o->multi.back();
}

void add_text(MockOutput *o, std::string text) {
    o->strings.push_back(std::move(text));
    agent_part &p = new_part(o);
    p.kind = AGENT_PART_TEXT;
    p.text = agent_str{o->strings.back().data(), o->strings.back().size()};
}

void add_file(MockOutput *o, const std::string &path, const char *mime) {
    o->strings.push_back(path);
    agent_part &p = new_part(o);
    p.kind = AGENT_PART_FILE;
    p.path = agent_str{o->strings.back().data(), o->strings.back().size()};
    p.mime = agent_str{mime, std::strlen(mime)};
}

void add_tensor(MockOutput *o, const char *name, std::vector<float> values, std::vector<int64_t> shape) {
    o->strings.push_back(name);
    o->fbufs.push_back(std::move(values));
    o->shapes.push_back(std::move(shape));
    agent_part &p = new_part(o);
    p.kind = AGENT_PART_TENSOR;
    p.name = agent_str{o->strings.back().data(), o->strings.back().size()};
    p.dtype = AGENT_DTYPE_F32;
    p.shape = o->shapes.back().data();
    p.ndim = o->shapes.back().size();
    p.data = o->fbufs.back().data();
    p.size = o->fbufs.back().size() * sizeof(float);
}

std::vector<std::string> split_words(std::string_view s) {
    std::vector<std::string> w;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\n') {
            if (!cur.empty()) w.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}

std::string num(double v, int digits) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.*f", digits, v);
    return buf;
}

// The option labels of a DECIDE request, from params_json: choice options, a score scale ("1".."5") or
// "Yes"/"No". Also returns the mode name.
std::vector<std::string> decide_options(std::string_view pj, std::string *mode) {
    std::vector<std::string> out;
    if (const size_t at = pj.find("\"options\""); at != std::string_view::npos) {
        *mode = "choice";
        const size_t open = pj.find('[', at), close = pj.find(']', at);
        if (open != std::string_view::npos && close != std::string_view::npos) {
            for (size_t i = open + 1; i < close;) {
                const size_t q1 = pj.find('"', i);
                if (q1 == std::string_view::npos || q1 >= close) break;
                const size_t q2 = pj.find('"', q1 + 1);
                if (q2 == std::string_view::npos || q2 > close) break;
                out.emplace_back(pj.substr(q1 + 1, q2 - q1 - 1));
                i = q2 + 1;
            }
        }
    } else if (const size_t sc = pj.find("\"scale\""); sc != std::string_view::npos) {
        *mode = "score";
        const size_t open = pj.find('[', sc), comma = pj.find(',', sc);
        if (open != std::string_view::npos && comma != std::string_view::npos) {
            const int lo = std::atoi(std::string(pj.substr(open + 1, comma - open - 1)).c_str());
            const int hi = std::atoi(std::string(pj.substr(comma + 1, 8)).c_str());
            for (int v = lo; v <= hi && v - lo < 101; ++v) out.push_back(std::to_string(v));
        }
    } else {
        *mode = "yesno";
        out = {"Yes", "No"};
    }
    if (out.size() < 2) out = {"Yes", "No"};
    return out;
}

int32_t run_other_task(const agent_request *req, agent_response *resp, MockOutput *o) {
    std::vector<std::string_view> texts, all_texts, per_message;
    bool any_media = false; // texts: the last user message; all_texts: every user message (rerank: query, then documents)
    std::string first_file;
    for (size_t i = 0; i < req->n_messages; ++i) {
        if (sv(req->messages[i].role) != "user") continue;
        texts.clear();
        std::string_view msg_text;
        for (size_t p = 0; p < req->messages[i].n_parts; ++p) {
            const agent_part &part = req->messages[i].parts[p];
            if (part.kind == AGENT_PART_TEXT) {
                texts.push_back(sv(part.text));
                all_texts.push_back(sv(part.text));
                if (msg_text.empty()) msg_text = sv(part.text);
            } else if (part.kind == AGENT_PART_FILE) {
                any_media = true;
                if (first_file.empty()) first_file = std::string(sv(part.path));
            }
        }
        per_message.push_back(msg_text);
    }
    const std::string out_dir(sv(req->output_dir));
    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    o->multi.clear();
    o->strings.clear();
    o->fbufs.clear();
    o->shapes.clear();
    // Parts and buffers are referenced by pointer: reserve so push_back never moves them.
    o->multi.reserve(256);
    o->strings.reserve(256);
    o->fbufs.reserve(256);
    o->shapes.reserve(256);
    for (uint64_t step = 1; step <= 4 && req->on_progress; ++step) req->on_progress(req->user_data, step, 4);
    auto fail_write = [&] {
        o->error = "mock: cannot write to output_dir";
        return AGENT_ERR_UNSUPPORTED;
    };
    auto done = [&] {
        resp->message.parts = o->multi.data();
        resp->message.n_parts = o->multi.size();
        resp->usage = agent_usage{0, 0};
        return AGENT_OK;
    };

    // Failure scenarios for any non-chat task: a text part that is exactly "mock-error" fails the call (a decode error, like a model
    // that cannot process its input), "mock-empty" succeeds with no output parts, "mock-slow" reports progress while honouring cancel.
    for (std::string_view t : texts) {
        if (t == "mock-error") {
            o->error = "mock: simulated engine failure";
            return AGENT_ERR_DECODE_FAILED;
        }
        if (t == "mock-empty") return done();
        if (t == "mock-slow") {
            for (uint64_t step = 1; step <= 100; ++step) {
                if (req->cancel && *req->cancel) {
                    o->error = "mock: cancelled";
                    return AGENT_ERR_CANCELLED;
                }
                if (req->on_progress) req->on_progress(req->user_data, step, 100);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            add_text(o, "slow task finished");
            return done();
        }
    }
    switch (req->task) {
    case AGENT_CAP_ASR:
        if (first_file.empty()) {
            o->error = "mock: ASR needs an audio part";
            return AGENT_ERR_INVALID_CONFIG;
        }
        add_text(o, "mock transcript");
        return done();
    case AGENT_CAP_OCR: add_text(o, "mock ocr text"); return done();
    case AGENT_CAP_TTS: {
        if (texts.empty()) {
            o->error = "mock: TTS needs text";
            return AGENT_ERR_INVALID_CONFIG;
        }
        if (req->on_chunk) {
            std::vector<float> pcm(2400, 0.0f);
            req->on_chunk(req->user_data, AGENT_CHUNK_AUDIO, pcm.data(), pcm.size() * sizeof(float));
        }
        const std::string path = out_dir + "/mock-tts-0.wav";
        if (!write_file(path, tiny_wav())) return fail_write();
        add_text(o, "Generated 0.10 s of speech (24000 Hz).");
        add_file(o, path, "audio/wav");
        return done();
    }
    case AGENT_CAP_VAD:
        add_text(o, R"([{"start":0.500,"end":1.500}])");
        add_tensor(o, "segments", {0.5f, 1.5f}, {1, 2});
        return done();
    case AGENT_CAP_DIARIZE:
        add_text(o, R"([{"start":0.00,"end":1.00,"speaker":0},{"start":1.00,"end":2.00,"speaker":1}])");
        add_tensor(o, "segments", {0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f}, {2, 3});
        return done();
    case AGENT_CAP_IMAGE_GEN:
    case AGENT_CAP_IMAGE_EDIT: {
        const std::string path = out_dir + "/mock-0.png";
        if (!write_file(path, tiny_png())) return fail_write();
        add_file(o, path, "image/png");
        add_text(o, "generated image: " + path); // sdcpp returns the FILE part, then this text part
        return done();
    }
    case AGENT_CAP_VIDEO_GEN: {
        const std::string path = out_dir + "/mock-0.mp4";
        if (!write_file(path, std::string("\0\0\0\x14"
                                          "ftypisom\0\0\0\0isom",
                                          20)))
            return fail_write();
        add_file(o, path, "video/mp4");
        return done();
    }
    case AGENT_CAP_MESH_GEN: {
        const std::string path = out_dir + "/mock-0.glb";
        if (!write_file(path, tiny_glb())) return fail_write();
        add_file(o, path, "model/gltf-binary");
        return done();
    }
    case AGENT_CAP_EMBED: {
        // One L2-normalised "embedding" tensor f32 [dim] per TEXT part, in order (llama: dim = n_embd).
        // Across every user message, like the llama engine (embed() sends one message per text).
        // With media in the request: one embedding per message (llama: run_embed_media).
        const std::vector<std::string_view> &rows = any_media ? per_message : all_texts;
        for (size_t t = 0; t < std::max<size_t>(1, rows.size()); ++t) {
            std::vector<float> v(8, 0.0f);
            const std::string_view s = t < rows.size() ? rows[t] : std::string_view();
            for (size_t i = 0; i < s.size(); ++i) v[i % 8] += static_cast<float>(static_cast<unsigned char>(s[i])) / 255.0f;
            float norm = 0;
            for (float x : v) norm += x * x;
            norm = norm > 0 ? std::sqrt(norm) : 1.0f;
            for (float &x : v) x /= norm;
            add_tensor(o, "embedding", std::move(v), {8});
        }
        return done();
    }
    case AGENT_CAP_RERANK: {
        // First TEXT = query, later TEXT parts = documents; one "scores" tensor f32 [n_docs].
        if (all_texts.size() < 2) {
            o->error = "mock: rerank needs a query and at least one document text part";
            return AGENT_ERR_INVALID_CONFIG;
        }
        const auto q = split_words(all_texts[0]);
        std::vector<float> scores;
        for (size_t d = 1; d < all_texts.size(); ++d) {
            const auto dw = split_words(all_texts[d]);
            size_t hit = 0;
            for (const auto &a : q)
                for (const auto &b : dw)
                    if (a == b) {
                        ++hit;
                        break;
                    }
            scores.push_back(q.empty() ? 0.0f : static_cast<float>(hit) / static_cast<float>(q.size()));
        }
        add_tensor(o, "scores", scores, {static_cast<int64_t>(scores.size())});
        return done();
    }
    case AGENT_CAP_DECIDE:
    case AGENT_CAP_CLASSIFY: {
        // Mirrors llama: a TEXT JSON summary plus one f32 "probs" tensor [n_options].
        std::string mode;
        const std::vector<std::string> options = decide_options(sv(req->params_json), &mode);
        std::vector<float> probs;
        float sum = 0;
        for (size_t i = 0; i < options.size(); ++i) {
            const float w = 1.0f / static_cast<float>(i + 1);
            probs.push_back(w);
            sum += w;
        }
        for (float &x : probs) x /= sum;
        std::string j = "{\"mode\":\"" + mode + "\",\"choice\":\"" + options[0] +
                        "\",\"index\":0,\"confidence\":" + num(static_cast<double>(probs[0]), 6);
        if (mode == "yesno") j += ",\"yes\":" + num(static_cast<double>(probs[0]), 6);
        j += ",\"options\":[";
        for (size_t i = 0; i < options.size(); ++i) j += (i ? ",\"" : "\"") + options[i] + "\"";
        j += "],\"probs\":[";
        for (size_t i = 0; i < probs.size(); ++i) j += (i ? "," : "") + num(static_cast<double>(probs[i]), 6);
        j += "]}";
        add_text(o, j);
        add_tensor(o, "probs", probs, {static_cast<int64_t>(probs.size())});
        return done();
    }
    default: o->error = "mock: unsupported task"; return AGENT_ERR_UNSUPPORTED;
    }
}
#endif

int32_t mock_run(void *instance, const agent_request *req, agent_response *resp) {
    MockOutput *inst = &thread_output();
    inst->error.clear();
#if defined(AGENT_MOCK_ALL)
    if (req->task != AGENT_CAP_TEXT_GEN && req->task != AGENT_CAP_TOOL_CALL && req->task != AGENT_CAP_GENERIC)
        return run_other_task(req, resp, inst);
#else
    if (req->task != AGENT_CAP_TEXT_GEN) {
        inst->error = "mock: only TEXT_GEN is supported";
        return AGENT_ERR_UNSUPPORTED;
    }
#endif
    if (static_cast<MockInstance *>(instance)->model.find("reject-media") != std::string::npos) {
        for (size_t i = 0; i < req->n_messages; ++i)
            for (size_t p = 0; p < req->messages[i].n_parts; ++p)
                if (req->messages[i].parts[p].kind == AGENT_PART_FILE) {
                    inst->error = "mock: media input is disabled (reject-media)";
                    return AGENT_ERR_UNSUPPORTED;
                }
    }

    std::string_view last_user;
    std::string media_lines;
    const agent_message *last_user_msg = nullptr;
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        if (sv(m.role) != "user") continue;
        last_user_msg = &m;
        media_lines.clear();
        for (size_t p = 0; p < m.n_parts; ++p) {
            if (m.parts[p].kind == AGENT_PART_TEXT) last_user = sv(m.parts[p].text);
            else if (m.parts[p].kind == AGENT_PART_FILE)
                media_lines += "\nmedia: " + std::string(sv(m.parts[p].mime)) + " " + std::string(sv(m.parts[p].path));
        }
    }

    if (last_user == "crash:segv") std::raise(SIGSEGV);
    if (last_user == "crash:abort") std::abort();
    if (last_user == "crash:exit") std::exit(3);
    if (last_user == "hang")
        for (;;) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (last_user.starts_with("sleep:")) {
        const auto until =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(std::atoi(std::string(last_user.substr(6)).c_str()));
        while (std::chrono::steady_clock::now() < until) {
            if (req->cancel && *req->cancel) {
                inst->error = "mock: cancelled";
                return AGENT_ERR_CANCELLED;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    if (last_user.starts_with("barrier:") && !barrier(last_user.substr(8))) {
        inst->error = "mock: barrier timed out (runs did not overlap)";
        return AGENT_ERR_DECODE_FAILED;
    }
    if (last_user.starts_with("stream:") && req->on_chunk) {
        const int n = std::atoi(std::string(last_user.substr(7)).c_str());
        for (int i = 0; i < n; ++i) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, "x", 1);
    }

    if (last_user == "slow") {
        for (uint64_t step = 1; step <= 100; ++step) {
            if (req->cancel && *req->cancel) {
                inst->error = "mock: cancelled";
                return AGENT_ERR_CANCELLED;
            }
            if (req->on_progress) req->on_progress(req->user_data, step, 100);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    agent_response_reset(resp);
    resp->message.role = agent_str{"assistant", 9};
    resp->usage = agent_usage{static_cast<int32_t>(last_user.size()), 1};

    const bool after_tool = req->n_messages > 0 && sv(req->messages[req->n_messages - 1].role) == "tool";
    if (!after_tool && last_user.starts_with("call:")) {
        // "call:<tool>" or "call:<tool> <json arguments>"
        std::string_view spec = last_user.substr(5);
        const size_t space = spec.find(' ');
        inst->tool_name = std::string(spec.substr(0, space));
        inst->tool_args = space == std::string_view::npos ? std::string("{}") : std::string(spec.substr(space + 1));
        inst->call = agent_tool_call{agent_str{"call_0", 6}, agent_str{inst->tool_name.data(), inst->tool_name.size()},
                                     agent_str{inst->tool_args.data(), inst->tool_args.size()}};
        resp->message.tool_calls = &inst->call;
        resp->message.n_tool_calls = 1;
        return AGENT_OK;
    }

    if (after_tool) {
        // "tool done", plus one "tool media: <mime> <path>" line per FILE part of the trailing tool messages.
        inst->reply = "tool done";
        size_t first_tool = req->n_messages;
        while (first_tool > 0 && sv(req->messages[first_tool - 1].role) == "tool") --first_tool;
        for (size_t i = first_tool; i < req->n_messages; ++i)
            for (size_t p = 0; p < req->messages[i].n_parts; ++p)
                if (req->messages[i].parts[p].kind == AGENT_PART_FILE)
                    inst->reply += "\ntool media: " + std::string(sv(req->messages[i].parts[p].mime)) + " " +
                                   std::string(sv(req->messages[i].parts[p].path));
    } else if (last_user == "show:tools") {
        // The tools_json the host sent with this request: lets tests see which tools the model is told about.
        inst->reply = std::string(sv(req->tools_json));
    } else if (last_user == "show:system") {
        // The system messages as the engine received them: "<count> system: <text>|<text>..."
        size_t n = 0;
        std::string text;
        for (size_t i = 0; i < req->n_messages; ++i) {
            if (sv(req->messages[i].role) != "system") continue;
            if (n++) text += '|';
            for (size_t p = 0; p < req->messages[i].n_parts; ++p)
                if (req->messages[i].parts[p].kind == AGENT_PART_TEXT) text += sv(req->messages[i].parts[p].text);
        }
        inst->reply = std::to_string(n) + " system: " + text;
    } else {
        inst->reply = "echo: " + std::string(last_user) + media_lines;
    }
    if (std::string probe = sandbox_probe(last_user); !after_tool && !probe.empty()) inst->reply = std::move(probe);
    if (req->on_chunk) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, inst->reply.data(), inst->reply.size());
    inst->part = agent_part{};
    inst->part.kind = AGENT_PART_TEXT;
    inst->part.text = agent_str{inst->reply.data(), inst->reply.size()};
    resp->message.parts = &inst->part;
    resp->message.n_parts = 1;

    if (last_user == "draw") {
        inst->out_path = std::string(sv(req->output_dir)) + "/mock_drawing.txt";
        if (FILE *f = std::fopen(inst->out_path.c_str(), "wb")) {
            std::fputs("mock drawing", f);
            std::fclose(f);
        }
        // parts[0] = text, parts[1] = file; both live in the instance until the next run().
        inst->parts[0] = inst->part;
        inst->parts[1] = agent_part{};
        inst->parts[1].kind = AGENT_PART_FILE;
        inst->parts[1].path = agent_str{inst->out_path.data(), inst->out_path.size()};
        inst->parts[1].mime = agent_str{"text/plain", 10};
        resp->message.parts = inst->parts;
        resp->message.n_parts = 2;
    }

    inst->tensor_names.clear();
    inst->tensor_shapes.clear();
    inst->tensor_data.clear();
    inst->tensor_dtypes.clear();
    for (size_t p = 0; last_user_msg && p < last_user_msg->n_parts; ++p) {
        const agent_part &in = last_user_msg->parts[p];
        if (in.kind != AGENT_PART_TENSOR) continue;
        inst->tensor_names.emplace_back(sv(in.name));
        inst->tensor_shapes.emplace_back(in.shape, in.shape + in.ndim);
        const auto *bytes = static_cast<const uint8_t *>(in.data);
        inst->tensor_data.emplace_back(bytes, bytes + (bytes ? in.size : 0));
        inst->tensor_dtypes.push_back(in.dtype);
    }
    if (!inst->tensor_names.empty()) {
        inst->out_parts.assign(resp->message.parts, resp->message.parts + resp->message.n_parts);
        for (size_t t = 0; t < inst->tensor_names.size(); ++t) {
            agent_part out{};
            out.kind = AGENT_PART_TENSOR;
            out.name = agent_str{inst->tensor_names[t].data(), inst->tensor_names[t].size()};
            out.dtype = inst->tensor_dtypes[t];
            out.shape = inst->tensor_shapes[t].data();
            out.ndim = inst->tensor_shapes[t].size();
            out.data = inst->tensor_data[t].data();
            out.size = inst->tensor_data[t].size();
            inst->out_parts.push_back(out);
        }
        resp->message.parts = inst->out_parts.data();
        resp->message.n_parts = inst->out_parts.size();
    }
    return AGENT_OK;
}

void mock_free_response(void *, agent_response *resp) { agent_response_reset(resp); }

const char *mock_last_error(void *) { return thread_output().error.c_str(); }

#if defined(AGENT_MOCK_ALL)
constexpr const char *kName = "mock_all";
constexpr uint32_t kFlags = AGENT_ENGINE_FLAG_CONCURRENT;
constexpr uint64_t kCaps = AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL | AGENT_CAP_VISION_IN | AGENT_CAP_AUDIO_IN | AGENT_CAP_VIDEO_IN |
                           AGENT_CAP_OCR | AGENT_CAP_ASR | AGENT_CAP_TTS | AGENT_CAP_VAD | AGENT_CAP_DIARIZE | AGENT_CAP_IMAGE_GEN |
                           AGENT_CAP_IMAGE_EDIT | AGENT_CAP_VIDEO_GEN | AGENT_CAP_MESH_GEN | AGENT_CAP_EMBED | AGENT_CAP_RERANK |
                           AGENT_CAP_DECIDE | AGENT_CAP_CLASSIFY | AGENT_CAP_GENERIC;
#elif defined(AGENT_MOCK_CONCURRENT)
constexpr const char *kName = "mock_concurrent";
constexpr uint32_t kFlags = AGENT_ENGINE_FLAG_CONCURRENT;
#else
constexpr const char *kName = "mock";
constexpr uint32_t kFlags = 0;
#endif
#if !defined(AGENT_MOCK_ALL)
constexpr uint64_t kCaps = AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL | AGENT_CAP_VISION_IN | AGENT_CAP_AUDIO_IN;
#endif

const agent_engine_v1 kMockEngine = {
    AGENT_ENGINE_ABI_VERSION, sizeof(agent_engine_v1), kName,   "1.0",   kCaps, kFlags, mock_create, mock_destroy, mock_run,
    mock_free_response,       mock_last_error,         nullptr, nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kMockEngine; }
