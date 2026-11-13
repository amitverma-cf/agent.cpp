// agent task: run one engine capability (image_gen, tts, asr, embed, mesh_gen, ...) once and print every
// output part. For smoke tests, scripts and comparing engines (or the mock engine) part by part.

#include "commands.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

namespace {

namespace fs = std::filesystem;

struct CapName {
    uint32_t bit;
    std::string_view name;
};
constexpr CapName kCaps[] = {
    {AGENT_CAP_TEXT_GEN, "text_gen"},
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

std::string mime_for(const fs::path &p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (e == ".png") return "image/png";
    if (e == ".jpg" || e == ".jpeg") return "image/jpeg";
    if (e == ".webp") return "image/webp";
    if (e == ".wav") return "audio/wav";
    if (e == ".mp3") return "audio/mpeg";
    if (e == ".flac") return "audio/flac";
    if (e == ".mp4") return "video/mp4";
    if (e == ".glb") return "model/gltf-binary";
    return "application/octet-stream";
}

const char *dtype_name(DType d) {
    switch (d) {
    case DType::Float32: return "f32";
    case DType::Float16: return "f16";
    case DType::Int8: return "i8";
    case DType::Int32: return "i32";
    case DType::Int64: return "i64";
    case DType::UInt8: return "u8";
    }
    return "?";
}

// An input tensor from --tensor name:dtype:shape=values, e.g. input_ids:i64:1,4=101,2023,2003,102
struct TensorBuf {
    std::string name;
    DType dtype = DType::Float32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

size_t dtype_size(DType d) {
    switch (d) {
    case DType::Float32:
    case DType::Int32: return 4;
    case DType::Float16: return 2;
    case DType::Int64: return 8;
    case DType::Int8:
    case DType::UInt8: return 1;
    }
    return 0;
}

std::vector<std::string> split(std::string_view s, char sep) {
    std::vector<std::string> out;
    size_t from = 0;
    while (from <= s.size()) {
        const size_t at = s.find(sep, from);
        out.emplace_back(s.substr(from, at == std::string_view::npos ? std::string_view::npos : at - from));
        if (at == std::string_view::npos) break;
        from = at + 1;
    }
    return out;
}

bool parse_tensor_arg(std::string_view arg, TensorBuf &out, std::string &err) {
    const size_t eq = arg.find('=');
    const auto head = split(arg.substr(0, eq), ':');
    if (head.size() != 3) {
        err = "expected name:dtype:shape=values";
        return false;
    }
    out.name = head[0];
    if (head[1] == "f32") out.dtype = DType::Float32;
    else if (head[1] == "i64") out.dtype = DType::Int64;
    else if (head[1] == "i32") out.dtype = DType::Int32;
    else if (head[1] == "u8") out.dtype = DType::UInt8;
    else {
        err = "dtype must be f32, i64, i32 or u8";
        return false;
    }
    size_t count = 1;
    for (const std::string &d : split(head[2], ',')) {
        const long long v = std::atoll(d.c_str());
        if (v <= 0) {
            err = "shape dimensions must be positive";
            return false;
        }
        out.shape.push_back(v);
        count *= static_cast<size_t>(v);
    }
    const auto values = eq == std::string_view::npos ? std::vector<std::string>{} : split(arg.substr(eq + 1), ',');
    if (values.size() != count) {
        err = "shape has " + std::to_string(count) + " elements but " + std::to_string(values.size()) + " values were given";
        return false;
    }
    out.bytes.resize(count * dtype_size(out.dtype));
    for (size_t i = 0; i < count; ++i) {
        uint8_t *dst = out.bytes.data() + i * dtype_size(out.dtype);
        switch (out.dtype) {
        case DType::Float32: {
            const float v = std::strtof(values[i].c_str(), nullptr);
            std::memcpy(dst, &v, 4);
            break;
        }
        case DType::Int64: {
            const int64_t v = std::strtoll(values[i].c_str(), nullptr, 10);
            std::memcpy(dst, &v, 8);
            break;
        }
        case DType::Int32: {
            const int32_t v = static_cast<int32_t>(std::strtol(values[i].c_str(), nullptr, 10));
            std::memcpy(dst, &v, 4);
            break;
        }
        default: *dst = static_cast<uint8_t>(std::strtol(values[i].c_str(), nullptr, 10)); break;
        }
    }
    return true;
}

bool g_all_values = false; // --all-values: print every element of a tensor, not the first four

void print_tensor(const TensorView &t) {
    std::string shape;
    for (size_t i = 0; i < t.shape.size(); ++i) shape += (i ? "," : "") + std::to_string(t.shape[i]);
    std::printf("tensor: %.*s %s [%s] %zu bytes", static_cast<int>(t.name.size()), t.name.data(), dtype_name(t.dtype), shape.c_str(),
                t.data.size());
    const size_t esize = dtype_size(t.dtype);
    if (esize && (t.dtype == DType::Float32 || t.dtype == DType::Int64 || t.dtype == DType::Int32) && t.data.size() >= esize) {
        const size_t n = g_all_values ? t.data.size() / esize : std::min<size_t>(4, t.data.size() / esize);
        std::printf("  first:");
        for (size_t i = 0; i < n; ++i) {
            if (t.dtype == DType::Float32) {
                float v = 0;
                std::memcpy(&v, t.data.data() + i * esize, esize);
                std::printf(" %.4g", static_cast<double>(v));
            } else if (t.dtype == DType::Int64) {
                int64_t v = 0;
                std::memcpy(&v, t.data.data() + i * esize, esize);
                std::printf(" %lld", static_cast<long long>(v));
            } else {
                int32_t v = 0;
                std::memcpy(&v, t.data.data() + i * esize, esize);
                std::printf(" %d", v);
            }
        }
    }
    std::printf("\n");
}

} // namespace

int cmd_task(const CommandLine &cl, Context &ctx) {
    g_all_values = cl.has("all-values");
    const std::string cap_name = cl.positionals[0];
    uint32_t task = 0;
    for (const auto &c : kCaps)
        if (c.name == cap_name) task = c.bit;
    if (!task) {
        std::string list;
        for (const auto &c : kCaps) list += (list.empty() ? "" : ", ") + std::string(c.name);
        return report_error(Error{ErrorCode::InvalidConfig, "unknown capability '" + cap_name + "' (one of: " + list + ")"});
    }

    // --model is optional: engines such as audiocpp take all their files from --engine-options.
    std::string model_path;
    if (cl.has("model")) {
        auto model = resolve_model(cl.get("model"), ctx.config.default_model, catalog_refs(ctx), path_exists);
        if (!model.ok) return report_error(model.error);
        model_path = model.value.path;
    }
    auto engine = load_engine_for(ctx, cl.get("engine"), model_path);
    if (!engine.ok) return report_error(engine.error);

    Config cfg;
    cfg.provider = AiProvider::Engine;
    cfg.engine = engine.value.name;
    cfg.engine_isolation = cli_engine_isolation(cl.has("inprocess"), engine.value.isolation);
    cfg.engine_sandbox.gpu = cli_engine_gpu(engine.value.backend);
    apply_sandbox_choice(cfg);
    cfg.engine_options = catalog_engine_options(ctx, model_path, std::string(cl.get("engine-options")));
    // audio.cpp takes one model file per role (tts, asr, vad, diarize): --model is the file of the role this task needs.
    if (engine.value.name == "audiocpp" && (cap_name == "tts" || cap_name == "asr" || cap_name == "vad" || cap_name == "diarize"))
        cfg.engine_options = with_role_model(cfg.engine_options, cap_name, model_path);
    cfg.model = model_path;
    cfg.auto_default_tools = false; // a task is not a chat turn: keep the tool instructions out of the prompt
    cfg.workspace_dir = workspace_dir(ctx, cl.get("workspace"));
    // The file the user named is theirs to give: let a sandboxed engine read that one file.
    if (cl.has("file")) {
        std::error_code ec;
        const auto abs = std::filesystem::absolute(std::string(cl.get("file")), ec);
        if (!ec) cfg.engine_sandbox.read_only_paths.push_back(abs.string()); // the file only, not its folder
    }

    // --tools FILE: a JSON array of {name, description, parameters}. They are offered to the model as tools; the call it makes is
    // printed (tool_call: name {args}) and nothing runs. For checking a model's function calling.
    static std::vector<std::string> tool_text; // Tool holds views: keep the strings alive
    if (cl.has("tools")) {
        std::ifstream in{std::string(cl.get("tools")), std::ios::binary};
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        simdjson::dom::array list;
        if (!in || parser.parse(body).get(doc) || doc.get(list)) {
            std::fprintf(stderr, "agent task: --tools expects a JSON array of tool definitions\n");
            return 2;
        }
        tool_text.reserve(list.size() * 3);
        for (simdjson::dom::element e : list) {
            simdjson::dom::object o;
            std::string_view name, description;
            if (e.get(o) || o["name"].get(name)) continue;
            if (o["description"].get(description)) description = "";
            simdjson::dom::element params;
            std::string schema = "{\"type\":\"object\",\"properties\":{}}";
            if (!o["parameters"].get(params)) schema = simdjson::minify(params);
            tool_text.emplace_back(name);
            tool_text.emplace_back(description);
            tool_text.emplace_back(schema);
            const size_t at = tool_text.size() - 3;
            cfg.tools.push_back(Tool{.name = tool_text[at],
                                     .description = tool_text[at + 1],
                                     .parameter_schema = tool_text[at + 2],
                                     .callback = [](std::string_view, void *) -> Result<std::string> { return ok(std::string()); }});
        }
    }
    cfg.context_window = ctx.config.context_window;
    cfg.max_tokens = ctx.config.max_tokens;
    cfg.temperature = ctx.config.temperature;
    auto session = init(cfg);
    if (!session.ok) return report_error(session.error);

    const std::string text(cl.get("text"));
    const std::string file(cl.get("file"));
    const std::string mime = file.empty() ? std::string() : mime_for(file);
    std::vector<MediaView> media;
    if (!file.empty()) media.push_back(MediaView{file, mime});

    // --tensor name:dtype:d1,d2=v1,v2,... (repeatable): raw named tensors for generic / onnx models.
    std::vector<TensorBuf> tensor_bufs;
    for (const Flag &f : cl.flags) {
        if (f.name != "tensor") continue;
        TensorBuf buf;
        if (std::string err; !parse_tensor_arg(f.value, buf, err))
            return report_error(Error{ErrorCode::InvalidConfig, "--tensor " + f.value + ": " + err});
        tensor_bufs.push_back(std::move(buf));
    }
    std::vector<TensorView> tensor_views;
    for (const TensorBuf &b : tensor_bufs) tensor_views.push_back(TensorView{b.name, b.dtype, b.shape, b.bytes});

    MessageView user;
    user.role = "user";
    user.content = text;
    user.media = media;
    user.tensors = tensor_views;
    const MessageView messages[] = {user};

    InferRequest req;
    req.messages = messages;
    req.task = task;
    req.params_json = cl.get("params");
    if (cl.has("max-tokens")) req.max_tokens = std::atoi(std::string(cl.get("max-tokens")).c_str());
    req.on_progress = [](uint64_t step, uint64_t total, void *) {
        if (total && (step == total || step % std::max<uint64_t>(1, total / 10) == 0))
            std::fprintf(stderr, "[progress] %llu/%llu\n", static_cast<unsigned long long>(step), static_cast<unsigned long long>(total));
    };

    if (task == AGENT_CAP_RERANK) {
        // rerank is query (--text) plus documents (--doc, repeatable); one score per document.
        std::vector<std::string_view> docs;
        for (const Flag &f : cl.flags)
            if (f.name == "doc") docs.push_back(f.value);
        auto scores = rerank(session.value, text, docs);
        if (!scores.ok) return report_error(scores.error);
        std::printf("task: rerank  engine: %s\n", engine.value.name.c_str());
        for (size_t i = 0; i < scores.value.size(); ++i)
            std::printf("score[%zu]: %.4f  %.*s\n", i, static_cast<double>(scores.value[i]), static_cast<int>(docs[i].size()),
                        docs[i].data());
        return 0;
    }

    auto res = run_task(session.value, task, req);
    if (!res.ok) return report_error(res.error);
    const InferResponse &r = res.value;

    std::printf("task: %s  engine: %s\n", cap_name.c_str(), engine.value.name.c_str());
    if (!r.message.content.empty()) std::printf("text: %.*s\n", static_cast<int>(r.message.content.size()), r.message.content.data());
    for (const MediaView &m : r.message.media) {
        std::error_code ec;
        const auto size = fs::file_size(std::string(m.path), ec);
        std::printf("media: %.*s %.*s %llu bytes\n", static_cast<int>(m.mime.size()), m.mime.data(), static_cast<int>(m.path.size()),
                    m.path.data(), static_cast<unsigned long long>(ec ? 0 : size));
    }
    // output_tensors mirrors the tensors of the message; print one copy.
    for (const TensorView &t : (r.output_tensors.empty() ? r.message.tensors : r.output_tensors)) print_tensor(t);
    for (const ToolCallView &c : r.message.tool_calls)
        std::printf("tool_call: %.*s %.*s\n", static_cast<int>(c.name.size()), c.name.data(), static_cast<int>(c.arguments.size()),
                    c.arguments.data());
    std::printf("usage: prompt=%d completion=%d\n", r.usage.prompt_tokens, r.usage.completion_tokens);
    if (const std::string footer = format_turn_footer(r.usage); !footer.empty()) std::puts(("stats: " + footer).c_str());
    return 0;
}

} // namespace agent::cli
