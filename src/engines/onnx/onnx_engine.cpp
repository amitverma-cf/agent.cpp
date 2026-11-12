// agent-engine-onnx: generic ONNX Runtime graph runner (AGENT_CAP_GENERIC).
//   - create(): config "model" = .onnx path; options "threads" (intra-op), "execution_provider"
//     ("cpu" | "cuda"), "device_id" (cuda), "graph_optimization" ("disable" | "basic" | "extended" | "all")
//   - run(): every model input is fed from the last AGENT_PART_TENSOR part with the same name across
//     the request messages; every model output comes back as an AGENT_PART_TENSOR part.
// Uses the ORT C API only (the C++ API throws). OrtSession::Run is thread-safe and every call keeps
// its own buffers, so the engine sets AGENT_ENGINE_FLAG_CONCURRENT.

#include <agent-cpp/engine_abi.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <onnxruntime_c_api.h>
#include <simdjson.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

// Errors are per thread: with concurrent run() calls, last_error() is read on the thread that failed.
thread_local std::string t_error;

int32_t set_error(int32_t code, std::string msg) {
    t_error = std::move(msg);
    return code;
}

const OrtApi *ort() {
    static const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return api;
}

// One OrtEnv per process, created on first use and kept until exit (engines are never unloaded).
OrtEnv *ort_env(std::string &err) {
    static std::once_flag once;
    static OrtEnv *env = nullptr;
    static std::string env_err;
    std::call_once(once, [] {
        if (OrtStatus *s = ort()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "agent-engine-onnx", &env)) {
            env_err = ort()->GetErrorMessage(s);
            ort()->ReleaseStatus(s);
            env = nullptr;
        }
    });
    if (!env) err = env_err;
    return env;
}

// Returns the ORT error message and releases the status; empty string when s is null.
std::string take(OrtStatus *s) {
    if (!s) return {};
    std::string msg = ort()->GetErrorMessage(s);
    ort()->ReleaseStatus(s);
    return msg.empty() ? std::string("unknown ONNX Runtime error") : msg;
}

bool to_agent_dtype(ONNXTensorElementDataType t, uint32_t &out, size_t &bytes) {
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: out = AGENT_DTYPE_F32, bytes = 4; return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: out = AGENT_DTYPE_F16, bytes = 2; return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: out = AGENT_DTYPE_I8, bytes = 1; return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: out = AGENT_DTYPE_I32, bytes = 4; return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: out = AGENT_DTYPE_I64, bytes = 8; return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: out = AGENT_DTYPE_U8, bytes = 1; return true;
    default: return false;
    }
}

const char *dtype_name(uint32_t d) {
    switch (d) {
    case AGENT_DTYPE_F32: return "f32";
    case AGENT_DTYPE_F16: return "f16";
    case AGENT_DTYPE_I8: return "i8";
    case AGENT_DTYPE_I32: return "i32";
    case AGENT_DTYPE_I64: return "i64";
    case AGENT_DTYPE_U8: return "u8";
    default: return "?";
    }
}

struct InputInfo {
    std::string name;
    uint32_t dtype = AGENT_DTYPE_F32;
    size_t elem_bytes = 4;
    ONNXTensorElementDataType ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    std::vector<int64_t> shape; // -1 for dynamic dimensions
};

// Everything one run() returns; owned by the instance until free_response().
struct CallResult {
    std::vector<OrtValue *> values;
    std::vector<std::vector<int64_t>> shapes; // per output
    std::vector<agent_part> parts;

    ~CallResult() {
        for (OrtValue *v : values)
            if (v) ort()->ReleaseValue(v);
    }
};

struct OnnxInstance {
    OrtSession *session = nullptr;
    OrtMemoryInfo *cpu_mem = nullptr;
    std::vector<InputInfo> inputs;
    std::vector<std::string> outputs;
    std::vector<const char *> input_names, output_names; // c_str views of the above, cached for Run()

    std::mutex live_mutex;
    std::vector<std::unique_ptr<CallResult>> live; // results handed out and not yet freed

    ~OnnxInstance() {
        live.clear();
        if (session) ort()->ReleaseSession(session);
        if (cpu_mem) ort()->ReleaseMemoryInfo(cpu_mem);
    }
};

std::string shape_str(const std::vector<int64_t> &shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ',';
        s += shape[i] < 0 ? std::string("?") : std::to_string(shape[i]);
    }
    return s + "]";
}

// Reads input/output names and input types once, so run() does no metadata queries.
std::string cache_io(OnnxInstance &inst) {
    const OrtApi *api = ort();
    OrtAllocator *alloc = nullptr;
    if (std::string e = take(api->GetAllocatorWithDefaultOptions(&alloc)); !e.empty()) return e;

    size_t n_in = 0, n_out = 0;
    if (std::string e = take(api->SessionGetInputCount(inst.session, &n_in)); !e.empty()) return e;
    if (std::string e = take(api->SessionGetOutputCount(inst.session, &n_out)); !e.empty()) return e;

    inst.inputs.resize(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        InputInfo &in = inst.inputs[i];
        char *name = nullptr;
        if (std::string e = take(api->SessionGetInputName(inst.session, i, alloc, &name)); !e.empty()) return e;
        in.name = name;
        alloc->Free(alloc, name);

        OrtTypeInfo *type_info = nullptr;
        if (std::string e = take(api->SessionGetInputTypeInfo(inst.session, i, &type_info)); !e.empty()) return e;
        const OrtTensorTypeAndShapeInfo *tinfo = nullptr;
        std::string e = take(api->CastTypeInfoToTensorInfo(type_info, &tinfo));
        size_t ndim = 0;
        if (e.empty() && !tinfo) e = "input '" + in.name + "' is not a tensor";
        if (e.empty()) e = take(api->GetTensorElementType(tinfo, &in.ort_type));
        if (e.empty()) e = take(api->GetDimensionsCount(tinfo, &ndim));
        if (e.empty()) {
            in.shape.resize(ndim);
            e = take(api->GetDimensions(tinfo, in.shape.data(), ndim));
        }
        api->ReleaseTypeInfo(type_info);
        if (!e.empty()) return e;
        if (!to_agent_dtype(in.ort_type, in.dtype, in.elem_bytes))
            return "input '" + in.name + "' has an element type agent_dtype cannot represent (" + std::to_string(in.ort_type) + ")";
    }

    inst.outputs.resize(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        char *name = nullptr;
        if (std::string e = take(api->SessionGetOutputName(inst.session, i, alloc, &name)); !e.empty()) return e;
        inst.outputs[i] = name;
        alloc->Free(alloc, name);
    }

    for (const InputInfo &in : inst.inputs) inst.input_names.push_back(in.name.c_str());
    for (const std::string &out : inst.outputs) inst.output_names.push_back(out.c_str());
    return {};
}

// ------------------------------------------------------------------ ABI functions

int32_t onnx_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    *out_instance = nullptr;
    const OrtApi *api = ort();
    if (!api) {
        std::snprintf(err, err_cap, "onnx: libonnxruntime is older than the headers this plugin was built with (needs ORT API %d)",
                      ORT_API_VERSION);
        return AGENT_ERR_INIT_FAILED;
    }

    simdjson::ondemand::parser parser;
    simdjson::padded_string json(sv(config_json));
    simdjson::ondemand::document doc;
    simdjson::ondemand::object cfg;
    if (parser.iterate(json).get(doc) || doc.get_object().get(cfg)) {
        std::snprintf(err, err_cap, "onnx: invalid config JSON");
        return AGENT_ERR_INVALID_CONFIG;
    }
    std::string_view model_sv;
    if (cfg["model"].get_string().get(model_sv) || model_sv.empty()) {
        std::snprintf(err, err_cap, "onnx: config.model (.onnx path) is required");
        return AGENT_ERR_INVALID_CONFIG;
    }
    const std::string model(model_sv);

    int64_t threads = 0, device_id = 0;
    std::string ep = "cpu", opt_level = "all";
    simdjson::ondemand::object options;
    if (!cfg["options"].get_object().get(options)) {
        double d = 0;
        std::string_view s;
        if (!options["threads"].get_double().get(d)) threads = static_cast<int64_t>(d);
        if (!options["device_id"].get_double().get(d)) device_id = static_cast<int64_t>(d);
        if (!options["execution_provider"].get_string().get(s)) ep = std::string(s);
        if (!options["graph_optimization"].get_string().get(s)) opt_level = std::string(s);
    }

    GraphOptimizationLevel level = ORT_ENABLE_ALL;
    if (opt_level == "disable") level = ORT_DISABLE_ALL;
    else if (opt_level == "basic") level = ORT_ENABLE_BASIC;
    else if (opt_level == "extended") level = ORT_ENABLE_EXTENDED;
    else if (opt_level != "all") {
        std::snprintf(err, err_cap, "onnx: graph_optimization must be disable, basic, extended or all (got '%s')", opt_level.c_str());
        return AGENT_ERR_INVALID_CONFIG;
    }
    if (ep != "cpu" && ep != "cuda") {
        std::snprintf(err, err_cap, "onnx: execution_provider must be cpu or cuda (got '%s')", ep.c_str());
        return AGENT_ERR_INVALID_CONFIG;
    }

    std::string env_err;
    OrtEnv *env = ort_env(env_err);
    if (!env) {
        std::snprintf(err, err_cap, "onnx: CreateEnv failed: %s", env_err.c_str());
        return AGENT_ERR_INIT_FAILED;
    }

    OrtSessionOptions *opts = nullptr;
    std::string e = take(api->CreateSessionOptions(&opts));
    if (e.empty() && threads > 0) e = take(api->SetIntraOpNumThreads(opts, static_cast<int>(threads)));
    if (e.empty()) e = take(api->SetSessionGraphOptimizationLevel(opts, level));
    if (!e.empty()) {
        if (opts) api->ReleaseSessionOptions(opts);
        std::snprintf(err, err_cap, "onnx: session options: %s", e.c_str());
        return AGENT_ERR_INIT_FAILED;
    }

    if (ep == "cuda") {
        // The CPU-only release builds still export this entry point; it fails when the CUDA EP
        // (libonnxruntime_providers_cuda) is not part of the ORT build or cannot be loaded.
        OrtCUDAProviderOptionsV2 *cuda = nullptr;
        e = take(api->CreateCUDAProviderOptions(&cuda));
        if (e.empty()) {
            const std::string dev = std::to_string(device_id);
            const char *keys[] = {"device_id"};
            const char *vals[] = {dev.c_str()};
            e = take(api->UpdateCUDAProviderOptions(cuda, keys, vals, 1));
        }
        if (e.empty()) e = take(api->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cuda));
        if (cuda) api->ReleaseCUDAProviderOptions(cuda);
        if (!e.empty()) {
            api->ReleaseSessionOptions(opts);
            std::snprintf(err, err_cap, "onnx: execution_provider 'cuda' is not available in this ONNX Runtime build: %s", e.c_str());
            return AGENT_ERR_UNSUPPORTED;
        }
    }

    auto inst = std::make_unique<OnnxInstance>();
#ifdef _WIN32
    const std::wstring wpath(model.begin(), model.end());
    e = take(api->CreateSession(env, wpath.c_str(), opts, &inst->session));
#else
    e = take(api->CreateSession(env, model.c_str(), opts, &inst->session));
#endif
    api->ReleaseSessionOptions(opts);
    if (!e.empty()) {
        std::snprintf(err, err_cap, "onnx: failed to load %s: %s", model.c_str(), e.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    if (e = take(api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &inst->cpu_mem)); !e.empty()) {
        std::snprintf(err, err_cap, "onnx: CreateCpuMemoryInfo: %s", e.c_str());
        return AGENT_ERR_INIT_FAILED;
    }
    if (e = cache_io(*inst); !e.empty()) {
        std::snprintf(err, err_cap, "onnx: %s: %s", model.c_str(), e.c_str());
        return AGENT_ERR_MODEL_LOAD_FAILED;
    }
    *out_instance = inst.release();
    return AGENT_OK;
}

void onnx_destroy(void *instance) { delete static_cast<OnnxInstance *>(instance); }

// The last TENSOR part named `name` across all messages (later messages win).
const agent_part *find_tensor(const agent_request *req, std::string_view name) {
    const agent_part *found = nullptr;
    for (size_t i = 0; i < req->n_messages; ++i) {
        const agent_message &m = req->messages[i];
        for (size_t p = 0; p < m.n_parts; ++p)
            if (m.parts[p].kind == AGENT_PART_TENSOR && sv(m.parts[p].name) == name) found = &m.parts[p];
    }
    return found;
}

int32_t onnx_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<OnnxInstance *>(instance);
    const OrtApi *api = ort();
    agent_response_reset(resp);
    if (req->task != AGENT_CAP_GENERIC)
        return set_error(AGENT_ERR_UNSUPPORTED, "onnx: only AGENT_CAP_GENERIC (named tensors) is supported");
    if (req->cancel && *req->cancel) return set_error(AGENT_ERR_CANCELLED, "onnx: cancelled");

    // --- inputs: wrap the borrowed tensor data in OrtValues (no copy) ---
    const size_t n_in = inst->inputs.size();
    std::vector<OrtValue *> in_values(n_in, nullptr);
    auto release_inputs = [&] {
        for (OrtValue *v : in_values)
            if (v) api->ReleaseValue(v);
    };
    for (size_t i = 0; i < n_in; ++i) {
        const InputInfo &info = inst->inputs[i];
        const agent_part *part = find_tensor(req, info.name);
        if (!part) {
            release_inputs();
            return set_error(AGENT_ERR_INVALID_CONFIG, "onnx: missing required input tensor '" + info.name + "' (" +
                                                           dtype_name(info.dtype) + " " + shape_str(info.shape) + ")");
        }
        if (part->dtype != info.dtype) {
            release_inputs();
            return set_error(AGENT_ERR_INVALID_CONFIG,
                             "onnx: input '" + info.name + "' must be " + dtype_name(info.dtype) + ", got " + dtype_name(part->dtype));
        }
        size_t elems = 1;
        for (size_t d = 0; d < part->ndim; ++d) elems *= part->shape[d] < 0 ? 0 : static_cast<size_t>(part->shape[d]);
        if (elems * info.elem_bytes != part->size) {
            release_inputs();
            return set_error(AGENT_ERR_INVALID_CONFIG,
                             "onnx: input '" + info.name + "': " + std::to_string(part->size) + " bytes do not match its shape");
        }
        if (std::string e = take(api->CreateTensorWithDataAsOrtValue(inst->cpu_mem, const_cast<void *>(part->data), part->size, part->shape,
                                                                     part->ndim, info.ort_type, &in_values[i]));
            !e.empty()) {
            release_inputs();
            return set_error(AGENT_ERR_INVALID_CONFIG, "onnx: input '" + info.name + "': " + e);
        }
    }

    // --- run ---
    auto result = std::make_unique<CallResult>();
    const size_t n_out = inst->outputs.size();
    result->values.assign(n_out, nullptr);
    std::string e = take(api->Run(inst->session, nullptr, inst->input_names.data(), in_values.data(), n_in, inst->output_names.data(),
                                  n_out, result->values.data()));
    release_inputs();
    if (!e.empty()) return set_error(AGENT_ERR_DECODE_FAILED, "onnx: Run failed: " + e);

    // --- outputs: point parts at the OrtValues' buffers, kept alive until free_response() ---
    result->shapes.resize(n_out);
    result->parts.resize(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        int is_tensor = 0;
        if (!take(api->IsTensor(result->values[i], &is_tensor)).empty() || !is_tensor)
            return set_error(AGENT_ERR_UNSUPPORTED,
                             "onnx: output '" + inst->outputs[i] + "' is not a tensor (sequence/map outputs are unsupported)");
        OrtTensorTypeAndShapeInfo *tinfo = nullptr;
        if (e = take(api->GetTensorTypeAndShape(result->values[i], &tinfo)); !e.empty())
            return set_error(AGENT_ERR_DECODE_FAILED, "onnx: " + e);
        ONNXTensorElementDataType t = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        size_t ndim = 0, elems = 0;
        std::vector<int64_t> &shape = result->shapes[i];
        e = take(api->GetTensorElementType(tinfo, &t));
        if (e.empty()) e = take(api->GetDimensionsCount(tinfo, &ndim));
        if (e.empty()) {
            shape.resize(ndim);
            e = take(api->GetDimensions(tinfo, shape.data(), ndim));
        }
        if (e.empty()) e = take(api->GetTensorShapeElementCount(tinfo, &elems));
        api->ReleaseTensorTypeAndShapeInfo(tinfo);
        if (!e.empty()) return set_error(AGENT_ERR_DECODE_FAILED, "onnx: " + e);

        uint32_t dtype = 0;
        size_t elem_bytes = 0;
        if (!to_agent_dtype(t, dtype, elem_bytes))
            return set_error(AGENT_ERR_UNSUPPORTED, "onnx: output '" + inst->outputs[i] +
                                                        "' has an element type agent_dtype cannot represent (" + std::to_string(t) + ")");
        void *data = nullptr;
        if (e = take(api->GetTensorMutableData(result->values[i], &data)); !e.empty())
            return set_error(AGENT_ERR_DECODE_FAILED, "onnx: " + e);

        agent_part &part = result->parts[i];
        part = agent_part{};
        part.kind = AGENT_PART_TENSOR;
        part.name = agent_str{inst->outputs[i].data(), inst->outputs[i].size()};
        part.dtype = dtype;
        part.shape = shape.data();
        part.ndim = ndim;
        part.data = data;
        part.size = elems * elem_bytes;
    }

    resp->message.role = agent_str{"assistant", 9};
    resp->message.parts = result->parts.data();
    resp->message.n_parts = result->parts.size();
    {
        std::lock_guard<std::mutex> lk(inst->live_mutex);
        inst->live.push_back(std::move(result));
    }
    return AGENT_OK;
}

void onnx_free_response(void *instance, agent_response *resp) {
    auto *inst = static_cast<OnnxInstance *>(instance);
    if (resp->message.parts) {
        std::unique_ptr<CallResult> done;
        {
            std::lock_guard<std::mutex> lk(inst->live_mutex);
            auto it = std::find_if(inst->live.begin(), inst->live.end(),
                                   [&](const std::unique_ptr<CallResult> &r) { return r->parts.data() == resp->message.parts; });
            if (it != inst->live.end()) {
                done = std::move(*it);
                inst->live.erase(it);
            }
        }
    }
    agent_response_reset(resp);
}

const char *onnx_last_error(void *) { return t_error.c_str(); }

const agent_engine_v1 kOnnxEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1),
    "onnx",
    "1.0",
    AGENT_CAP_GENERIC,
    AGENT_ENGINE_FLAG_CONCURRENT,
    onnx_create,
    onnx_destroy,
    onnx_run,
    onnx_free_response,
    onnx_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kOnnxEngine; }
