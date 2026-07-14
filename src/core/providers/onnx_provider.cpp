#include "../executor/ambient_turn.hpp"
#include "provider_ops.hpp"

#include <agent-cpp/agent.hpp>

#ifdef AGENT_HAS_ONNX
#include <cstring>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <string>
#include <vector>

namespace agent::providers {

static ONNXTensorElementDataType to_ort_type(DType dtype) {
    switch (dtype) {
    case DType::Float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::Float16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DType::Int8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DType::Int32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DType::Int64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DType::UInt8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

static DType from_ort_type(ONNXTensorElementDataType t) {
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DType::Float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DType::Float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return DType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return DType::UInt8;
    default: return DType::Float32;
    }
}

static size_t dtype_bytes(DType dtype) {
    switch (dtype) {
    case DType::Float32: return 4;
    case DType::Float16: return 2;
    case DType::Int8: return 1;
    case DType::Int32: return 4;
    case DType::Int64: return 8;
    case DType::UInt8: return 1;
    }
    return 4;
}

#define ORT_CHECK(api, expr)                                                                                         \
    do {                                                                                                             \
        OrtStatus *_s = (expr);                                                                                      \
        if (_s) {                                                                                                    \
            std::string _msg((api)->GetErrorMessage(_s));                                                           \
            (api)->ReleaseStatus(_s);                                                                                \
            return fail<InferResponse>(ErrorCode::DecodeFailed, _msg);                                               \
        }                                                                                                            \
    } while (0)

#define ORT_CHECK_INIT(api, expr)                                                                                    \
    do {                                                                                                             \
        OrtStatus *_s = (expr);                                                                                      \
        if (_s) {                                                                                                    \
            std::string _msg((api)->GetErrorMessage(_s));                                                           \
            (api)->ReleaseStatus(_s);                                                                                \
            return fail(ErrorCode::ModelLoadFailed, _msg);                                                          \
        }                                                                                                            \
    } while (0)

struct OnnxState {
    const OrtApi *api = nullptr;
    OrtEnv *env = nullptr;
    OrtSessionOptions *session_opts = nullptr;
    OrtSession *ort_session = nullptr;

    ~OnnxState() {
        if (!api) return;
        if (ort_session) api->ReleaseSession(ort_session);
        if (session_opts) api->ReleaseSessionOptions(session_opts);
        if (env) api->ReleaseEnv(env);
    }
};

Result<void> init_onnx(Session &session) {
    if (session.config.model.empty()) { return fail(ErrorCode::InvalidConfig, "OnnxRuntime requires a model path in Config::model."); }

    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) { return fail(ErrorCode::ProviderInitFailed, "Failed to get ORT API pointer."); }

    auto state = std::make_shared<OnnxState>();
    state->api = api;

    ORT_CHECK_INIT(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "agent_onnx", &state->env));
    ORT_CHECK_INIT(api, api->CreateSessionOptions(&state->session_opts));
    ORT_CHECK_INIT(api, api->SetIntraOpNumThreads(state->session_opts, 1));

#ifdef _WIN32
    std::wstring wpath(session.config.model.begin(), session.config.model.end());
    ORT_CHECK_INIT(api, api->CreateSession(state->env, wpath.c_str(), state->session_opts, &state->ort_session));
#else
    ORT_CHECK_INIT(api, api->CreateSession(state->env, session.config.model.c_str(), state->session_opts, &state->ort_session));
#endif

    session.provider_state = state;
    return ok();
}

Result<InferResponse> infer_onnx(Session &session, const InferRequest &request) {
    auto state = std::static_pointer_cast<OnnxState>(session.provider_state);
    if (!state || !state->ort_session) { return fail<InferResponse>(ErrorCode::ProviderInitFailed, "ONNX session not initialized."); }

    const OrtApi *api = state->api;
    OrtAllocator *allocator = nullptr;
    api->GetAllocatorWithDefaultOptions(&allocator);

    size_t input_count = 0;
    ORT_CHECK(api, api->SessionGetInputCount(state->ort_session, &input_count));

    std::vector<char *> raw_input_names(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        ORT_CHECK(api, api->SessionGetInputName(state->ort_session, i, allocator, &raw_input_names[i]));
    }

    OrtMemoryInfo *cpu_mem = nullptr;
    ORT_CHECK(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_mem));

    std::vector<OrtValue *> input_values(input_count, nullptr);
    for (size_t i = 0; i < input_count; ++i) {
        std::string_view expected(raw_input_names[i]);
        for (const auto &mv : request.messages) {
            for (const auto &tv : mv.tensors) {
                if (tv.name != expected) continue;
                OrtValue *val = nullptr;
                ORT_CHECK(api, api->CreateTensorWithDataAsOrtValue(cpu_mem, const_cast<void *>(static_cast<const void *>(tv.data.data())),
                                                                   tv.data.size_bytes(), tv.shape.data(), tv.shape.size(),
                                                                   to_ort_type(tv.dtype), &val));
                input_values[i] = val;
                break;
            }
            if (input_values[i]) break;
        }
    }
    api->ReleaseMemoryInfo(cpu_mem);

    size_t output_count = 0;
    ORT_CHECK(api, api->SessionGetOutputCount(state->ort_session, &output_count));

    std::vector<char *> raw_output_names(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        ORT_CHECK(api, api->SessionGetOutputName(state->ort_session, i, allocator, &raw_output_names[i]));
    }

    std::vector<const char *> in_ptrs(input_count), out_ptrs(output_count);
    for (size_t i = 0; i < input_count; ++i) in_ptrs[i] = raw_input_names[i];
    for (size_t i = 0; i < output_count; ++i) out_ptrs[i] = raw_output_names[i];

    std::vector<OrtValue *> output_values(output_count, nullptr);
    ORT_CHECK(api, api->Run(state->ort_session, nullptr, in_ptrs.data(), input_values.data(), input_count, out_ptrs.data(), output_count,
                            output_values.data()));

    for (size_t i = 0; i < input_count; ++i) {
        allocator->Free(allocator, raw_input_names[i]);
        if (input_values[i]) api->ReleaseValue(input_values[i]);
    }

    std::span<TensorView> out_views = ambient::current_arena(session).allocate_span<TensorView>(output_count);

    for (size_t i = 0; i < output_count; ++i) {
        OrtValue *oval = output_values[i];

        OrtTensorTypeAndShapeInfo *info = nullptr;
        api->GetTensorTypeAndShape(oval, &info);

        size_t ndim = 0;
        api->GetDimensionsCount(info, &ndim);

        std::span<int64_t> shape_span = ambient::current_arena(session).allocate_span<int64_t>(ndim);
        api->GetDimensions(info, shape_span.data(), ndim);

        ONNXTensorElementDataType ort_type;
        api->GetTensorElementType(info, &ort_type);
        api->ReleaseTensorTypeAndShapeInfo(info);

        DType dtype = from_ort_type(ort_type);

        size_t elem_count = 1;
        for (size_t d = 0; d < ndim; ++d) elem_count *= static_cast<size_t>(shape_span[d]);
        size_t byte_count = elem_count * dtype_bytes(dtype);

        void *raw_data = nullptr;
        api->GetTensorMutableData(oval, &raw_data);

        std::span<uint8_t> data_span = ambient::current_arena(session).allocate_span<uint8_t>(byte_count);
        std::memcpy(data_span.data(), raw_data, byte_count);

        std::string_view name_view = ambient::current_arena(session).allocate_string(std::string_view(raw_output_names[i]));

        out_views[i] = TensorView{.name = name_view, .dtype = dtype, .shape = shape_span, .data = data_span};

        allocator->Free(allocator, raw_output_names[i]);
        api->ReleaseValue(oval);
    }

    MessageView msg;
    msg.role = ambient::current_arena(session).allocate_string("assistant");
    msg.content = ambient::current_arena(session).allocate_string("");
    msg.tool_calls = {};
    msg.tool_call_id = {};
    msg.tensors = out_views;

    return ok(InferResponse{.message = msg, .usage = TokenUsage{}, .output_tensors = out_views});
}

} // namespace agent::providers
#endif
