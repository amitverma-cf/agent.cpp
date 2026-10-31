// Reference engine plugin used by the test suite. Shows the minimum an engine implements.
//   - echoes the last user text back as "echo: <text>"
//   - a user message starting with "call:<tool>" produces a tool call to <tool> with {}
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

#include <agent-cpp/engine_abi.h>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct MockInstance {
    std::string model;
    std::string reply;
    std::string tool_name;
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
};

std::string_view sv(agent_str s) { return s.data ? std::string_view(s.data, s.size) : std::string_view(); }

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

int32_t mock_run(void *instance, const agent_request *req, agent_response *resp) {
    auto *inst = static_cast<MockInstance *>(instance);
    if (req->task != AGENT_CAP_TEXT_GEN) {
        inst->error = "mock: only TEXT_GEN is supported";
        return AGENT_ERR_UNSUPPORTED;
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

    *resp = agent_response{};
    resp->message.role = agent_str{"assistant", 9};
    resp->usage = agent_usage{static_cast<int32_t>(last_user.size()), 1};

    const bool after_tool = req->n_messages > 0 && sv(req->messages[req->n_messages - 1].role) == "tool";
    if (!after_tool && last_user.starts_with("call:")) {
        inst->tool_name = std::string(last_user.substr(5));
        inst->call = agent_tool_call{agent_str{"call_0", 6}, agent_str{inst->tool_name.data(), inst->tool_name.size()}, agent_str{"{}", 2}};
        resp->message.tool_calls = &inst->call;
        resp->message.n_tool_calls = 1;
        return AGENT_OK;
    }

    inst->reply = after_tool ? "tool done" : "echo: " + std::string(last_user) + media_lines;
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

void mock_free_response(void *, agent_response *resp) { *resp = agent_response{}; }

const char *mock_last_error(void *instance) { return static_cast<MockInstance *>(instance)->error.c_str(); }

const agent_engine_v1 kMockEngine = {
    AGENT_ENGINE_ABI_VERSION,
    "mock",
    "1.0",
    AGENT_CAP_TEXT_GEN | AGENT_CAP_TOOL_CALL,
    0,
    mock_create,
    mock_destroy,
    mock_run,
    mock_free_response,
    mock_last_error,
    nullptr,
    nullptr,
};

} // namespace

extern "C" AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kMockEngine; }
