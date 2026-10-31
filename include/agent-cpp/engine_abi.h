/*
 * agent.cpp engine plugin ABI.
 *
 * An engine is a shared library (agent-engine-<name>.so / .dll / .dylib) wrapping an inference
 * project (llama.cpp, whisper.cpp, stable-diffusion.cpp, ...). It exports one symbol:
 *
 *     AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void);
 *
 * This header is plain C so plugins built with a different compiler or C++ runtime (MSVC vs
 * MinGW, libstdc++ vs libc++) and with their own copy of ggml/ONNX Runtime can be loaded by any
 * host. Nothing crosses the boundary except C structs, pointers and function pointers.
 *
 * Ownership: everything the host passes in is borrowed for the duration of the call.
 * Everything the engine returns in agent_response stays valid until free_response() (or the
 * next run() on the same instance) and is owned by the engine.
 */
#ifndef AGENT_CPP_ENGINE_ABI_H
#define AGENT_CPP_ENGINE_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGENT_ENGINE_ABI_VERSION 1

#if defined(_WIN32)
#define AGENT_ENGINE_EXPORT __declspec(dllexport)
#else
#define AGENT_ENGINE_EXPORT __attribute__((visibility("default")))
#endif

/* Status codes. Values match agent::ErrorCode where they overlap. */
enum agent_status {
    AGENT_OK = 0,
    AGENT_ERR_INVALID_CONFIG = 1,
    AGENT_ERR_UNSUPPORTED = 2,
    AGENT_ERR_FILESYSTEM = 3,
    AGENT_ERR_INIT_FAILED = 4,
    AGENT_ERR_MODEL_LOAD_FAILED = 5,
    AGENT_ERR_DECODE_FAILED = 6,
    AGENT_ERR_PARSE = 9,
    AGENT_ERR_CANCELLED = 10,
};

/* Capability bits (agent_engine_v1::caps, agent_request::task). */
enum agent_capability {
    AGENT_CAP_TEXT_GEN = 1u << 0,   /* chat / completion */
    AGENT_CAP_TOOL_CALL = 1u << 1,  /* emits structured tool calls */
    AGENT_CAP_VISION_IN = 1u << 2,  /* image parts in input */
    AGENT_CAP_AUDIO_IN = 1u << 3,   /* audio parts in input */
    AGENT_CAP_VIDEO_IN = 1u << 4,   /* video parts in input */
    AGENT_CAP_OCR = 1u << 5,
    AGENT_CAP_ASR = 1u << 6,        /* speech to text */
    AGENT_CAP_TTS = 1u << 7,        /* text to speech */
    AGENT_CAP_VAD = 1u << 8,
    AGENT_CAP_DIARIZE = 1u << 9,
    AGENT_CAP_IMAGE_GEN = 1u << 10,
    AGENT_CAP_IMAGE_EDIT = 1u << 11,
    AGENT_CAP_VIDEO_GEN = 1u << 12,
    AGENT_CAP_MESH_GEN = 1u << 13,  /* 3D */
    AGENT_CAP_EMBED = 1u << 14,
    AGENT_CAP_RERANK = 1u << 15,
    AGENT_CAP_DECIDE = 1u << 16,    /* choice / score / yes-no decision models */
    AGENT_CAP_CLASSIFY = 1u << 17,
    AGENT_CAP_GENERIC = 1u << 18,   /* raw named tensors in / out */
};

/* Engine flags (agent_engine_v1::flags). */
enum agent_engine_flag {
    /* run() may be called concurrently on the same instance from several threads. */
    AGENT_ENGINE_FLAG_CONCURRENT = 1u << 0,
    /* The engine formats agent_request::tools_json into the prompt itself. Without this flag the
     * host prepends its plain-text tool/skill instructions to the system message. */
    AGENT_ENGINE_FLAG_NATIVE_TOOLS = 1u << 1,
};

/* Non-owning UTF-8 string view. data need not be NUL-terminated. */
typedef struct agent_str {
    const char *data;
    size_t size;
} agent_str;

enum agent_part_kind {
    AGENT_PART_TEXT = 0,   /* text */
    AGENT_PART_FILE = 1,   /* media on disk: path + mime (image/png, audio/wav, video/mp4, model/gltf-binary) */
    AGENT_PART_BYTES = 2,  /* media in memory: data/size + mime */
    AGENT_PART_TENSOR = 3, /* raw tensor: name + dtype + shape + data */
};

enum agent_dtype {
    AGENT_DTYPE_F32 = 0,
    AGENT_DTYPE_F16 = 1,
    AGENT_DTYPE_I8 = 2,
    AGENT_DTYPE_I32 = 3,
    AGENT_DTYPE_I64 = 4,
    AGENT_DTYPE_U8 = 5,
};

typedef struct agent_part {
    uint32_t kind;          /* agent_part_kind */
    agent_str text;         /* TEXT */
    agent_str path;         /* FILE */
    agent_str mime;         /* FILE, BYTES */
    agent_str name;         /* TENSOR (input/output name) */
    uint32_t dtype;         /* TENSOR, agent_dtype */
    const int64_t *shape;   /* TENSOR */
    size_t ndim;            /* TENSOR */
    const void *data;       /* BYTES, TENSOR */
    size_t size;            /* BYTES, TENSOR: bytes */
} agent_part;

typedef struct agent_tool_call {
    agent_str id;
    agent_str name;
    agent_str arguments; /* JSON object */
} agent_tool_call;

typedef struct agent_message {
    agent_str role; /* "system" | "user" | "assistant" | "tool" */
    const agent_part *parts;
    size_t n_parts;
    const agent_tool_call *tool_calls; /* assistant messages */
    size_t n_tool_calls;
    agent_str tool_call_id; /* tool messages */
} agent_message;

enum agent_chunk_kind {
    AGENT_CHUNK_TEXT = 0,  /* UTF-8 token text; size = bytes */
    AGENT_CHUNK_AUDIO = 1, /* PCM f32 mono samples at the engine's output rate; size = bytes (4 per sample) */
};

/* size is always in bytes, whatever the kind. */
typedef void (*agent_chunk_fn)(void *user_data, uint32_t kind, const void *data, size_t size);
typedef void (*agent_progress_fn)(void *user_data, uint64_t step, uint64_t total);

typedef struct agent_request {
    uint32_t task; /* the single agent_capability this call performs, e.g. AGENT_CAP_TEXT_GEN */
    const agent_message *messages;
    size_t n_messages;
    agent_str tools_json;   /* JSON array of tool definitions, empty if none */
    agent_str params_json;  /* per-call parameters, e.g. {"max_tokens":512,"temperature":0.7} */
    agent_str output_dir;   /* where FILE outputs must be written */
    agent_chunk_fn on_chunk;       /* optional streaming */
    agent_progress_fn on_progress; /* optional, for long jobs (image/video/3D) */
    void *user_data;
    const volatile int32_t *cancel; /* optional; engine stops with AGENT_ERR_CANCELLED when *cancel != 0 */
} agent_request;

typedef struct agent_usage {
    int32_t prompt_tokens;
    int32_t completion_tokens;
} agent_usage;

typedef struct agent_response {
    agent_message message; /* assistant output: text / file / tensor parts, tool calls */
    agent_usage usage;
} agent_response;

typedef struct agent_engine_v1 {
    uint32_t abi_version; /* AGENT_ENGINE_ABI_VERSION */
    const char *name;     /* "llama", "whisper", "sdcpp", ... */
    const char *version;
    uint64_t caps;        /* agent_capability bits */
    uint32_t flags;       /* agent_engine_flag bits */

    /* Load a model. config_json: {"model": "...", "context_window": 4096, "threads": 8, ...}.
     * On failure write a NUL-terminated message into err (capacity err_cap) and return non-zero. */
    int32_t (*create)(agent_str config_json, void **out_instance, char *err, size_t err_cap);
    void (*destroy)(void *instance);

    /* Execute one request. On failure return non-zero; last_error() describes it. */
    int32_t (*run)(void *instance, const agent_request *request, agent_response *response);
    void (*free_response)(void *instance, agent_response *response);
    const char *(*last_error)(void *instance);

    /* Optional (may be NULL). */
    int32_t (*count_tokens)(void *instance, agent_str text, int32_t *out_count);
    int32_t (*reset_cache)(void *instance);
} agent_engine_v1;

typedef const agent_engine_v1 *(*agent_engine_get_v1_fn)(void);

#define AGENT_ENGINE_ENTRY_POINT "agent_engine_get_v1"

#ifdef __cplusplus
}
#endif

#endif /* AGENT_CPP_ENGINE_ABI_H */
