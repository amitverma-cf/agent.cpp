/* agent_c.h: the agent.cpp library from plain C (and from anything that can call C: Python ctypes, Rust, Go, C#, Zig, ...).
 *
 * It wraps the C++ API in agent.hpp: open a session on an engine and a model, talk to it (streamed chat turns with tools, embeddings,
 * one-shot engine tasks such as speech recognition or image generation), close it. Everything is opaque handles, plain strings and
 * caller-provided buffers: no C++ types cross this boundary, no exceptions, no allocation that the caller has to free except through
 * the functions below. Errors come back as a non-zero status and a message in the buffer you pass.
 *
 * Threads: a session is for one thread at a time (the same rule as the C++ API). Different sessions are independent.
 */
#ifndef AGENT_CPP_AGENT_C_H
#define AGENT_CPP_AGENT_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(AGENT_C_BUILD_DLL)
#define AGENT_C_API __declspec(dllexport)
#elif defined(_WIN32) && defined(AGENT_C_USE_DLL)
#define AGENT_C_API __declspec(dllimport)
#else
#define AGENT_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct agent_c_session agent_c_session;
typedef struct agent_c_conversation agent_c_conversation;

/* 0 is success; every other value is an error (the message is in the buffer you passed). */
#define AGENT_C_OK 0
#define AGENT_C_ERROR 1
#define AGENT_C_CANCELLED 2
#define AGENT_C_BUFFER_TOO_SMALL 3 /* the answer did not fit the buffer; *needed (when given) says how large it must be */

/* The library version, e.g. "0.1.2". Static string. */
AGENT_C_API const char *agent_c_version(void);

/* Opens a session. engine: a registered engine name or the path of an engine plugin (agent-engine-llama.so / .dll).
 * model_path: the model file the engine loads (may be NULL for engines that take all their files from options_json).
 * options_json: engine options as a JSON object, e.g. {"mmproj":"...","mode":"embed"} or {"tts":{"model":"..."}}; NULL or "" for none.
 * workspace_dir: a writable folder for the session's files (created if missing). context_window / max_tokens: 0 = the library defaults.
 * Returns NULL on failure with the reason in err. */
AGENT_C_API agent_c_session *agent_c_open(const char *engine, const char *model_path, const char *options_json, const char *workspace_dir,
                                          int context_window, int max_tokens, char *err, size_t err_cap);
AGENT_C_API void agent_c_close(agent_c_session *session);

/* A conversation: the message history of one chat. Create one per chat; the session can serve many. */
AGENT_C_API agent_c_conversation *agent_c_conversation_new(const char *id);
AGENT_C_API void agent_c_conversation_free(agent_c_conversation *conversation);

/* Called with each piece of the answer while it is generated (text only; tool calls are handled inside). */
typedef void (*agent_c_token_fn)(const char *text, size_t length, void *user_data);

/* One chat turn: sends prompt, lets the model call tools as needed, writes the final answer (NUL terminated) to reply.
 * on_token may be NULL. If the answer is longer than reply_cap - 1 bytes it is cut, AGENT_C_BUFFER_TOO_SMALL is returned and
 * *needed (when not NULL) is the size required including the NUL. */
AGENT_C_API int agent_c_run_turn(agent_c_session *session, agent_c_conversation *conversation, const char *prompt,
                                 agent_c_token_fn on_token, void *user_data, char *reply, size_t reply_cap, size_t *needed, char *err,
                                 size_t err_cap);

/* Cancels the turn running on this session from another thread. Safe to call at any time. */
AGENT_C_API void agent_c_cancel(agent_c_session *session);

/* An embedding of text from a session opened on an embedding model ({"mode":"embed"}). Writes up to out_cap floats to out and the
 * vector's length to *dims. AGENT_C_BUFFER_TOO_SMALL when out_cap is smaller than the vector (*dims says how many are needed). */
AGENT_C_API int agent_c_embed(agent_c_session *session, const char *text, float *out, size_t out_cap, size_t *dims, char *err,
                              size_t err_cap);

/* A one-shot engine task. task is one of the AGENT_CAP_* bits of engine_abi.h (speech recognition, synthesis, voice activity, image
 * generation, ...). text and media_path (a file the engine reads, e.g. a WAV or an image) are both optional. The engine's text answer goes to
 * text_out; the first file it produced (a WAV, a PNG, a GLB) has its path written to file_out, empty when there is none. */
AGENT_C_API int agent_c_run_task(agent_c_session *session, uint32_t task, const char *text, const char *media_path, char *text_out,
                                 size_t text_cap, char *file_out, size_t file_cap, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_CPP_AGENT_C_H */
