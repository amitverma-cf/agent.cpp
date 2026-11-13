# C API

`include/agent-cpp/agent_c.h` exposes the library to plain C and to anything that can call C: Python (`ctypes`), Rust, Go, C#, Zig.
It wraps the C++ API in [`agent.hpp`](../api.md) for the common cases: chat turns with tools, embeddings and one-shot engine tasks
(speech, images, ...). It is part of `libagent`, so there is nothing extra to link.

Handles are opaque, strings are NUL-terminated, and buffers belong to the caller. No C++ type crosses the boundary and no function throws.

```c
#include <agent-cpp/agent_c.h>
#include <stdio.h>

static void on_token(const char *text, size_t length, void *user) { fwrite(text, 1, length, stdout); }

int main(void) {
    char err[512];
    agent_c_session *s = agent_c_open("llama", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", NULL, ".workspace", 4096, 512,
                                      err, sizeof err);
    if (!s) { fprintf(stderr, "%s\n", err); return 1; }

    agent_c_conversation *chat = agent_c_conversation_new("demo");
    char reply[4096];
    if (agent_c_run_turn(s, chat, "Say hi in five words.", on_token, NULL, reply, sizeof reply, NULL, err, sizeof err) != AGENT_C_OK)
        fprintf(stderr, "%s\n", err);

    agent_c_conversation_free(chat);
    agent_c_close(s);
    return 0;
}
```

The engine name `"llama"` is found at run time next to the program, the same way as in C++ (see [engines.md](engines.md)). A path to an
engine plugin also works. Engine files are not named in code.

## Functions

| Function | What it does |
|---|---|
| `agent_c_version()` | The library version, a static string. |
| `agent_c_open(...)` | Opens a session on an engine and a model. See [Sessions](#sessions). |
| `agent_c_close(session)` | Closes a session. `NULL` is allowed. |
| `agent_c_conversation_new(id)`, `agent_c_conversation_free(c)` | Creates and frees the history of one chat. |
| `agent_c_run_turn(...)` | One chat turn, with tools. See [Chat turns](#chat-turns). |
| `agent_c_cancel(session)` | Cancels the running call. Safe from any thread. |
| `agent_c_embed(...)` | The embedding of a text. See [Embeddings](#embeddings). |
| `agent_c_run_task(...)` | A one-shot engine task. See [One-shot tasks](#one-shot-tasks). |

The header uses `__declspec(dllimport)` or `__declspec(dllexport)` on Windows when `AGENT_C_USE_DLL` or `AGENT_C_BUILD_DLL` is
defined. Only the Windows shared-library build needs them. The CMake build does not set either macro.

## Status codes

| Constant | Value | Meaning |
|---|---|---|
| `AGENT_C_OK` | 0 | Success. |
| `AGENT_C_ERROR` | 1 | Failure. The message is in the `err` buffer. |
| `AGENT_C_CANCELLED` | 2 | The turn or task was stopped with `agent_c_cancel`. |
| `AGENT_C_BUFFER_TOO_SMALL` | 3 | The answer did not fit the buffer. `*needed` (when given) is the size required, including the NUL. |

Errors are reported in the `err` buffer you pass, cut to `err_cap - 1` bytes plus the NUL.

## Sessions

```c
agent_c_session *agent_c_open(const char *engine, const char *model_path, const char *options_json, const char *workspace_dir,
                              int context_window, int max_tokens, char *err, size_t err_cap);
void agent_c_close(agent_c_session *session);
```

- `engine`: a registered engine name (`"llama"`, `"audiocpp"`, `"sdcpp"`, `"trellis"`, `"onnx"`) or the path of an engine plugin.
  Required.
- `model_path`: the model file, passed as the engine's `model` role. May be `NULL` for engines that take all their files from
  `options_json`.
- `options_json`: the engine options as a JSON object, for example `{"mmproj":"clip.gguf"}`, `{"mode":"embed"}` or
  `{"tts":{"model":"tts.gguf"}}`. `NULL` or `""` for none. The string is passed to the engine unchanged.
- `workspace_dir`: a folder for the session's files. Required; created if missing.
- `context_window`, `max_tokens`: `0` keeps the library defaults (2048 and 512).

Returns `NULL` on failure, with the reason in `err`. `agent_c_close` accepts `NULL`.

```c
char err[256];
agent_c_session *s = agent_c_open("audiocpp", NULL, "{\"asr\":{\"model\":\"asr.gguf\"}}", ".workspace", 0, 0, err, sizeof err);
```

## Conversations

```c
agent_c_conversation *agent_c_conversation_new(const char *id);
void agent_c_conversation_free(agent_c_conversation *conversation);
```

A conversation is the message history of one chat. Create one per chat; one session can serve many conversations. The `id` is used for
the saved history, as in the C++ `Conversation::id`.

## Chat turns

```c
typedef void (*agent_c_token_fn)(const char *text, size_t length, void *user_data);

int agent_c_run_turn(agent_c_session *session, agent_c_conversation *conversation, const char *prompt,
                     agent_c_token_fn on_token, void *user_data, char *reply, size_t reply_cap, size_t *needed, char *err,
                     size_t err_cap);
```

Sends `prompt` and lets the model call tools as needed, the same as `agent::run_turn`. `on_token` (may be `NULL`) receives each piece of
the answer as it is generated; tool calls are handled inside the call. The final answer is written to `reply` as a NUL-terminated string.

If the answer does not fit `reply_cap`, it is cut, the call returns `AGENT_C_BUFFER_TOO_SMALL`, and `*needed` (when not `NULL`) is the
size that would have fit. The turn itself has completed and is in the history, so you do not need to send the prompt again.

```c
char reply[4096];
size_t needed = 0;
int rc = agent_c_run_turn(s, chat, "Summarise the plan.", NULL, NULL, reply, sizeof reply, &needed, err, sizeof err);
if (rc == AGENT_C_BUFFER_TOO_SMALL) {
    /* needed holds the size of the full answer, including the NUL */
}
```

## Cancelling

```c
void agent_c_cancel(agent_c_session *session);
```

Stops the turn or task running on the session. It is safe to call from another thread at any time, and the running call returns
`AGENT_C_CANCELLED`. The C API has no way to clear the cancel flag (the C++ `clear_cancel`), so after a cancel, every later call on that
session fails with `AGENT_C_CANCELLED`. Close the session and open a new one.

## Embeddings

```c
int agent_c_embed(agent_c_session *session, const char *text, float *out, size_t out_cap, size_t *dims, char *err, size_t err_cap);
```

Returns the embedding of `text` from a session opened on an embedding model, with the engine option `{"mode":"embed"}`. The vector is
written to `out` (up to `out_cap` floats) and its length to `*dims`. If `out_cap` is smaller than the vector, the call returns
`AGENT_C_BUFFER_TOO_SMALL` and `*dims` says how many floats are needed.

```c
char err[256];
agent_c_session *emb = agent_c_open("llama", "models/bge-small-en-v1.5-q8_0.gguf", "{\"mode\":\"embed\"}", ".workspace", 512, 0,
                                    err, sizeof err);
float vec[1024];
size_t dims = 0;
if (emb && agent_c_embed(emb, "How many vacation days?", vec, 1024, &dims, err, sizeof err) == AGENT_C_OK) {
    /* dims floats in vec, L2-normalized */
}
agent_c_close(emb);
```

## One-shot tasks

```c
int agent_c_run_task(agent_c_session *session, uint32_t task, const char *text, const char *media_path, char *text_out,
                     size_t text_cap, char *file_out, size_t file_cap, char *err, size_t err_cap);
```

Runs one engine task. `task` is one `AGENT_CAP_*` bit from `<agent-cpp/engine_abi.h>`, for example `AGENT_CAP_ASR` (speech to text),
`AGENT_CAP_TTS` (text to speech) or `AGENT_CAP_IMAGE_GEN`. `text` and `media_path` (a file the engine reads, such as a WAV or a PNG)
are both optional and may be `NULL`.

The engine's text answer goes to `text_out`. The path of the first file the engine wrote (a WAV, a PNG, a GLB) goes to `file_out`, which
is empty when there is none. Either output may be `NULL` when you do not need it. The file is under the session's `artifacts` folder.

```c
#include <agent-cpp/engine_abi.h>

char err[256], heard[1024];
agent_c_session *asr = agent_c_open("audiocpp", NULL, "{\"asr\":{\"model\":\"asr.gguf\"}}", ".workspace", 0, 0, err, sizeof err);
if (asr && agent_c_run_task(asr, AGENT_CAP_ASR, NULL, "clip.wav", heard, sizeof heard, NULL, 0, err, sizeof err) == AGENT_C_OK)
    printf("%s\n", heard);
agent_c_close(asr);
```

## Not in the C API

The C API covers chat with tools, embeddings and one-shot tasks. These are C++ only for now: graphs ([graph.md](graph.md)), the model
registry ([models.md](models.md)), custom tools and the permission gate ([tools.md](tools.md)), MCP servers ([mcp.md](mcp.md)) and
`agent::decide` ([decide.md](decide.md)).

Writing an engine plugin uses a separate C ABI, `engine_abi.h`. See [engines.md](engines.md) and
[writing-an-engine.md](../guides/writing-an-engine.md). Using the C API does not need it.
