# C API

`include/agent-cpp/agent_c.h` is the library from plain C, and from anything that can call C functions: Python (`ctypes`), Rust,
Go, C#, Zig. It wraps the C++ API in [`agent.hpp`](../api.md). Handles are opaque, strings are NUL-terminated, buffers belong to the
caller, and no C++ type crosses the boundary. It is part of `libagent` (the same shared or static library); there is nothing extra to
link.

```c
#include <agent-cpp/agent_c.h>
#include <stdio.h>

static void on_token(const char *text, size_t length, void *user) { fwrite(text, 1, length, stdout); }

int main(void) {
    char err[512];
    agent_c_session *s = agent_c_open("agent-engine-llama.so", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", NULL, ".workspace", 4096, 512,
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

## Functions

| Function | What it does |
|---|---|
| `agent_c_version()` | the library version, a static string |
| `agent_c_open(engine, model_path, options_json, workspace_dir, context_window, max_tokens, err, err_cap)` | opens a session on an engine (a registered name, or the path of an engine plugin) and a model; `NULL` and the reason in `err` on failure. `options_json` is the engine's options object (`{"mmproj":"..."}`, `{"mode":"embed"}`, `{"tts":{"model":"..."}}`); `0` for the two sizes means the defaults |
| `agent_c_close(session)` | closes it; `NULL` is allowed |
| `agent_c_conversation_new(id)` / `agent_c_conversation_free()` | the message history of one chat; a session serves many |
| `agent_c_run_turn(session, conversation, prompt, on_token, user_data, reply, reply_cap, needed, err, err_cap)` | one chat turn: tools the model calls run inside; `on_token` (may be `NULL`) gets the answer as it is generated; the final answer is written to `reply`. If it does not fit, it is cut, `AGENT_C_BUFFER_TOO_SMALL` is returned and `*needed` is the size required |
| `agent_c_cancel(session)` | cancels the running turn; callable from another thread at any time |
| `agent_c_embed(session, text, out, out_cap, dims, err, err_cap)` | the embedding of a text from an embedding model; `*dims` is its length |
| `agent_c_run_task(session, task, text, media_path, text_out, text_cap, file_out, file_cap, err, err_cap)` | a one-shot engine task: speech recognition, text to speech, image generation, ... `task` is one of the `AGENT_CAP_*` bits of [`engine_abi.h`](engines.md); the text answer goes to `text_out`, the path of the first file produced (a WAV, a PNG, a GLB) to `file_out` |

Every function that can fail returns `AGENT_C_OK` (0), `AGENT_C_ERROR`, `AGENT_C_CANCELLED` or `AGENT_C_BUFFER_TOO_SMALL`, and puts a
message in the `err` buffer you pass. A session is for one thread at a time, the same rule as the C++ API; separate sessions are
independent.

## What it does not cover

The C API is the part of the library most programs need: chat with tools, embeddings and one-shot tasks. Graphs, the model registry,
custom tools and the permission gate are C++ APIs for now. Engine plugins are a separate C ABI,
[`engine_abi.h`](engines.md), for writing engines rather than using them.
