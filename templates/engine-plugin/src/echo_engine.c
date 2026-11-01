/*
 * agent-engine-echo: a minimal agent.cpp engine plugin in plain C. Start your own engine from here.
 *
 *   AGENT_CAP_TEXT_GEN: replies "echo: <text of the last user message>", streamed word by word.
 *
 * Replace echo_create / echo_run with calls into your inference library. Everything that crosses the
 * boundary is a C struct from <agent-cpp/engine_abi.h>; see docs/guides/writing-an-engine.md.
 */
#include <agent-cpp/engine_abi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct echo_instance {
    char *reply; /* owned output, valid until free_response() or the next run() */
    size_t reply_cap;
    agent_part part;
    char error[256];
} echo_instance;

static int str_eq(agent_str s, const char *lit) { return s.size == strlen(lit) && memcmp(s.data, lit, s.size) == 0; }

static int32_t echo_create(agent_str config_json, void **out_instance, char *err, size_t err_cap) {
    /* config_json: {"model": "...", "context_window": N, ..., "options": {...Config::engine_options...}}.
     * Load your model here; on failure write a message into err and return an AGENT_ERR_* code. */
    echo_instance *inst;
    (void)config_json;
    inst = (echo_instance *)calloc(1, sizeof(echo_instance));
    if (!inst) {
        snprintf(err, err_cap, "echo: out of memory");
        return AGENT_ERR_INIT_FAILED;
    }
    *out_instance = inst;
    return AGENT_OK;
}

static void echo_destroy(void *instance) {
    echo_instance *inst = (echo_instance *)instance;
    free(inst->reply);
    free(inst);
}

static int32_t echo_run(void *instance, const agent_request *req, agent_response *resp) {
    echo_instance *inst = (echo_instance *)instance;
    agent_str text = {"", 0};
    size_t i, p, need, start;

    inst->error[0] = '\0';
    if (req->task != AGENT_CAP_TEXT_GEN) {
        snprintf(inst->error, sizeof(inst->error), "echo: unsupported task %u", (unsigned)req->task);
        return AGENT_ERR_UNSUPPORTED;
    }

    /* The last TEXT part of the last user message. */
    for (i = 0; i < req->n_messages; ++i) {
        const agent_message *m = &req->messages[i];
        if (!str_eq(m->role, "user")) continue;
        for (p = 0; p < m->n_parts; ++p)
            if (m->parts[p].kind == AGENT_PART_TEXT) text = m->parts[p].text;
    }

    need = 6 + text.size + 1; /* "echo: " + text + NUL */
    if (need > inst->reply_cap) {
        char *grown = (char *)realloc(inst->reply, need);
        if (!grown) {
            snprintf(inst->error, sizeof(inst->error), "echo: out of memory");
            return AGENT_ERR_DECODE_FAILED;
        }
        inst->reply = grown;
        inst->reply_cap = need;
    }
    memcpy(inst->reply, "echo: ", 6);
    if (text.size) memcpy(inst->reply + 6, text.data, text.size);
    inst->reply[need - 1] = '\0';

    /* Stream: one chunk per word (the space travels with the word before it). Check cancel between
     * chunks, as a real engine would between tokens. */
    for (start = 0, i = 0; i < need - 1; ++i) {
        if (inst->reply[i] != ' ' && i + 1 < need - 1) continue;
        if (req->cancel && *req->cancel) {
            snprintf(inst->error, sizeof(inst->error), "echo: cancelled");
            return AGENT_ERR_CANCELLED;
        }
        if (req->on_chunk) req->on_chunk(req->user_data, AGENT_CHUNK_TEXT, inst->reply + start, i + 1 - start);
        start = i + 1;
    }

    /* Never assign a whole agent_response: agent_response_reset keeps the host's struct_size. */
    agent_response_reset(resp);
    resp->message.role.data = "assistant";
    resp->message.role.size = 9;
    memset(&inst->part, 0, sizeof(inst->part));
    inst->part.kind = AGENT_PART_TEXT;
    inst->part.text.data = inst->reply;
    inst->part.text.size = need - 1;
    resp->message.parts = &inst->part;
    resp->message.n_parts = 1;
    resp->usage.prompt_tokens = (int32_t)text.size;
    resp->usage.completion_tokens = 1;
    return AGENT_OK;
}

static void echo_free_response(void *instance, agent_response *resp) {
    (void)instance;
    agent_response_reset(resp);
}

static const char *echo_last_error(void *instance) { return ((echo_instance *)instance)->error; }

static const agent_engine_v1 kEchoEngine = {
    AGENT_ENGINE_ABI_VERSION,
    sizeof(agent_engine_v1), /* struct_size as compiled: the host refuses tables below the v2 baseline */
    "echo",
    "0.1.0",
    AGENT_CAP_TEXT_GEN,
    0, /* flags: run() is serialized by the host; no native tool formatting */
    echo_create,
    echo_destroy,
    echo_run,
    echo_free_response,
    echo_last_error,
    NULL, /* count_tokens (optional) */
    NULL, /* reset_cache (optional) */
};

AGENT_ENGINE_EXPORT const agent_engine_v1 *agent_engine_get_v1(void) { return &kEchoEngine; }
