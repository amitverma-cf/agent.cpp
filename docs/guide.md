# Usage Guide

This guide covers every major feature of agent.cpp with concrete code examples. For API signatures see [API Reference](api.md). For how things work internally see [Architecture](architecture.md).

---

## Workspace layout (required)

Every session requires `workspace_dir`. `init()` creates this structure automatically:

```
.workspace/
  models/    <-- model files (.gguf, .onnx, ...)
  memory/    <-- SQLite database (db.sqlite3)
  logs/      <-- structured log files (if enable_file_logging = true)
  tmp/       <-- scratch files for tools
```

All relative paths in filesystem and terminal tool calls resolve inside `workspace_dir`. The path is validated at `init()` for shell metacharacters — it must not contain `"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, or newlines.

---

## Session lifecycle

A `Session` is created once, used for the lifetime of a conversation (or many conversations), and destroyed implicitly when it goes out of scope. Sessions are move-only — they cannot be copied.

```cpp
agent::AgentRuntime runtime;    // init global backends; must outlive all sessions

agent::Config cfg;
cfg.provider      = agent::AiProvider::LlamaCpp;
cfg.model         = ".workspace/models/my.gguf";
cfg.workspace_dir = ".workspace";
cfg.context_window = 4096;
cfg.max_tokens    = 512;
cfg.temperature   = 0.7f;

auto res = agent::init(cfg);
if (!res.ok) {
    fprintf(stderr, "init: %s\n", res.error.message.c_str());
    return 1;
}
agent::Session session = std::move(res.value);

// use session ...

// session destroyed here -> provider state, file handles, memory cleaned up
```

---

## Providers

### Mock

Always available. Returns fixed text. Useful for testing without a model.

```cpp
cfg.provider = agent::AiProvider::Mock;
```

### LlamaCpp

Local on-device inference from GGUF files.

```cpp
cfg.provider       = agent::AiProvider::LlamaCpp;
cfg.model          = ".workspace/models/mistral-7b-q4_k_m.gguf";
cfg.context_window = 8192;
cfg.max_tokens     = 1024;
cfg.temperature    = 0.8f;
```

### OpenAI-compatible

Any HTTP API that speaks the OpenAI Chat Completions format.

```cpp
cfg.provider = agent::AiProvider::OpenAICompatible;
cfg.base_url = "https://api.openai.com";
cfg.api_key  = std::getenv("OPENAI_API_KEY");
cfg.model    = "gpt-4o";
```

Works with Ollama, LM Studio, Together AI, Groq, Mistral, Anyscale, and any other compatible provider.

Network failures and HTTP 429/5xx responses are automatically retried up to `retry_max_attempts` times (default 3) with exponential backoff starting at `retry_base_delay_ms` (default 500 ms).

### ONNX Runtime

Multimodal models (image, audio, embeddings) via ONNX Runtime.

```cpp
cfg.provider = agent::AiProvider::OnnxRuntime;
cfg.model    = ".workspace/models/clip-vit-b32.onnx";
```

---

## Low-level inference: infer

`infer` is stateless. You build an `InferRequest` with a span of `MessageView`s and get an `InferResponse` back. The response views are arena-backed and valid only until the next call to `infer`/`run_turn` on the same session (or the same thread's worker arena slot, under `AgentScheduler`).

```cpp
agent::MessageView msg;
msg.role    = "user";
msg.content = "What is the capital of France?";

agent::InferRequest req;
req.messages = std::span<const agent::MessageView>(&msg, 1);
req.stream   = false;

auto res = agent::infer(session, req);
if (res.ok)
    printf("%s\n", std::string(res.value.message.content).c_str());
```

Streaming:

```cpp
req.stream          = true;
req.on_token        = [](std::string_view tok, void*) { fwrite(tok.data(), 1, tok.size(), stdout); };
req.token_user_data = nullptr;
agent::infer(session, req);
```

---

## High-level conversations: run_turn

`run_turn` manages everything: history accumulation, context pruning, automatic tool call loops (ReAct), and memory persistence. It returns an owned `std::string` — the assistant's final reply.

```cpp
agent::FlowMemory conv;
conv.id = "my_session";        // used as memory key "history:my_session"

// optional: seed with a system message
agent::Message sys;
sys.role    = "system";
sys.content = "You are a helpful assistant with access to shell tools.";
conv.history.push_back(std::move(sys));

// multi-turn loop
while (true) {
    std::string input;
    std::cout << "> ";
    if (!std::getline(std::cin, input) || input == "/quit") break;

    auto r = agent::run_turn(session, conv, input,
        true,                                               // stream = true
        [](std::string_view tok, void*){ std::cout << tok; });
    if (!r.ok) {
        fprintf(stderr, "error: %s\n", r.error.message.c_str());
    }
    std::cout << "\n";
}
```

Each call:
1. Appends the user message to `conv.history`.
2. Optionally compresses if context is > 70% full.
3. Prunes old messages to fit the context budget.
4. Calls `infer`.
5. If the model returns tool calls, executes each one and loops back.
6. Appends the final assistant message.
7. Serialises `conv.history` to memory under `"history:<conv.id>"`.
8. Returns the assistant reply as an owned `std::string`.

---

## Custom tools

Tools are plain C function pointers. The callback receives the JSON arguments string and a `void*` user_data pointer, and returns `Result<std::string>`.

For tools backed by a `DataSource` (RAG/SQL/vector search — see [data-sources.md](api/data-sources.md)), use `agent::bind_data_source_tool(session, source, ...)`: it retains the `shared_ptr<DataSource>` on the `Session` for you, so the tool's `user_data` can't outlive the data it points to.

```cpp
struct WeatherCtx { std::string api_key; };
WeatherCtx ctx{ .api_key = "..." };

agent::Tool weather{
    .name             = "get_weather",
    .description      = "Get current weather for a city.",
    .parameter_schema =
        R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
    .callback = [](std::string_view args, void *ud) -> agent::Result<std::string> {
        auto *c = static_cast<WeatherCtx*>(ud);
        // parse args, call API, return JSON result
        return agent::ok(std::string(R"({"temp":22,"condition":"Sunny"})"));
    },
    .user_data = &ctx,   // must outlive the session
};

cfg.tools = { weather };    // auto_default_tools adds filesystem/terminal too
```

Disable default tools:

```cpp
cfg.auto_default_tools = false;
cfg.tools = { weather };    // only this tool
```

After `init()`, rebuild the O(1) dispatch index if you modify `session.config.tools` at runtime:

```cpp
session.config.tools.push_back(new_tool);
agent::rebuild_tool_index(session);
```

---

## Permissions

Every tool call — native, `bind_data_source_tool`-backed, or a skill's `use_skill` — passes through a per-call Allow/Ask/Deny gate before its callback runs, checked in `execute_tool` right before dispatch. By default `Config::permission_check` is `nullptr`, meaning **everything is allowed**, matching prior behavior exactly.

```cpp
agent::PermissionDecision my_policy(std::string_view tool_name, std::string_view arguments, void *) {
    if (tool_name == "delete_path")
        return agent::PermissionDecision::Deny;      // never allowed
    if (tool_name == "run_command")
        return agent::PermissionDecision::Ask;        // ask the user first
    return agent::PermissionDecision::Allow;          // everything else proceeds
}

bool confirm_with_user(std::string_view tool_name, std::string_view arguments, void *) {
    printf("Allow %s(%s)? [y/N] ", std::string(tool_name).c_str(), std::string(arguments).c_str());
    return getchar() == 'y';
}

cfg.permission_check = my_policy;
cfg.permission_prompt = confirm_with_user;   // only consulted when my_policy returns Ask
```

`Ask` with no `permission_prompt` set is treated as `Deny` — there's no silent fallback to Allow. A denied or declined call returns `ErrorCode::PermissionDenied` without ever invoking the tool's callback (not just skipping its result — the callback genuinely never runs).

A ready-made example policy ships as `agent::tools::default_permission_policy` — Ask before `run_command`/`delete_path`/`move_path`, Allow everything else. It's not wired in automatically; opt in explicitly:

```cpp
cfg.permission_check = agent::tools::default_permission_policy;
```

This composes with, but doesn't replace, sandbox enforcement (`Config::sandbox_filesystem`): the permission gate decides *whether* a call is attempted; the filesystem sandbox constrains *what paths* a filesystem tool can touch once it runs. Neither can stop a tool's own native code from doing something outside agent.cpp's view entirely — a `Tool::callback` is a plain function pointer with the full privileges of the host process. If a tool's code isn't fully trusted, the permission gate (ask/deny before it's ever called) is the mitigation this library provides; there's no additional runtime isolation layer underneath it.

---

## Default tools reference

When `auto_default_tools = true`, these tools are registered automatically:

| Tool | Required args | Optional args | Returns |
|---|---|---|---|
| `read_file` | `path` | `max_bytes` (default 16384) | file content string |
| `write_file` | `path`, `content` | `create_dirs` (default true) | confirmation string |
| `append_file` | `path`, `content` | — | confirmation string |
| `list_dir` | — | `path` (default: workspace root) | JSON array of `{name,type,size}` |
| `create_dir` | `path` | — | confirmation string |
| `delete_path` | `path` | — | confirmation string |
| `move_path` | `from`, `to` | — | confirmation string |
| `file_info` | `path` | — | JSON `{exists, type, size, path}` |
| `run_command` | `command` | `timeout_seconds` (default 30) | JSON `{exit_code, output}` |
| `compress_context` | — | `keep_recent` (default 6) | `"Context compressed."` |

---

## Skills

Skills are [SKILL.md](https://agentskills.io)-format capability packages: a directory with a `SKILL.md` file (YAML frontmatter — at minimum `name` and `description` — plus free-form Markdown instructions). This is an open, cross-vendor format; skills authored for Claude Code, Codex CLI, or Gemini CLI work here unmodified.

Set `Config::skills_dir` to a directory containing one subdirectory per skill:

```
my_skills/
  pdf-tools/
    SKILL.md
  xlsx-tools/
    SKILL.md
```

```cpp
cfg.skills_dir = "my_skills";
auto sess = agent::init(cfg);
```

At `init()`, each skill's `name`+`description` (only — not the full body) is folded into the tools system prompt, and a `use_skill` tool is registered automatically. The model calls `use_skill({"name": "pdf-tools"})` to load a skill's full `SKILL.md` content only when it's actually relevant — this progressive-disclosure shape keeps unused skills' token cost to a couple of lines each.

You can also drive this directly, e.g. from your own application code or a `FlowState::context_provider`:

```cpp
auto res = agent::load_skills_dir(session, "my_skills");   // populate session.skills + register use_skill
auto body = agent::read_skill_body(session, "pdf-tools");  // full SKILL.md content on demand
```

Like `session.config.tools`, `session.skills` isn't mutex-guarded: call `load_skills_dir` during single-threaded setup (at `init()`, or before starting `AgentScheduler` dispatch), not concurrently with an active turn or scheduler `pump()` on another thread.

---

## FSM executor

The FSM executor drives multi-state agent workflows. Each state declares:
- `system_prompt` — injected as the system message for the duration of that state only.
- `allowed_tools` — restricts the session's tool set; empty = all tools.
- `hooks` — pre/post callbacks around tool calls in that state.
- `on_transition` — the function that does the actual LLM call(s) and returns the next state ID.

```cpp
static const std::string_view kResearchTools[] = { "read_file", "run_command" };
static const std::string_view kWriteTools[]    = { "write_file", "append_file" };

agent::Result<std::string_view> research_fn(
        agent::Session &session, agent::FlowMemory &conv, void *) {
    auto r = agent::run_turn(session, conv,
        "Research the topic and gather all relevant information.");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("write"));
}

agent::Result<std::string_view> write_fn(
        agent::Session &session, agent::FlowMemory &conv, void *) {
    auto r = agent::run_turn(session, conv,
        "Write a comprehensive report based on your research.");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("exit"));
}

std::vector<agent::FlowState> states = {
    {
        .state_id      = "research",
        .system_prompt = "You are a research assistant. Gather facts, read files, run commands.",
        .allowed_tools = kResearchTools,
        .on_transition = research_fn,
    },
    {
        .state_id      = "write",
        .system_prompt = "You are a technical writer. Produce a clear, structured report.",
        .allowed_tools = kWriteTools,
        .on_transition = write_fn,
    },
};

agent::Flow flow;
flow.states    = states;
flow.memory.id = "report_agent";

auto res = agent::run_flow(session, nullptr, flow);
if (!res.ok) fprintf(stderr, "FSM failed: %s\n", res.error.message.c_str());
```

The FSM can also be stepped manually — useful for interactive flows or when you want to inspect the conversation between state transitions:

```cpp
while (true) {
    auto step = agent::step_flow(session, nullptr, flow);
    if (!step.ok) { /* handle error */ break; }
    if (!step.value) { /* FSM finished */ break; }
    // inspect flow.memory.history here
}
```

---

## FSM hooks (pre/post tool calls)

Hooks fire before (`Pre`) or after (`Post`) each tool call while a particular FSM state is active. Use them for logging, auditing, or modifying tool results.

```cpp
static void audit_pre(agent::Session &, agent::FlowMemory &,
                      std::string_view name, std::string_view args,
                      std::string_view result, void *) {
    printf("[PRE]  tool=%s args=%s\n", std::string(name).c_str(), std::string(args).c_str());
}

static void audit_post(agent::Session &, agent::FlowMemory &,
                       std::string_view name, std::string_view args,
                       std::string_view result, void *) {
    printf("[POST] tool=%s result=%s\n", std::string(name).c_str(), std::string(result).c_str());
}

static const agent::FlowHook kHooks[] = {
    { .when = agent::FlowHook::When::Pre,  .tool_name = "", .callback = audit_pre  },
    { .when = agent::FlowHook::When::Post, .tool_name = "", .callback = audit_post },
};

agent::FlowState my_state{
    .state_id      = "audited",
    .hooks         = kHooks,
    .on_transition = my_transition_fn,
};
```

`tool_name` is empty in the examples above, which means "fire for every tool call". Set it to a specific tool name to narrow the hook:

```cpp
{ .when = agent::FlowHook::When::Post, .tool_name = "run_command", .callback = log_commands }
```

---

## Parallel subagents with scheduling

`AgentScheduler` runs multiple `Flow`s. For `AiProvider::OpenAICompatible`, subagents run with real thread-pooled parallelism (bounded by `Config::max_parallel_subagents`); for every other provider (a local model can only decode one sequence at a time), subagents are interleaved in a cooperative round-robin instead. All subagents share the same `Session`. See [docs/api/scheduler.md](api/scheduler.md) for details.

```cpp
agent::AgentScheduler scheduler;

// Spawn two parallel research subagents
agent::Flow agent_a;
agent_a.states    = states_a;
agent_a.memory.id = "worker_a";
scheduler.spawn("worker_a", std::move(agent_a));

agent::Flow agent_b;
agent_b.states    = states_b;
agent_b.memory.id = "worker_b";
scheduler.spawn("worker_b", std::move(agent_b));

// Add a cron task that fires every 10 seconds
scheduler.add_cron("heartbeat",
    [](agent::Session &s, void *) -> agent::Result<void> {
        agent::utils::log(s, agent::LogLevel::Info, "scheduler heartbeat");
        return agent::ok();
    },
    nullptr,
    std::chrono::seconds(0),      // delay before first fire
    std::chrono::seconds(10));    // repeat interval (0 = one-shot)

// Run until all subagents finish
auto res = scheduler.run_until_done(session);

// Collect results
for (const auto &r : scheduler.results()) {
    printf("subagent '%s': %s  (turns=%d, tool_calls=%d)\n",
           r.id.c_str(),
           r.ok ? "OK" : r.error.message.c_str(),
           r.stats.turns,
           r.stats.tool_calls);
}
```

You can also drive the scheduler manually in your own loop — useful when you need to interleave scheduler ticks with other work:

```cpp
while (!scheduler.all_done()) {
    auto res = scheduler.pump(session);
    if (!res.ok) break;
    // do other work here
}
```

---

## Cron tasks

Cron tasks are standalone callbacks attached to the scheduler. A one-shot task fires once after a delay; a recurring task fires repeatedly on an interval.

```cpp
// One-shot: fire after 5 seconds
scheduler.add_cron("init_data",
    [](agent::Session &s, void *ctx) -> agent::Result<void> {
        auto *data = static_cast<MyData*>(ctx);
        // load initial data
        return agent::ok();
    },
    &my_data,
    std::chrono::seconds(5),
    std::chrono::milliseconds(0));   // interval = 0 means one-shot

// Recurring: fire every minute
scheduler.add_cron("monitor",
    [](agent::Session &s, void *) -> agent::Result<void> {
        // check system health
        return agent::ok();
    },
    nullptr,
    std::chrono::seconds(0),
    std::chrono::minutes(1));

// Cancel a cron task
scheduler.cancel_cron("monitor");
```

---

## Event hooks

Event hooks observe lifecycle events without modifying behaviour. They are useful for telemetry, logging, and dashboards.

```cpp
agent::register_event_hook(session, agent::EventType::OnToolCall,
    [](const agent::Event &ev, void *) {
        std::string name(reinterpret_cast<const char*>(ev.payload.data()), ev.payload.size());
        printf("tool called: %s\n", name.c_str());
    });

agent::register_event_hook(session, agent::EventType::OnInferenceComplete,
    [](const agent::Event &, void *) {
        printf("inference done\n");
    });
```

Available event types: `OnTurnStart`, `OnInferenceSubmit`, `OnInferenceComplete`, `OnToolCall`, `OnToolResult`, `OnStateTransition`, `OnCompression`, `OnPrune`, `OnCronFire`, `OnSubAgentComplete`.

---

## Memory

Conversation history is automatically persisted to the memory backend after every turn (under key `"history:<conv.id>"`, plus a `"history_stats:<conv.id>"` sidecar for `FlowMemory::stats`) via `save_flow_memory`. To resume a conversation after a process restart, load it back explicitly:

```cpp
agent::FlowMemory conv;
conv.id = "my_session";
auto load_res = agent::load_flow_memory(session, conv);   // fills conv.history + conv.stats
if (!load_res.ok) {
    // no saved history yet (or it failed to parse) -- start fresh
}
```

`save_flow_memory`/`load_flow_memory` are also available for calling directly if you want to snapshot or restore a `FlowMemory` outside the normal `run_turn` flow.

You can also use the memory API directly for arbitrary key-value storage:

```cpp
// Store
agent::store_memory(session, "user:preferences", R"({"theme":"dark"})");

// Retrieve
auto r = agent::retrieve_memory(session, "user:preferences");
if (r.ok) printf("%s\n", r.value.c_str());

// Clear everything
agent::clear_memory(session);
```

### Sqlite (the only provider)

A bounded in-memory write-back cache in front of a SQLite database. Reads/writes always hit the cache first; a background thread flushes dirty entries on a size-or-time trigger. See [docs/api/memory.md](api/memory.md) for the full design.

```cpp
// Durable, on disk:
cfg.memory = {
    .provider = agent::MemoryProvider::Sqlite,
    .db_path  = ".workspace/memory/db.sqlite3",  // auto-set if left as the default
};

// Ephemeral, for tests/short-lived agents -- data is lost when the session is destroyed:
cfg.memory = { .provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:" };
```

---

## Context compression

Compression is triggered automatically at 70% context fill, but you can also call it manually:

```cpp
// Compress, keeping the 6 most recent messages verbatim
agent::compress_context(session, conv);

// Keep only 2 recent messages
agent::compress_context(session, conv, 2);
```

The LLM is asked to summarize the oldest messages into a short `[Compressed history]: ...` entry. If the summarization call fails, the function returns `ok()` and the sliding-window pruner handles overflow.

---

## KV cache management

For llama.cpp, clearing the KV cache forces a full context rebuild on the next turn. This trades speed for memory:

```cpp
// Clear after a major topic shift to free VRAM
agent::clear_provider_kv_cache(session);
```

For other providers this is a no-op.

---

## Session statistics

```cpp
agent::SessionStats stats = agent::get_stats(session);
printf("prompt tokens:     %d\n", stats.total_prompt_tokens);
printf("completion tokens: %d\n", stats.total_completion_tokens);
printf("tool calls:        %d\n", stats.total_tool_calls);
printf("turns:             %d\n", stats.total_turns);
printf("compressions:      %d\n", stats.total_compressions);
printf("prune cycles:      %d\n", stats.total_prune_cycles);

agent::reset_stats(session);    // zero all counters
```

Per-conversation stats are on `FlowMemory::stats` and are updated automatically by `run_turn`.

---

## File logging

```cpp
cfg.enable_file_logging = true;    // creates workspace_dir/logs/agent_YYYY-MM-DD_HH-MM-SS.log
```

You can also supply a custom logger callback for real-time log capture:

```cpp
cfg.logger = [](agent::LogLevel lvl, std::string_view msg, void *) {
    const char *prefix = lvl == agent::LogLevel::Error   ? "ERROR" :
                         lvl == agent::LogLevel::Warning ? "WARN"  :
                         lvl == agent::LogLevel::Debug   ? "DEBUG" : "INFO";
    fprintf(stderr, "[%s] %s\n", prefix, std::string(msg).c_str());
};
```

Both can be active simultaneously. The custom logger fires first, then the file write occurs.

---

## Error handling

Every fallible function returns `Result<T>`. There are no exceptions anywhere in the library.

```cpp
auto res = agent::run_turn(session, conv, "hello");
if (!res.ok) {
    switch (res.error.code) {
    case agent::ErrorCode::ToolCallLimitExceeded:
        printf("agent got stuck in a tool loop\n");
        break;
    case agent::ErrorCode::NetworkError:
        printf("network failed: %s\n", res.error.message.c_str());
        break;
    default:
        printf("error [%d]: %s\n", (int)res.error.code, res.error.message.c_str());
    }
}
```

Full error code list: `Ok`, `InvalidConfig`, `UnsupportedProvider`, `FilesystemError`, `ProviderInitFailed`, `ModelLoadFailed`, `DecodeFailed`, `NetworkError`, `HttpError`, `ParseError`, `Cancelled`, `KeyNotFound`, `ToolCallLimitExceeded`, `SandboxViolation`, `PermissionDenied`.
