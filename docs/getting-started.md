# Getting Started

## Prerequisites

- CMake 3.21+
- C++20-capable compiler (GCC 12+, Clang 14+, MSVC 19.30+)
- Ninja (recommended) or Make
- OpenSSL (for networking; install `libssl-dev` on Debian/Ubuntu)

On Windows, build tools must run under WSL (Debian/Ubuntu). Git and file operations run natively on Windows.

## Build

```bash
# Configure (from repo root, run under WSL)
wsl cmake -B build -G Ninja

# Build everything
wsl cmake --build build

# Run tests
cd build && ctest --output-on-failure
```

### Build without llama.cpp (API-only or OpenAI)

```bash
wsl cmake -B build -G Ninja -DAGENT_ENABLE_LLAMACPP=OFF
wsl cmake --build build
```

### Build with ONNX Runtime

```bash
wsl cmake -B build -G Ninja -DAGENT_ENABLE_ONNX=ON
wsl cmake --build build
```

### Enable sanitizers

```bash
wsl cmake -B build -G Ninja -DAGENT_ENABLE_SANITIZERS=ON
wsl cmake --build build
```

## Workspace layout

Every session requires a `workspace_dir`. The library creates this directory structure automatically on `init()`:

```
.workspace/
├── models/      # put your .gguf / .onnx model files here
├── memory/      # SQLite database (db.sqlite3)
├── logs/        # structured log files (if enable_file_logging = true)
└── tmp/         # scratch space for tools
```

Pass the workspace path at session creation:

```cpp
cfg.workspace_dir = ".workspace";   // relative or absolute — no shell metacharacters
```

The path is validated at `init()` time. It must not contain `"`, `'`, `&`, `|`, `;`, `` ` ``, `$`, newlines, or carriage returns, because the terminal tool interpolates it into shell commands.

## Your first program

### Step 1 — link the library

```cmake
find_package(agent REQUIRED)       # or add_subdirectory if vendored
target_link_libraries(my_app PRIVATE agent)
```

### Step 2 — initialise a session

```cpp
#include <agent-cpp/agent.hpp>

agent::AgentRuntime runtime;   // must outlive all sessions

agent::Config cfg;
cfg.provider      = agent::AiProvider::LlamaCpp;
cfg.model         = ".workspace/models/my.gguf";
cfg.workspace_dir = ".workspace";
cfg.context_window = 4096;
cfg.max_tokens    = 512;
cfg.temperature   = 0.7f;

auto res = agent::init(cfg);
if (!res.ok) {
    fprintf(stderr, "init failed: %s\n", res.error.message.c_str());
    return 1;
}
agent::Session session = std::move(res.value);
```

### Step 3 — run a conversation

```cpp
agent::FlowMemory conv;
conv.id = "my_chat";

// Add an optional system message
agent::Message sys;
sys.role = "system";
sys.content = "You are a helpful assistant.";
conv.history.push_back(std::move(sys));

// Each call appends the user message and gets an assistant reply
auto r1 = agent::run_turn(session, conv, "Hello! What can you do?");
if (r1.ok) printf("Agent: %s\n", r1.value.c_str());

auto r2 = agent::run_turn(session, conv, "List the files in my workspace.");
if (r2.ok) printf("Agent: %s\n", r2.value.c_str());
```

### Step 4 — use an FSM for structured agent flows

```cpp
agent::Result<std::string_view> plan_state(
        agent::Session &session, agent::FlowMemory &conv, void *) {
    auto r = agent::run_turn(session, conv, "Make a plan.");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("execute"));
}

agent::Result<std::string_view> execute_state(
        agent::Session &session, agent::FlowMemory &conv, void *) {
    auto r = agent::run_turn(session, conv, "Execute the plan step by step.");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("exit"));
}

std::vector<agent::FlowState> states = {
    {.state_id = "plan",    .on_transition = plan_state},
    {.state_id = "execute", .on_transition = execute_state},
};

agent::Flow flow;
flow.states    = states;
flow.memory.id = "fsm_demo";

auto flow_res = agent::run_flow(session, nullptr, flow);
```

### Step 5 — use the OpenAI-compatible provider

```cpp
cfg.provider = agent::AiProvider::OpenAICompatible;
cfg.base_url = "https://api.openai.com";
cfg.api_key  = std::getenv("OPENAI_API_KEY");
cfg.model    = "gpt-4o";
```

Any endpoint that speaks the OpenAI Chat Completions protocol works (Ollama, LM Studio, Together AI, etc.).

