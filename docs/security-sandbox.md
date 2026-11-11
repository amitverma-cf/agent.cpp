# OS-level sandbox

The permission gate (`Config::permission_check`) decides *whether* a tool call runs; the path checks
of the filesystem tools (`Config::sandbox_filesystem`) cover only those tools. Neither limits what a
child process can touch once it runs: a shell command, an engine plugin, or an MCP server. The OS
sandbox does. It is applied to three kinds of children:

| Child | Policy | Default |
|---|---|---|
| `agent-engine-host` (`EngineIsolation::Process`) | `Config::engine_sandbox` | **on**: plugin + model read-only, workspace read-only, `workspace/artifacts` read-write, no network, no `/tmp` |
| `run_command` (default tool set) | `Config::tool_sandbox` | **on**: workspace read-write, system dirs read-only, `/tmp` read-write, network allowed |
| MCP servers | `McpServerConfig::sandbox` (`"sandbox"` per server in [agent.json](api/agent-json.md#mcp_servers)) | **off** (opt-in per server) |

In-process engines are not sandboxed (they are the calling process). Neither is anything your own
tools start.

## Policy

```cpp
enum class SandboxNetwork { Allow, Deny };

struct SandboxPolicy {
    bool enabled = false;
    bool filesystem = true;                     // false: no filesystem rules at all
    std::vector<std::string> read_only_paths;   // extra, on top of what each use adds
    std::vector<std::string> read_write_paths;
    bool system_paths = true;                   // /usr, /lib*, selected /etc files, /proc, /sys, /dev basics
    bool allow_tmp = true;                      // $TMPDIR (or /tmp) and /var/tmp read-write
    SandboxNetwork network = SandboxNetwork::Allow;
    uint64_t memory_limit_bytes = 0;            // RLIMIT_AS / Job Object memory, 0 = none
    int max_processes = 0;                      // RLIMIT_NPROC / Job Object process count, 0 = none
    bool gpu = false;                           // GPU device nodes read-write (opt-in, see below)
};

cfg.tool_sandbox   = {.enabled = true};                                                         // default
cfg.engine_sandbox = {.enabled = true, .allow_tmp = false, .network = SandboxNetwork::Deny};    // default
```

What each use adds to its policy's lists:

- **Engine host**: the host executable's directory and `<host dir>/../lib{,64}`; the plugin's
  directory (bundled runtime libraries); the model's directory (or the model directory itself), plus
  the directory of the symlink target when the model is a symlink (Hugging Face cache snapshots);
  every string in `Config::engine_options` that names an existing path (its directory); the
  workspace read-only (media inputs); `workspace/artifacts` read-write. `memory_limit_bytes` is the
  host's address-space limit. A plugin that needs anything else (a cache directory, a model file
  outside these, temporary files) needs it listed in `engine_sandbox.read_only_paths` /
  `read_write_paths`, or `engine_sandbox.allow_tmp = true`.
- **run_command**: the workspace read-write. With `Config::sandbox_filesystem = false` the
  filesystem part is off (`filesystem = false`), the rest still applies. Stdin is `/dev/null` and the
  command runs in its own session (no controlling terminal, so no `TIOCSTI` keystroke injection).
  The network is allowed by default because package managers, `git` and `curl` are common tool
  uses; set `cfg.tool_sandbox.network = SandboxNetwork::Deny` to block it. The home directory is
  **not** readable: `git` (which reads `~/.gitconfig`), `npm`, `pip` and friends need their config
  and cache directories listed explicitly.
- **MCP server** (when `sandbox.enabled`): the program's directory (and its symlink target's
  directory) read-only and `working_directory` read-write. Servers launched through `npx`/`uvx`
  need their runtime (`node`/`python` prefix), package cache (`~/.npm`, `~/.cache/uv`) and whatever
  directories they serve listed. That is why it is off by default.

Setting `enabled = false` restores the old, unrestricted behaviour for that child.

## Linux (enforced and tested)

Applied in the child after `fork()` and before `exec()` (`hal::apply_prepared_sandbox`; the
Landlock ruleset and the seccomp program are built in the parent, the child only makes raw
syscalls):

1. `setsid()`: a new session and process group. Terminating the child (`ChildProcess::terminate`)
   signals the whole group.
2. `RLIMIT_CORE = 0`; `RLIMIT_AS = memory_limit_bytes`; `RLIMIT_NPROC = max_processes`.
3. Every inherited descriptor above 2 is marked close-on-exec (`close_range(CLOSE_RANGE_CLOEXEC)`).
4. `PR_SET_NO_NEW_PRIVS`: set-uid binaries (`sudo`, `su`, `ping` on some systems) cannot gain
   privileges.
5. **Landlock** (`landlock_restrict_self`), the ABI detected at runtime:
   - ABI 1+: filesystem allow-list. Read-only paths get execute/read-file/read-dir; read-write paths
     get every filesystem right the kernel knows (create, remove, rename across dirs (ABI 2),
     truncate (ABI 3), device ioctls (ABI 5)). Access is checked on the resolved inode, so absolute
     paths, `..` and symlinks cannot escape.
   - ABI 4+: TCP bind/connect are denied when `network = Deny` (defence in depth next to seccomp).
   - ABI 6+: the child cannot signal processes outside its sandbox, and with `network = Deny` cannot
     connect to abstract Unix sockets outside it (D-Bus, X11, ...).
6. **seccomp-bpf** when `network = Deny`: `socket()` for any family other than `AF_UNIX` fails with
   `EACCES`; `io_uring_setup/enter/register` fail with `ENOSYS` (io_uring can create and connect
   sockets without the `socket` syscall); syscalls of a foreign ABI (i386 / x32 on x86_64) fail with
   `ENOSYS`. Everything else is allowed: this is a network filter, not a syscall allow-list.
   x86_64 and aarch64 only.

The system read-only list is `/usr /bin /sbin /lib /lib32 /lib64 /libx32 /opt /nix/store /snap /proc
/sys`, the dynamic loader files in `/etc` (`ld.so.*`), `passwd`, `group`, `nsswitch.conf`,
`localtime`, `alternatives`, locale and GPU ICD directories, and, when the network is allowed, the
resolver and TLS files (`hosts`, `resolv.conf`, `/etc/ssl`, `/run/systemd/resolve`, ...). Not
`/etc/shadow`, not `/home`, not `/root`. Read-write devices: `/dev/null`, `/dev/zero`, `/dev/full`,
`/dev/random`, `/dev/urandom`, `/dev/shm`. See `hal::sandbox_system_read_only_paths()`.

**GPUs are opt-in** (`SandboxPolicy::gpu`, default `false` for every child): with it, `/dev/dri`
(DRM card and render nodes: Vulkan, Intel, ROCm), `/dev/kfd` (ROCm compute) and every `/dev/nvidia*`
node (`nvidiactl`, `nvidia-uvm`, `nvidia0`, ...) become read-write. GPU drivers are a large kernel
attack surface, so a CPU engine should not be able to reach them. The driver libraries (`/usr/lib`),
ICD files (`/etc/vulkan`, `/etc/OpenCL`, `/usr/share/vulkan`) and `/sys` / `/proc` (device
enumeration) are readable either way. A CUDA, ROCm or Vulkan build of llama.cpp / sd.cpp, or ORT
with the CUDA provider, needs `cfg.engine_sandbox.gpu = true`; the `agent` CLI sets it whenever the
engine's `backend` in `engines.json` (or `config.json`) is anything but `"cpu"`. CUDA also keeps a
JIT cache in `~/.nv/ComputeCache` (`$CUDA_CACHE_PATH`): without a read-write entry for it the cache
is simply not written (kernels are re-JITed per start). Untested: this machine has no GPU.

**Graceful degradation.** A kernel without Landlock (older than 5.13, or Landlock not in the active
LSM list) or without seccomp still spawns the child; what could not be enforced is returned as
warnings (`ChildProcess::sandbox_warnings()`), logged through `Config::logger` for the engine host
and MCP servers, and reported in run_command's result as a `"sandbox_warning"` field. Detection
uses the `landlock_create_ruleset(..., LANDLOCK_CREATE_RULESET_VERSION)` syscall, not
`/sys/kernel/security/lsm` (securityfs is often not mounted in containers).

**Caveats.**

- `RLIMIT_NPROC` counts every process and thread of the *user*, not just this child's tree, and root
  ignores it. Leave `max_processes = 0` unless the agent runs as a dedicated user.
- `RLIMIT_AS` limits virtual address space, not resident memory. Engines that `mmap` large models
  (llama.cpp) need a limit well above the model size. AddressSanitizer reserves terabytes of shadow
  memory, so an ASan-built `agent-engine-host` cannot start under any practical `memory_limit_bytes`.
- Descriptors are closed on exec, not before it: anything the parent passed deliberately (stdio)
  stays. A socket inherited as stdio would still work under the network filter.
- `/proc` is readable (needed for CPU detection and sanitizers). Other users' processes are
  protected by the usual `/proc` permissions, not by the sandbox.

## macOS (written, untested)

The same `setsid` / rlimit / close-on-exec steps, then `sandbox_init()` with an SBPL profile:
`(allow default)`, `(deny file-read* file-write*)`, `(allow file-read-metadata)` (stat / realpath /
dyld walk the parents of allowed paths; names and attributes stay visible, contents do not), then
the allow-lists re-allowed as `subpath` rules on their canonical paths (`/tmp` becomes
`/private/tmp`; system list: `/System`, `/usr`, `/bin`, `/sbin`, `/Library`, `/opt/homebrew`,
`/private/etc`, `/private/var/db/dyld`, `/dev`, ...), and `(deny network*)` except Unix sockets when
the network is denied. `sandbox_init` is deprecated but still present; the profile has never been
run. Every sandboxed spawn reports the warning "the macOS sandbox_init() profile is untested".

Known gaps, from review only:

- `sandbox_init()` runs in the child between `fork()` and `exec()`. It compiles the profile, which
  allocates; in a multithreaded parent that is not async-signal-safe and can deadlock on a malloc
  lock held by another thread at fork time. The fix is to exec through `/usr/bin/sandbox-exec -p
  <profile>` (deprecated too) or to compile the profile in the parent (private
  `sandbox_compile_string` / `sandbox_apply`).
- `SandboxPolicy::gpu` does nothing: Metal goes through IOKit / Mach services, which `(allow
  default)` leaves open anyway. Mach lookups, IPC and `process-exec` are not restricted at all.
- `RLIMIT_AS` is not enforced by the macOS kernel for most allocations.

## Windows

The child is created suspended, assigned to a **Job Object** and resumed: kill-on-job-close (the
whole tree dies with the `ChildProcess` / when run_command returns), no breakaway for descendants,
`JOB_OBJECT_LIMIT_PROCESS_MEMORY` = `memory_limit_bytes`, `JOB_OBJECT_LIMIT_ACTIVE_PROCESS` =
`max_processes`, and UI restrictions (no clipboard, desktop switching, global atoms, system
parameter changes).

**Reads, writes and network (MCP servers and `run_command`, tested on Windows 11).** With `SandboxPolicy::windows_low_integrity`
(the name is kept for compatibility; it means "confine the files of this child") the child runs in an **AppContainer**, the
mechanism Windows uses for its own sandboxed apps (no third-party sandbox library does this better: the C++ ones are Linux-only).
The container's SID is added to the DACL of each folder the policy names: the workspace and `read_write_paths` get read, write and
delete, `read_only_paths` and the folder of the program being started get read and execute, both inherited by what is created
below. Everything else the user owns, such as the rest of the profile, other drives, other projects and the SSH keys, refuses even a
read, while `C:\Windows` and `Program Files` stay readable (they grant `ALL APPLICATION PACKAGES`). A command that needs a tool
installed outside those places (Git's `cat`, a Node install under the profile) is refused until the folder is granted, which is
what the permission gate asks the user about (`run_command` names its absolute paths; a grant is a folder added to
`read_only_paths` or `read_write_paths`). The network follows `SandboxPolicy::network`: `Allow` adds the `internetClient` and
`privateNetworkClientServer` capabilities, `Deny` adds none and every connection fails; loopback (127.0.0.1) is blocked by Windows
for any AppContainer. `test_sandbox_windows.cpp` covers the write, read and grant cases; `AGENT_TEST_NETWORK=1` also runs the
network pair (curl succeeds with `Allow`, fails with `Deny`).

If the container or an ACL cannot be set up (a folder the user cannot change, such as one an administrator made under `C:\`),
the child falls back to the **Low integrity level** (writes confined to the allowed folders; reads and network not restricted)
and `sandbox_warnings` says so. Engine hosts leave the flag off: they need more of the system than a container allows, so they get
the Job Object only.

The container grants are kept: the profile `agent-cpp.sandbox` and the ACEs stay on the folders after a run, so a later run is
quick. Removing them is `icacls <folder> /remove "<container sid>"`, or deleting the folder.

Still open on Windows: `bInheritHandles = TRUE` lets the child inherit every inheritable handle of the parent, not only its three
pipe ends (add `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` to the same attribute list).
Other review notes: a failed `AssignProcessToJobObject` (e.g. a parent job that forbids nesting on
Windows 7) degrades to a warning and the child runs without limits; `SandboxPolicy::gpu` has no
meaning (no device files).

## Verifying

`tests/unit/test_sandbox.cpp` (Linux; Landlock and seccomp cases SKIP when unavailable):

- a sandboxed `cat` reads an allowed file but not a secret outside the list, nor `/etc/shadow`;
- writes outside the read-write paths fail, `/dev/null` keeps working;
- under the network filter `socket(AF_INET)` / `AF_INET6` fail with `EACCES`, `AF_UNIX` works;
- `RLIMIT_AS` makes a large allocation fail;
- the child leads its own session and has `ulimit -c` 0;
- `run_command` cannot read a file outside the workspace by absolute path, `..` or a symlink created
  inside the workspace, cannot write outside it, and works inside it; with
  `sandbox_filesystem = false` the same read succeeds; with `network = Deny` bash's `/dev/tcp` fails;
- the engine host runs the mock engine under its default policy: the model file is readable,
  `/etc/shadow`, a temp secret and writes outside `artifacts/` are not, `socket(AF_INET)` fails
  (mock triggers `read:<path>`, `write:<path>`, `net`);
- an MCP server with an opted-in sandbox still serves tool calls; `agent.json`'s per-server
  `"sandbox"` object is parsed, validated and reaches the spawn (`test_graph_json.cpp`);
- GPU device nodes are only in the read-write list with `gpu = true`.

`tests/unit/test_engine_sandbox.cpp` runs every real engine through the sandboxed host under the
untouched default policy (gated on the model environment variables; see the next section).

## Real engines under the default engine policy

Every engine in this repository works under the untouched `Config::engine_sandbox` (no extra paths,
no `/tmp`, no network, no GPU) when its files live where the policy looks: the model file's
directory, the plugin's directory (bundled runtime libraries) and any path named in
`engine_options`. Verified on Linux 6.18 (Landlock ABI 7, seccomp) with x86_64 CPU builds on
2026-11-01, by `tests/unit/test_engine_sandbox.cpp` (one real request each through
`agent-engine-host`, asserting that no `sandbox:` warning was logged), and by
`strace -f -e status=failed` over the same run to catch silent fallbacks:

| Engine | Request | Denied accesses seen (all harmless) |
|---|---|---|
| llama (Qwen2.5-0.5B, SmolVLM + mmproj, bge-small, jina reranker) | chat turn; image from the workspace; embed; rerank | none. ggml's CPU feature detection, OpenMP and the model `mmap` stay inside `/proc`, `/sys`, `/usr/lib` and the model directory |
| llama vision, secret file | the same image placed outside every allow-list | the read fails with `EACCES` (`Unable to open file ...: Permission denied`), the turn fails cleanly, the host keeps serving |
| onnx (ORT 1.31.1, generated `Add` model) | `infer()` | ORT's 1DS telemetry: `$TMPDIR/.ses`, `~/.cache/Microsoft/DeveloperTools/.onnxruntime/onnxruntime.db`, `/run/systemd/container`. ORT logs "Failed to persist telemetry device ID; using an in-memory identifier" and carries on. `DisableTelemetryEvents()` cannot avoid it: the accesses happen inside `CreateEnv` |
| sherpa-onnx 2.0.1 (Kokoro TTS with espeak-ng data, Whisper tiny, Silero VAD, pyannote + 3D-Speaker diarization) | TTS into `artifacts/`, then ASR, VAD and diarization of that WAV | none. No temporary files: the WAV goes to the request's output directory, espeak-ng data is read from the Kokoro directory |
| sdcpp (SDXS-512 Q8_0) | 64x64, 1-step text-to-image | none |
| trellis (no weights here) | `create()` with an empty and with a garbage model directory | none; the errors name the missing / invalid `dino_f16.gguf`, so the host could see the directory |

What a user may still have to add, all outside what the policy can derive:

- **Model files outside the model's directory and not named in `engine_options`**, e.g. a file
  referenced only from another config file. Paths in `engine_options` are found as JSON strings
  containing `/`; a bare file name relative to the host's working directory is not.
- **GPU builds**: `engine_sandbox.gpu = true` (the CLI derives it from the engine's `backend`).
  CUDA's JIT cache (`~/.nv/ComputeCache`) is not writable; list it read-write to keep it.
- **Symlinks**: a symlinked model (Hugging Face snapshot) gets its target's directory added, and so
  does a symlinked path in `engine_options`. A symlinked *directory* inside the model directory
  pointing elsewhere is not followed.
- **`memory_limit_bytes`** is off by default; when set it must exceed the mapped model size plus
  compute buffers (llama `mmap`s the whole GGUF).

## Permission gate: paths outside the workspace

The sandbox is not an all-or-nothing wall. A sandboxed tool (the file tools, `run_command`, MCP servers) can use a path outside
the workspace when the host allows it, per path and per kind of access (read or write):

- **Grants** (`Config::path_grants`): `{path, write, consumer}` where consumer is `*`, `files`, `run_command` or `mcp:<server>`. A
  grant covers the path and everything below it (symlinks are resolved first), and a read grant never allows a write.
- **The gate** (`Config::path_gate`): asked for any other path outside the workspace. It answers Deny, AllowOnce (asked again next
  time) or AllowAlways (remembered for the session; the host can also save it). `run_command` asks about each absolute path its
  command line names (writes are guessed from `>`, `cp`, `mv`, `rm`, `mkdir`, ...); the file tools ask when a path resolves outside.
  MCP servers get grants only, because their paths are not known when they start.
- **The CLI** keeps grants in `<home>/sandbox-grants.json` (and `<project>/.agents/sandbox.json`) and asks on the terminal:
  `[o] allow once  [a] allow always  [n] no`. Without a terminal the answer is no and the message names the command that grants it:
  `agent sandbox grant PATH [--write] [--for run_command|files|mcp:NAME]`, `agent sandbox ls`, `agent sandbox revoke PATH`.

On Windows a granted folder gets an access entry for the AppContainer, so the same grants work there (reads included). If this user may not change a folder's permissions (it is owned by an administrator, for example under `C:\`), the program falls back to Low integrity and the tool result carries a `sandbox_warning` saying so.
