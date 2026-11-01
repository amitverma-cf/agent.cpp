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

## Windows (written, untested)

The child is created suspended, assigned to a **Job Object** and resumed: kill-on-job-close (the
whole tree dies with the `ChildProcess` / when run_command returns), no breakaway for descendants,
`JOB_OBJECT_LIMIT_PROCESS_MEMORY` = `memory_limit_bytes`, `JOB_OBJECT_LIMIT_ACTIVE_PROCESS` =
`max_processes`, and UI restrictions (no clipboard, desktop switching, global atoms, system
parameter changes).

**Not enforced on Windows:** the filesystem allow-list and network denial. Spawns report a warning
saying so. This code has never been compiled.

AppContainer is the planned way to enforce both, and why it is not done yet:

- An AppContainer process can open only objects whose DACL grants its SID (or `ALL APPLICATION
  PACKAGES`). Every allowed directory (plugin, model, workspace, artifacts) needs an inheritable ACE
  for the container SID: a persistent change to the user's files that must be added before the spawn
  and removed afterwards (or kept, keyed by a stable container name from
  `CreateAppContainerProfile`). Parallel sessions sharing a directory have to coordinate the ACEs.
- `C:\Windows` and `Program Files` are already readable through `ALL APPLICATION PACKAGES`; the
  user profile, other drives and temp directories are not, which is the point.
- The network is denied unless the `internetClient` / `privateNetworkClientServer` capabilities are
  added, so `network = Allow` maps to those; loopback needs a firewall exemption
  (`NetworkIsolationSetAppContainerConfig`), so a local MCP server over TCP would not work without it.
- The child is started with `STARTUPINFOEX` + `PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES`. The same
  attribute list should carry `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`: today `CreateProcessW(...,
  bInheritHandles = TRUE, ...)` lets the child inherit *every* inheritable handle of the parent, not
  only its three pipe ends (the Linux path marks everything else close-on-exec).

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
