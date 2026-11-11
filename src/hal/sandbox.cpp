#include "hal/sandbox.hpp"

#include "hal/platform.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>

#if defined(AGENT_HAL_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(AGENT_HAL_LINUX)
#include <dirent.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#endif

#if defined(AGENT_HAL_MACOS)
extern "C" int sandbox_init(const char *profile, uint64_t flags, char **errorbuf);
extern "C" void sandbox_free_error(char *errorbuf);
#endif

namespace agent::hal {

// ------------------------------------------------------------------ PreparedSandbox

PreparedSandbox::~PreparedSandbox() {
#if !defined(AGENT_HAL_WINDOWS)
    if (landlock_fd >= 0) ::close(landlock_fd);
#endif
}

PreparedSandbox::PreparedSandbox(PreparedSandbox &&other) noexcept
    : active(other.active),
      new_session(other.new_session),
      landlock_fd(std::exchange(other.landlock_fd, -1)),
      seccomp(std::move(other.seccomp)),
      memory_limit_bytes(other.memory_limit_bytes),
      max_processes(other.max_processes),
      macos_profile(std::move(other.macos_profile)),
      low_integrity(other.low_integrity),
      low_write_paths(std::move(other.low_write_paths)),
      warnings(std::move(other.warnings)) {}

PreparedSandbox &PreparedSandbox::operator=(PreparedSandbox &&other) noexcept {
    if (this != &other) {
#if !defined(AGENT_HAL_WINDOWS)
        if (landlock_fd >= 0) ::close(landlock_fd);
#endif
        active = other.active;
        new_session = other.new_session;
        landlock_fd = std::exchange(other.landlock_fd, -1);
        seccomp = std::move(other.seccomp);
        memory_limit_bytes = other.memory_limit_bytes;
        max_processes = other.max_processes;
        macos_profile = std::move(other.macos_profile);
        low_integrity = other.low_integrity;
        low_write_paths = std::move(other.low_write_paths);
        warnings = std::move(other.warnings);
    }
    return *this;
}

namespace {

[[maybe_unused]] std::string temp_dir() {
#if defined(AGENT_HAL_WINDOWS)
    std::error_code ec;
    auto p = std::filesystem::temp_directory_path(ec);
    return ec ? std::string() : p.string();
#else
    const char *t = std::getenv("TMPDIR");
    return (t && *t) ? std::string(t) : std::string("/tmp");
#endif
}

} // namespace

// ------------------------------------------------------------------ system paths

std::vector<std::string> sandbox_system_read_only_paths(bool network) {
#if defined(AGENT_HAL_LINUX)
    std::vector<std::string> out = {"/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/opt", "/nix/store", "/snap",
                                    // The dynamic loader, name service, locale and time zone, alternatives symlinks, GPU ICDs.
                                    "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/ld.so.preload", "/etc/localtime",
                                    "/etc/timezone", "/etc/passwd", "/etc/group", "/etc/nsswitch.conf", "/etc/alternatives",
                                    "/etc/locale.alias", "/etc/locale.conf", "/etc/mime.types", "/etc/os-release", "/etc/lsb-release",
                                    "/etc/terminfo", "/etc/OpenCL", "/etc/vulkan", "/etc/glvnd", "/etc/gitconfig",
                                    // Process and hardware introspection (CPU count, /proc/self/maps for sanitizers, ...).
                                    "/proc", "/sys"};
    if (network) {
        for (const char *p : {"/etc/hosts", "/etc/resolv.conf", "/etc/host.conf", "/etc/gai.conf", "/etc/services", "/etc/protocols",
                              "/etc/ssl", "/etc/ca-certificates", "/etc/pki", "/etc/crypto-policies", "/run/systemd/resolve"})
            out.emplace_back(p);
    }
    return out;
#elif defined(AGENT_HAL_MACOS)
    (void)network;
    return {
        "/System", "/usr", "/bin", "/sbin", "/Library", "/opt/homebrew", "/private/etc", "/private/var/db/dyld", "/private/var/db/timezone",
        "/dev"};
#else
    (void)network;
    return {};
#endif
}

std::vector<std::string> sandbox_system_read_write_paths(bool gpu) {
#if defined(AGENT_HAL_LINUX)
    std::vector<std::string> out = {"/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/shm"};
    if (!gpu) return out;
    // DRM render/card nodes (Vulkan, ROCm, Intel), the AMD compute node, and the NVIDIA nodes
    // (/dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm, /dev/nvidia-caps, ...).
    out.emplace_back("/dev/dri");
    out.emplace_back("/dev/kfd");
    if (DIR *d = ::opendir("/dev")) {
        while (dirent *e = ::readdir(d))
            if (std::strncmp(e->d_name, "nvidia", 6) == 0) out.push_back(std::string("/dev/") + e->d_name);
        ::closedir(d);
    }
    return out;
#elif defined(AGENT_HAL_MACOS)
    (void)gpu; // Metal goes through IOKit / Mach services, not device files
    return {"/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"};
#else
    (void)gpu;
    return {};
#endif
}

// ------------------------------------------------------------------ Linux

#if defined(AGENT_HAL_LINUX)

namespace {

// Kernel ABI (include/uapi/linux/landlock.h), restated so older system headers still build.
#ifndef SYS_landlock_create_ruleset
#define SYS_landlock_create_ruleset 444
#endif
#ifndef SYS_landlock_add_rule
#define SYS_landlock_add_rule 445
#endif
#ifndef SYS_landlock_restrict_self
#define SYS_landlock_restrict_self 446
#endif
#ifndef SYS_close_range
#define SYS_close_range 436
#endif
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif

struct LandlockRulesetAttr {
    uint64_t handled_access_fs;
    uint64_t handled_access_net;
    uint64_t scoped;
};

struct __attribute__((packed)) LandlockPathBeneathAttr {
    uint64_t allowed_access;
    int32_t parent_fd;
};

constexpr uint32_t kLandlockCreateRulesetVersion = 1u << 0;
constexpr int kLandlockRulePathBeneath = 1;

constexpr uint64_t kFsExecute = 1ull << 0;
constexpr uint64_t kFsWriteFile = 1ull << 1;
constexpr uint64_t kFsReadFile = 1ull << 2;
constexpr uint64_t kFsReadDir = 1ull << 3;
constexpr uint64_t kFsAbi1 = (1ull << 13) - 1; // EXECUTE .. MAKE_SYM
constexpr uint64_t kFsRefer = 1ull << 13;      // ABI 2
constexpr uint64_t kFsTruncate = 1ull << 14;   // ABI 3
constexpr uint64_t kFsIoctlDev = 1ull << 15;   // ABI 5
constexpr uint64_t kFsReadOnly = kFsExecute | kFsReadFile | kFsReadDir;
// Rights that make sense on a non-directory (anything else is EINVAL on a file rule).
constexpr uint64_t kFsFileRights = kFsExecute | kFsWriteFile | kFsReadFile | kFsTruncate | kFsIoctlDev;

constexpr uint64_t kNetBindTcp = 1ull << 0;    // ABI 4
constexpr uint64_t kNetConnectTcp = 1ull << 1; // ABI 4

constexpr uint64_t kScopeAbstractUnixSocket = 1ull << 0; // ABI 6
constexpr uint64_t kScopeSignal = 1ull << 1;             // ABI 6

constexpr unsigned kCloseRangeCloexec = 1u << 2;

#if defined(__x86_64__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
constexpr uint32_t kNativeArch = AUDIT_ARCH_X86_64;
constexpr bool kSeccompArch = true;
#elif defined(__aarch64__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
constexpr uint32_t kNativeArch = AUDIT_ARCH_AARCH64;
constexpr bool kSeccompArch = true;
#else
constexpr uint32_t kNativeArch = 0;
constexpr bool kSeccompArch = false;
#endif

SandboxSupport probe_support() {
    SandboxSupport s;
    long abi = ::syscall(SYS_landlock_create_ruleset, nullptr, 0, kLandlockCreateRulesetVersion);
    s.landlock_abi = abi > 0 ? static_cast<int>(abi) : 0;
    s.seccomp = kSeccompArch && ::prctl(PR_GET_SECCOMP, 0, 0, 0, 0) >= 0;
    return s;
}

// socket() for anything but AF_UNIX fails with EACCES; io_uring (which can create and connect
// sockets without going through socket()) and syscalls of a foreign ABI (i386 / x32 on x86_64) fail
// with ENOSYS. Everything else is allowed: this filter is about the network, not a syscall allow-list.
std::vector<BpfInsn> network_deny_filter() {
    std::vector<BpfInsn> p;
    auto stmt = [&](uint16_t code, uint32_t k) { p.push_back(BpfInsn{code, 0, 0, k}); };
    auto jump = [&](uint16_t code, uint32_t k) {
        p.push_back(BpfInsn{static_cast<uint16_t>(BPF_JMP | code | BPF_K), 0, 0, k});
        return p.size() - 1;
    };
    std::vector<std::pair<size_t, int>> to_enosys, to_socket; // (insn, 0 = jt / 1 = jf)
    auto ret_errno = [&](int e) { stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (static_cast<uint32_t>(e) & SECCOMP_RET_DATA)); };

    stmt(BPF_LD | BPF_W | BPF_ABS, static_cast<uint32_t>(offsetof(seccomp_data, arch)));
    to_enosys.push_back({jump(BPF_JEQ, kNativeArch), 1}); // foreign arch -> ENOSYS
    stmt(BPF_LD | BPF_W | BPF_ABS, static_cast<uint32_t>(offsetof(seccomp_data, nr)));
#if defined(__x86_64__)
    to_enosys.push_back({jump(BPF_JGE, 0x40000000u), 0}); // x32 ABI
#endif
    to_socket.push_back({jump(BPF_JEQ, static_cast<uint32_t>(__NR_socket)), 0});
    for (long nr :
         {static_cast<long>(__NR_io_uring_setup), static_cast<long>(__NR_io_uring_enter), static_cast<long>(__NR_io_uring_register)})
        to_enosys.push_back({jump(BPF_JEQ, static_cast<uint32_t>(nr)), 0});
    stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);

    const size_t socket_label = p.size();
    // args[0] (domain), low 32 bits on a little-endian machine.
    stmt(BPF_LD | BPF_W | BPF_ABS, static_cast<uint32_t>(offsetof(seccomp_data, args)));
    p.push_back(BpfInsn{static_cast<uint16_t>(BPF_JMP | BPF_JEQ | BPF_K), 1, 0, static_cast<uint32_t>(AF_UNIX)});
    ret_errno(EACCES);
    stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);

    const size_t enosys_label = p.size();
    ret_errno(ENOSYS);

    auto patch = [&](const std::vector<std::pair<size_t, int>> &list, size_t label) {
        for (auto [i, which] : list) {
            const auto off = static_cast<uint8_t>(label - i - 1);
            if (which == 0) p[i].jt = off;
            else p[i].jf = off;
        }
    };
    patch(to_enosys, enosys_label);
    patch(to_socket, socket_label);
    return p;
}

void add_rule(int ruleset, const std::string &path, uint64_t access, uint64_t handled, bool warn_missing,
              std::vector<std::string> &warnings) {
    if (path.empty()) return;
    const int fd = ::open(path.c_str(), O_PATH | O_CLOEXEC);
    if (fd < 0) {
        if (warn_missing)
            warnings.push_back("sandbox: allow-list path '" + path + "' is not accessible (" + std::strerror(errno) + "); skipped.");
        return;
    }
    struct stat st {};
    if (::fstat(fd, &st) == 0 && !S_ISDIR(st.st_mode)) access &= kFsFileRights;
    access &= handled;
    if (access) {
        LandlockPathBeneathAttr attr{access, fd};
        if (::syscall(SYS_landlock_add_rule, ruleset, kLandlockRulePathBeneath, &attr, 0) != 0)
            warnings.push_back("sandbox: cannot add a Landlock rule for '" + path + "' (" + std::strerror(errno) + ").");
    }
    ::close(fd);
}

} // namespace

SandboxSupport sandbox_support() {
    static const SandboxSupport s = probe_support();
    return s;
}

Result<PreparedSandbox> prepare_sandbox(const SandboxPolicy &policy) {
    PreparedSandbox out;
    if (!policy.enabled) return ok(std::move(out));
    out.active = true;
    out.memory_limit_bytes = policy.memory_limit_bytes;
    out.max_processes = policy.max_processes > 0 ? policy.max_processes : 0;

    const SandboxSupport sup = sandbox_support();
    const bool deny_net = policy.network == SandboxNetwork::Deny;
    const int abi = sup.landlock_abi;

    uint64_t fs = 0;
    if (policy.filesystem && abi >= 1) {
        fs = kFsAbi1;
        if (abi >= 2) fs |= kFsRefer;
        if (abi >= 3) fs |= kFsTruncate;
        if (abi >= 5) fs |= kFsIoctlDev;
    }
    const uint64_t net = (deny_net && abi >= 4) ? (kNetBindTcp | kNetConnectTcp) : 0;
    const uint64_t scoped = abi >= 6 ? (kScopeSignal | (deny_net ? kScopeAbstractUnixSocket : 0)) : 0;

    if (policy.filesystem && abi == 0)
        out.warnings.push_back("sandbox: Landlock is not available on this kernel; the filesystem allow-list is NOT enforced.");

    if (fs || net || scoped) {
        LandlockRulesetAttr attr{fs, net, scoped};
        const long fd = ::syscall(SYS_landlock_create_ruleset, &attr, sizeof(attr), 0);
        if (fd < 0) {
            out.warnings.push_back(std::string("sandbox: landlock_create_ruleset failed (") + std::strerror(errno) +
                                   "); the filesystem allow-list is NOT enforced.");
        } else {
            out.landlock_fd = static_cast<int>(fd);
            if (fs) {
                for (const auto &p : policy.read_only_paths) add_rule(out.landlock_fd, p, kFsReadOnly, fs, true, out.warnings);
                for (const auto &p : policy.read_write_paths) add_rule(out.landlock_fd, p, fs, fs, true, out.warnings);
                if (policy.system_paths) {
                    for (const auto &p : sandbox_system_read_only_paths(!deny_net))
                        add_rule(out.landlock_fd, p, kFsReadOnly, fs, false, out.warnings);
                    for (const auto &p : sandbox_system_read_write_paths(policy.gpu))
                        add_rule(out.landlock_fd, p, fs, fs, false, out.warnings);
                }
                if (policy.allow_tmp) {
                    add_rule(out.landlock_fd, temp_dir(), fs, fs, false, out.warnings);
                    add_rule(out.landlock_fd, "/var/tmp", fs, fs, false, out.warnings);
                }
            }
        }
    }

    if (deny_net) {
        if (sup.seccomp) out.seccomp = network_deny_filter();
        else if (net)
            out.warnings.push_back("sandbox: seccomp is not available; only TCP is blocked (Landlock), UDP and raw sockets are not.");
        else out.warnings.push_back("sandbox: neither seccomp nor Landlock ABI >= 4 is available; the network is NOT blocked.");
    }
    return ok(std::move(out));
}

int apply_prepared_sandbox(const PreparedSandbox &p) noexcept {
    if (!p.active) return 0;
    if (p.new_session) (void)::setsid(); // fails only if already a group leader

    auto lower = [](int resource, rlim_t value) {
        rlimit cur{};
        if (::getrlimit(resource, &cur) != 0) return errno;
        if (cur.rlim_max != RLIM_INFINITY && cur.rlim_max < value) value = cur.rlim_max;
        rlimit lim{value, value};
        return ::setrlimit(resource, &lim) == 0 ? 0 : errno;
    };
    if (int e = lower(RLIMIT_CORE, 0)) return e;
    if (p.memory_limit_bytes)
        if (int e = lower(RLIMIT_AS, static_cast<rlim_t>(p.memory_limit_bytes))) return e;
    if (p.max_processes > 0)
        if (int e = lower(RLIMIT_NPROC, static_cast<rlim_t>(p.max_processes))) return e;

    // Descriptors the parent leaked without O_CLOEXEC must not reach the sandboxed program.
    if (::syscall(SYS_close_range, 3u, ~0u, kCloseRangeCloexec) != 0)
        for (int fd = 3; fd < 4096; ++fd) (void)::fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) return errno;
    if (p.landlock_fd >= 0 && ::syscall(SYS_landlock_restrict_self, p.landlock_fd, 0) != 0) return errno;
    if (!p.seccomp.empty()) {
        sock_fprog prog{static_cast<unsigned short>(p.seccomp.size()),
                        reinterpret_cast<sock_filter *>(const_cast<BpfInsn *>(p.seccomp.data()))};
        if (::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog, 0, 0) != 0) return errno;
    }
    return 0;
}

static_assert(sizeof(BpfInsn) == sizeof(sock_filter), "BpfInsn must mirror sock_filter");

// ------------------------------------------------------------------ macOS (written, untested)

#elif defined(AGENT_HAL_MACOS)

namespace {

std::string sbpl_quote(const std::string &path) {
    std::error_code ec;
    std::string real = std::filesystem::exists(path, ec) ? std::filesystem::canonical(path, ec).string() : path;
    if (ec) real = path;
    std::string q = "\"";
    for (char c : real) {
        if (c == '"' || c == '\\') q.push_back('\\');
        q.push_back(c);
    }
    q.push_back('"');
    return q;
}

} // namespace

SandboxSupport sandbox_support() { return {}; }

Result<PreparedSandbox> prepare_sandbox(const SandboxPolicy &policy) {
    PreparedSandbox out;
    if (!policy.enabled) return ok(std::move(out));
    out.active = true;
    out.memory_limit_bytes = policy.memory_limit_bytes;
    out.max_processes = policy.max_processes > 0 ? policy.max_processes : 0;

    // Later rules take precedence in SBPL: start permissive, deny file access, re-allow the lists.
    std::string prof = "(version 1)\n(allow default)\n";
    if (policy.filesystem) {
        // file-read-metadata stays allowed everywhere (stat, realpath, getcwd and dyld walk the parents
        // of allowed paths); it exposes names and attributes, not contents.
        prof += "(deny file-read* file-write*)\n(allow file-read-metadata)\n(allow file-read* file-map-executable (literal \"/\")";
        auto ro = policy.read_only_paths;
        if (policy.system_paths)
            for (auto &p : sandbox_system_read_only_paths(policy.network == SandboxNetwork::Allow)) ro.push_back(p);
        for (const auto &p : ro) prof += " (subpath " + sbpl_quote(p) + ")";
        prof += ")\n(allow file-read* file-write*";
        auto rw = policy.read_write_paths;
        if (policy.system_paths)
            for (auto &p : sandbox_system_read_write_paths(policy.gpu)) rw.push_back(p);
        if (policy.allow_tmp) {
            rw.push_back(temp_dir());
            rw.push_back("/private/tmp");
            rw.push_back("/private/var/folders");
        }
        for (const auto &p : rw) prof += " (subpath " + sbpl_quote(p) + ")";
        prof += ")\n";
    }
    if (policy.network == SandboxNetwork::Deny) prof += "(deny network*)\n(allow network* (remote unix-socket))\n";
    out.macos_profile = std::move(prof);
    out.warnings.push_back("sandbox: the macOS sandbox_init() profile is untested.");
    return ok(std::move(out));
}

int apply_prepared_sandbox(const PreparedSandbox &p) noexcept {
    if (!p.active) return 0;
    if (p.new_session) (void)::setsid();
    auto lower = [](int resource, rlim_t value) {
        rlimit cur{};
        if (::getrlimit(resource, &cur) != 0) return errno;
        if (cur.rlim_max != RLIM_INFINITY && cur.rlim_max < value) value = cur.rlim_max;
        rlimit lim{value, value};
        return ::setrlimit(resource, &lim) == 0 ? 0 : errno;
    };
    if (int e = lower(RLIMIT_CORE, 0)) return e;
    if (p.memory_limit_bytes)
        if (int e = lower(RLIMIT_AS, static_cast<rlim_t>(p.memory_limit_bytes))) return e;
    if (p.max_processes > 0)
        if (int e = lower(RLIMIT_NPROC, static_cast<rlim_t>(p.max_processes))) return e;
    for (int fd = 3; fd < 4096; ++fd) (void)::fcntl(fd, F_SETFD, FD_CLOEXEC);
    if (!p.macos_profile.empty()) {
        char *err = nullptr;
        if (::sandbox_init(p.macos_profile.c_str(), 0, &err) != 0) {
            if (err) ::sandbox_free_error(err);
            return EPERM;
        }
    }
    return 0;
}

// ------------------------------------------------------------------ Windows (written, untested)

#elif defined(AGENT_HAL_WINDOWS)

SandboxSupport sandbox_support() { return {}; }

Result<PreparedSandbox> prepare_sandbox(const SandboxPolicy &policy) {
    PreparedSandbox out;
    if (!policy.enabled) return ok(std::move(out));
    out.active = true;
    out.memory_limit_bytes = policy.memory_limit_bytes;
    out.max_processes = policy.max_processes > 0 ? policy.max_processes : 0;
    if (policy.filesystem && policy.windows_low_integrity) {
        // Writes are confined to the read-write folders by the Low integrity level. Reads are not restricted on Windows.
        out.low_integrity = true;
        out.low_write_paths = policy.read_write_paths;
        out.warnings.push_back("sandbox: on Windows only writes are confined to the allowed folders (Low integrity level); reads are not restricted.");
    }
    if (policy.network == SandboxNetwork::Deny) out.warnings.push_back("sandbox: network denial is not enforced on Windows.");
    return ok(std::move(out));
}

int apply_prepared_sandbox(const PreparedSandbox &) noexcept { return 0; }

bool label_low_integrity(const std::string &path, std::string &error) {
    PSECURITY_DESCRIPTOR sd = nullptr;
    // S: = a system ACL; ML = mandatory label; OICI = inherited by files and folders created below; NW = no write up;
    // LW = Low integrity.
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"S:(ML;OICI;NW;;;LW)", SDDL_REVISION_1, &sd, nullptr)) {
        error = "cannot build the integrity label (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    BOOL present = FALSE, defaulted = FALSE;
    PACL sacl = nullptr;
    bool ok_label = GetSecurityDescriptorSacl(sd, &present, &sacl, &defaulted) && present;
    DWORD rc = ok_label ? ERROR_SUCCESS : GetLastError();
    if (ok_label) {
        std::wstring wide(path.begin(), path.end());
        int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (n > 0) {
            wide.assign(static_cast<size_t>(n - 1), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), n);
        }
        rc = SetNamedSecurityInfoW(wide.data(), SE_FILE_OBJECT, LABEL_SECURITY_INFORMATION, nullptr, nullptr, nullptr, sacl);
    }
    LocalFree(sd);
    if (rc != ERROR_SUCCESS) {
        error = "cannot label " + path + " for the sandbox (error " + std::to_string(rc) + ")";
        return false;
    }
    return true;
}

void *create_low_integrity_token(std::string &error) {
    HANDLE self = nullptr, primary = nullptr;
    PSID low = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY | TOKEN_ADJUST_DEFAULT, &self)) {
        error = "OpenProcessToken failed (error " + std::to_string(GetLastError()) + ")";
        return nullptr;
    }
    bool ok_token = DuplicateTokenEx(self, MAXIMUM_ALLOWED, nullptr, SecurityImpersonation, TokenPrimary, &primary) &&
                    ConvertStringSidToSidW(L"S-1-16-4096", &low);
    if (ok_token) {
        TOKEN_MANDATORY_LABEL tml{};
        tml.Label.Attributes = SE_GROUP_INTEGRITY;
        tml.Label.Sid = low;
        ok_token = SetTokenInformation(primary, TokenIntegrityLevel, &tml, sizeof(tml) + GetLengthSid(low)) != 0;
    }
    const DWORD last = GetLastError();
    if (low) LocalFree(low);
    CloseHandle(self);
    if (!ok_token) {
        if (primary) CloseHandle(primary);
        error = "cannot create a Low integrity token (error " + std::to_string(last) + ")";
        return nullptr;
    }
    return primary;
}

void *attach_job_object(PreparedSandbox &p, void *process) {
    if (!p.active) return nullptr;
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job) {
        p.warnings.push_back("sandbox: CreateJobObject failed (error " + std::to_string(GetLastError()) + ").");
        return nullptr;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    // No JOB_OBJECT_LIMIT_BREAKAWAY_OK / SILENT_BREAKAWAY_OK: descendants stay in the job.
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
    if (p.memory_limit_bytes) {
        info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = static_cast<SIZE_T>(p.memory_limit_bytes);
    }
    if (p.max_processes > 0) {
        info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
        info.BasicLimitInformation.ActiveProcessLimit = static_cast<DWORD>(p.max_processes);
    }
    JOBOBJECT_BASIC_UI_RESTRICTIONS ui{};
    ui.UIRestrictionsClass = JOB_OBJECT_UILIMIT_DESKTOP | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS | JOB_OBJECT_UILIMIT_EXITWINDOWS |
                             JOB_OBJECT_UILIMIT_GLOBALATOMS | JOB_OBJECT_UILIMIT_HANDLES | JOB_OBJECT_UILIMIT_READCLIPBOARD |
                             JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS | JOB_OBJECT_UILIMIT_WRITECLIPBOARD;
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info)) ||
        !SetInformationJobObject(job, JobObjectBasicUIRestrictions, &ui, sizeof(ui)) ||
        !AssignProcessToJobObject(job, static_cast<HANDLE>(process))) {
        p.warnings.push_back("sandbox: configuring the Job Object failed (error " + std::to_string(GetLastError()) + ").");
        CloseHandle(job);
        return nullptr;
    }
    return job;
}

#endif

// ------------------------------------------------------------------ self-restriction

Result<void> apply_sandbox(const SandboxPolicy &policy) {
    auto prepared = prepare_sandbox(policy);
    if (!prepared.ok) return fail(prepared.error.code, prepared.error.message);
    prepared.value.new_session = false; // a running process keeps its session
    if (int e = apply_prepared_sandbox(prepared.value); e != 0)
        return fail(ErrorCode::SandboxViolation, std::string("cannot apply the sandbox: ") + std::strerror(e));
    return ok();
}

} // namespace agent::hal
