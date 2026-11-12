#include "engine_shm.hpp"

#include "hal/platform.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>

#if defined(AGENT_HAL_WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace agent::engine_shm {

namespace {

std::atomic<uint64_t> g_seq{0};

long current_pid() {
#if defined(AGENT_HAL_WINDOWS)
    return static_cast<long>(GetCurrentProcessId());
#else
    return static_cast<long>(::getpid());
#endif
}

// The OS-level name: "/acpp-..." on POSIX, "Local\acpp-..." on Windows. false if it doesn't fit.
bool os_name(std::string_view name, char (&out)[64]) {
#if defined(AGENT_HAL_WINDOWS)
    constexpr std::string_view prefix = "Local\\";
#else
    constexpr std::string_view prefix = "/";
#endif
    if (name.empty() || prefix.size() + name.size() >= sizeof(out)) return false;
    for (char c : name)
        if (c == '/' || c == '\\' || c == '\0') return false;
    std::memcpy(out, prefix.data(), prefix.size());
    std::memcpy(out + prefix.size(), name.data(), name.size());
    out[prefix.size() + name.size()] = '\0';
    return true;
}

} // namespace

Segment &Segment::operator=(Segment &&other) noexcept {
    if (this == &other) return *this;
    unmap();
    addr_ = other.addr_;
    size_ = other.size_;
    handle_ = other.handle_;
    std::memcpy(name_, other.name_, sizeof(name_));
    name_len_ = other.name_len_;
    other.addr_ = nullptr;
    other.size_ = 0;
    other.handle_ = nullptr;
    other.name_len_ = 0;
    return *this;
}

bool Segment::create(size_t size, char side) {
    unmap();
    if (size == 0) return false;
    const int n = std::snprintf(name_, sizeof(name_), "acpp-%c%lx-%llx", side, current_pid(),
                                static_cast<unsigned long long>(g_seq.fetch_add(1, std::memory_order_relaxed)));
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(name_)) return false;
    name_len_ = static_cast<size_t>(n);
    char os[64];
    if (!os_name(name(), os)) return false;
#if defined(AGENT_HAL_WINDOWS)
    const unsigned long long sz = size;
    HANDLE h = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>(sz >> 32),
                                  static_cast<DWORD>(sz & 0xffffffffu), os);
    if (!h) return false;
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        CloseHandle(h);
        return false;
    }
    void *p = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) {
        CloseHandle(h);
        return false;
    }
    handle_ = h;
    addr_ = p;
#else
    const int fd = ::shm_open(os, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) return false;
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
        ::close(fd);
        ::shm_unlink(os);
        return false;
    }
#if defined(MAP_POPULATE)
    const int flags = MAP_SHARED | MAP_POPULATE; // the sender writes every page right away
#else
    const int flags = MAP_SHARED;
#endif
    void *p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        ::shm_unlink(os);
        return false;
    }
    addr_ = p;
#endif
    size_ = size;
    return true;
}

bool Segment::open(std::string_view name, size_t size) {
    unmap();
    if (size == 0 || name.size() >= sizeof(name_)) return false;
    char os[64];
    if (!os_name(name, os)) return false;
#if defined(AGENT_HAL_WINDOWS)
    HANDLE h = OpenFileMappingA(FILE_MAP_READ, FALSE, os);
    if (!h) return false;
    void *p = MapViewOfFile(h, FILE_MAP_READ, 0, 0, size);
    if (!p) {
        CloseHandle(h);
        return false;
    }
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(p, &info, sizeof(info)) == 0 || info.RegionSize < size) {
        UnmapViewOfFile(p);
        CloseHandle(h);
        return false;
    }
    handle_ = h;
    addr_ = p;
#else
    const int fd = ::shm_open(os, O_RDONLY, 0);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 0 || static_cast<size_t>(st.st_size) < size) {
        ::close(fd);
        return false;
    }
#if defined(MAP_POPULATE)
    const int flags = MAP_SHARED | MAP_POPULATE;
#else
    const int flags = MAP_SHARED;
#endif
    void *p = ::mmap(nullptr, size, PROT_READ, flags, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) return false;
    addr_ = p;
#endif
    size_ = size;
    std::memcpy(name_, name.data(), name.size());
    name_len_ = name.size();
    return true;
}

void Segment::unmap() {
    if (addr_) {
#if defined(AGENT_HAL_WINDOWS)
        UnmapViewOfFile(addr_);
#else
        ::munmap(addr_, size_);
#endif
    }
#if defined(AGENT_HAL_WINDOWS)
    if (handle_) CloseHandle(static_cast<HANDLE>(handle_));
#endif
    addr_ = nullptr;
    handle_ = nullptr;
    size_ = 0;
    name_len_ = 0;
}

void unlink(std::string_view name) {
#if defined(AGENT_HAL_WINDOWS)
    (void)name;
#else
    char os[64];
    if (os_name(name, os)) ::shm_unlink(os);
#endif
}

void sweep(char side, long pid) {
#if defined(AGENT_HAL_LINUX)
    char prefix[32];
    const int n = std::snprintf(prefix, sizeof(prefix), "acpp-%c%lx-", side, pid);
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(prefix)) return;
    DIR *dir = ::opendir("/dev/shm");
    if (!dir) return;
    while (const dirent *e = ::readdir(dir))
        if (std::strncmp(e->d_name, prefix, static_cast<size_t>(n)) == 0) unlink(e->d_name);
    ::closedir(dir);
#else
    (void)side;
    (void)pid;
#endif
}

bool available() {
    static const bool ok = [] {
        Segment s;
        if (!s.create(4096, 't')) return false;
        s.data()[0] = 1;
        Segment r;
        const bool opened = r.open(s.name(), 4096) && r.data()[0] == 1;
        unlink(s.name());
        return opened;
    }();
    return ok;
}

} // namespace agent::engine_shm
