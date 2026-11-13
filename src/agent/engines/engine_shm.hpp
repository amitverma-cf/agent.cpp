#pragma once

// Named shared-memory segments for the engine IPC protocol (engine_ipc.hpp): tensor and BYTES payloads
// of at least kShmThreshold bytes travel in a segment the sender creates and the receiver maps, instead
// of being copied through the pipe. POSIX: shm_open + mmap (the name lives in /dev/shm on Linux).
// Windows: CreateFileMapping / OpenFileMapping in the session's Local\ namespace (run by the Windows test suite).
//
// Lifetime: the sender creates a segment and sends its name; the receiver maps it on first use. Both
// sides keep their mappings and reuse the segment for later calls (growing means a new segment and a
// "retire" of the old one), so steady-state calls do no shm syscalls. On POSIX the core unlinks every
// name as soon as both sides have it mapped (its request segments after the first exchange, the host's
// response segments when it maps them), so nothing is left in /dev/shm when a process dies; names the
// host created but the core never mapped are swept after a host crash (sweep()). Windows has no unlink:
// a mapping lives while either side holds a handle.
//
// No heap allocation: names are fixed-size buffers.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace agent::engine_shm {

// Payloads (one tensor / BYTES part) at least this large go through shared memory.
inline constexpr size_t kShmThreshold = 64 * 1024;
// Offsets of payloads inside a segment are aligned to this.
inline constexpr size_t kShmAlign = 64;

class Segment {
  public:
    Segment() = default;
    ~Segment() { unmap(); }
    Segment(const Segment &) = delete;
    Segment &operator=(const Segment &) = delete;
    Segment(Segment &&other) noexcept { *this = static_cast<Segment &&>(other); }
    Segment &operator=(Segment &&other) noexcept;

    // Creates a new segment of size bytes, mapped read/write, named "acpp-<side><pid>-<seq>"
    // (side: 'c' core, 'h' host). false (nothing left behind) on failure.
    bool create(size_t size, char side);
    // Maps an existing segment read-only; fails if it is smaller than size.
    bool open(std::string_view name, size_t size);
    // Unmaps (and on Windows closes the mapping handle). Does not unlink.
    void unmap();

    char *data() const { return static_cast<char *>(addr_); }
    size_t size() const { return size_; }
    std::string_view name() const { return std::string_view(name_, name_len_); }
    bool mapped() const { return addr_ != nullptr; }

  private:
    void *addr_ = nullptr;
    size_t size_ = 0;
    void *handle_ = nullptr; // Windows mapping handle
    char name_[48] = {};
    size_t name_len_ = 0;
};

// Removes the segment's name (POSIX shm_unlink; ignored if it is already gone). No-op on Windows.
void unlink(std::string_view name);

// Linux: unlinks every leftover segment created by process pid on the given side (after that process
// died). No-op elsewhere.
void sweep(char side, long pid);

// true when this process can create, map and remove a segment (checked once, then cached).
bool available();

inline size_t align_up(size_t n) { return (n + kShmAlign - 1) & ~(kShmAlign - 1); }

} // namespace agent::engine_shm
