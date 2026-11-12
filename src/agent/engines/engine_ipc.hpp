#pragma once

// Wire protocol between the core (src/agent/engines/engine_remote.cpp) and agent-engine-host
// (apps/engine-host). Both ends are the same build on the same machine, so integers are sent in
// native byte order and layout; nothing here is a stable cross-version format (see kProtocolVersion).
//
// Frame:   u32 payload_size | u32 type | u64 id | payload
//          id: the request a frame belongs to (0 for startup frames). Every reply, chunk and progress
//          frame of a request carries that request's id, so several requests can be in flight at once.
// str:     u32 size | bytes                     (agent_str; not NUL-terminated)
// blob:    u64 size | bytes                     (Chunk payloads)
// data:    u64 size | u32 loc | loc 0: bytes[size] | loc 1: u64 offset into the frame's segment
//          (tensor / BYTES payloads; loc 1 when the payload is >= kShmThreshold and the frame has a
//          shared-memory segment, see engine_shm.hpp)
// shm:     str segment_name | u64 segment_size | str retire_name
//          (empty segment_name = no segment; retire_name: a segment the receiver should unmap, it is
//          replaced by a bigger one and will not be named again)
// part:    u32 kind | str text | str path | str mime | str name | u32 dtype | u32 ndim | i64 shape[ndim] | data
// call:    str id | str name | str arguments
// message: str role | u32 n_parts | part* | u32 n_calls | call* | str tool_call_id
// request: u32 task | u32 request_flags | u32 run_flags | u32 n_messages | message* | str tools_json | str params_json | str output_dir
// response: message | i32 prompt_tokens | i32 completion_tokens | agent_run_stats (raw)
//
// Every encoder appends to a caller-owned std::string whose capacity is reused across calls, and
// every decoder returns views into the received frame (or the mapped segment); the agent_message /
// agent_part arrays come from an Arena. Decoders validate every length and offset against the frame
// and the segment and fail instead of reading past them.

#include "engine_shm.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/engine_abi.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace agent::engine_ipc {

inline constexpr uint32_t kProtocolVersion = 3;
inline constexpr size_t kHeaderSize = 16;
// Largest frame either side accepts (payloads that don't go through shared memory travel inline).
inline constexpr uint32_t kMaxFrame = 1u << 30;

enum class Msg : uint32_t {
    // core -> host
    Create = 1,      // u32 client_flags | str config_json  -> Created | Error
    Run = 2,         // shm | request                       -> Chunk* Progress* (Response | Error)
    Cancel = 3,      // (empty) out of band: sets the cancel flag request <id> polls
    CountTokens = 4, // str text                            -> Count | Error
    ResetCache = 5,  // (empty)                             -> Done | Error
    Destroy = 6,     // (empty) destroys the instance       -> Done
    Shutdown = 7,    // (empty) destroys and exits
    Release = 8,     // (empty) the core is done with Response <id>'s segment; the host may reuse it
    // host -> core
    Ready = 100,    // u32 protocol | u32 abi | u64 caps | u32 flags | u32 optional_fns | u32 host_flags | u64 pid | str name | str version
    Error = 101,    // i32 status | str message
    Created = 102,  // (empty)
    Chunk = 103,    // u32 kind | blob data
    Progress = 104, // u64 step | u64 total
    Response = 105, // shm | response
    Count = 106,    // i32 count
    Done = 107,     // (empty)
};

// Ready::optional_fns bits.
inline constexpr uint32_t kHasCountTokens = 1u << 0;
inline constexpr uint32_t kHasResetCache = 1u << 1;

// Ready::host_flags and Create::client_flags bits.
inline constexpr uint32_t kShmCapable = 1u << 0; // can create and map shared-memory segments

// Where a Writer puts large payloads: the frame's segment (base/size), filled from used upwards.
struct ShmOut {
    char *base = nullptr;
    size_t size = 0;
    size_t used = 0;
    size_t threshold = engine_shm::kShmThreshold;
    size_t count = 0; // payloads placed in the segment
};

// Run::run_flags bits.
inline constexpr uint32_t kRunStream = 1u << 0;   // forward on_chunk
inline constexpr uint32_t kRunProgress = 1u << 1; // forward on_progress

// ------------------------------------------------------------------ encoding

class Writer {
  public:
    explicit Writer(std::string &out) : out_(out) {}

    // Starts a frame; finish() patches its size. One frame per Writer.
    void begin(Msg type, uint64_t id = 0) {
        start_ = out_.size();
        u32(0);
        u32(static_cast<uint32_t>(type));
        u64(id);
    }
    // Large tensor / BYTES payloads of the frame go to shm (nullptr: everything inline).
    void set_shm(ShmOut *shm) { shm_ = shm; }
    // false when the payload exceeds kMaxFrame.
    bool finish() {
        const size_t payload = out_.size() - start_ - kHeaderSize;
        if (payload > kMaxFrame) return false;
        const uint32_t n = static_cast<uint32_t>(payload);
        std::memcpy(out_.data() + start_, &n, sizeof(n));
        return true;
    }

    void raw(const void *p, size_t n) {
        if (n) out_.append(static_cast<const char *>(p), n);
    }
    void u32(uint32_t v) { raw(&v, sizeof(v)); }
    void i32(int32_t v) { raw(&v, sizeof(v)); }
    void u64(uint64_t v) { raw(&v, sizeof(v)); }
    void i64(int64_t v) { raw(&v, sizeof(v)); }
    void str(agent_str s) { str(s.data ? std::string_view(s.data, s.size) : std::string_view()); }
    void str(std::string_view s) {
        u32(static_cast<uint32_t>(s.size()));
        raw(s.data(), s.size());
    }
    void blob(const void *p, size_t n) {
        u64(n);
        if (p) raw(p, n);
    }
    void data(const void *p, size_t n) {
        u64(n);
        if (p && shm_ && shm_->base && n >= shm_->threshold && n <= shm_->size - shm_->used) {
            std::memcpy(shm_->base + shm_->used, p, n);
            u32(1);
            u64(shm_->used);
            shm_->used = engine_shm::align_up(shm_->used + n);
            if (shm_->used > shm_->size) shm_->used = shm_->size;
            ++shm_->count;
            return;
        }
        u32(0);
        if (p) raw(p, n);
    }

    void part(const agent_part &p) {
        u32(p.kind);
        str(p.text);
        str(p.path);
        str(p.mime);
        str(p.name);
        u32(p.dtype);
        const size_t ndim = p.shape ? p.ndim : 0;
        u32(static_cast<uint32_t>(ndim));
        for (size_t i = 0; i < ndim; ++i) i64(p.shape[i]);
        data(p.data, p.data ? p.size : 0);
    }

    void message(const agent_message &m) {
        str(m.role);
        const size_t n_parts = m.parts ? m.n_parts : 0;
        u32(static_cast<uint32_t>(n_parts));
        for (size_t i = 0; i < n_parts; ++i) part(m.parts[i]);
        const size_t n_calls = m.tool_calls ? m.n_tool_calls : 0;
        u32(static_cast<uint32_t>(n_calls));
        for (size_t i = 0; i < n_calls; ++i) {
            str(m.tool_calls[i].id);
            str(m.tool_calls[i].name);
            str(m.tool_calls[i].arguments);
        }
        str(m.tool_call_id);
    }

    void request(const agent_request &r, uint32_t run_flags) {
        u32(r.task);
        u32(r.flags);
        u32(run_flags);
        const size_t n = r.messages ? r.n_messages : 0;
        u32(static_cast<uint32_t>(n));
        for (size_t i = 0; i < n; ++i) message(r.messages[i]);
        str(r.tools_json);
        str(r.params_json);
        str(r.output_dir);
    }

    void response(const agent_response &r) {
        message(r.message);
        i32(r.usage.prompt_tokens);
        i32(r.usage.completion_tokens);
        agent_run_stats stats{};
        if (AGENT_HAS_FIELD(&r, agent_response, stats)) stats = r.stats;
        raw(&stats, sizeof(stats));
    }

  private:
    std::string &out_;
    size_t start_ = 0;
    ShmOut *shm_ = nullptr;
};

// Segment bytes a Writer needs to place every payload of at least threshold bytes in shm.
inline size_t shm_bytes(const agent_message &m, size_t threshold) {
    size_t n = 0;
    for (size_t i = 0; m.parts && i < m.n_parts; ++i)
        if (m.parts[i].data && m.parts[i].size >= threshold) n += engine_shm::align_up(m.parts[i].size);
    return n;
}
inline size_t shm_bytes(const agent_request &r, size_t threshold) {
    size_t n = 0;
    for (size_t i = 0; r.messages && i < r.n_messages; ++i) n += shm_bytes(r.messages[i], threshold);
    return n;
}

// ------------------------------------------------------------------ decoding

class Reader {
  public:
    Reader(const char *data, size_t size) : p_(data), end_(data + size) {}

    bool ok() const { return ok_; }
    bool at_end() const { return ok_ && p_ == end_; }
    // The mapped segment loc-1 payloads point into (none: such payloads are rejected).
    void set_shm(const char *base, size_t size) {
        shm_base_ = base;
        shm_size_ = size;
    }

    bool raw(void *dst, size_t n) {
        if (!take(n)) return false;
        std::memcpy(dst, p_ - n, n);
        return true;
    }
    uint32_t u32() { return scalar<uint32_t>(); }
    int32_t i32() { return scalar<int32_t>(); }
    uint64_t u64() { return scalar<uint64_t>(); }

    agent_str str() {
        const uint32_t n = u32();
        if (!take(n)) return agent_str{nullptr, 0};
        return agent_str{n ? p_ - n : nullptr, n};
    }
    // A view of the next n bytes; nullptr (with ok() false) if the frame is shorter.
    const char *bytes(uint64_t n) {
        if (n > static_cast<uint64_t>(end_ - p_)) {
            ok_ = false;
            return nullptr;
        }
        take(static_cast<size_t>(n));
        return p_ - n;
    }

    bool part(agent_part &out, Arena &arena) {
        out = agent_part{};
        out.kind = u32();
        out.text = str();
        out.path = str();
        out.mime = str();
        out.name = str();
        out.dtype = u32();
        const uint32_t ndim = u32();
        if (!ok_ || ndim > (end_ - p_) / sizeof(int64_t)) return fail();
        if (ndim) {
            auto shape = arena.allocate_span<int64_t>(ndim);
            if (shape.size() != ndim) return fail(true);
            raw(shape.data(), ndim * sizeof(int64_t));
            out.shape = shape.data();
            out.ndim = ndim;
        }
        const uint64_t size = u64();
        const uint32_t loc = u32();
        const char *data = nullptr;
        if (loc == 0) {
            data = bytes(size);
        } else if (loc == 1) {
            const uint64_t off = u64();
            if (!ok_ || !shm_base_ || off > shm_size_ || size > shm_size_ - off) return fail();
            data = shm_base_ + off;
        } else {
            return fail();
        }
        if (!ok_) return false;
        out.data = size ? data : nullptr;
        out.size = static_cast<size_t>(size);
        return true;
    }

    bool message(agent_message &out, Arena &arena) {
        out = agent_message{};
        out.role = str();
        const uint32_t n_parts = u32();
        // Every part needs at least 40 bytes on the wire: reject counts the frame can't hold
        // before allocating for them.
        if (!ok_ || n_parts > (end_ - p_) / 40) return fail();
        if (n_parts) {
            auto parts = arena.allocate_span<agent_part>(n_parts);
            if (parts.size() != n_parts) return fail(true);
            for (auto &pt : parts)
                if (!part(pt, arena)) return false;
            out.parts = parts.data();
            out.n_parts = n_parts;
        }
        const uint32_t n_calls = u32();
        if (!ok_ || n_calls > (end_ - p_) / 12) return fail();
        if (n_calls) {
            auto calls = arena.allocate_span<agent_tool_call>(n_calls);
            if (calls.size() != n_calls) return fail(true);
            for (auto &c : calls) c = agent_tool_call{str(), str(), str()};
            out.tool_calls = calls.data();
            out.n_tool_calls = n_calls;
        }
        out.tool_call_id = str();
        return ok_;
    }

    // Fills everything but the callbacks, user_data and cancel.
    bool request(agent_request &out, uint32_t &run_flags, Arena &arena) {
        out = agent_request{};
        out.struct_size = sizeof(agent_request);
        out.task = u32();
        out.flags = u32();
        run_flags = u32();
        const uint32_t n = u32();
        if (!ok_ || n > (end_ - p_) / 16) return fail();
        if (n) {
            auto msgs = arena.allocate_span<agent_message>(n);
            if (msgs.size() != n) return fail(true);
            for (auto &m : msgs)
                if (!message(m, arena)) return false;
            out.messages = msgs.data();
            out.n_messages = n;
        }
        out.tools_json = str();
        out.params_json = str();
        out.output_dir = str();
        return ok_;
    }

    bool response(agent_response &out, Arena &arena) {
        out = agent_response{};
        out.struct_size = sizeof(agent_response);
        if (!message(out.message, arena)) return false;
        out.usage.prompt_tokens = i32();
        out.usage.completion_tokens = i32();
        out.stats = scalar<agent_run_stats>();
        return ok_;
    }

    // true when the last failure was the arena running out (not a malformed frame).
    bool arena_exhausted() const { return arena_exhausted_; }

  private:
    template <typename T> T scalar() {
        T v{};
        raw(&v, sizeof(v));
        return v;
    }
    bool take(size_t n) {
        if (!ok_ || n > static_cast<size_t>(end_ - p_)) {
            ok_ = false;
            return false;
        }
        p_ += n;
        return true;
    }
    bool fail(bool arena = false) {
        ok_ = false;
        arena_exhausted_ = arena_exhausted_ || arena;
        return false;
    }

    const char *p_;
    const char *end_;
    const char *shm_base_ = nullptr;
    size_t shm_size_ = 0;
    bool ok_ = true;
    bool arena_exhausted_ = false;
};

// Arena bytes a decoder may need for a payload of n bytes: every decoded struct is at most 4x the
// minimum size of its wire encoding (agent_message: 64 bytes vs 16, agent_part: 112 vs 40), plus
// alignment padding.
inline size_t decode_arena_bound(size_t payload_size) { return payload_size * 5 + 4096; }

struct FrameHeader {
    uint32_t size = 0;
    Msg type{};
    uint64_t id = 0;
};

// Header of the frame at data (needs kHeaderSize bytes).
inline FrameHeader read_header(const char *data) {
    FrameHeader h;
    uint32_t type = 0;
    std::memcpy(&h.size, data, sizeof(h.size));
    std::memcpy(&type, data + 4, sizeof(type));
    std::memcpy(&h.id, data + 8, sizeof(h.id));
    h.type = static_cast<Msg>(type);
    return h;
}

} // namespace agent::engine_ipc
