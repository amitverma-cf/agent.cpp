#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cstring>

namespace agent {

Arena::Arena(size_t capacity) : capacity_(capacity), offset_(0) {
    ptr_ = capacity > 0 ? new char[capacity] : nullptr;
}

Arena::~Arena() {
    delete[] ptr_;
}

Arena::Arena(Arena &&other) noexcept : capacity_(other.capacity_), ptr_(other.ptr_), offset_(other.offset_) {
    other.ptr_ = nullptr;
    other.offset_ = 0;
    other.capacity_ = 0;
}

Arena &Arena::operator=(Arena &&other) noexcept {
    if (this != &other) {
        delete[] ptr_;
        ptr_ = other.ptr_;
        offset_ = other.offset_;
        capacity_ = other.capacity_;
        other.ptr_ = nullptr;
        other.offset_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

void *Arena::allocate(size_t size, size_t alignment) {
    if (!ptr_ || size == 0)
        return nullptr;
    size_t current_ptr = reinterpret_cast<size_t>(ptr_ + offset_);
    size_t mask = alignment - 1;
    size_t aligned_ptr = (current_ptr + mask) & ~mask;
    size_t aligned_offset = aligned_ptr - reinterpret_cast<size_t>(ptr_);
    if (aligned_offset + size > capacity_) {
        return nullptr;
    }
    offset_ = aligned_offset + size;
    return reinterpret_cast<void *>(aligned_ptr);
}

void Arena::reset() {
    offset_ = 0;
}

std::string_view Arena::allocate_string(std::string_view src) {
    if (src.empty())
        return {};
    void *mem = allocate(src.size());
    if (!mem)
        return {};
    std::memcpy(mem, src.data(), src.size());
    return std::string_view(static_cast<const char *>(mem), src.size());
}

} // namespace agent
