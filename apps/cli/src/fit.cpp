#include "fit.hpp"

#include "cli_core.hpp"
#include "doctor.hpp"

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
// clang-format on
#elif defined(__linux__)
#include <cstdlib>
#endif

namespace agent::cli {

namespace {

// ------------------------------------------------------------------ GGUF reader

struct Reader {
    std::string_view d;
    size_t at = 0;
    bool ok = true;

    bool need(size_t n) {
        if (!ok || d.size() - at < n || at > d.size()) ok = false;
        return ok;
    }
    template <class T> T get() {
        T v{};
        if (!need(sizeof(T))) return v;
        std::memcpy(&v, d.data() + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
    std::string_view str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || n > d.size() || !need(static_cast<size_t>(n))) {
            ok = false;
            return {};
        }
        std::string_view s = d.substr(at, static_cast<size_t>(n));
        at += static_cast<size_t>(n);
        return s;
    }
    void skip(uint64_t n) {
        if (n > d.size() || !need(static_cast<size_t>(n))) {
            ok = false;
            return;
        }
        at += static_cast<size_t>(n);
    }
};

size_t scalar_size(uint32_t type) {
    switch (type) {
    case 0:
    case 1:
    case 7: return 1;
    case 2:
    case 3: return 2;
    case 4:
    case 5:
    case 6: return 4;
    case 10:
    case 11:
    case 12: return 8;
    default: return 0;
    }
}

// An integer scalar as uint64 (0 for anything else).
uint64_t read_uint(Reader &r, uint32_t type) {
    switch (type) {
    case 0: return r.get<uint8_t>();
    case 1: return static_cast<uint64_t>(std::max<int64_t>(0, r.get<int8_t>()));
    case 2: return r.get<uint16_t>();
    case 3: return static_cast<uint64_t>(std::max<int64_t>(0, r.get<int16_t>()));
    case 4: return r.get<uint32_t>();
    case 5: return static_cast<uint64_t>(std::max<int64_t>(0, r.get<int32_t>()));
    case 10: return r.get<uint64_t>();
    case 11: return static_cast<uint64_t>(std::max<int64_t>(0, r.get<int64_t>()));
    default: r.skip(scalar_size(type)); return 0;
    }
}

bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

} // namespace

GgufInfo parse_gguf_header(std::string_view bytes) {
    GgufInfo info;
    Reader r{bytes};
    if (bytes.size() < 24 || bytes.substr(0, 4) != "GGUF") return info;
    r.at = 4;
    const uint32_t version = r.get<uint32_t>();
    if (version < 2 || version > 3) return info;
    info.tensor_count = r.get<uint64_t>();
    const uint64_t kv_count = r.get<uint64_t>();
    info.valid = true;

    for (uint64_t i = 0; i < kv_count && r.ok; ++i) {
        const std::string key(r.str());
        const uint32_t type = r.get<uint32_t>();
        if (!r.ok) break;
        if (type == 8) {
            const std::string_view value = r.str();
            if (r.ok && key == "general.architecture") info.architecture = std::string(value);
            continue;
        }
        if (type == 9) {
            const uint32_t elem = r.get<uint32_t>();
            const uint64_t count = r.get<uint64_t>();
            if (!r.ok) break;
            const bool numeric = elem != 8 && elem != 9 && scalar_size(elem) != 0;
            if (numeric && count <= 4096 && ends_with(key, ".attention.head_count_kv")) {
                uint64_t largest = 0;
                for (uint64_t k = 0; k < count && r.ok; ++k) largest = std::max(largest, read_uint(r, elem));
                if (r.ok) info.head_count_kv = largest;
            } else if (numeric) {
                r.skip(count * scalar_size(elem));
            } else if (elem == 8) {
                for (uint64_t k = 0; k < count && r.ok; ++k) r.str(); // tokens, merges: long, and after the hyper-parameters
            } else {
                r.ok = false; // nested arrays do not occur in practice
            }
            continue;
        }
        const uint64_t value = read_uint(r, type);
        if (!r.ok) break;
        if (ends_with(key, ".block_count")) info.block_count = value;
        else if (ends_with(key, ".embedding_length")) info.embedding_length = value;
        else if (ends_with(key, ".attention.head_count")) info.head_count = value;
        else if (ends_with(key, ".attention.head_count_kv")) info.head_count_kv = value;
        else if (ends_with(key, ".context_length")) info.context_length = value;
        else if (ends_with(key, ".attention.key_length")) info.key_length = value;
        else if (ends_with(key, ".expert_count")) info.expert_count = value;
        else if (ends_with(key, ".expert_used_count")) info.expert_used_count = value;
    }
    info.complete = r.ok;
    return info;
}

// ------------------------------------------------------------------ the machine

namespace {

uint64_t free_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys : 0;
#elif defined(__linux__)
    std::FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "MemAvailable: %llu kB", reinterpret_cast<unsigned long long *>(&kb)) == 1) break;
    std::fclose(f);
    return kb * 1024;
#else
    return 0;
#endif
}

std::string capture(const char *command) {
#if defined(_WIN32)
    std::FILE *pipe = _popen(command, "r");
#else
    std::FILE *pipe = popen(command, "r");
#endif
    std::string out;
    if (!pipe) return out;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return out;
}

// nvidia-smi reports MiB. Other vendors: not read (0 = unknown).
uint64_t gpu_memory_bytes() {
    auto out = capture("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>"
#if defined(_WIN32)
                       "nul"
#else
                       "/dev/null"
#endif
    );
    uint64_t best = 0;
    size_t at = 0;
    while (at < out.size()) {
        size_t end = out.find('\n', at);
        if (end == std::string::npos) end = out.size();
        const std::string line = out.substr(at, end - at);
        char *stop = nullptr;
        const unsigned long long mib = std::strtoull(line.c_str(), &stop, 10);
        if (stop != line.c_str()) best = std::max<uint64_t>(best, mib << 20);
        at = end + 1;
    }
    return best;
}

// Reads a slab per thread a few times and keeps the best aggregate rate: what token generation, which is memory-bound, can get.
double measure_bandwidth() {
    const int threads = std::max(1, std::min(8, static_cast<int>(std::thread::hardware_concurrency())));
    const size_t slab = 48u << 20;
    std::vector<std::vector<uint64_t>> data(static_cast<size_t>(threads), std::vector<uint64_t>(slab / sizeof(uint64_t), 1));
    std::atomic<uint64_t> sink{0};
    double best = 0;
    for (int pass = 0; pass < 3; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;
        for (int t = 0; t < threads; ++t)
            workers.emplace_back([&, t] {
                uint64_t sum = 0;
                for (uint64_t v : data[static_cast<size_t>(t)]) sum += v;
                sink += sum;
            });
        for (auto &w : workers) w.join();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (seconds > 0) best = std::max(best, static_cast<double>(slab) * threads / seconds);
    }
    return best;
}

} // namespace

Machine probe_machine(bool with_bandwidth) {
    Machine m;
    m.ram_total = total_ram_bytes();
    m.ram_free = free_ram_bytes();
    m.vram_total = gpu_memory_bytes();
    m.cores = static_cast<int>(std::thread::hardware_concurrency());
    if (with_bandwidth) m.mem_bandwidth = measure_bandwidth();
    return m;
}

// ------------------------------------------------------------------ the estimate

FitResult estimate_fit(const GgufInfo &model, uint64_t file_bytes, const Machine &machine, int context, KvType kv) {
    FitResult r;
    r.weights = file_bytes;

    const double bytes_per_element = kv == KvType::F16 ? 2.0 : kv == KvType::Q8 ? 1.0625 : 0.5625;
    if (model.has_shape()) {
        const uint64_t kv_heads = model.head_count_kv ? model.head_count_kv : model.head_count;
        const uint64_t head_dim = model.key_length ? model.key_length : model.embedding_length / model.head_count;
        r.kv_cache =
            static_cast<uint64_t>(2.0 * static_cast<double>(model.block_count * kv_heads * head_dim) * context * bytes_per_element);
    } else {
        r.note = "the model header has no layer shape; the KV cache is not counted";
    }
    r.buffers = static_cast<uint64_t>(0.08 * static_cast<double>(file_bytes)) + (256ull << 20);
    r.need = r.weights + r.kv_cache + r.buffers;

    // Where it would run: the GPU when the whole thing fits there, else system memory.
    const uint64_t vram_limit = static_cast<uint64_t>(0.9 * static_cast<double>(machine.vram_total));
    const uint64_t ram_basis = machine.ram_free ? machine.ram_free : static_cast<uint64_t>(0.7 * static_cast<double>(machine.ram_total));
    const uint64_t ram_limit = static_cast<uint64_t>(0.9 * static_cast<double>(ram_basis));
    if (machine.vram_total && r.need <= vram_limit) {
        r.on_gpu = true;
        r.limit = vram_limit;
        r.fits = true;
    } else {
        r.limit = ram_limit;
        r.fits = r.need <= ram_limit;
    }
    if (!r.fits) {
        r.note = "needs " + format_bytes(r.need) + ", this machine offers " + format_bytes(r.limit) +
                 (machine.vram_total ? " (GPU memory too small too)" : "");
        return r;
    }

    // Generation reads the active weights once per token: speed is memory bandwidth over those bytes, at about 60 % efficiency.
    if (!r.on_gpu && machine.mem_bandwidth > 0) {
        double active = 1.0;
        if (model.expert_count > 1 && model.expert_used_count > 0)
            active = 0.35 + 0.65 * static_cast<double>(model.expert_used_count) / static_cast<double>(model.expert_count);
        r.tokens_per_second = 0.6 * machine.mem_bandwidth / (static_cast<double>(file_bytes) * active);
    } else if (r.on_gpu) {
        r.note = "runs on the GPU backend; speed is not estimated";
    }
    if (r.fits && r.need > static_cast<uint64_t>(0.7 * static_cast<double>(r.limit)))
        r.note += (r.note.empty() ? "" : "; ") + std::string("tight: other programs may push it into swap");
    return r;
}

} // namespace agent::cli
