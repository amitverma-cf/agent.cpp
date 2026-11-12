#pragma once

// Will this model run on this machine, and about how fast? Everything here is an estimate from the model's own header and a short
// memory-bandwidth measurement; there is no model catalogue and no labels. The numbers say "about".

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// ------------------------------------------------------------------ GGUF header

struct GgufInfo {
    bool valid = false;       // the magic and version were right
    bool complete = false;    // every metadata entry was read (false: the buffer ended first, the fields found so far are kept)
    std::string architecture; // general.architecture
    uint64_t block_count = 0;
    uint64_t embedding_length = 0;
    uint64_t head_count = 0;
    uint64_t head_count_kv = 0; // the largest per-layer value when the file stores an array
    uint64_t context_length = 0;
    uint64_t key_length = 0; // per-head key size; 0 = embedding_length / head_count
    uint64_t expert_count = 0;
    uint64_t expert_used_count = 0;
    uint64_t tensor_count = 0;
    bool has_shape() const { return block_count && embedding_length && head_count; }
};

// Reads the metadata of a GGUF file from its first bytes (a few MiB are enough: tokenizer data comes after the hyper-parameters).
GgufInfo parse_gguf_header(std::string_view bytes);

// ------------------------------------------------------------------ the machine

struct Machine {
    uint64_t ram_total = 0;
    uint64_t ram_free = 0;    // available for new allocations; 0 = unknown
    uint64_t vram_total = 0;  // the largest GPU's memory; 0 = no GPU found or unknown
    double mem_bandwidth = 0; // bytes per second, measured; 0 = not measured
    int cores = 0;
};

// Reads RAM, GPU memory (nvidia-smi when present) and core count. measure_bandwidth runs a ~0.3 s multi-threaded read test.
Machine probe_machine(bool measure_bandwidth);

// ------------------------------------------------------------------ the estimate

enum class KvType { F16, Q8, Q4 };

struct FitResult {
    bool fits = false;
    bool on_gpu = false; // fits in GPU memory (a GPU backend would be used)
    uint64_t weights = 0;
    uint64_t kv_cache = 0;
    uint64_t buffers = 0;
    uint64_t need = 0;            // weights + kv_cache + buffers
    uint64_t limit = 0;           // what this machine offers (free RAM, or VRAM when on_gpu)
    double tokens_per_second = 0; // generation, about; 0 = not estimated
    std::string note;             // why it does not fit, or a caution
};

// file_bytes: the model file(s). context: tokens of context to reserve KV cache for.
FitResult estimate_fit(const GgufInfo &model, uint64_t file_bytes, const Machine &machine, int context, KvType kv = KvType::Q8);

} // namespace agent::cli
