// Model registry: named Config templates loaded lazily into Sessions under a RAM/VRAM budget, with
// LRU eviction of idle, unpinned models. Layered on agent::init; no engine-specific code here beyond
// the memory estimate heuristics (documented in docs/api/models.md).

#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <simdjson.h>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2 // GetProcessMemoryInfo -> K32GetProcessMemoryInfo in kernel32 (no psapi.lib)
#endif
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

namespace agent {

namespace {

namespace fs = std::filesystem;

constexpr uint64_t kMiB = 1024ull * 1024ull;

int64_t wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string mib(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(bytes) / static_cast<double>(kMiB));
    return buf;
}

// -------------------------- Memory estimate --------------------------

// Per-engine heuristics: resident bytes ~= file_bytes * num / den + overhead (+ KV cache for llama).
struct EngineFactor {
    std::string_view kind;
    uint64_t num, den;
    uint64_t overhead;
};

constexpr EngineFactor kFactors[] = {
    {"mock", 1, 1, 0},             // test engine: the file size, exactly
    {"llama", 1, 1, 64 * kMiB},    // mmap'ed weights + compute buffers; KV cache added separately
    {"onnx", 3, 2, 32 * kMiB},     // ORT keeps an initializer copy + arena
    {"sdcpp", 6, 5, 256 * kMiB},   // weights + large diffusion compute buffers
    {"trellis", 6, 5, 512 * kMiB}, // several sub-models + 3D decoding buffers
    {"sherpa", 3, 2, 32 * kMiB},   // ORT sessions
    {"", 5, 4, 64 * kMiB},         // unknown engine
};

std::string lower(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

const EngineFactor &factor_for(const Config &config) {
    if (config.provider == AiProvider::Mock) return kFactors[0];
    // A registered name ("llama") or a plugin path (".../agent-engine-llama.so").
    const std::string name = lower(fs::path(config.engine).filename().string());
    for (const auto &f : kFactors)
        if (!f.kind.empty() && name.find(f.kind) != std::string::npos) return f;
    return kFactors[std::size(kFactors) - 1];
}

uint64_t file_size_or_zero(const fs::path &p) {
    std::error_code ec;
    const auto size = fs::file_size(p, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

uint64_t directory_bytes(const fs::path &dir) {
    uint64_t total = 0;
    std::error_code ec;
    fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code ec2;
        if (it->is_regular_file(ec2)) total += file_size_or_zero(it->path());
    }
    return total;
}

// "<stem>-00001-of-00003.gguf": the sizes of the other shards (the first one is counted by the caller).
uint64_t other_shard_bytes(const fs::path &first) {
    const std::string name = first.filename().string();
    const size_t of = name.rfind("-of-");
    if (of == std::string::npos || of < 6) return 0;
    const size_t dash = name.rfind('-', of - 1);
    if (dash == std::string::npos) return 0;
    const std::string idx = name.substr(dash + 1, of - dash - 1);
    const size_t dot = name.find('.', of);
    const std::string cnt = name.substr(of + 4, dot == std::string::npos ? std::string::npos : dot - of - 4);
    if (idx.empty() || idx.size() != cnt.size() || idx.find_first_not_of('0') != idx.size() - 1 || idx.back() != '1') return 0;
    int total = 0;
    for (char c : cnt) {
        if (c < '0' || c > '9' || total > 100000) return 0;
        total = total * 10 + (c - '0');
    }
    uint64_t bytes = 0;
    for (int i = 2; i <= total; ++i) {
        std::string n = std::to_string(i);
        if (n.size() < idx.size()) n.insert(0, idx.size() - n.size(), '0');
        bytes += file_size_or_zero(first.parent_path() / (name.substr(0, dash + 1) + n + name.substr(of)));
    }
    return bytes;
}

// Every string value in the engine_options JSON that names an existing regular file (mmproj, vocoder,
// tokens, ...), plus the top-level "gpu_layers" number.
void scan_options(simdjson::dom::element el, std::set<std::string> &files, const std::string &model) {
    using simdjson::dom::element_type;
    switch (el.type()) {
    case element_type::OBJECT: {
        simdjson::dom::object obj;
        if (el.get(obj)) return;
        for (auto field : obj) scan_options(field.value, files, model);
        break;
    }
    case element_type::ARRAY: {
        simdjson::dom::array arr;
        if (el.get(arr)) return;
        for (auto v : arr) scan_options(v, files, model);
        break;
    }
    case element_type::STRING: {
        std::string_view s;
        if (el.get(s) || s.empty() || s.size() > 4096 || s == model) return;
        std::error_code ec;
        if (fs::is_regular_file(fs::path(s), ec)) files.insert(std::string(s));
        break;
    }
    default: break;
    }
}

int gpu_layers_option(simdjson::dom::element root) {
    simdjson::dom::object obj;
    if (root.get(obj)) return -1;
    int64_t i = 0;
    if (!obj["gpu_layers"].get(i)) return static_cast<int>(std::clamp<int64_t>(i, -1, 1 << 20));
    double d = 0;
    if (!obj["gpu_layers"].get(d)) return static_cast<int>(std::clamp(d, -1.0, 1048576.0));
    return -1;
}

} // namespace

Result<ModelMemoryEstimate> estimate_model_memory(const ModelSpec &spec) {
    const Config &cfg = spec.config;
    int gpu_layers = -1;
    std::set<std::string> extra_files;
    if (!cfg.engine_options.empty()) {
        simdjson::dom::parser parser;
        simdjson::dom::element root;
        if (parser.parse(cfg.engine_options).get(root))
            return fail<ModelMemoryEstimate>(ErrorCode::InvalidConfig, "model '" + spec.name + "': engine_options is not valid JSON");
        gpu_layers = gpu_layers_option(root);
        if (spec.estimated_bytes == 0) scan_options(root, extra_files, cfg.model);
    }

    ModelMemoryEstimate est;
    const EngineFactor &f = factor_for(cfg);
    uint64_t total = spec.estimated_bytes;
    if (total == 0) {
        if (!cfg.model.empty()) {
            const fs::path model(cfg.model);
            std::error_code ec;
            if (fs::is_directory(model, ec)) {
                est.file_bytes += directory_bytes(model);
            } else if (fs::is_regular_file(model, ec)) {
                est.file_bytes += file_size_or_zero(model) + other_shard_bytes(model);
            } else if (cfg.provider == AiProvider::Engine) {
                return fail<ModelMemoryEstimate>(ErrorCode::FilesystemError, "model '" + spec.name + "': no such file: " + cfg.model +
                                                                                 " (set ModelSpec::estimated_bytes to skip the estimate)");
            }
        }
        for (const auto &p : extra_files) est.file_bytes += file_size_or_zero(p);
        total = est.file_bytes / f.den * f.num + (est.file_bytes % f.den) * f.num / f.den;
        if (f.kind == "llama") {
            // KV cache: n_ctx = context_window per sequence slot x max_parallel_subagents slots. Bytes per
            // token scale with model size; file/32768 matches current GQA models within ~2x (Qwen2.5-0.5B
            // Q4: ~12 KiB/token, Llama-3-8B Q4: ~128 KiB/token). Older MHA models need more.
            const uint64_t slots = static_cast<uint64_t>(std::max(1, cfg.max_parallel_subagents));
            const uint64_t ctx = static_cast<uint64_t>(std::max(1, cfg.context_window));
            const uint64_t per_token = std::max<uint64_t>(est.file_bytes / 32768, 64);
            total += ctx * slots * per_token;
        }
        total += f.overhead;
    }
    if (gpu_layers > 0) {
        // Best effort: offloaded weights and KV live in VRAM; only the host-side overhead stays in RAM.
        est.vram_bytes = total;
        est.ram_bytes = std::min(total, f.overhead);
    } else {
        est.ram_bytes = total;
    }
    return ok(est);
}

uint64_t process_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return static_cast<uint64_t>(pmc.WorkingSetSize);
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
    return static_cast<uint64_t>(info.resident_size);
#else
    std::FILE *f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long size = 0, resident = 0;
    const int n = std::fscanf(f, "%llu %llu", &size, &resident);
    std::fclose(f);
    const long page = sysconf(_SC_PAGESIZE);
    if (n != 2 || page <= 0) return 0;
    return static_cast<uint64_t>(resident) * static_cast<uint64_t>(page);
#endif
}

// -------------------------- Registry --------------------------

namespace {

enum class EntryStatus { Unloaded, Loading, Loaded };

struct Entry {
    ModelSpec spec;
    EntryStatus status = EntryStatus::Unloaded;
    std::unique_ptr<Session> session;
    std::shared_ptr<std::mutex> turn_mutex;
    uint64_t ram = 0, vram = 0; // charged while Loading / Loaded
    int leases = 0;
    uint64_t use_tick = 0; // LRU order
    int64_t loaded_at_ms = 0, last_used_ms = 0, load_time_ms = 0;
};

struct RegistryState {
    std::mutex mu;
    std::condition_variable cv;
    ModelBudget budget;
    // unique_ptr: leases point at their Entry; entries are never erased.
    std::map<std::string, std::unique_ptr<Entry>, std::less<>> entries;
    uint64_t ram_used = 0, vram_used = 0;           // Loading + Loaded + still being destroyed
    uint64_t ram_releasing = 0, vram_releasing = 0; // evicted Sessions not destroyed yet
    uint64_t tick = 0;
};

// Sessions taken out of the registry under the lock, destroyed after it is dropped (model teardown can
// take a while). Their bytes stay charged until then.
struct Doomed {
    std::unique_ptr<Session> session;
    uint64_t ram = 0, vram = 0;
};

RegistryState &state_of(const ModelRegistry &r) { return *static_cast<RegistryState *>(r.state.get()); }

void touch(RegistryState &st, Entry &e) {
    e.use_tick = ++st.tick;
    e.last_used_ms = wall_ms();
}

// Caller holds st.mu. Moves a loaded, idle model's Session out; its bytes become "releasing".
Doomed take_out(RegistryState &st, Entry &e) {
    Doomed d{std::move(e.session), e.ram, e.vram};
    st.ram_releasing += e.ram;
    st.vram_releasing += e.vram;
    e.status = EntryStatus::Unloaded;
    e.ram = e.vram = 0;
    e.turn_mutex.reset();
    return d;
}

// Destroys the Sessions without the lock, then uncharges them. Takes the lock itself.
void destroy(RegistryState &st, std::unique_lock<std::mutex> &lock, std::vector<Doomed> &doomed) {
    if (doomed.empty()) return;
    lock.unlock();
    uint64_t ram = 0, vram = 0;
    for (auto &d : doomed) {
        d.session.reset();
        ram += d.ram;
        vram += d.vram;
    }
    doomed.clear();
    lock.lock();
    st.ram_used -= ram;
    st.vram_used -= vram;
    st.ram_releasing -= ram;
    st.vram_releasing -= vram;
    st.cv.notify_all();
}

bool fits(uint64_t limit, uint64_t used, uint64_t need) { return limit == 0 || used + need <= limit; }

// Idle, unpinned, loaded models, least recently used first.
std::vector<Entry *> evictable(RegistryState &st, const Entry *except) {
    std::vector<Entry *> out;
    for (auto &[name, e] : st.entries)
        if (e.get() != except && e->status == EntryStatus::Loaded && e->leases == 0 && !e->spec.pinned) out.push_back(e.get());
    std::sort(out.begin(), out.end(), [](const Entry *a, const Entry *b) { return a->use_tick < b->use_tick; });
    return out;
}

// The LRU prefix of candidates whose eviction makes need fit; false when even all of them do not suffice.
// used_* exclude nothing: bytes still being destroyed count as used.
bool plan(const RegistryState &st, const std::vector<Entry *> &candidates, uint64_t need_ram, uint64_t need_vram, size_t &count) {
    uint64_t ram = st.ram_used, vram = st.vram_used;
    for (count = 0;; ++count) {
        if (fits(st.budget.ram_bytes, ram, need_ram) && fits(st.budget.vram_bytes, vram, need_vram)) return true;
        if (count == candidates.size()) return false;
        ram -= candidates[count]->ram;
        vram -= candidates[count]->vram;
    }
}

std::string out_of_memory_message(const RegistryState &st, const Entry &e, const ModelMemoryEstimate &est) {
    std::string msg = "model '" + e.spec.name + "' needs " + mib(est.ram_bytes) + " RAM";
    if (est.vram_bytes) msg += " + " + mib(est.vram_bytes) + " VRAM";
    msg += "; budget " + (st.budget.ram_bytes ? mib(st.budget.ram_bytes) : std::string("unlimited")) + " RAM";
    if (st.budget.vram_bytes) msg += " / " + mib(st.budget.vram_bytes) + " VRAM";
    msg += ", and the loaded models that cannot be evicted use " + mib(st.ram_used);
    std::string held;
    for (const auto &[name, other] : st.entries) {
        if (other->status == EntryStatus::Unloaded || other.get() == &e) continue;
        held += held.empty() ? " (" : ", ";
        held += name + " " + mib(other->ram + other->vram);
        if (other->status == EntryStatus::Loading) held += " loading";
        else if (other->leases) held += " in use";
        else if (other->spec.pinned) held += " pinned";
    }
    if (!held.empty()) msg += held + ")";
    return msg + ". Release a lease, unload a model, raise the budget or set allow_overcommit.";
}

ModelLease make_lease(const std::shared_ptr<void> &state, Entry &e) {
    ModelLease lease;
    lease.session = e.session.get();
    lease.name = e.spec.name;
    lease.turn_mutex = e.turn_mutex;
    auto st = std::static_pointer_cast<RegistryState>(state);
    lease.hold = std::shared_ptr<void>(static_cast<void *>(&e), [st](void *p) {
        auto *entry = static_cast<Entry *>(p);
        std::lock_guard<std::mutex> lock(st->mu);
        --entry->leases;
        touch(*st, *entry);
        st->cv.notify_all();
    });
    return lease;
}

// Unloads e (idle) and waits out a Loading state. Caller holds lock.
Result<void> unload_locked(RegistryState &st, std::unique_lock<std::mutex> &lock, Entry &e) {
    st.cv.wait(lock, [&e] { return e.status != EntryStatus::Loading; });
    if (e.status == EntryStatus::Unloaded) return ok();
    if (e.leases > 0)
        return fail(ErrorCode::InvalidConfig, "model '" + e.spec.name + "' is in use (" + std::to_string(e.leases) + " lease(s))");
    std::vector<Doomed> doomed;
    doomed.push_back(take_out(st, e));
    destroy(st, lock, doomed);
    return ok();
}

} // namespace

ModelRegistry::ModelRegistry(ModelBudget budget) {
    auto st = std::make_shared<RegistryState>();
    st->budget = budget;
    state = std::move(st);
}

Result<void> register_model(ModelRegistry &registry, ModelSpec spec) {
    if (spec.name.empty()) return fail(ErrorCode::InvalidConfig, "register_model: the model name is empty");
    RegistryState &st = state_of(registry);
    std::unique_lock<std::mutex> lock(st.mu);
    auto it = st.entries.find(spec.name);
    if (it == st.entries.end()) {
        auto e = std::make_unique<Entry>();
        std::string name = spec.name;
        e->spec = std::move(spec);
        st.entries.emplace(std::move(name), std::move(e));
        return ok();
    }
    Entry &e = *it->second;
    if (auto r = unload_locked(st, lock, e); !r.ok) return fail(r.error.code, "register_model: " + r.error.message);
    // unload_locked may have dropped the lock; someone may have loaded it again meanwhile.
    if (e.status != EntryStatus::Unloaded) return fail(ErrorCode::InvalidConfig, "register_model: model '" + e.spec.name + "' is in use");
    e.spec = std::move(spec);
    return ok();
}

bool has_model(const ModelRegistry &registry, std::string_view name) {
    RegistryState &st = state_of(registry);
    std::lock_guard<std::mutex> lock(st.mu);
    return st.entries.find(name) != st.entries.end();
}

Result<ModelLease> acquire_model(ModelRegistry &registry, std::string_view name) {
    const std::shared_ptr<void> keep = registry.state; // the handle may be dropped while we load
    RegistryState &st = state_of(registry);
    std::unique_lock<std::mutex> lock(st.mu);
    auto it = st.entries.find(name);
    if (it == st.entries.end()) return fail<ModelLease>(ErrorCode::KeyNotFound, "no model named '" + std::string(name) + "' is registered");
    Entry &e = *it->second;

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, st.budget.wait_ms));
    ModelMemoryEstimate est;
    for (;;) {
        if (e.status == EntryStatus::Loaded) {
            ++e.leases;
            touch(st, e);
            return ok(make_lease(keep, e));
        }
        if (e.status == EntryStatus::Loading) {
            st.cv.wait(lock, [&e] { return e.status != EntryStatus::Loading; });
            continue;
        }

        auto estimate = estimate_model_memory(e.spec);
        if (!estimate.ok) return fail<ModelLease>(estimate.error.code, estimate.error.message);
        est = estimate.value;

        std::vector<Entry *> candidates = evictable(st, &e);
        size_t count = 0;
        const bool ok_plan = plan(st, candidates, est.ram_bytes, est.vram_bytes, count);
        if (!ok_plan) {
            const bool frees_pending = st.ram_releasing > 0 || st.vram_releasing > 0;
            if (frees_pending) { // another thread is destroying evicted models: wait for it, then re-plan
                st.cv.wait(lock);
                continue;
            }
            const bool can_wait = std::chrono::steady_clock::now() < deadline;
            if (!st.budget.allow_overcommit && can_wait) {
                st.cv.wait_until(lock, deadline);
                continue;
            }
            if (!st.budget.allow_overcommit)
                return fail<ModelLease>(ErrorCode::OutOfMemory, "acquire_model: " + out_of_memory_message(st, e, est));
            count = candidates.size(); // overcommit: get as close to the budget as possible
        }

        std::vector<Doomed> doomed;
        for (size_t i = 0; i < count; ++i) doomed.push_back(take_out(st, *candidates[i]));
        e.status = EntryStatus::Loading; // reserve before dropping the lock
        e.ram = est.ram_bytes;
        e.vram = est.vram_bytes;
        st.ram_used += e.ram;
        st.vram_used += e.vram;
        destroy(st, lock, doomed);
        break;
    }

    // Load without the lock: other models stay usable meanwhile.
    const ModelSpec spec = e.spec;
    lock.unlock();
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<Session> session;
    Error error;
    auto created = init(spec.config);
    if (created.ok) {
        session = std::make_unique<Session>(std::move(created.value));
        if (spec.on_load) {
            if (auto r = spec.on_load(*session, spec, spec.on_load_user_data); !r.ok) {
                error = r.error;
                error.message = "model '" + spec.name + "': on_load: " + error.message;
                session.reset();
            }
        }
    } else {
        error = created.error;
        error.message = "model '" + spec.name + "': " + error.message;
    }
    const int64_t load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    lock.lock();

    if (!session) {
        st.ram_used -= e.ram;
        st.vram_used -= e.vram;
        e.ram = e.vram = 0;
        e.status = EntryStatus::Unloaded;
        st.cv.notify_all();
        return fail<ModelLease>(error.code, error.message);
    }
    e.session = std::move(session);
    e.turn_mutex = std::make_shared<std::mutex>();
    e.status = EntryStatus::Loaded;
    e.leases = 1;
    e.loaded_at_ms = wall_ms();
    e.load_time_ms = load_ms;
    touch(st, e);
    st.cv.notify_all();
    return ok(make_lease(keep, e));
}

void release_model(ModelLease &lease) {
    lease.hold.reset();
    lease.turn_mutex.reset();
    lease.session = nullptr;
}

Result<void> unload_model(ModelRegistry &registry, std::string_view name) {
    RegistryState &st = state_of(registry);
    std::unique_lock<std::mutex> lock(st.mu);
    auto it = st.entries.find(name);
    if (it == st.entries.end()) return fail(ErrorCode::KeyNotFound, "no model named '" + std::string(name) + "' is registered");
    if (auto r = unload_locked(st, lock, *it->second); !r.ok) return fail(r.error.code, "unload_model: " + r.error.message);
    return ok();
}

std::vector<LoadedModelInfo> loaded_models(const ModelRegistry &registry) {
    RegistryState &st = state_of(registry);
    std::vector<std::pair<uint64_t, LoadedModelInfo>> rows;
    {
        std::lock_guard<std::mutex> lock(st.mu);
        for (const auto &[name, e] : st.entries) {
            if (e->status == EntryStatus::Unloaded) continue;
            rows.push_back({e->use_tick, LoadedModelInfo{.name = name,
                                                         .ram_bytes = e->ram,
                                                         .vram_bytes = e->vram,
                                                         .pinned = e->spec.pinned,
                                                         .leases = e->leases,
                                                         .loaded_at_ms = e->loaded_at_ms,
                                                         .last_used_ms = e->last_used_ms,
                                                         .load_time_ms = e->load_time_ms}});
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<LoadedModelInfo> out;
    out.reserve(rows.size());
    for (auto &r : rows) out.push_back(std::move(r.second));
    return out;
}

void set_budget(ModelRegistry &registry, const ModelBudget &budget) {
    RegistryState &st = state_of(registry);
    std::unique_lock<std::mutex> lock(st.mu);
    st.budget = budget;
    std::vector<Entry *> candidates = evictable(st, nullptr);
    size_t count = 0;
    if (!plan(st, candidates, 0, 0, count)) count = candidates.size();
    std::vector<Doomed> doomed;
    for (size_t i = 0; i < count; ++i) doomed.push_back(take_out(st, *candidates[i]));
    destroy(st, lock, doomed);
}

ModelBudget get_budget(const ModelRegistry &registry) {
    RegistryState &st = state_of(registry);
    std::lock_guard<std::mutex> lock(st.mu);
    return st.budget;
}

} // namespace agent
