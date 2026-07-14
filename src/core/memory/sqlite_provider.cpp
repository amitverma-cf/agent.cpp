#include "memory_ops.hpp"

#include <agent-cpp/agent.hpp>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <list>
#include <mutex>
#include <sqlite-vec.h>
#include <sqlite3.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace agent::memory {

namespace {

struct CacheEntry {
    std::string value;
    bool dirty = false;
    std::list<std::string>::iterator lru_it;
};

Result<void> exec_simple(sqlite3 *db, const char *sql) {
    char *errmsg = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string msg = errmsg ? errmsg : "unknown sqlite error";
        sqlite3_free(errmsg);
        return fail(ErrorCode::FilesystemError, "SQLite error: " + msg);
    }
    return ok();
}

} // namespace

struct SqliteMemoryState {
    sqlite3 *write_conn = nullptr;
    sqlite3 *read_conn = nullptr;

    std::mutex cache_mutex;
    std::unordered_map<std::string, CacheEntry> cache;
    std::list<std::string> lru;
    size_t cache_bytes = 0;
    size_t dirty_bytes = 0;

    std::mutex signal_mutex;
    std::condition_variable cv;
    bool wake_requested = false;
    bool stop_requested = false;
    bool clear_requested = false;
    bool clear_done = false;
    Result<void> clear_result;

    std::thread writer_thread;

    size_t max_cache_bytes = 16 * 1024 * 1024;
    size_t flush_dirty_threshold_bytes = 4 * 1024 * 1024;
    std::chrono::milliseconds flush_interval{500};

    ~SqliteMemoryState() {
        {
            std::lock_guard<std::mutex> lk(signal_mutex);
            stop_requested = true;
            cv.notify_all();
        }
        if (writer_thread.joinable()) writer_thread.join();

        if (write_conn) sqlite3_close(write_conn);
        if (read_conn) sqlite3_close(read_conn);
    }
};

namespace {

std::shared_ptr<SqliteMemoryState> get_state(Session &session) { return std::static_pointer_cast<SqliteMemoryState>(session.memory_state); }

void evict_if_needed_locked(SqliteMemoryState &state) {
    while (state.cache_bytes > state.max_cache_bytes && !state.lru.empty()) {
        bool evicted_any = false;
        for (auto rit = state.lru.rbegin(); rit != state.lru.rend(); ++rit) {
            auto cit = state.cache.find(*rit);
            if (cit != state.cache.end() && !cit->second.dirty) {
                size_t sz = cit->first.size() + cit->second.value.size();
                state.lru.erase(std::next(rit).base());
                state.cache.erase(cit);
                state.cache_bytes -= sz;
                evicted_any = true;
                break;
            }
        }
        if (!evicted_any) break;
    }
}

void flush_dirty(SqliteMemoryState &state) {
    std::vector<std::pair<std::string, std::string>> batch;
    {
        std::lock_guard<std::mutex> lk(state.cache_mutex);
        for (auto &[key, entry] : state.cache) {
            if (entry.dirty) batch.emplace_back(key, entry.value);
        }
    }
    if (batch.empty()) return;

    auto begin = exec_simple(state.write_conn, "BEGIN;");
    if (!begin.ok) return;

    bool write_ok = true;
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(state.write_conn, "INSERT OR REPLACE INTO kv(key,value) VALUES(?,?);", -1, &stmt, nullptr) != SQLITE_OK) {
        write_ok = false;
    } else {
        for (const auto &[key, value] : batch) {
            sqlite3_bind_text(stmt, 1, key.data(), static_cast<int>(key.size()), SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
            int rc = sqlite3_step(stmt);
            if (rc != SQLITE_DONE) {
                write_ok = false;
                break;
            }
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
    }

    if (!write_ok) {
        exec_simple(state.write_conn, "ROLLBACK;");
        return;
    }

    auto commit = exec_simple(state.write_conn, "COMMIT;");
    if (!commit.ok) {
        exec_simple(state.write_conn, "ROLLBACK;");
        return;
    }

    {
        std::lock_guard<std::mutex> lk(state.cache_mutex);
        for (const auto &[key, value] : batch) {
            auto it = state.cache.find(key);
            if (it == state.cache.end()) continue;
            if (it->second.dirty && it->second.value == value) {
                it->second.dirty = false;
                size_t sz = key.size() + value.size();
                if (state.dirty_bytes >= sz) state.dirty_bytes -= sz;
                else state.dirty_bytes = 0;
            }
        }
        evict_if_needed_locked(state);
    }
}

void writer_loop(SqliteMemoryState *state) {
    while (true) {
        std::unique_lock<std::mutex> lk(state->signal_mutex);
        state->cv.wait_for(lk, state->flush_interval,
                           [&] { return state->wake_requested || state->stop_requested || state->clear_requested; });
        bool do_stop = state->stop_requested;
        bool do_clear = state->clear_requested;
        state->wake_requested = false;
        lk.unlock();

        if (do_clear) {
            Result<void> r = exec_simple(state->write_conn, "DELETE FROM kv;");
            std::lock_guard<std::mutex> lk2(state->signal_mutex);
            state->clear_result = r;
            state->clear_requested = false;
            state->clear_done = true;
            state->cv.notify_all();
        }

        flush_dirty(*state);

        if (do_stop) break;
    }
}

} // namespace

Result<void> init_sqlite(Session &session) {
    const std::string &db_path = session.config.memory.db_path;
    if (db_path.empty()) return fail(ErrorCode::InvalidConfig, "SQLite memory requires a non-empty db_path.");

    std::string open_uri = db_path;
    int open_flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    if (db_path == ":memory:") {
        open_uri = "file::memory:?cache=shared";
        open_flags |= SQLITE_OPEN_URI;
    } else {
        std::filesystem::path p(db_path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            if (ec) return fail(ErrorCode::FilesystemError, "Failed to create memory directory: " + ec.message());
        }
    }

    auto state = std::make_shared<SqliteMemoryState>();
    state->max_cache_bytes = session.config.memory.max_cache_bytes;
    state->flush_dirty_threshold_bytes = session.config.memory.flush_dirty_threshold_bytes;
    state->flush_interval = std::chrono::milliseconds(session.config.memory.flush_interval_ms);

    if (sqlite3_open_v2(open_uri.c_str(), &state->write_conn, open_flags, nullptr) != SQLITE_OK) {
        std::string err = state->write_conn ? sqlite3_errmsg(state->write_conn) : "unknown error";
        return fail(ErrorCode::ProviderInitFailed, "Failed to open SQLite db (write): " + err);
    }
    sqlite3_vec_init(state->write_conn, nullptr, nullptr);

    if (auto r = exec_simple(state->write_conn, "PRAGMA journal_mode=WAL;"); !r.ok) return r;
    if (auto r = exec_simple(state->write_conn, "PRAGMA synchronous=NORMAL;"); !r.ok) return r;
    if (auto r = exec_simple(state->write_conn, "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);"); !r.ok) return r;

    if (sqlite3_open_v2(open_uri.c_str(), &state->read_conn, open_flags, nullptr) != SQLITE_OK) {
        std::string err = state->read_conn ? sqlite3_errmsg(state->read_conn) : "unknown error";
        return fail(ErrorCode::ProviderInitFailed, "Failed to open SQLite db (read): " + err);
    }
    sqlite3_vec_init(state->read_conn, nullptr, nullptr);

    state->writer_thread = std::thread(writer_loop, state.get());

    session.memory_state = state;
    return ok();
}

Result<void> store_sqlite(Session &session, std::string_view key, std::string_view value) {
    auto state = get_state(session);
    if (!state) return fail(ErrorCode::ProviderInitFailed, "SQLite memory not initialized.");

    std::string k(key);
    bool need_wake = false;
    {
        std::lock_guard<std::mutex> lk(state->cache_mutex);
        auto it = state->cache.find(k);
        if (it != state->cache.end()) {
            size_t old_size = k.size() + it->second.value.size();
            size_t new_size = k.size() + value.size();
            if (it->second.dirty) state->dirty_bytes = state->dirty_bytes - old_size + new_size;
            else state->dirty_bytes += new_size;
            state->cache_bytes = state->cache_bytes - old_size + new_size;
            it->second.value.assign(value);
            it->second.dirty = true;
            state->lru.erase(it->second.lru_it);
            state->lru.push_front(k);
            it->second.lru_it = state->lru.begin();
        } else {
            state->lru.push_front(k);
            CacheEntry entry;
            entry.value.assign(value);
            entry.dirty = true;
            entry.lru_it = state->lru.begin();
            size_t sz = k.size() + value.size();
            state->cache_bytes += sz;
            state->dirty_bytes += sz;
            state->cache.emplace(std::move(k), std::move(entry));
        }
        if (state->dirty_bytes >= state->flush_dirty_threshold_bytes) need_wake = true;
    }

    if (need_wake) {
        std::lock_guard<std::mutex> lk(state->signal_mutex);
        state->wake_requested = true;
        state->cv.notify_one();
    }

    return ok();
}

Result<std::string> retrieve_sqlite(Session &session, std::string_view key) {
    auto state = get_state(session);
    if (!state) return fail<std::string>(ErrorCode::ProviderInitFailed, "SQLite memory not initialized.");

    std::string k(key);
    {
        std::lock_guard<std::mutex> lk(state->cache_mutex);
        auto it = state->cache.find(k);
        if (it != state->cache.end()) {
            state->lru.erase(it->second.lru_it);
            state->lru.push_front(k);
            it->second.lru_it = state->lru.begin();
            return ok(it->second.value);
        }
    }

    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(state->read_conn, "SELECT value FROM kv WHERE key = ?;", -1, &stmt, nullptr) != SQLITE_OK)
        return fail<std::string>(ErrorCode::FilesystemError, "SQLite prepare failed.");
    sqlite3_bind_text(stmt, 1, k.data(), static_cast<int>(k.size()), SQLITE_TRANSIENT);

    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return fail<std::string>(ErrorCode::KeyNotFound, "Key not found: " + k);
    }
    const auto *text = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
    int len = sqlite3_column_bytes(stmt, 0);
    std::string value(text, static_cast<size_t>(len));
    sqlite3_finalize(stmt);

    {
        std::lock_guard<std::mutex> lk(state->cache_mutex);
        if (state->cache.find(k) == state->cache.end()) {
            state->lru.push_front(k);
            CacheEntry entry;
            entry.value = value;
            entry.dirty = false;
            entry.lru_it = state->lru.begin();
            state->cache_bytes += k.size() + value.size();
            state->cache.emplace(k, std::move(entry));
            evict_if_needed_locked(*state);
        }
    }

    return ok(value);
}

Result<void> clear_sqlite(Session &session) {
    auto state = get_state(session);
    if (!state) return fail(ErrorCode::ProviderInitFailed, "SQLite memory not initialized.");

    {
        std::lock_guard<std::mutex> lk(state->cache_mutex);
        state->cache.clear();
        state->lru.clear();
        state->cache_bytes = 0;
        state->dirty_bytes = 0;
    }

    std::unique_lock<std::mutex> lk(state->signal_mutex);
    state->clear_requested = true;
    state->clear_done = false;
    state->cv.notify_one();

    state->cv.wait(lk, [&] { return state->clear_done; });
    Result<void> result = state->clear_result;
    lk.unlock();
    return result;
}

} // namespace agent::memory
