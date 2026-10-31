// SQLite CheckpointerOps: one connection per opened checkpointer (WAL), serialized by a mutex.

#include <agent-cpp/agent.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <sqlite3.h>
#include <string>
#include <vector>

namespace agent::store {

namespace {

struct SqliteCheckpointer {
    std::mutex mutex;
    sqlite3 *db = nullptr;
    ~SqliteCheckpointer() {
        if (db) sqlite3_close(db);
    }
};

int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

Error sqlite_error(sqlite3 *db, std::string_view what) {
    return make_error(ErrorCode::FilesystemError, "sqlite checkpointer: " + std::string(what) + ": " + sqlite3_errmsg(db));
}

Result<void> exec(sqlite3 *db, const char *sql) {
    char *msg = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &msg) != SQLITE_OK) {
        std::string text = msg ? msg : "unknown error";
        sqlite3_free(msg);
        return fail(ErrorCode::FilesystemError, "sqlite checkpointer: " + text);
    }
    return ok();
}

// Finalizes on scope exit.
struct Stmt {
    sqlite3_stmt *s = nullptr;
    ~Stmt() {
        if (s) sqlite3_finalize(s);
    }
    bool prepare(sqlite3 *db, const char *sql) { return sqlite3_prepare_v2(db, sql, -1, &s, nullptr) == SQLITE_OK; }
    void text(int i, std::string_view v) { sqlite3_bind_text(s, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT); }
    void int64(int i, int64_t v) { sqlite3_bind_int64(s, i, v); }
    std::string column_text(int i) {
        const auto *p = reinterpret_cast<const char *>(sqlite3_column_text(s, i));
        return p ? std::string(p, static_cast<size_t>(sqlite3_column_bytes(s, i))) : std::string();
    }
};

constexpr const char *kSchema = "CREATE TABLE IF NOT EXISTS graph_checkpoints ("
                                " thread_id TEXT NOT NULL,"
                                " step INTEGER NOT NULL,"
                                " created_ms INTEGER NOT NULL,"
                                " metadata TEXT NOT NULL,"
                                " blob TEXT NOT NULL,"
                                " PRIMARY KEY (thread_id, step)) WITHOUT ROWID;"
                                "CREATE INDEX IF NOT EXISTS graph_checkpoints_created ON graph_checkpoints (thread_id, created_ms);";

Result<std::shared_ptr<void>> cp_open(const CheckpointerConfig &config) {
    using R = Result<std::shared_ptr<void>>;
    const std::string path = config.path.empty() ? std::string(":memory:") : config.path;
    if (path != ":memory:") {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            if (ec)
                return fail<std::shared_ptr<void>>(ErrorCode::FilesystemError,
                                                   "sqlite checkpointer: cannot create directory: " + ec.message());
        }
    }
    auto state = std::make_shared<SqliteCheckpointer>();
    if (sqlite3_open_v2(path.c_str(), &state->db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr) !=
        SQLITE_OK) {
        std::string err = state->db ? sqlite3_errmsg(state->db) : "unknown error";
        return fail<std::shared_ptr<void>>(ErrorCode::ProviderInitFailed, "sqlite checkpointer: cannot open '" + path + "': " + err);
    }
    sqlite3_busy_timeout(state->db, 5000);
    for (const char *sql : {"PRAGMA journal_mode=WAL;", "PRAGMA synchronous=NORMAL;", kSchema}) {
        auto r = exec(state->db, sql);
        if (!r.ok) return fail<std::shared_ptr<void>>(r.error.code, r.error.message);
    }
    return R{.ok = true, .value = std::move(state), .error = {}};
}

Result<void> cp_put(void *state, std::string_view thread_id, uint64_t step, std::string_view blob, std::string_view metadata) {
    auto *s = static_cast<SqliteCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    Stmt st;
    if (!st.prepare(s->db,
                    "INSERT OR REPLACE INTO graph_checkpoints (thread_id, step, created_ms, metadata, blob) VALUES (?, ?, ?, ?, ?);"))
        return Result<void>{.ok = false, .error = sqlite_error(s->db, "prepare put")};
    st.text(1, thread_id);
    st.int64(2, static_cast<int64_t>(step));
    st.int64(3, now_ms());
    st.text(4, metadata);
    st.text(5, blob);
    if (sqlite3_step(st.s) != SQLITE_DONE) return Result<void>{.ok = false, .error = sqlite_error(s->db, "put")};
    return ok();
}

Result<std::string> select_blob(SqliteCheckpointer *s, const char *sql, std::string_view thread_id, const uint64_t *step) {
    std::lock_guard<std::mutex> lock(s->mutex);
    Stmt st;
    if (!st.prepare(s->db, sql)) return Result<std::string>{.ok = false, .value = {}, .error = sqlite_error(s->db, "prepare get")};
    st.text(1, thread_id);
    if (step) st.int64(2, static_cast<int64_t>(*step));
    const int rc = sqlite3_step(st.s);
    if (rc == SQLITE_ROW) return ok(st.column_text(0));
    if (rc == SQLITE_DONE) {
        std::string what = "no checkpoint";
        if (step) what += " at step " + std::to_string(*step);
        return fail<std::string>(ErrorCode::KeyNotFound, what + " for thread '" + std::string(thread_id) + "'");
    }
    return Result<std::string>{.ok = false, .value = {}, .error = sqlite_error(s->db, "get")};
}

Result<std::string> cp_get_latest(void *state, std::string_view thread_id) {
    return select_blob(static_cast<SqliteCheckpointer *>(state),
                       "SELECT blob FROM graph_checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1;", thread_id, nullptr);
}

Result<std::string> cp_get(void *state, std::string_view thread_id, uint64_t step) {
    return select_blob(static_cast<SqliteCheckpointer *>(state), "SELECT blob FROM graph_checkpoints WHERE thread_id = ? AND step = ?;",
                       thread_id, &step);
}

Result<std::vector<GraphCheckpointInfo>> cp_list(void *state, std::string_view thread_id) {
    using R = Result<std::vector<GraphCheckpointInfo>>;
    auto *s = static_cast<SqliteCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    Stmt st;
    if (!st.prepare(s->db, "SELECT step, created_ms, metadata FROM graph_checkpoints WHERE thread_id = ? ORDER BY step;"))
        return R{.ok = false, .value = {}, .error = sqlite_error(s->db, "prepare list")};
    st.text(1, thread_id);
    std::vector<GraphCheckpointInfo> out;
    int rc = 0;
    while ((rc = sqlite3_step(st.s)) == SQLITE_ROW)
        out.push_back(GraphCheckpointInfo{.step = static_cast<uint64_t>(sqlite3_column_int64(st.s, 0)),
                                          .timestamp_ms = sqlite3_column_int64(st.s, 1),
                                          .metadata = st.column_text(2)});
    if (rc != SQLITE_DONE) return R{.ok = false, .value = {}, .error = sqlite_error(s->db, "list")};
    return ok(std::move(out));
}

Result<void> cp_delete_thread(void *state, std::string_view thread_id) {
    auto *s = static_cast<SqliteCheckpointer *>(state);
    std::lock_guard<std::mutex> lock(s->mutex);
    Stmt st;
    if (!st.prepare(s->db, "DELETE FROM graph_checkpoints WHERE thread_id = ?;"))
        return Result<void>{.ok = false, .error = sqlite_error(s->db, "prepare delete")};
    st.text(1, thread_id);
    if (sqlite3_step(st.s) != SQLITE_DONE) return Result<void>{.ok = false, .error = sqlite_error(s->db, "delete")};
    return ok();
}

constexpr CheckpointerOps kSqliteCheckpointer{
    .open = cp_open, .put = cp_put, .get_latest = cp_get_latest, .get = cp_get, .list = cp_list, .delete_thread = cp_delete_thread};

} // namespace

const CheckpointerOps *sqlite_checkpointer() { return &kSqliteCheckpointer; }

} // namespace agent::store
