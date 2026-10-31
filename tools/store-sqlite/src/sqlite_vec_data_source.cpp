#include "core/utils/utils.hpp"
#include "hal/path.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <filesystem>
#include <mutex>
#include <simdjson.h>
#include <sqlite-vec.h>
#include <sqlite3.h>
#include <string>

namespace agent {

namespace {

struct SqliteVecState {
    sqlite3 *conn = nullptr;
    std::string vec_table;
    std::string meta_table;
    int dimensions = 0;
    std::mutex mutex;

    ~SqliteVecState() {
        if (conn) sqlite3_close(conn);
    }
};

Result<void> exec_simple(sqlite3 *db, const std::string &sql) {
    char *errmsg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string msg = errmsg ? errmsg : "unknown sqlite error";
        sqlite3_free(errmsg);
        return fail(ErrorCode::FilesystemError, "sqlite-vec: " + msg);
    }
    return ok();
}

std::string vector_to_json(std::span<const float> v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) s += ',';
        s += std::to_string(v[i]);
    }
    s += ']';
    return s;
}

Result<std::string> sqlite_vec_query(void *state_ptr, std::string_view query, void *) {
    auto *state = static_cast<SqliteVecState *>(state_ptr);
    if (!state) return fail<std::string>(ErrorCode::InvalidConfig, "sqlite-vec: data source not initialized.");

    simdjson::ondemand::parser parser;
    simdjson::padded_string json(query.data(), query.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc)) return fail<std::string>(ErrorCode::ParseError, "sqlite-vec: invalid query JSON.");

    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj)) return fail<std::string>(ErrorCode::ParseError, "sqlite-vec: expected a JSON object.");

    std::vector<float> vec;
    int64_t top_k = 5;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error()) continue;
        if (k.value() == "vector") {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr)) continue;
            for (auto elem : arr) {
                double d = 0.0;
                if (!elem.get_double().get(d)) vec.push_back(static_cast<float>(d));
            }
        } else if (k.value() == "top_k") {
            int64_t v;
            if (!field.value().get_int64().get(v) && v > 0) top_k = v;
        }
    }

    if (vec.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "sqlite-vec: query requires a non-empty 'vector'.");
    if (static_cast<int>(vec.size()) != state->dimensions)
        return fail<std::string>(ErrorCode::InvalidConfig, "sqlite-vec: query vector dimensionality (" + std::to_string(vec.size()) +
                                                               ") does not match the data "
                                                               "source's dimensions (" +
                                                               std::to_string(state->dimensions) + ").");

    std::string vec_json = vector_to_json(vec);
    std::lock_guard<std::mutex> lock(state->mutex);

    std::string sql = "SELECT rowid, distance FROM " + state->vec_table + " WHERE embedding MATCH vec_f32(?) ORDER BY distance LIMIT ?;";
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(state->conn, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
        return fail<std::string>(ErrorCode::FilesystemError, "sqlite-vec: query prepare failed.");
    sqlite3_bind_text(stmt, 1, vec_json.data(), static_cast<int>(vec_json.size()), SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, top_k);

    std::string result = "[";
    bool first = true;
    std::string meta_sql = "SELECT text FROM " + state->meta_table + " WHERE rowid = ?;";
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        sqlite3_int64 rowid = sqlite3_column_int64(stmt, 0);
        double distance = sqlite3_column_double(stmt, 1);

        std::string text;
        sqlite3_stmt *meta_stmt = nullptr;
        if (sqlite3_prepare_v2(state->conn, meta_sql.c_str(), -1, &meta_stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(meta_stmt, 1, rowid);
            if (sqlite3_step(meta_stmt) == SQLITE_ROW) {
                const auto *t = reinterpret_cast<const char *>(sqlite3_column_text(meta_stmt, 0));
                int len = sqlite3_column_bytes(meta_stmt, 0);
                if (t && len > 0) text.assign(t, static_cast<size_t>(len));
            }
            sqlite3_finalize(meta_stmt);
        }

        if (!first) result += ',';
        first = false;
        result += "{\"text\":\"";
        utils::escape_json_string(text, result);
        result += "\",\"distance\":";
        result += std::to_string(distance);
        result += '}';
    }
    sqlite3_finalize(stmt);
    result += ']';
    return ok(result);
}

} // namespace

Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(std::string name, std::string db_path, int dimensions) {
    if (!hal::is_safe_identifier(name))
        return fail<std::shared_ptr<DataSource>>(ErrorCode::InvalidConfig, "sqlite-vec: name must match [A-Za-z_][A-Za-z0-9_]{0,63}.");
    if (dimensions <= 0) return fail<std::shared_ptr<DataSource>>(ErrorCode::InvalidConfig, "sqlite-vec: dimensions must be > 0.");

    if (db_path != ":memory:") {
        std::filesystem::path p(db_path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            if (ec)
                return fail<std::shared_ptr<DataSource>>(ErrorCode::FilesystemError,
                                                         "sqlite-vec: failed to create directory: " + ec.message());
        }
    }

    auto state = std::make_shared<SqliteVecState>();
    state->dimensions = dimensions;
    state->vec_table = name + "_vec";
    state->meta_table = name + "_meta";

    if (sqlite3_open_v2(db_path.c_str(), &state->conn, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) != SQLITE_OK) {
        std::string err = state->conn ? sqlite3_errmsg(state->conn) : "unknown error";
        return fail<std::shared_ptr<DataSource>>(ErrorCode::ProviderInitFailed, "sqlite-vec: failed to open db: " + err);
    }
    sqlite3_vec_init(state->conn, nullptr, nullptr);

    std::string create_vec =
        "CREATE VIRTUAL TABLE IF NOT EXISTS " + state->vec_table + " USING vec0(embedding FLOAT[" + std::to_string(dimensions) + "]);";
    if (auto r = exec_simple(state->conn, create_vec); !r.ok) return fail<std::shared_ptr<DataSource>>(r.error.code, r.error.message);

    std::string create_meta = "CREATE TABLE IF NOT EXISTS " + state->meta_table + " (rowid INTEGER PRIMARY KEY, text TEXT);";
    if (auto r = exec_simple(state->conn, create_meta); !r.ok) return fail<std::shared_ptr<DataSource>>(r.error.code, r.error.message);

    auto source = std::make_shared<DataSource>();
    source->name = std::move(name);
    source->state = state;
    source->query = sqlite_vec_query;
    source->user_data = nullptr;
    return ok(source);
}

Result<void> sqlite_vec_insert(DataSource &source, std::string_view text, std::span<const float> embedding) {
    auto *state = static_cast<SqliteVecState *>(source.state.get());
    if (!state) return fail(ErrorCode::InvalidConfig, "sqlite-vec: data source not initialized.");
    if (static_cast<int>(embedding.size()) != state->dimensions)
        return fail(ErrorCode::InvalidConfig, "sqlite-vec: embedding dimensionality (" + std::to_string(embedding.size()) +
                                                  ") does not match the data source's dimensions (" + std::to_string(state->dimensions) +
                                                  ").");

    std::lock_guard<std::mutex> lock(state->mutex);

    if (auto r = exec_simple(state->conn, "BEGIN;"); !r.ok) return r;

    std::string meta_sql = "INSERT INTO " + state->meta_table + "(text) VALUES (?);";
    sqlite3_stmt *meta_stmt = nullptr;
    if (sqlite3_prepare_v2(state->conn, meta_sql.c_str(), -1, &meta_stmt, nullptr) != SQLITE_OK) {
        exec_simple(state->conn, "ROLLBACK;");
        return fail(ErrorCode::FilesystemError, "sqlite-vec: meta insert prepare failed.");
    }
    sqlite3_bind_text(meta_stmt, 1, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT);
    if (sqlite3_step(meta_stmt) != SQLITE_DONE) {
        sqlite3_finalize(meta_stmt);
        exec_simple(state->conn, "ROLLBACK;");
        return fail(ErrorCode::FilesystemError, "sqlite-vec: meta insert failed.");
    }
    sqlite3_finalize(meta_stmt);
    sqlite3_int64 rowid = sqlite3_last_insert_rowid(state->conn);

    std::string vec_json = vector_to_json(embedding);
    std::string vec_sql = "INSERT INTO " + state->vec_table + "(rowid, embedding) VALUES (?, vec_f32(?));";
    sqlite3_stmt *vec_stmt = nullptr;
    if (sqlite3_prepare_v2(state->conn, vec_sql.c_str(), -1, &vec_stmt, nullptr) != SQLITE_OK) {
        exec_simple(state->conn, "ROLLBACK;");
        return fail(ErrorCode::FilesystemError, "sqlite-vec: vector insert prepare failed.");
    }
    sqlite3_bind_int64(vec_stmt, 1, rowid);
    sqlite3_bind_text(vec_stmt, 2, vec_json.data(), static_cast<int>(vec_json.size()), SQLITE_TRANSIENT);
    if (sqlite3_step(vec_stmt) != SQLITE_DONE) {
        sqlite3_finalize(vec_stmt);
        exec_simple(state->conn, "ROLLBACK;");
        return fail(ErrorCode::FilesystemError, "sqlite-vec: vector insert failed.");
    }
    sqlite3_finalize(vec_stmt);

    if (auto r = exec_simple(state->conn, "COMMIT;"); !r.ok) {
        exec_simple(state->conn, "ROLLBACK;");
        return r;
    }
    return ok();
}

// -------------------------- VectorStoreOps over sqlite-vec --------------------------
// <name>_vectors: vec0(embedding float[d] distance_metric=cosine), keyed by rowid.
// <name>_chunks:  rowid -> (id UNIQUE, text, metadata).

namespace {

constexpr int kVecMaxK = 4096; // sqlite-vec's limit on a knn query's k

struct SqliteVectorStore {
    SqliteVecState db; // connection, table names, dimensions, mutex
    std::string sql_find, sql_insert_chunk, sql_update_chunk, sql_delete_chunk, sql_insert_vec, sql_delete_vec, sql_query;
};

// Finalizes on scope exit.
struct Stmt {
    sqlite3_stmt *s = nullptr;
    Stmt(sqlite3 *db, const std::string &sql) {
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &s, nullptr) != SQLITE_OK) s = nullptr;
    }
    ~Stmt() { sqlite3_finalize(s); }
    Stmt(const Stmt &) = delete;
    Stmt &operator=(const Stmt &) = delete;
};

Result<void> sqlite_error(sqlite3 *db, std::string_view what) {
    return fail(ErrorCode::FilesystemError, "sqlite-vec: " + std::string(what) + ": " + sqlite3_errmsg(db));
}

void bind_text(sqlite3_stmt *s, int i, std::string_view v) {
    sqlite3_bind_text(s, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT);
}

std::string column_text(sqlite3_stmt *s, int i) {
    const auto *t = reinterpret_cast<const char *>(sqlite3_column_text(s, i));
    const int len = sqlite3_column_bytes(s, i);
    return t && len > 0 ? std::string(t, static_cast<size_t>(len)) : std::string();
}

// rowid of id, 0 when absent, -1 on error.
sqlite3_int64 find_rowid(SqliteVectorStore &st, std::string_view id) {
    Stmt q(st.db.conn, st.sql_find);
    if (!q.s) return -1;
    bind_text(q.s, 1, id);
    const int rc = sqlite3_step(q.s);
    if (rc == SQLITE_ROW) return sqlite3_column_int64(q.s, 0);
    return rc == SQLITE_DONE ? 0 : -1;
}

Result<void> step_done(SqliteVectorStore &st, const std::string &sql, std::string_view what, sqlite3_int64 rowid, std::string_view a = {},
                       std::string_view b = {}, std::string_view c = {}, int n_text = 0) {
    Stmt q(st.db.conn, sql);
    if (!q.s) return sqlite_error(st.db.conn, what);
    int i = 1;
    if (rowid != 0) sqlite3_bind_int64(q.s, i++, rowid);
    const std::string_view texts[] = {a, b, c};
    for (int t = 0; t < n_text; ++t) bind_text(q.s, i++, texts[t]);
    if (sqlite3_step(q.s) != SQLITE_DONE) return sqlite_error(st.db.conn, what);
    return ok();
}

Result<std::shared_ptr<void>> vs_open(const VectorStoreConfig &config) {
    using R = std::shared_ptr<void>;
    if (!hal::is_safe_identifier(config.name))
        return fail<R>(ErrorCode::InvalidConfig, "sqlite-vec: VectorStoreConfig::name must match [A-Za-z_][A-Za-z0-9_]{0,63}.");
    if (config.dimensions <= 0) return fail<R>(ErrorCode::InvalidConfig, "sqlite-vec: dimensions must be > 0.");
    const std::string path = config.path.empty() ? std::string(":memory:") : config.path;
    if (path != ":memory:") {
        std::filesystem::path p(path);
        std::error_code ec;
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) return fail<R>(ErrorCode::FilesystemError, "sqlite-vec: failed to create directory: " + ec.message());
    }

    auto st = std::make_shared<SqliteVectorStore>();
    st->db.dimensions = config.dimensions;
    st->db.vec_table = config.name + "_vectors";
    st->db.meta_table = config.name + "_chunks";
    if (sqlite3_open_v2(path.c_str(), &st->db.conn, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr) !=
        SQLITE_OK) {
        std::string err = st->db.conn ? sqlite3_errmsg(st->db.conn) : "unknown error";
        return fail<R>(ErrorCode::ProviderInitFailed, "sqlite-vec: failed to open db: " + err);
    }
    sqlite3_vec_init(st->db.conn, nullptr, nullptr);

    const std::string &vec = st->db.vec_table, &chunks = st->db.meta_table;
    const std::string schema = "CREATE VIRTUAL TABLE IF NOT EXISTS " + vec + " USING vec0(embedding float[" +
                               std::to_string(config.dimensions) + "] distance_metric=cosine);" + "CREATE TABLE IF NOT EXISTS " + chunks +
                               " (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, text TEXT, metadata TEXT);";
    if (auto r = exec_simple(st->db.conn, schema); !r.ok) return fail<R>(r.error.code, r.error.message);

    st->sql_find = "SELECT rowid FROM " + chunks + " WHERE id = ?1;";
    st->sql_insert_chunk = "INSERT INTO " + chunks + "(id, text, metadata) VALUES (?1, ?2, ?3);";
    st->sql_update_chunk = "UPDATE " + chunks + " SET text = ?2, metadata = ?3 WHERE rowid = ?1;";
    st->sql_delete_chunk = "DELETE FROM " + chunks + " WHERE rowid = ?1;";
    st->sql_insert_vec = "INSERT INTO " + vec + "(rowid, embedding) VALUES (?1, ?2);";
    st->sql_delete_vec = "DELETE FROM " + vec + " WHERE rowid = ?1;";
    st->sql_query = "WITH knn AS (SELECT rowid, distance FROM " + vec +
                    " WHERE embedding MATCH ?1 AND k = ?2) SELECT c.id, c.text, c.metadata, knn.distance FROM knn JOIN " + chunks +
                    " c ON c.rowid = knn.rowid ORDER BY knn.distance;";
    return ok(R(std::move(st)));
}

Result<void> vs_upsert_locked(SqliteVectorStore &st, std::string_view id, std::string_view text, std::span<const float> embedding,
                              std::string_view metadata_json) {
    sqlite3_int64 rowid = find_rowid(st, id);
    if (rowid < 0) return sqlite_error(st.db.conn, "lookup failed");
    if (rowid > 0) {
        if (auto r = step_done(st, st.sql_update_chunk, "chunk update failed", rowid, text, metadata_json, {}, 2); !r.ok) return r;
        if (auto r = step_done(st, st.sql_delete_vec, "vector delete failed", rowid); !r.ok) return r;
    } else {
        if (auto r = step_done(st, st.sql_insert_chunk, "chunk insert failed", 0, id, text, metadata_json, 3); !r.ok) return r;
        rowid = sqlite3_last_insert_rowid(st.db.conn);
    }
    Stmt q(st.db.conn, st.sql_insert_vec);
    if (!q.s) return sqlite_error(st.db.conn, "vector insert prepare failed");
    sqlite3_bind_int64(q.s, 1, rowid);
    sqlite3_bind_blob(q.s, 2, embedding.data(), static_cast<int>(embedding.size_bytes()), SQLITE_TRANSIENT);
    if (sqlite3_step(q.s) != SQLITE_DONE) return sqlite_error(st.db.conn, "vector insert failed");
    return ok();
}

Result<void> vs_upsert(void *state, std::string_view id, std::string_view text, std::span<const float> embedding,
                       std::string_view metadata_json) {
    auto &st = *static_cast<SqliteVectorStore *>(state);
    if (id.empty()) return fail(ErrorCode::InvalidConfig, "sqlite-vec: id must not be empty.");
    if (static_cast<int>(embedding.size()) != st.db.dimensions)
        return fail(ErrorCode::InvalidConfig, "sqlite-vec: embedding has " + std::to_string(embedding.size()) + " dimensions, the store " +
                                                  std::to_string(st.db.dimensions) + ".");
    std::lock_guard<std::mutex> lock(st.db.mutex);
    if (auto r = exec_simple(st.db.conn, "BEGIN;"); !r.ok) return r;
    auto r = vs_upsert_locked(st, id, text, embedding, metadata_json);
    if (r.ok) r = exec_simple(st.db.conn, "COMMIT;");
    if (!r.ok) exec_simple(st.db.conn, "ROLLBACK;");
    return r;
}

Result<std::vector<VectorHit>> vs_query(void *state, std::span<const float> embedding, int k) {
    using R = std::vector<VectorHit>;
    auto &st = *static_cast<SqliteVectorStore *>(state);
    if (static_cast<int>(embedding.size()) != st.db.dimensions)
        return fail<R>(ErrorCode::InvalidConfig, "sqlite-vec: query has " + std::to_string(embedding.size()) + " dimensions, the store " +
                                                     std::to_string(st.db.dimensions) + ".");
    if (k <= 0) return ok(R{});
    std::lock_guard<std::mutex> lock(st.db.mutex);
    Stmt q(st.db.conn, st.sql_query);
    if (!q.s) {
        auto e = sqlite_error(st.db.conn, "query prepare failed");
        return fail<R>(e.error.code, e.error.message);
    }
    sqlite3_bind_blob(q.s, 1, embedding.data(), static_cast<int>(embedding.size_bytes()), SQLITE_TRANSIENT);
    sqlite3_bind_int(q.s, 2, std::min(k, kVecMaxK));
    R hits;
    int rc = 0;
    while ((rc = sqlite3_step(q.s)) == SQLITE_ROW)
        hits.push_back(VectorHit{.id = column_text(q.s, 0),
                                 .text = column_text(q.s, 1),
                                 .metadata_json = column_text(q.s, 2),
                                 .score = static_cast<float>(1.0 - sqlite3_column_double(q.s, 3))});
    if (rc != SQLITE_DONE) {
        auto e = sqlite_error(st.db.conn, "query failed");
        return fail<R>(e.error.code, e.error.message);
    }
    return ok(std::move(hits));
}

Result<bool> vs_erase(void *state, std::string_view id) {
    auto &st = *static_cast<SqliteVectorStore *>(state);
    std::lock_guard<std::mutex> lock(st.db.mutex);
    if (auto r = exec_simple(st.db.conn, "BEGIN;"); !r.ok) return fail<bool>(r.error.code, r.error.message);
    const sqlite3_int64 rowid = find_rowid(st, id);
    Result<void> r = rowid < 0 ? sqlite_error(st.db.conn, "lookup failed") : ok();
    if (r.ok && rowid > 0) r = step_done(st, st.sql_delete_vec, "vector delete failed", rowid);
    if (r.ok && rowid > 0) r = step_done(st, st.sql_delete_chunk, "chunk delete failed", rowid);
    if (r.ok) r = exec_simple(st.db.conn, "COMMIT;");
    if (!r.ok) {
        exec_simple(st.db.conn, "ROLLBACK;");
        return fail<bool>(r.error.code, r.error.message);
    }
    return ok(rowid > 0);
}

Result<void> vs_clear(void *state) {
    auto &st = *static_cast<SqliteVectorStore *>(state);
    std::lock_guard<std::mutex> lock(st.db.mutex);
    return exec_simple(st.db.conn, "BEGIN; DELETE FROM " + st.db.vec_table + "; DELETE FROM " + st.db.meta_table + "; COMMIT;");
}

constexpr VectorStoreOps kSqliteVectorStore{
    .open = vs_open, .upsert = vs_upsert, .query_topk = vs_query, .erase = vs_erase, .clear = vs_clear};

} // namespace

namespace store {

const VectorStoreOps *sqlite_vector_store() { return &kSqliteVectorStore; }

} // namespace store

} // namespace agent
