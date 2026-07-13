#include "../utils/utils.hpp"
#include "hal/path.hpp"

#include <agent-cpp/agent.hpp>

#include <sqlite-vec.h>
#include <sqlite3.h>
#include <simdjson.h>

#include <filesystem>
#include <mutex>
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
        if (conn)
            sqlite3_close(conn);
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
        if (i > 0)
            s += ',';
        s += std::to_string(v[i]);
    }
    s += ']';
    return s;
}

Result<std::string> sqlite_vec_query(void *state_ptr, std::string_view query, void *) {
    auto *state = static_cast<SqliteVecState *>(state_ptr);
    if (!state)
        return fail<std::string>(ErrorCode::InvalidConfig, "sqlite-vec: data source not initialized.");

    simdjson::ondemand::parser parser;
    simdjson::padded_string json(query.data(), query.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc))
        return fail<std::string>(ErrorCode::ParseError, "sqlite-vec: invalid query JSON.");

    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj))
        return fail<std::string>(ErrorCode::ParseError, "sqlite-vec: expected a JSON object.");

    std::vector<float> vec;
    int64_t top_k = 5;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        if (k.value() == "vector") {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr))
                continue;
            for (auto elem : arr) {
                double d = 0.0;
                if (!elem.get_double().get(d))
                    vec.push_back(static_cast<float>(d));
            }
        } else if (k.value() == "top_k") {
            int64_t v;
            if (!field.value().get_int64().get(v) && v > 0)
                top_k = v;
        }
    }

    if (vec.empty())
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "sqlite-vec: query requires a non-empty 'vector'.");
    if (static_cast<int>(vec.size()) != state->dimensions)
        return fail<std::string>(ErrorCode::InvalidConfig,
                                 "sqlite-vec: query vector dimensionality (" +
                                     std::to_string(vec.size()) + ") does not match the data "
                                     "source's dimensions (" + std::to_string(state->dimensions) +
                                     ").");

    std::string vec_json = vector_to_json(vec);
    std::lock_guard<std::mutex> lock(state->mutex);

    std::string sql = "SELECT rowid, distance FROM " + state->vec_table +
                      " WHERE embedding MATCH vec_f32(?) ORDER BY distance LIMIT ?;";
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
        if (sqlite3_prepare_v2(state->conn, meta_sql.c_str(), -1, &meta_stmt, nullptr) ==
            SQLITE_OK) {
            sqlite3_bind_int64(meta_stmt, 1, rowid);
            if (sqlite3_step(meta_stmt) == SQLITE_ROW) {
                const auto *t = reinterpret_cast<const char *>(sqlite3_column_text(meta_stmt, 0));
                int len = sqlite3_column_bytes(meta_stmt, 0);
                if (t && len > 0)
                    text.assign(t, static_cast<size_t>(len));
            }
            sqlite3_finalize(meta_stmt);
        }

        if (!first)
            result += ',';
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

Result<std::shared_ptr<DataSource>> make_sqlite_vec_data_source(std::string name,
                                                                std::string db_path,
                                                                int dimensions) {
    if (!hal::is_safe_identifier(name))
        return fail<std::shared_ptr<DataSource>>(
            ErrorCode::InvalidConfig,
            "sqlite-vec: name must match [A-Za-z_][A-Za-z0-9_]{0,63}.");
    if (dimensions <= 0)
        return fail<std::shared_ptr<DataSource>>(ErrorCode::InvalidConfig,
                                                 "sqlite-vec: dimensions must be > 0.");

    if (db_path != ":memory:") {
        std::filesystem::path p(db_path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            if (ec)
                return fail<std::shared_ptr<DataSource>>(
                    ErrorCode::FilesystemError,
                    "sqlite-vec: failed to create directory: " + ec.message());
        }
    }

    auto state = std::make_shared<SqliteVecState>();
    state->dimensions = dimensions;
    state->vec_table = name + "_vec";
    state->meta_table = name + "_meta";

    if (sqlite3_open_v2(db_path.c_str(), &state->conn, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                        nullptr) != SQLITE_OK) {
        std::string err = state->conn ? sqlite3_errmsg(state->conn) : "unknown error";
        return fail<std::shared_ptr<DataSource>>(ErrorCode::ProviderInitFailed,
                                                 "sqlite-vec: failed to open db: " + err);
    }
    sqlite3_vec_init(state->conn, nullptr, nullptr);

    std::string create_vec = "CREATE VIRTUAL TABLE IF NOT EXISTS " + state->vec_table +
                             " USING vec0(embedding FLOAT[" + std::to_string(dimensions) + "]);";
    if (auto r = exec_simple(state->conn, create_vec); !r.ok)
        return fail<std::shared_ptr<DataSource>>(r.error.code, r.error.message);

    std::string create_meta = "CREATE TABLE IF NOT EXISTS " + state->meta_table +
                              " (rowid INTEGER PRIMARY KEY, text TEXT);";
    if (auto r = exec_simple(state->conn, create_meta); !r.ok)
        return fail<std::shared_ptr<DataSource>>(r.error.code, r.error.message);

    auto source = std::make_shared<DataSource>();
    source->name = std::move(name);
    source->state = state;
    source->query = sqlite_vec_query;
    source->user_data = nullptr;
    return ok(source);
}

Result<void> sqlite_vec_insert(DataSource &source, std::string_view text,
                               std::span<const float> embedding) {
    auto *state = static_cast<SqliteVecState *>(source.state.get());
    if (!state)
        return fail(ErrorCode::InvalidConfig, "sqlite-vec: data source not initialized.");
    if (static_cast<int>(embedding.size()) != state->dimensions)
        return fail(ErrorCode::InvalidConfig,
                    "sqlite-vec: embedding dimensionality (" + std::to_string(embedding.size()) +
                        ") does not match the data source's dimensions (" +
                        std::to_string(state->dimensions) + ").");

    std::lock_guard<std::mutex> lock(state->mutex);

    if (auto r = exec_simple(state->conn, "BEGIN;"); !r.ok)
        return r;

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
    std::string vec_sql =
        "INSERT INTO " + state->vec_table + "(rowid, embedding) VALUES (?, vec_f32(?));";
    sqlite3_stmt *vec_stmt = nullptr;
    if (sqlite3_prepare_v2(state->conn, vec_sql.c_str(), -1, &vec_stmt, nullptr) != SQLITE_OK) {
        exec_simple(state->conn, "ROLLBACK;");
        return fail(ErrorCode::FilesystemError, "sqlite-vec: vector insert prepare failed.");
    }
    sqlite3_bind_int64(vec_stmt, 1, rowid);
    sqlite3_bind_text(vec_stmt, 2, vec_json.data(), static_cast<int>(vec_json.size()),
                      SQLITE_TRANSIENT);
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

} // namespace agent
