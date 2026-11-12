#pragma once

// One place for how agent.cpp opens SQLite files: durable, quick, and small in RAM.
//
//  * WAL journal. Writes append to <db>-wal and readers never block the writer. With synchronous=NORMAL a commit is
//    durable against an application crash, and a power cut loses at most the last few commits but never corrupts the
//    database (the same trade-off most WAL deployments make). The <db>-shm file is the WAL index, memory-mapped by SQLite.
//  * The page cache is capped at 2 MiB per connection; reads go through a 64 MiB read-only memory map instead, which is
//    the operating system's page cache: shared between connections, counted as reclaimable file cache, not heap.
//  * Temporary tables and sort files go to disk, the WAL is truncated after checkpoints (journal_size_limit), and a busy
//    database waits 5 s before failing.

#include <sqlite3.h>

namespace agent::store {

inline void apply_sqlite_tuning(sqlite3 *db, bool writer) {
    if (writer) {
        sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA wal_autocheckpoint=1000;", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "PRAGMA journal_size_limit=16777216;", nullptr, nullptr, nullptr);
    }
    sqlite3_exec(db, "PRAGMA busy_timeout=5000;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA cache_size=-2048;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA mmap_size=67108864;", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA temp_store=FILE;", nullptr, nullptr, nullptr);
}

} // namespace agent::store
