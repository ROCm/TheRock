// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "../diagnostics/rocfft_aot_transaction.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <utility>

using namespace rocfft_build_profile;

static void sql(sqlite3 *db, const char *command) {
  if (sqlite3_exec(db, command, nullptr, nullptr, nullptr) != SQLITE_OK)
    throw std::runtime_error(sqlite3_errmsg(db));
}
static int scalar(sqlite3 *db, const char *command) {
  sqlite3_stmt *stmt = nullptr;
  assert(sqlite3_prepare_v2(db, command, -1, &stmt, nullptr) == SQLITE_OK);
  assert(sqlite3_step(stmt) == SQLITE_ROW);
  int value = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return value;
}
static int deny_commit(void *, int action, const char *operation, const char *,
                       const char *, const char *) {
  return action == SQLITE_TRANSACTION && operation &&
                 std::strcmp(operation, "COMMIT") == 0
             ? SQLITE_DENY
             : SQLITE_OK;
}
int main(int argc, char **argv) {
  assert(argc == 2);
  assert(!private_aot_batch_requested("", "custom.db", true));
  assert(!private_aot_batch_requested("0", "", false));
  assert(private_aot_batch_requested("1", "", false));
  for (auto values : {std::pair{"1", "custom.db"}, std::pair{"yes", ""}}) {
    bool failed = false;
    try {
      private_aot_batch_requested(values.first, values.second, false);
    } catch (const std::invalid_argument &) {
      failed = true;
    }
    assert(failed);
  }
  assert(!private_aot_batch_requested("1", "", true));
  sqlite3 *db = nullptr, *observer = nullptr;
  assert(sqlite3_open(argv[1], &db) == SQLITE_OK);
  sql(db, "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; "
          "CREATE TABLE cache_v1(k TEXT PRIMARY KEY, code BLOB)");
  assert(sqlite3_open(argv[1], &observer) == SQLITE_OK);
  sql(db, "INSERT INTO cache_v1 VALUES('baseline', X'0001ff')");
  assert(scalar(observer, "SELECT count(*) FROM cache_v1") == 1);
  private_aot_transaction(db, true);
  sql(db, "INSERT INTO cache_v1 VALUES('candidate', X'ff007f')");
  assert(scalar(db, "SELECT count(*) FROM cache_v1") == 2);
  assert(scalar(observer, "SELECT count(*) FROM cache_v1") == 1);
  assert(scalar(db, "PRAGMA synchronous") == 2);
  private_aot_transaction(db, false);
  assert(scalar(observer, "SELECT count(*) FROM cache_v1") == 2);
  assert(
      scalar(observer,
             "SELECT hex(code)='FF007F' FROM cache_v1 WHERE k='candidate'") ==
      1);
  assert(scalar(db, "PRAGMA synchronous") == 2);

  // A real SQLite commit error must propagate, never report successful output.
  private_aot_transaction(db, true);
  sql(db, "INSERT INTO cache_v1 VALUES('must_rollback', X'1234')");
  sqlite3_set_authorizer(db, deny_commit, nullptr);
  bool failed = false;
  try {
    private_aot_transaction(db, false);
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed && !sqlite3_get_autocommit(db));
  sqlite3_set_authorizer(db, nullptr, nullptr);
  // Closing an uncommitted private cache rolls back the entire incomplete
  // batch.
  assert(sqlite3_close(db) == SQLITE_OK);
  assert(scalar(observer, "SELECT count(*) FROM cache_v1") == 2);
  assert(sqlite3_close(observer) == SQLITE_OK);
  std::cout << "private guards, visibility, blob identity, FULL policy, commit "
               "failure and rollback passed\n";
}
