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
static void owned_chain_guards(const std::filesystem::path &root) {
  namespace fs = std::filesystem;
  const std::string id(32, 'a');
  const auto directory = fs::canonical(root) / "owned";
  fs::create_directory(directory);
  const auto path = directory / "rocfft_temp_cache.db";
  const auto marker = directory / ".rocfft-aot-owner";
  auto write_marker = [&](const std::string &text) {
    std::ofstream stream(marker, std::ios::binary | std::ios::trunc);
    stream << text;
  };
  auto reject = [&](const std::string &supplied, const std::string &cache,
                    const std::string &owner) {
    bool failed = false;
    try {
      select_private_aot_cache(supplied, cache, owner);
    } catch (const std::exception &) {
      failed = true;
    }
    assert(failed);
  };
  reject("", path.string(), id); // Missing marker must not create/adopt it.
  write_marker(id + "\n");
  const auto first = select_private_aot_cache("", path.string(), id);
  assert(first.owned && !first.existed && first.path == path.string());
  assert(private_aot_batch_requested("all", "", false, first.owned));
  std::ofstream(path, std::ios::binary) << "growing private DB";
  const auto later = select_private_aot_cache("", path.string(), id);
  assert(later.owned && later.existed);
  assert(private_aot_batch_requested("all", "", true, later.owned));
  assert(!private_aot_batch_requested("1", "", true, later.owned));
  assert(!private_aot_batch_requested("0", "", true, later.owned));
  reject("custom.db", path.string(), id);
  reject("", path.string(), std::string(32, 'A'));
  reject("", "", id);
  reject("", (directory / "other.db").string(), id);
  reject("", "relative/rocfft_temp_cache.db", id);
  write_marker(id + "\nextra");
  reject("", path.string(), id);
  write_marker(id + "x");
  reject("", path.string(), id);
  write_marker(id + "\n");
  bool failed = false;
  try {
    private_aot_batch_requested("all", "", false);
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed);
  const auto alias = directory / "alias";
  for (const auto &guarded : {marker, path, fs::path(path.string() + "-wal"),
                              fs::path(path.string() + "-shm")}) {
    if (!fs::exists(guarded))
      std::ofstream(guarded) << "private sidecar";
    fs::create_hard_link(guarded, alias);
    reject("", path.string(), id);
    fs::remove(alias);
    // Real symlink guards are mandatory on Linux. Windows developer accounts
    // may lack symlink privileges; report that separately rather than faking
    // it.
    fs::rename(guarded, alias);
    std::error_code error;
    fs::create_symlink(alias, guarded, error);
    if (!error) {
      reject("", path.string(), id);
      fs::remove(guarded);
    } else {
#ifndef _WIN32
      assert(!error);
#else
      std::cout << "Windows symlink fixture unavailable: " << error.message()
                << '\n';
#endif
    }
    fs::rename(alias, guarded);
  }
  assert(select_private_aot_cache("", path.string(), id).existed);
  assert(select_private_aot_cache("custom.db", "", "").path == "custom.db");
  // Guard validation neither rewrites nor truncates a growing chain.
  std::ifstream original(path);
  std::string content((std::istreambuf_iterator<char>(original)), {});
  assert(content == "growing private DB");
}
int main(int argc, char **argv) {
  assert(argc == 2);
  owned_chain_guards(std::filesystem::path(argv[1]).parent_path());
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
