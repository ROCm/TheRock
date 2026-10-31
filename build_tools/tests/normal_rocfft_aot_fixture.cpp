// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "../diagnostics/rocfft_normal_aot.h"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace rocfft_build_profile;

static void sql(sqlite3 *db, const char *text) {
  if (sqlite3_exec(db, text, nullptr, nullptr, nullptr) != SQLITE_OK)
    throw std::runtime_error(sqlite3_errmsg(db));
}
static int count(sqlite3 *db, const char *query) {
  return std::stoi(pragma_text(db, query));
}
static int deny_commit(void *, int action, const char *operation, const char *,
                       const char *, const char *) {
  return action == SQLITE_TRANSACTION && operation &&
                 std::strcmp(operation, "COMMIT") == 0
             ? SQLITE_DENY
             : SQLITE_OK;
}
static void rollback_fixture(const std::filesystem::path &path) {
  sqlite3 *db = nullptr, *observer = nullptr;
  assert(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK);
  sql(db, "PRAGMA journal_mode=WAL;PRAGMA synchronous=FULL;CREATE TABLE "
          "t(k);INSERT INTO t VALUES(1)");
  assert(sqlite3_open(path.string().c_str(), &observer) == SQLITE_OK);
  private_aot_policy(db);
  private_aot_transaction(db, true);
  sql(db, "INSERT INTO t VALUES(2)");
  assert(count(observer, "SELECT count(*) FROM t") == 1);
  sqlite3_set_authorizer(db, deny_commit, nullptr);
  bool copied = false, failed = false;
  try {
    private_aot_transaction(db, false);
    copied = true; // The production patch also places output copy AFTER commit.
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed && !copied && !sqlite3_get_autocommit(db));
  sqlite3_set_authorizer(db, nullptr, nullptr);
  assert(sqlite3_close(db) == SQLITE_OK);
  assert(count(observer, "SELECT count(*) FROM t") == 1);
  assert(pragma_text(observer, "PRAGMA synchronous") == "2");
  assert(sqlite3_close(observer) == SQLITE_OK);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const std::filesystem::path output_dir(argv[1]);
  // Disabled/default and caller-supplied paths do not add filesystem probes.
  const char *controls[] = {
      "ROCFFT_AOT_PRIVATE_CACHE_PATH", "ROCFFT_AOT_PRIVATE_CACHE_ID",
      "ROCFFT_AOT_BATCH_TRANSACTION", "ROCFFT_AOT_RECEIPT_DIR"};
  std::vector<std::string> saved;
  for (const auto name : controls) {
    saved.push_back(environment(name));
    unsetenv(name);
  }
  NormalAotContext legacy("/missing/custom/parent/cache.db", {});
  assert(!legacy.enabled);
  for (size_t i = 0; i < 4; ++i)
    setenv(controls[i], saved[i].c_str(), 1);
  rollback_fixture(output_dir / "rollback.db");
  const char *sum =
      "5a96d3dd1f2fdcbfdacf246a427050105a2b7b3b68b73c6a387d343db92d8653";
  std::vector<unsigned char> generator;
  for (size_t i = 0; i < 64; i += 2)
    generator.push_back(std::stoul(std::string(sum + i, 2), nullptr, 16));
  for (int i = 0; i < 7; ++i) {
    const auto arch = "gfx103" + std::to_string(i);
    NormalAotContext context("", {arch});
    assert(context.enabled && context.cache.existed == (i != 0));
    sqlite3 *db = nullptr;
    assert(sqlite3_open(context.cache.path.c_str(), &db) == SQLITE_OK);
    sql(db,
        "PRAGMA journal_mode=WAL;PRAGMA synchronous=FULL;"
        "CREATE TABLE IF NOT EXISTS "
        "cache_v1(kernel_name,arch,hip_version,generator_sum,code,timestamp)");
    context.before_policy = private_aot_policy(db);
    if (context.batch) {
      private_aot_transaction(db, true);
      context.began = true;
    }
    sqlite3_stmt *stmt = nullptr;
    assert(sqlite3_prepare_v2(
               db, "INSERT INTO cache_v1 VALUES('kernel',?,123,?,X'00ff7f',0)",
               -1, &stmt, nullptr) == SQLITE_OK);
    sqlite3_bind_text(stmt, 1, arch.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 2, generator.data(), generator.size(),
                      SQLITE_TRANSIENT);
    assert(sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    if (context.batch) {
      private_aot_transaction(db, false);
      context.committed = true;
    }
    context.after_policy = private_aot_policy(db);
    assert(count(db, "SELECT count(*) FROM cache_v1") == i + 1);
    const auto output = output_dir / ("rocfft_kernel_cache_" + arch + ".db");
    assert(sqlite3_prepare_v2(db, "ATTACH DATABASE ? AS published", -1, &stmt,
                              nullptr) == SQLITE_OK);
    sqlite3_bind_text(stmt, 1, output.string().c_str(), -1, SQLITE_TRANSIENT);
    assert(sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    sql(db, ("CREATE TABLE published.cache_v1 AS SELECT * FROM main.cache_v1 "
             "WHERE arch='" +
             arch + "';DETACH published")
                .c_str());
    assert(sqlite3_close(db) == SQLITE_OK);
    context.publish_success(output.string());
  }
  // Ownership validation must reject aliases without touching the original DB.
  const auto cache =
      std::filesystem::path(environment("ROCFFT_AOT_PRIVATE_CACHE_PATH"));
  const auto alias = cache.parent_path() / "fixture-alias";
  std::filesystem::create_hard_link(cache, alias);
  bool rejected = false;
  try {
    NormalAotContext invalid("", {"gfx1030"});
  } catch (const std::exception &) {
    rejected = true;
  }
  assert(rejected);
  std::filesystem::remove(alias);
  const auto metadata =
      std::filesystem::path(environment("ROCFFT_AOT_RECEIPT_DIR")) /
      "gfx1030.json";
  assert(std::filesystem::is_regular_file(metadata));
  std::cout << "real SQLite FULL visibility/COMMIT rollback, seven "
               "growing-cache outputs, owner guard and receipts passed\n";
}
