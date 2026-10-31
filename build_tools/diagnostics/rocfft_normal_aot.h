// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

// Experiment only: a private, regenerable AOT intermediate. Runtime caches and
// caller-supplied paths must keep their existing transaction/durability policy.
#include "sqlite3.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocfft_build_profile {
struct AotCacheSelection {
  std::string path;
  bool owned;
  bool existed;
};

// Accidental-reuse guard for a controller-created directory, not a security
// boundary against an adversarial process changing files concurrently.
inline AotCacheSelection select_private_aot_cache(const std::string &supplied,
                                                  const std::string &owned_path,
                                                  const std::string &owner_id) {
  namespace fs = std::filesystem;
  if (owned_path.empty()) {
    if (!owner_id.empty())
      throw std::invalid_argument("AOT owner ID requires an owned cache path");
    const auto path = supplied.empty() ? "rocfft_temp_cache.db" : supplied;
    // Preserve the frozen legacy probe, including for caller-supplied paths.
    return {path, false, fs::exists("rocfft_temp_cache.db")};
  }
  if (!supplied.empty() || owner_id.size() != 32 ||
      !std::all_of(owner_id.begin(), owner_id.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    throw std::invalid_argument(
        "Owned AOT cache requires default argv and owner ID");
  const fs::path path(owned_path), directory = path.parent_path();
  const auto marker = directory / ".rocfft-aot-owner";
  if (!path.is_absolute() || path.filename() != "rocfft_temp_cache.db" ||
      fs::is_symlink(directory) || !fs::is_directory(directory) ||
      fs::canonical(directory) != directory.lexically_normal() ||
      fs::is_symlink(marker) || !fs::is_regular_file(marker) ||
      fs::hard_link_count(marker) != 1 || fs::is_symlink(path) ||
      (fs::exists(path) &&
       (!fs::is_regular_file(path) || fs::hard_link_count(path) != 1)))
    throw std::invalid_argument("Invalid owned AOT cache path or marker");
  for (const char *suffix : {"-wal", "-shm"}) {
    const fs::path sidecar(path.string() + suffix);
    if (fs::is_symlink(sidecar) ||
        (fs::exists(sidecar) &&
         (!fs::is_regular_file(sidecar) || fs::hard_link_count(sidecar) != 1)))
      throw std::invalid_argument("Invalid owned AOT cache sidecar");
  }
  std::ifstream input(marker, std::ios::binary);
  std::array<char, 34> text{};
  input.read(text.data(), text.size());
  if (input.gcount() != 33 || std::string(text.data(), 33) != owner_id + "\n")
    throw std::invalid_argument(
        "Owned AOT cache marker does not match owner ID");
  return {path.string(), true, fs::exists(path)};
}

inline bool private_aot_batch_requested(const std::string &setting,
                                        const std::string &supplied_path,
                                        bool default_cache_exists,
                                        bool owned = false) {
  if (setting.empty() || setting == "0")
    return false;
  if (setting == "all") {
    if (!owned)
      throw std::invalid_argument(
          "All-helper batching requires an owned cache");
    return true;
  }
  if (setting != "1" || !supplied_path.empty())
    throw std::invalid_argument("AOT batching requires value 1 and the "
                                "helper-owned default temp cache");
  // Later architecture helpers reuse the growing private DB with original
  // autocommit behavior. Only the first fresh helper is the experimental arm.
  return !default_cache_exists;
}

inline void private_aot_transaction(sqlite3 *db, bool begin) {
  if (!db || (sqlite3_get_autocommit(db) != 0) != begin)
    throw std::runtime_error("Unexpected private AOT transaction state");
  // Do not change journal_mode, synchronous, busy timeout or checkpoint policy.
  // A failure aborts the helper; SQLite rolls back an unfinished private batch
  // when its connection is closed. No incomplete output is published as
  // success.
  const int result = sqlite3_exec(db, begin ? "BEGIN IMMEDIATE" : "COMMIT",
                                  nullptr, nullptr, nullptr);
  if (result != SQLITE_OK)
    throw std::runtime_error(std::string("Private AOT transaction failed: ") +
                             sqlite3_errmsg(db));
  if ((sqlite3_get_autocommit(db) != 0) != !begin)
    throw std::runtime_error(
        "Private AOT transaction did not reach expected state");
}
inline std::string quote(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32) {
      const char *hex = "0123456789abcdef";
      out << "\\u00" << hex[c >> 4] << hex[c & 15];
    } else
      out << c;
  }
  out << '"';
  return out.str();
}

inline std::string pragma_text(sqlite3 *db, const char *query) {
  sqlite3_stmt *raw = nullptr;
  if (!db || sqlite3_prepare_v2(db, query, -1, &raw, nullptr) != SQLITE_OK)
    throw std::runtime_error("Cannot read private AOT SQLite policy");
  std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(
      raw, sqlite3_finalize);
  if (sqlite3_step(raw) != SQLITE_ROW || !sqlite3_column_text(raw, 0))
    throw std::runtime_error("Missing private AOT SQLite policy");
  return reinterpret_cast<const char *>(sqlite3_column_text(raw, 0));
}

inline std::string private_aot_policy(sqlite3 *db) {
  const auto journal = pragma_text(db, "PRAGMA journal_mode");
  const auto sync = pragma_text(db, "PRAGMA synchronous");
  const auto checkpoint = pragma_text(db, "PRAGMA wal_autocheckpoint");
  const auto page = pragma_text(db, "PRAGMA page_size");
  if (journal != "wal" || sync != "2" || !sqlite3_get_autocommit(db))
    throw std::runtime_error(
        "Owned AOT cache requires WAL/FULL and no open transaction");
  return "{\"journal_mode\":" + quote(journal) +
         ",\"synchronous\":" + quote(sync) +
         ",\"wal_autocheckpoint\":" + quote(checkpoint) +
         ",\"page_size\":" + quote(page) + ",\"autocommit\":true}";
}

inline std::string environment(const char *name) {
  const char *value = std::getenv(name);
  return value ? value : "";
}
struct NormalFallbackErrors {
  std::atomic<unsigned> count{0};
  std::mutex mutex;
  std::string first;
};
inline NormalFallbackErrors &normal_fallback_errors() {
  static NormalFallbackErrors errors;
  return errors;
}
inline void note_normal_fallback(const char *reason) {
  if (environment("ROCFFT_AOT_PRIVATE_CACHE_PATH").empty())
    return;
  auto &errors = normal_fallback_errors();
  errors.count.fetch_add(1);
  std::lock_guard<std::mutex> lock(errors.mutex);
  if (errors.first.empty())
    errors.first = std::string(reason).substr(0, 160);
}
// End-only receipts; no per-kernel counters, timers or generated-source logs.
struct NormalAotContext {
  bool enabled = false, batch = false, began = false, committed = false;
  AotCacheSelection cache{"", false, false};
  std::string owner, arch, mode, before_policy, after_policy;
  std::filesystem::path receipts;

  NormalAotContext(const std::string &supplied,
                   const std::vector<std::string> &arches) {
    const auto path = environment("ROCFFT_AOT_PRIVATE_CACHE_PATH");
    owner = environment("ROCFFT_AOT_PRIVATE_CACHE_ID");
    mode = environment("ROCFFT_AOT_BATCH_TRANSACTION");
    const auto directory = environment("ROCFFT_AOT_RECEIPT_DIR");
    if (path.empty() && owner.empty() && directory.empty() &&
        (mode.empty() || mode == "0"))
      return; // Legacy/custom behavior, without a new filesystem probe.
    if (path.empty() || (mode != "0" && mode != "all") || arches.size() != 1)
      throw std::invalid_argument("Incomplete owned AOT normal-build controls");
    cache = select_private_aot_cache(supplied, path, owner);
    arch = arches.front();
    if (arch.empty() ||
        !std::all_of(arch.begin(), arch.end(), [](unsigned char c) {
          return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' ||
                 c == '_';
        }))
      throw std::invalid_argument("Invalid owned AOT architecture name");
    receipts = directory;
    namespace fs = std::filesystem;
    if (!receipts.is_absolute() || fs::is_symlink(receipts) ||
        !fs::is_directory(receipts) ||
        fs::canonical(receipts) != receipts.lexically_normal())
      throw std::invalid_argument("Invalid owned AOT receipt directory");
    enabled = true;
    batch =
        private_aot_batch_requested(mode, supplied, cache.existed, cache.owned);
  }

  void publish_success(const std::string &output) const {
    if (!enabled)
      return;
    namespace fs = std::filesystem;
    if (began != batch || committed != batch || before_policy.empty() ||
        after_policy.empty())
      throw std::runtime_error(
          "Owned AOT transaction/output gate is incomplete");
    select_private_aot_cache("", cache.path, owner);
    if (fs::is_symlink(output) || !fs::is_regular_file(output) ||
        fs::hard_link_count(output) != 1 || !fs::file_size(output))
      throw std::runtime_error(
          "Owned AOT output was not successfully published");
    const auto final = receipts / (arch + ".json");
    const auto pending = receipts / (arch + ".pending");
    if (fs::exists(final) || fs::is_symlink(final) ||
        !fs::create_directory(pending))
      throw std::runtime_error("Owned AOT receipt already exists");
    const auto temporary = pending / "receipt.json";
    std::ofstream stream(temporary, std::ios::binary);
    stream << "{\"schema_version\":1,\"success\":true,\"arch\":" << quote(arch)
           << ",\"mode\":" << quote(mode) << ",\"owner_id\":" << quote(owner)
           << ",\"cache_path\":" << quote(cache.path)
           << ",\"cache_existed_before\":" << (cache.existed ? "true" : "false")
           << ",\"begin_succeeded\":" << (began ? "true" : "false")
           << ",\"commit_succeeded\":" << (committed ? "true" : "false")
           << ",\"policy_before\":" << before_policy
           << ",\"policy_after\":" << after_policy
           << ",\"comgr_cache\":" << quote(environment("AMD_COMGR_CACHE"))
           << ",\"comgr_cache_dir\":"
           << quote(environment("AMD_COMGR_CACHE_DIR"))
           << ",\"comgr_cache_policy\":"
           << quote(environment("AMD_COMGR_CACHE_POLICY"))
           << ",\"subprocess_fallback_errors\":"
           << normal_fallback_errors().count.load()
           << ",\"first_fallback_error\":"
           << quote(normal_fallback_errors().first)
           << ",\"output_path\":" << quote(output)
           << ",\"output_bytes\":" << fs::file_size(output)
           << ",\"cache_bytes\":" << fs::file_size(cache.path) << "}\n";
    stream.flush();
    stream.close();
    if (!stream)
      throw std::runtime_error("Cannot write owned AOT completion receipt");
    fs::rename(temporary, final);
    fs::remove(pending);
  }
};
} // namespace rocfft_build_profile
